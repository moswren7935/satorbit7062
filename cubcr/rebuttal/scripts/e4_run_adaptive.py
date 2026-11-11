#!/usr/bin/env python3


from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import numpy as np
import pandas as pd

import rb
from e4_common import (RAW, LOGS, SEEDS30, PAPER_SEEDS, game_list, load_games, table_context, write_csv_gz,
                       run_chunks, nice_worker)
from e4_estimators import metrics, fmt_vec

OUTDIR = RAW / "adaptive"
EPS_DIVS = (20, 100)
OC_KW = dict(batch_perms=16, alpha=0.05, max_perms=2_000_000, time_budget=1800.0, antithetic=True,
             value_range="monotone", two_sided=True, bonferroni=True, sequential=True, ci="prpl_eb_cs",
             batch_schedule="geometric")


def stored_record(g, seed):

    if seed not in PAPER_SEEDS:
        return None, None
    if g["family"] == "overcooked":
        p = rb.REPO / "results" / "e14" / f"e14__overcooked_cramped_room_k10__cert_gpu_eps_vs_exact_s{seed}.json"
        if not p.exists():
            return None, None
        r = rb._read(p)
        return {"estimates": np.asarray(r["estimates"], dtype=np.float64).reshape(-1),
                "half_widths": np.asarray(r["half_widths"], dtype=np.float64).reshape(-1),
                "n_perms": r["n_perms"], "n_solves": r["n_solves"], "batch_sizes": r.get("batch_sizes"),
                "eps": r["eps_abs"]}, str(p.relative_to(rb.REPO))
    if not g.get("paper_task"):
        return None, None
    for r in rb.load_e9v3(arms=["cert_gpu"]):
        if r["instance_id"] == g["table_id"] and int(r["seed"]) == int(seed):
            return {"estimates": np.asarray(r["shapley_sampled"], dtype=np.float64),
                    "half_widths": np.asarray(r["half_widths"], dtype=np.float64), "n_perms": r["n_perms"],
                    "n_solves": r["n_solves"], "batch_sizes": r["batch_sizes"], "eps": r["eps_abs"]}, r["_file"]
    return None, None


def run_chunk(table_id: str, eps_div: int, seeds: list, out: str):
    nice_worker()
    from cubcr.shapley_sampled import sample_shapley
    g = load_games()[table_id]
    ctx = table_context(table_id)
    d = ctx.d
    ups = float(ctx.table[-1] - ctx.table[0])
    eps = ups / eps_div
    rows = []
    t_chunk = time.perf_counter()
    for seed in seeds:
        eng = rb.TableEngine(ctx.table, compiled=None)
        if g["family"] == "overcooked":
            kw = dict(OC_KW, kind="agent", n=d, k=ctx.k, seed=int(seed), eps=eps, fp_bound=ctx.fp)
            config = "E14 overcooked k=10 (batch_perms=16), 20-player bitset form"
        else:
            kw = rb.paper_sampler_kwargs(eps, seed, d, ctx.k, fp_bound=ctx.fp)
            config = "paper (rb.PAPER_SAMPLER_KW)"
        t0 = time.perf_counter()
        r = sample_shapley(eng, **kw)
        wall = time.perf_counter() - t0
        hw = np.asarray(r.half_widths, dtype=np.float64)
        row = {"table_id": table_id, "d": d, "family": g["family"], "eps_div": eps_div, "eps": eps, "upsilon": ups,
               "seed": int(seed), "paper_seed": int(seed) in PAPER_SEEDS, "config": config,
               "status": r.status, "certified": bool(r.certified), "n_units": int(r.n_samples),
               "n_perms": int(r.n_perms), "n_requested": int(r.n_coalitions_requested), "n_unique": int(r.n_solves),
               "unique_frac": float(r.n_solves) / float(1 << d), "n_batches": int(r.n_batches),
               "batch_sizes": " ".join(map(str, r.batch_sizes)), "hw_max": float(np.max(hw)),
               "covers_exact": bool(np.all(np.abs(r.estimates - ctx.phi) <= hw)), "host_wall_s": wall,
               "fp_term": float(r.fp_term), "value_range": fmt_vec(r.value_range),
               "phi_hat": fmt_vec(r.estimates), "L": fmt_vec(r.cs_lower), "R": fmt_vec(r.cs_upper),
               "half_widths": fmt_vec(hw), "phi_exact": fmt_vec(ctx.phi)}
        row.update(metrics(r.estimates, ctx.phi, ups))
        row["max_abs_over_eps"] = row["max_abs"] / eps
        rec, src = stored_record(g, seed) if eps_div == 20 else (None, None)
        if rec is not None:
            row["stored_record"] = src
            row["stored_eps_equal"] = bool(rec["eps"] == eps)
            row["stored_match_estimates"] = bool(np.array_equal(rec["estimates"], r.estimates))
            row["stored_match_half_widths"] = bool(np.array_equal(rec["half_widths"], hw))
            row["stored_match_n_perms"] = bool(int(rec["n_perms"]) == int(r.n_perms))
            row["stored_match_n_solves"] = bool(int(rec["n_solves"]) == int(r.n_solves))
            row["stored_match_batch_sizes"] = (None if rec["batch_sizes"] is None
                                               else bool(list(rec["batch_sizes"]) == list(r.batch_sizes)))
            row["stored_max_est_dev"] = float(np.max(np.abs(rec["estimates"] - r.estimates)))
        rows.append(row)
    df = pd.DataFrame(rows)
    write_csv_gz(df, Path(out))
    rb.write_json(Path(out).with_suffix("").with_suffix(".json"),
                  {"table_id": table_id, "eps_div": eps_div, "seeds": list(map(int, seeds)), "created": rb.now_iso(),
                   "hardware": rb.hardware_tags(), "load_snapshot": rb.load_snapshot(),
                   "chunk_wall_s": time.perf_counter() - t_chunk,
                   "note": "host_wall_s is table-lookup time on a shared host (not an engine timing)"})
    st = df[[c for c in df.columns if c.startswith("stored_match")]]
    nm = int(st.all(axis=1).sum()) if len(st.columns) else 0
    return (f"{table_id} div{eps_div}: median units {int(df.n_units.median())}, median requested "
            f"{int(df.n_requested.median())}, unique {int(df.n_unique.median())} ({df.unique_frac.median():.3f} of 2^d); "
            f"certified {int(df.certified.sum())}/{len(df)}; stored-record matches {nm}/{int(df.get('stored_record', pd.Series(dtype=str)).notna().sum())}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--tables", nargs="*", default=None)
    ap.add_argument("--eps-divs", type=int, nargs="*", default=list(EPS_DIVS))
    ap.add_argument("--seeds", type=int, default=30)
    a = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    seeds = SEEDS30[:a.seeds]
    chunks = []
    for g in sorted(game_list(), key=lambda g: -g["d"]):
        if a.tables and g["table_id"] not in a.tables:
            continue
        for div in a.eps_divs:
            out = OUTDIR / f"{g['table_id']}__div{div}.csv.gz"
            if out.exists():
                continue
            chunks.append((g["table_id"], div, seeds, str(out)))
    run_chunks(chunks, run_chunk, a.workers, "E4 adaptive", LOGS / "adaptive.log")


if __name__ == "__main__":
    main()
