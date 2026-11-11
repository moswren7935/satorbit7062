#!/usr/bin/env python3


from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import numpy as np
import pandas as pd

import rb
from e4_common import (RAW, LOGS, SEEDS30, CORE_ESTIMATORS, EXTRA_ESTIMATORS, game_list, load_games, table_context,
                       write_csv_gz, run_chunks, nice_worker, log_line)
from e4_estimators import ESTIMATORS, run_estimator, metrics, budget_grid, fmt_vec, min_feasible_budget

OUTDIR = RAW / "grid"

COST = {"shapiq_SVARM": 6.6, "Adalina": 2.8, "shapiq_OwenSamplingSV": 2.7, "shapiq_kADDSHAP": 2.7,
        "shapiq_UnbiasedKernelSHAP": 2.4, "shapiq_KernelSHAP": 2.3, "shapiq_StratifiedSamplingSV": 2.0,
        "LeverageSHAP": 0.6, "shapiq_PermutationSamplingSV": 0.25, "StratifiedSVARM": 0.11,
        "cubcr_paired_perm": 0.05, "plain_perm": 0.05}


def run_one(est, ctx, g, T, seed) -> dict:
    row = {"table_id": ctx.table_id, "d": ctx.d, "family": g["family"], "estimator": est, "budget": int(T),
           "seed": int(seed)}
    try:
        o = run_estimator(est, ctx, T, seed)
    except Exception as e:
        row.update(status="ERROR", error=repr(e)[:300], traceback=traceback.format_exc()[-600:])
        return row
    row["status"] = o["status"]
    if o["status"] == "INFEASIBLE":
        row["min_feasible_budget"] = min_feasible_budget(est, ctx.d)
        return row
    row.update(n_total=o["n_total"], n_unique=o["n_unique"], n_units=o.get("n_units"), n_batches=o.get("n_batches"),
               host_wall_s=o["wall_s"], hw_max=o.get("hw_max"))
    row.update(metrics(o["phi_hat"], ctx.phi, ctx.upsilon))
    row["phi_hat"] = fmt_vec(o["phi_hat"])
    if o.get("L") is not None:
        row["L"] = fmt_vec(o["L"]); row["R"] = fmt_vec(o["R"])
    return row


def run_chunk(table_id: str, est: str, seeds: list, budgets: list, out: str):
    nice_worker()
    g = load_games()[table_id]
    ctx = table_context(table_id)
    t0 = time.perf_counter()
    rows = [run_one(est, ctx, g, T, s) for T in budgets for s in seeds]
    df = pd.DataFrame(rows)
    write_csv_gz(df, Path(out))
    nerr = int((df.status == "ERROR").sum())
    return f"{table_id} {est}: {len(df)} runs, {nerr} errors, {time.perf_counter() - t0:.1f}s"


def seeds_for(est: str, tiered: bool, n_core: int, n_extra: int) -> list:
    if tiered and est in EXTRA_ESTIMATORS:
        return SEEDS30[:n_extra]
    return SEEDS30[:n_core]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--estimators", nargs="*", default=ESTIMATORS)
    ap.add_argument("--tables", nargs="*", default=None)
    ap.add_argument("--ds", type=int, nargs="*", default=None)
    ap.add_argument("--no-tiered", action="store_true")
    ap.add_argument("--seeds-core", type=int, default=30)
    ap.add_argument("--seeds-extra", type=int, default=15)
    ap.add_argument("--time-sweep", default=None, help="table id: one seed, all arms, all budgets; print costs")
    a = ap.parse_args(argv)
    if a.time_sweep:
        os.nice(19)
        g = load_games()[a.time_sweep]
        ctx = table_context(a.time_sweep)
        tot = 0.0
        for est in a.estimators:
            t0 = time.perf_counter()
            for T in budget_grid(ctx.d):
                run_one(est, ctx, g, T, SEEDS30[0])
            dt = time.perf_counter() - t0
            tot += dt
            print(f"{est:30s} sweep {dt:7.2f}s", flush=True)
        print(f"TOTAL one (table, seed) sweep at d={ctx.d}: {tot:.1f}s; load {rb.load_snapshot()['loadavg']}")
        return
    OUTDIR.mkdir(parents=True, exist_ok=True)
    chunks = []
    for g in game_list(a.ds):
        if a.tables and g["table_id"] not in a.tables:
            continue
        budgets = budget_grid(g["d"])
        for est in a.estimators:
            out = OUTDIR / f"{g['table_id']}__{est}.csv.gz"
            if out.exists():
                continue
            seeds = seeds_for(est, not a.no_tiered, a.seeds_core, a.seeds_extra)
            chunks.append((g["table_id"], est, seeds, budgets, str(out)))

    chunks.sort(key=lambda c: -(load_games()[c[0]]["d"] ** 2 * COST.get(c[1], 1.0) * len(c[2])))
    log_line(LOGS / "grid.log", f"E4 grid: {len(chunks)} chunks to run with {a.workers} workers "
                                f"(tiered={not a.no_tiered}); hardware {rb.hardware_tags()}")
    run_chunks(chunks, run_chunk, a.workers, "E4 grid", LOGS / "grid.log", round_size=30)


if __name__ == "__main__":
    main()
