#!/usr/bin/env python3


from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import argparse
import glob
import math
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "lib"))
sys.path.insert(0, str(HERE))
import numpy as np
import pandas as pd

import rb
from e4_common import (RAW, LOGS, SEEDS30, CORE_ESTIMATORS, EXTRA_ESTIMATORS, load_games, table_context,
                       write_csv_gz, run_chunks, nice_worker, log_line, read_csv_gz)
from e4_estimators import ESTIMATORS, min_feasible_budget
import e4_run_grid as G

OUTDIR = RAW / "equal_budget"
DISTINCT = {"shapiq_KernelSHAP", "shapiq_UnbiasedKernelSHAP", "shapiq_kADDSHAP", "shapiq_SVARM", "LeverageSHAP"}


def stopping_budgets() -> pd.DataFrame:
    df = pd.concat([read_csv_gz(f) for f in sorted(glob.glob(str(RAW / "adaptive" / "*.csv.gz")))], ignore_index=True)
    return df.groupby(["table_id", "d", "eps_div"]).agg(B_req=("n_requested", "median"), B_uniq=("n_unique", "median"),
                                                        n_runs=("seed", "size")).reset_index()


def calibrate(table_id: str, est: str, B: float):

    f = RAW / "grid" / f"{table_id}__{est}.csv.gz"
    g = read_csv_gz(f)
    g = g[g.status == "OK"].groupby("budget").agg(u=("n_unique", "median"), t=("n_total", "median")).reset_index()
    g = g.sort_values("u")
    u, t = np.log(g.u.values.astype(float)), np.log(g.budget.values.astype(float))
    x = math.log(B)
    if x <= u[-1]:
        return float(np.exp(np.interp(x, u, t))), "interpolated"
    slope = (t[-1] - t[-2]) / (u[-1] - u[-2])
    return float(np.exp(t[-1] + slope * (x - u[-1]))), "extrapolated"


def run_chunk(table_id: str, est: str, eps_div: int, acct: str, B: int, T: int, how: str, seeds: list, out: str):
    nice_worker()
    g = load_games()[table_id]
    ctx = table_context(table_id)
    t0 = time.perf_counter()
    rows = []
    for s in seeds:
        r = G.run_one(est, ctx, g, T, s)
        r.update(eps_div=eps_div, acct=acct, B=B, T_nominal=T, calibration=how,
                 near_exhaustive=bool(B >= (1 << ctx.d) // 2))
        rows.append(r)
    df = pd.DataFrame(rows)
    write_csv_gz(df, Path(out))
    rb.write_json(Path(str(out).replace(".csv.gz", ".json")),
                  {"table_id": table_id, "estimator": est, "eps_div": eps_div, "acct": acct, "B": B, "T_nominal": T,
                   "calibration": how, "seeds": list(map(int, seeds)), "gen_seed": g.get("gen_seed"),
                   "created": rb.now_iso(), "hardware": rb.hardware_tags(), "load_snapshot": rb.load_snapshot(),
                   "chunk_wall_s": time.perf_counter() - t0,
                   "note": "host_wall_s columns are table-lookup times on a shared host (not engine timings)"})
    uq = df.n_unique.median() if "n_unique" in df else float("nan")
    return f"{table_id} div{eps_div} {acct} B={B} T={T} ({how}) {est}: {len(rows)} runs, median unique {uq:.0f}, " \
           f"{time.perf_counter() - t0:.1f}s"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--divs", type=int, nargs="*", default=[20, 100])
    a = ap.parse_args(argv)
    OUTDIR.mkdir(parents=True, exist_ok=True)
    sb = stopping_budgets()
    games = load_games()
    plan, chunks = [], []
    for _, r in sb.iterrows():
        tid, d, div = r.table_id, int(r.d), int(r.eps_div)
        if div not in a.divs:
            continue
        for acct in ("uniq", "req"):
            B = int(r.B_uniq if acct == "uniq" else r.B_req)
            B -= B % 2
            skip = None
            if B >= (1 << d) // 2:
                skip = "B >= 2^d/2 (near-exhaustive / exhaustive enumeration cheaper)"
            elif div == 100 and acct != "uniq":
                skip = "scope cut: Upsilon/100 equal-budget only in the uniq accounting"
            ests = ESTIMATORS if div == 20 else CORE_ESTIMATORS
            for est in ests:
                seeds = (SEEDS30[:15] if est in EXTRA_ESTIMATORS else SEEDS30) if div == 20 else SEEDS30[:15]
                if skip:
                    plan.append(dict(table_id=tid, d=d, eps_div=div, acct=acct, estimator=est, B=B, T=None, how=None,
                                     skipped=skip))
                    continue
                if acct == "req" or est in DISTINCT:
                    T, how = B, "T = B"
                elif est == "cubcr_paired_perm":
                    T, how = int(r.B_req), "T = B_req (the adaptive run's pairs)"
                else:
                    T, how = calibrate(tid, est, B)
                    T = int(round(T))
                T -= T % 2
                if T < min_feasible_budget(est, d):
                    plan.append(dict(table_id=tid, d=d, eps_div=div, acct=acct, estimator=est, B=B, T=T, how=how,
                                     skipped="below the arm's minimum feasible budget"))
                    continue
                plan.append(dict(table_id=tid, d=d, eps_div=div, acct=acct, estimator=est, B=B, T=T, how=how,
                                 skipped=None, n_seeds=len(seeds)))
                out = OUTDIR / f"{tid}__div{div}__{acct}__{est}.csv.gz"
                if out.exists():
                    continue
                chunks.append((tid, est, div, acct, B, T, how, seeds, str(out)))
    pd.DataFrame(plan).to_csv(RAW / "equal_budget_plan.csv", index=False)
    chunks.sort(key=lambda c: -(c[5] * games[c[0]]["d"] * G.COST.get(c[1], 1.0) * len(c[7])))
    log_line(LOGS / "equal_budget.log", f"E4 equal-budget: {len(chunks)} chunks, {a.workers} workers")
    run_chunks(chunks, run_chunk, a.workers, "E4 equal-budget", LOGS / "equal_budget.log", round_size=40)


if __name__ == "__main__":
    main()
