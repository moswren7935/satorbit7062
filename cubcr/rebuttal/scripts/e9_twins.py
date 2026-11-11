#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))

import argparse
import json

import numpy as np
import pandas as pd

import rb
import e9_gpu_estimators as e9

E9 = rb.results_dir("E9")
TAB = E9 / "raw" / "table"
GPU = E9 / "raw" / "gpu"
E4RUNS = rb.RB / "results" / "E4" / "raw" / "equal_budget_runs.csv.gz"


def run_name(iid, level, arm, seed):
    return f"{iid}__{level}__{arm}__s{seed}.json"


def _nice():
    try:
        os.nice(19 - os.nice(0))
    except Exception:
        pass


def work(iid, rows, seeds):

    _nice()
    import shapiq
    task = [t for t in rb.paper_tasks(include_records=False) if t["iid"] == iid][0]
    comp, _, _, _ = rb.build_task(task)
    tab = np.load(rb.table_path(iid))
    done = 0
    for r in rows:
        for seed in seeds:
            p = TAB / run_name(iid, r["level"], r["arm"], seed)
            if p.exists():
                continue
            nominal = {"m": int(r["nominal_m"])} if r["arm"].startswith("paired") else {"T": int(r["nominal_T"])}
            te = e9.TimedEngine(rb.TableEngine(tab, compiled=comp), keep=False)
            out = e9.run_arm(r["arm"], te, task["n"], task["k"], nominal, seed)
            rec = {"exp": "E9", "kind": "table", "iid": iid, "level": r["level"], "arm": r["arm"], "seed": int(seed),
                   "nominal": nominal, "phi_hat": out["phi_hat"].tolist(),
                   "half_widths": None if out["half_widths"] is None else out["half_widths"].tolist(),
                   "batch_sizes": out["batch_sizes"], "n_requested": out["n_requested"], "n_engine": out["n_engine"],
                   "engine_calls": out["engine_calls"], "host_wall_s_table": out["wall_s"],
                   "metrics": e9.error_metrics(out["phi_hat"], task["phi_exact"], task["upsilon"]),
                   "intervals": e9.interval_checks(out, task["phi_exact"]), "git_commit": rb.git_commit(),
                   "created": rb.now_iso()}
            rb.write_json(p, rec)
            done += 1
    return f"{iid}: {done} twins written"


def compare():

    gpu_rows, e4_rows = [], []
    chunks = sorted((E4RUNS.parent / "equal_budget").glob("*.csv.gz"))
    inputs = chunks or [E4RUNS]
    e4 = pd.concat([pd.read_csv(p, compression="gzip", float_precision="round_trip")
                    for p in inputs], ignore_index=True)
    e4 = e4[(e4.eps_div == 20) & (e4.acct == "uniq")]
    keys = ["table_id", "estimator", "seed"]
    if e4.duplicated(keys).any():
        raise ValueError("duplicate E4 equal-budget comparison rows")
    games_path = E4RUNS.parent / "games.json"
    e4_tasks = ({g["table_id"] for g in json.loads(games_path.read_text())["games"]}
                if games_path.exists() else set(e4.table_id))
    for p in sorted(TAB.glob("*.json")):
        tw = json.load(open(p))
        g = GPU / p.name
        if g.exists():
            gr = json.load(open(g))
            if gr.get("status_run") == "OK":
                same = {"phi_hat": gr["phi_hat"] == tw["phi_hat"], "half_widths": gr.get("half_widths") == tw["half_widths"],
                        "batch_sizes": gr.get("batch_sizes") == tw["batch_sizes"],
                        "n_requested": gr["n_requested"] == tw["n_requested"], "n_engine": gr["n_engine"] == tw["n_engine"],
                        "engine_calls": gr["engine_calls"] == tw["engine_calls"]}
                diff = float(np.max(np.abs(np.asarray(gr["phi_hat"]) - np.asarray(tw["phi_hat"]))))
                gpu_rows.append(dict(run=p.stem, iid=tw["iid"], level=tw["level"], arm=tw["arm"], seed=tw["seed"],
                                     all_same=all(same.values()), max_abs_diff=diff, **{f"same_{k}": v for k, v in same.items()},
                                     resolve_same_values=all(r["same_values"] for r in gr["resolve"])))
        if tw["level"] == "Bstop" and tw["arm"] in e9.E4_NAME and tw["iid"] in e4_tasks:
            sel = e4[(e4.table_id == tw["iid"]) & (e4.estimator == e9.E4_NAME[tw["arm"]]) & (e4.seed == tw["seed"])]
            if len(sel) != 1:
                raise ValueError(f"missing E4 equal-budget comparison for {tw['iid']} {tw['arm']} seed {tw['seed']}")
            if len(sel) == 1:
                s = sel.iloc[0]
                e4phi = [float(v) for v in str(s.phi_hat).split()]
                nominal_e4 = int(s.T_nominal)
                same_nom = (tw["nominal"].get("T") == nominal_e4) if "T" in tw["nominal"] else \
                    ((nominal_e4 - 2) // (2 * (len(e4phi) + 1)) == tw["nominal"]["m"])
                e4_rows.append(dict(run=p.stem, iid=tw["iid"], arm=tw["arm"], seed=tw["seed"], e4_estimator=s.estimator,
                                    e4_T_nominal=nominal_e4, same_nominal=bool(same_nom),
                                    phi_bitwise=e4phi == tw["phi_hat"],
                                    max_abs_diff=float(np.max(np.abs(np.asarray(e4phi) - np.asarray(tw["phi_hat"])))),
                                    same_n_total=int(s.n_total) == tw["n_requested"],
                                    same_n_unique=int(s.n_unique) == tw["n_engine"]))
    gdf = pd.DataFrame(gpu_rows); edf = pd.DataFrame(e4_rows)
    gdf.to_csv(E9 / "raw" / "validation_gpu_vs_table.csv", index=False)
    edf.to_csv(E9 / "raw" / "validation_table_vs_e4.csv", index=False)
    summ = {"created": rb.now_iso(),
            "gpu_vs_table": {"n": int(len(gdf)), "all_same": int(gdf.all_same.sum()) if len(gdf) else 0,
                             "phi_bitwise": int(gdf.same_phi_hat.sum()) if len(gdf) else 0,
                             "max_abs_diff": float(gdf.max_abs_diff.max()) if len(gdf) else None,
                             "resolve_same_values": int(gdf.resolve_same_values.sum()) if len(gdf) else 0,
                             "by_arm": gdf.groupby("arm").all_same.agg(["sum", "count"]).reset_index().to_dict("records") if len(gdf) else []},
            "table_vs_e4": {"n": int(len(edf)), "phi_bitwise": int(edf.phi_bitwise.sum()) if len(edf) else 0,
                            "same_nominal": int(edf.same_nominal.sum()) if len(edf) else 0,
                            "same_counts": int((edf.same_n_total & edf.same_n_unique).sum()) if len(edf) else 0,
                            "max_abs_diff": float(edf.max_abs_diff.max()) if len(edf) else None}}
    rb.write_json(E9 / "raw" / "validation.json", summ)
    print(json.dumps(summ, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--compare-only", action="store_true")
    a = ap.parse_args()
    TAB.mkdir(parents=True, exist_ok=True)
    if not a.compare_only:
        plan = json.load(open(E9 / "raw" / "plan.json"))
        seeds = plan["seeds"]
        by_task = {}
        for r in plan["rows"]:
            if r["n"] <= 24:
                by_task.setdefault(r["iid"], []).append(r)
        from joblib import Parallel, delayed
        res = Parallel(n_jobs=min(a.workers, 6), verbose=0)(delayed(work)(iid, rows, seeds) for iid, rows in by_task.items())
        for x in res:
            print(x, flush=True)
    compare()


if __name__ == "__main__":
    main()
