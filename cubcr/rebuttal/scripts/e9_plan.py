#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import json
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import numpy as np
import pandas as pd

import rb
import e9_gpu_estimators as e9

E9 = rb.results_dir("E9")
SEEDS = list(rb.PAPER_SAMPLER_SEEDS)
TASK_SPEC = [(16, 1000), (16, 10000), (20, 1000), (20, 10000), (24, 1000), (24, 10000), (28, 1000)]
GEN_SEED = 20260829
K = 50
E4_PLAN = rb.RB / "results" / "E4" / "raw" / "equal_budget_plan.csv"


def e9_tasks():
    out = []
    ts = rb.paper_tasks(include_records=True)
    for n, S in TASK_SPEC:
        t = [x for x in ts if x["n"] == n and x["states_target"] == S and x["k"] == K and x["gen_seed"] == GEN_SEED]
        if len(t) != 1:
            raise RuntimeError(f"task n={n} S={S} not found")
        out.append(t[0])
    return out


def main():
    e4 = pd.read_csv(E4_PLAN)
    rows = []
    for t in e9_tasks():
        n, iid = t["n"], t["iid"]
        cg = t["cert_gpu"]
        levels = {"B4k": 4096, "Bhi": 32768 if n == 16 else 65536}
        e4rows = e4[(e4.table_id == iid) & (e4.eps_div == 20) & (e4.acct == "uniq")]
        if len(e4rows):
            B = int(e4rows.B.iloc[0])
            T_of = {r.estimator: int(r.T) for r in e4rows.itertuples() if np.isfinite(r.T)}
            stop = {"B": B, "source": "E4 equal-budget plan (median over 30 table-replayed online runs)",
                    "paired_d": {"m": (T_of["cubcr_paired_perm"] - 2) // (2 * (n + 1)),
                                 "how": "E4: T = B_req of the online runs"},
                    "stratsvarm": {"T": T_of["StratifiedSVARM"], "how": "E4: interpolated on the 30-seed grid"},
                    "kadd2": {"T": T_of["shapiq_kADDSHAP"]}, "leverage": {"T": T_of["LeverageSHAP"]}}
        else:
            B = int(np.median([cg[s]["n_solves"] for s in SEEDS]))
            m = int(np.median([cg[s]["n_perms"] // 2 for s in SEEDS]))
            stop = {"B": B, "source": "median over the paper's 3 cert_gpu records (results/e9v3)",
                    "paired_d": {"m": m, "how": "median pairs of the paper's 3 cert_gpu runs"}}
        levels["Bstop"] = stop["B"]
        for lev in ("B4k", "Bstop", "Bhi"):
            B = int(levels[lev])
            nom = {}
            if lev == "Bstop" and "paired_d" in stop:
                nom["paired_d"] = dict(stop["paired_d"])
            else:
                r = e9.paired_m_for_distinct(n, B, SEEDS)
                nom["paired_d"] = {"m": r["m"], "how": "median distinct over 3 seeds >= B"}
            nom["paired_r"] = {"m": e9.paired_m_requested(n, B), "how": "2 + 2m(n+1) <= B"}
            if lev == "Bstop" and "stratsvarm" in stop:
                nom["stratsvarm"] = dict(stop["stratsvarm"])
            else:
                c = e9.stratsvarm_nominal_for_distinct(n, B, SEEDS)
                nom["stratsvarm"] = {"T": int(c["T"]), "how": "median distinct over 3 seeds >= B (bisection)"}
            nom["kadd2"] = {"T": int(stop["kadd2"]["T"]) if (lev == "Bstop" and "kadd2" in stop) else B}
            nom["leverage"] = {"T": int(stop["leverage"]["T"]) if (lev == "Bstop" and "leverage" in stop) else B}
            for arm in e9.ARMS:
                d = nom[arm]
                if arm.startswith("paired"):
                    curves = [e9.paired_distinct_curve(n, s, d["m"])[-1] for s in SEEDS]
                    pred_req = 2 + 2 * d["m"] * (n + 1)
                elif arm == "stratsvarm":
                    curves = [e9.stratsvarm_distinct(n, d["T"], s) for s in SEEDS]
                    pred_req = d["T"]
                else:
                    curves = [d["T"] - (d["T"] % 2 if arm == "leverage" else 0)] * len(SEEDS)
                    pred_req = curves[0]
                rows.append({"iid": iid, "n": n, "states": t["states"], "states_target": t["states_target"], "k": t["k"],
                             "gen_seed": t["gen_seed"], "level": lev, "B": B,
                             "B_source": stop["source"] if lev == "Bstop" else "fixed",
                             "arm": arm, "nominal_m": d.get("m"), "nominal_T": d.get("T"), "how": d.get("how", "T = B"),
                             "pred_requested": int(pred_req), "pred_distinct_per_seed": [int(x) for x in curves],
                             "pred_distinct_median": float(np.median(curves))})
                print(iid, lev, B, arm, d, "pred distinct", curves, flush=True)
    plan = {"created": rb.now_iso(), "seeds": SEEDS, "git_commit": rb.git_commit(), "rows": rows}
    rb.write_json(E9 / "raw" / "plan.json", plan)
    df = pd.DataFrame(rows)
    df["pred_distinct_per_seed"] = df["pred_distinct_per_seed"].apply(lambda x: " ".join(map(str, x)))
    df.to_csv(E9 / "raw" / "plan.csv", index=False)
    print("wrote", E9 / "raw" / "plan.csv", len(df), "rows")


if __name__ == "__main__":
    main()
