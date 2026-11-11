#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e1_models
from e1_ext import deff_of

OUT = rb.results_dir("E1")
VALIDATOR_TOL = 1e-9
CSG_VALIDATE_MAX_ROWS = 2_000_000
REPO = rb.REPO


def _stored_checks(spec, comp, iid):

    fam = spec["family"]
    import glob
    if fam == "primary":
        return "iid==e9v3 record (rb.build_task check)", True
    if fam == "tables13":
        return f"iid==table file name ({spec['table_file']})", iid == spec["model_id"]
    if fam == "loo90":
        p = REPO / "results" / "e12" / f"e12__ticketbench_new__{iid}.json"
        ok = p.exists()
        if ok:
            d = json.load(open(p))
            ok = int(d["states"]) == int(comp.n_states) and int(d["n"]) == int(comp.n_agents) and int(d["k"]) == spec["k"]
        return "results/e12 record exists, |S|, n, k match", ok
    if fam == "decision6":
        fs = glob.glob(str(REPO / "results" / "e4v2" / f"e4__{iid}__eps0.005__s20260829.json"))
        ok = bool(fs) and int(json.load(open(fs[0]))["states"]) == int(comp.n_states)
        return "results/e4v2 record exists, |S| matches", ok
    if fam == "overcooked":
        p = REPO / "results" / "e14" / (f"e14__{iid}__exact_step.json" if spec["k"] == 10 else f"e14__{iid}__agent_exact.json")
        d = json.load(open(p))
        return f"{os.path.relpath(p, REPO)} |S| matches", int(d["states"]) == int(comp.n_states)
    if fam == "prism9":
        p = REPO / "results" / "e3b" / f"e3__{iid}.json"
        d = json.load(open(p))
        return "results/e3b record n, |S|, k match", (int(d["states"]) == int(comp.n_states) and int(d["n"]) == int(comp.n_agents)
                                                      and int(d["k"]) == spec["k"])
    if fam == "storm6":
        p = REPO / "results" / "e3storm" / f"e3storm__{iid}.json"
        ok = p.exists() and int(json.load(open(p))["states"]) == int(comp.n_states)
        return "results/e3storm record exists, |S| matches", ok
    return "", None


def check_arrays(comp) -> dict:
    S = int(comp.n_states)
    n = int(comp.n_agents)
    kind = np.asarray(comp.state_kind).astype(np.int64)
    val = np.asarray(comp.val, dtype=np.float64)
    col = np.asarray(comp.col, dtype=np.int64)
    rp = np.asarray(comp.row_ptr, dtype=np.int64)
    srs = np.asarray(comp.state_row_start, dtype=np.int64)
    W = np.asarray(comp.weights, dtype=np.float64)
    agent = np.asarray(comp.state_agent).astype(np.int64)
    radix = np.asarray(comp.state_radix).astype(np.int64)
    out = {}
    out["finite_P"] = bool(np.all(np.isfinite(val)))
    out["finite_pi"] = bool(np.all(np.isfinite(W)))
    out["n_nonfinite"] = int(np.sum(~np.isfinite(val)) + np.sum(~np.isfinite(W)))
    out["min_P"] = float(val.min()) if val.size else None
    out["n_negative_P"] = int(np.sum(val < 0))
    nnz_row = np.diff(rp)
    out["n_empty_rows"] = int(np.sum(nnz_row == 0))
    out["max_nnz_row"] = int(nnz_row.max()) if nnz_row.size else 0
    rs = np.add.reduceat(val.astype(np.longdouble), rp[:-1]) if val.size else np.zeros(0, np.longdouble)
    rs = np.where(nnz_row == 0, np.longdouble(0), rs)
    dev = np.abs(rs - np.longdouble(1))
    out["max_row_sum_dev"] = float(dev.max()) if dev.size else 0.0
    out["n_rows_dev_gt_1e-15"] = int(np.sum(dev > 1e-15))
    out["n_rows_dev_gt_1e-12"] = int(np.sum(dev > 1e-12))
    out["n_rows_dev_gt_1e-9"] = int(np.sum(dev > 1e-9))
    out["col_in_range"] = bool(col.size == 0 or (col.min() >= 0 and col.max() < S))

    deff = deff_of(comp)
    rows_per_state = np.diff(srs)
    prod_rad = np.prod(np.where(np.arange(3)[None, :] < deff[:, None], radix, 1), axis=1)
    out["rows_eq_prod_radix"] = bool(np.all(rows_per_state == prod_rad))
    out["max_rows_per_state"] = int(rows_per_state.max())
    out["max_active"] = int(deff.max())
    out["max_actions"] = int(np.max(np.where(np.arange(3)[None, :] < deff[:, None], radix, 1)))
    used = np.arange(3)[None, :] < deff[:, None]
    out["slots_contiguous"] = bool(np.all((agent >= 0) == used))
    out["agents_in_range"] = bool(np.all(agent[used] < n) and np.all(agent[used] >= 0))
    dup = False
    for l1 in range(3):
        for l2 in range(l1 + 1, 3):
            both = used[:, l1] & used[:, l2]
            if np.any(agent[both, l1] == agent[both, l2]):
                dup = True
    out["no_duplicate_agents"] = not dup
    out["limits_ok"] = bool(out["max_rows_per_state"] <= 64 and out["max_active"] <= 3 and out["max_actions"] <= 8 and n <= 64)

    normal = kind == 0
    pol_dev, pol_min, beyond = 0.0, np.inf, 0.0
    n_pol = 0
    n_neg_pi = 0
    for l in range(3):
        sel = normal & used[:, l]
        if not np.any(sel):
            continue
        Wl = W[sel, l, :]
        rl = radix[sel, l]
        inside = np.arange(W.shape[2])[None, :] < rl[:, None]
        s = np.where(inside, Wl, 0.0).astype(np.longdouble).sum(axis=1)
        pol_dev = max(pol_dev, float(np.max(np.abs(s - np.longdouble(1)))))
        pol_min = min(pol_min, float(np.min(np.where(inside, Wl, np.inf))))
        beyond = max(beyond, float(np.max(np.abs(np.where(inside, 0.0, Wl)))))
        n_neg_pi += int(np.sum(np.where(inside, Wl, 0.0) < 0))
        n_pol += int(sel.sum())
    out["n_policies"] = n_pol
    out["max_policy_sum_dev"] = pol_dev
    out["min_pi"] = None if n_pol == 0 else pol_min
    out["n_negative_pi"] = int(n_neg_pi)
    out["max_pi_mass_beyond_radix"] = beyond
    unused = ~used
    ref = np.zeros(W.shape[2]); ref[0] = 1.0
    out["unused_slots_exact_1_0"] = bool(np.all(W[unused] == ref[None, :])) if np.any(unused) else True

    fail = np.flatnonzero(kind == 1)
    out["n_fail_states"] = int(fail.size)
    out["n_avoid_states"] = int(np.sum(kind == 2))
    out["init_is_fail"] = bool(kind[int(comp.init_state)] == 1)
    absorbing = 0
    for s in fail.tolist():
        r0, r1 = srs[s], srs[s + 1]
        if r1 - r0 == 1 and rp[r0 + 1] - rp[r0] == 1 and int(col[rp[r0]]) == s and float(val[rp[r0]]) == 1.0:
            absorbing += 1
    out["n_fail_self_loop_p1"] = int(absorbing)
    out["fail_engine_fixed_at_1"] = True
    out["fail_structurally_absorbing"] = bool(absorbing == fail.size)
    return out


def worker(spec):
    try:
        os.nice(19)
    except Exception:
        pass
    t0 = time.perf_counter()
    row = {"family": spec["family"], "spec_model_id": spec.get("model_id"), "source": spec.get("source")}
    try:
        comp, csg, info = e1_models.build(spec)
        iid = info["iid"]
        row.update(model_id=iid, n_agents=int(comp.n_agents), n_states=int(comp.n_states), n_rows=int(comp.n_rows),
                   nnz=int(comp.nnz), k=int(spec["k"]), horizon_k_field=int(comp.horizon_k),
                   fingerprint=comp.fingerprint(), validated_at_build=info["validated_at_build"],
                   build_path=info["build_path"],
                   params=json.dumps({kk: spec[kk] for kk in ("n", "states_target", "k", "gen_seed", "pattern",
                                                               "hazard", "consts", "fail", "npz") if kk in spec},
                                     default=str))
        row.update(check_arrays(comp))
        what, ok = _stored_checks(spec, comp, iid)
        row["stored_record_check"] = what
        row["stored_record_match"] = ok

        csg_res, same_arrays = "skipped", None
        if spec["family"] in ("primary", "tables13", "loo90", "decision6"):
            from generators.ticketbench import build_instance
            g = build_instance(spec["n"], spec["states_target"], spec["k"], 0.05, spec["pattern"], spec["gen_seed"],
                               hazard=spec.get("hazard", 0.0))
            try:
                g.validate(tol=VALIDATOR_TOL)
                c2 = g.compile()
                same_arrays = c2.fingerprint() == comp.fingerprint()
                csg_res = "pass (build_instance CSG)"
            except Exception as e:
                csg_res = f"FAIL: {e}"
        elif csg is not None:
            try:
                csg.validate(tol=VALIDATOR_TOL)
                csg_res = "pass (CSG at build)"
                same_arrays = True
            except Exception as e:
                csg_res = f"FAIL: {e}"
        elif comp.n_rows <= CSG_VALIDATE_MAX_ROWS:
            try:
                e1_models.compiled_to_csg(comp).validate(tol=VALIDATOR_TOL)
                csg_res = "pass (generic CSG conversion)"
            except Exception as e:
                csg_res = f"FAIL: {e}"
        else:
            csg_res = f"skipped (> {CSG_VALIDATE_MAX_ROWS} rows; vectorised checks only)"
        row["csg_validate_tol"] = VALIDATOR_TOL
        row["csg_validate"] = csg_res
        row["csg_compile_equals_build_arrays"] = same_arrays
        ok = (row["finite_P"] and row["finite_pi"] and row["n_negative_P"] == 0 and row["n_negative_pi"] == 0
              and row["max_row_sum_dev"] <= VALIDATOR_TOL and row["max_policy_sum_dev"] <= VALIDATOR_TOL
              and row["max_pi_mass_beyond_radix"] == 0.0 and row["unused_slots_exact_1_0"] and row["col_in_range"]
              and row["rows_eq_prod_radix"] and row["n_empty_rows"] == 0 and row["limits_ok"]
              and row["slots_contiguous"] and row["agents_in_range"] and row["no_duplicate_agents"]
              and row["n_fail_states"] >= 1 and row["fail_structurally_absorbing"] and not csg_res.startswith("FAIL")
              and row["stored_record_match"] is not False)
        row["verdict"] = "PASS" if ok else "FAIL"
        row["status"] = "OK"
    except Exception as e:
        row["status"] = "ERROR"
        row["error"] = repr(e)[:500]
        row["traceback"] = traceback.format_exc()[-2000:]
        row["verdict"] = "ERROR"
    row["check_s"] = time.perf_counter() - t0
    return row


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--families", default="")
    a = ap.parse_args(argv)
    specs = e1_models.all_specs()
    if a.families:
        fams = set(a.families.split(","))
        specs = [s for s in specs if s["family"] in fams]

    t0 = time.perf_counter()
    with Pool(min(a.workers, 6)) as pool:
        rows = pool.map(worker, specs, chunksize=1)
    import pandas as pd
    df = pd.DataFrame(rows)

    df["game_key"] = df["fingerprint"].astype(str) + "_k" + df["k"].astype(str)
    first = {}
    dup_of = []
    for _, r in df.iterrows():
        g = r["game_key"]
        if g in first:
            dup_of.append(first[g])
        else:
            first[g] = f"{r['family']}:{r['model_id']}"
            dup_of.append("")
    df["same_game_as"] = dup_of
    cols_first = ["family", "model_id", "verdict", "n_agents", "n_states", "n_rows", "nnz", "k", "validated_at_build",
                  "min_P", "max_row_sum_dev", "min_pi", "max_policy_sum_dev", "max_pi_mass_beyond_radix",
                  "unused_slots_exact_1_0", "finite_P", "finite_pi", "n_fail_states", "fail_engine_fixed_at_1",
                  "fail_structurally_absorbing", "csg_validate", "csg_validate_tol", "csg_compile_equals_build_arrays",
                  "stored_record_match", "same_game_as"]
    cols = cols_first + [c for c in df.columns if c not in cols_first]
    df = df[cols]
    df.to_csv(OUT / "input_validity.csv", index=False)
    fam_rows = []
    for fam, g in df.groupby("family", sort=False):
        fam_rows.append({
            "family": fam, "n_models": len(g), "n_distinct_games": g["game_key"].nunique(),
            "n_pass": int((g["verdict"] == "PASS").sum()), "n_fail_or_error": int((g["verdict"] != "PASS").sum()),
            "validated_at_build": ",".join(sorted(set(str(x) for x in g["validated_at_build"]))),
            "min_P": float(g["min_P"].min()), "max_row_sum_dev": float(g["max_row_sum_dev"].max()),
            "min_pi": float(g["min_pi"].min()), "max_policy_sum_dev": float(g["max_policy_sum_dev"].max()),
            "max_pi_mass_beyond_radix": float(g["max_pi_mass_beyond_radix"].max()),
            "all_finite": bool(g["finite_P"].all() and g["finite_pi"].all()),
            "fail_absorbing_all": bool(g["fail_structurally_absorbing"].all()),
            "csg_validate": ";".join(sorted(set(str(x).split(" (")[0] for x in g["csg_validate"]))),
            "validator_tol": VALIDATOR_TOL,
            "states_range": f"{int(g['n_states'].min())}-{int(g['n_states'].max())}",
            "k_range": f"{int(g['k'].min())}-{int(g['k'].max())}",
            "n_agents_range": f"{int(g['n_agents'].min())}-{int(g['n_agents'].max())}",
        })
    fdf = pd.DataFrame(fam_rows)
    fdf.to_csv(OUT / "input_validity_families.csv", index=False)
    tot = {"n_models": int(len(df)), "n_distinct_games": int(df["game_key"].nunique()),
           "n_distinct_arrays": int(df["fingerprint"].nunique()), "n_pass": int((df["verdict"] == "PASS").sum()),
           "max_row_sum_dev": float(df["max_row_sum_dev"].max()), "max_policy_sum_dev": float(df["max_policy_sum_dev"].max()),
           "min_P": float(df["min_P"].min()), "min_pi": float(df["min_pi"].min()), "validator_tol": VALIDATOR_TOL,
           "wall_s": time.perf_counter() - t0, "git_commit": rb.git_commit(), "hardware": rb.hardware_tags(),
           "load": rb.load_snapshot(), "time": rb.now_iso()}
    rb.write_json(OUT / "raw" / "input_validity_raw.json", {"totals": tot, "rows": df.to_dict(orient="records")})
    print(json.dumps({k: v for k, v in tot.items() if k not in ("hardware", "load")}, indent=1))
    print(fdf.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
