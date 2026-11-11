#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import glob
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

OUT = rb.results_dir("E1")
REPO = rb.REPO


def specs():
    out = []
    for s in e1_models.specs_primary():
        out.append(dict(s, role="primary (cert runs, beta = eps_X)"))
    for s in e1_models.specs_decision6():
        out.append(dict(s, role="decision-time (Table 1 runs, beta = eps_X)"))
    out.append(dict(e1_models.specs_overcooked()[0], role="Overcooked k=10 (cert runs, beta = eps_X)"))
    for s in e1_models.specs_tables13():
        out.append(dict(s, role="13 tables (E11 runs used beta = 0)"))
    for s in e1_models.specs_prism9():
        out.append(dict(s, role="PRISM-games (value cross-checks only, no sampling runs)"))
    return out


def used_eps_x_map():

    m = {"primary": {}, "decision6": {}, "overcooked": {}}
    for r in rb.load_e9v3(arms=["cert_gpu", "cert_cpu"], paper_only=True):
        m["primary"].setdefault(r["instance_id"], set()).add(float(r["fp_term"]))
    for f in glob.glob(str(REPO / "results" / "e4v2" / "e4__*.json")):
        d = json.load(open(f))
        if "sampler_v2" in d:
            m["decision6"].setdefault(d["instance_id"], set()).add(float(d["sampler_v2"]["fp_term"]))
    m["overcooked"]["overcooked_cramped_room_k10"] = {float(np.load(f)["fp_term"]) for f in
                                                      glob.glob(str(OUT / "raw" / "units" / "overcooked_k10__s*.npz"))}
    src = {"primary": "results/e9v3 cert_gpu + cert_cpu records (fp_term)",
           "decision6": "results/e4v2 records (sampler_v2.fp_term)",
           "overcooked": "E1 bitwise re-runs (raw/units/overcooked_k10__s*.npz; the k=10 records do not store fp_term)"}
    return {fam: {iid: (sorted(v), src[fam]) for iid, v in d.items()} for fam, d in m.items()}


def _used_eps_x(spec, iid):
    fam = spec["family"]
    if fam in ("primary", "decision6", "overcooked"):
        return spec["_used"].get(iid, ([], "NOT FOUND"))
    if fam == "tables13":
        return [0.0], "E11 runs: oracle without the model, beta = 0 (experiments/e11_bakeoff.py:38)"
    return [], "no sampling runs"


def worker(spec):
    try:
        os.nice(19)
    except Exception:
        pass
    import e1_rho
    t0 = time.perf_counter()
    row = dict(family=spec["family"], role=spec["role"], model_id=spec.get("model_id"), k=int(spec["k"]))
    try:
        comp, _, info = e1_models.build(spec)
        iid = info["iid"]
        row.update(model_id=iid, n_agents=int(comp.n_agents), n_states=int(comp.n_states), n_rows=int(comp.n_rows),
                   nnz=int(comp.nnz))
        o = e1_rho.outward_bound(comp, int(spec["k"]), perms_per_unit=2)
        eps_x_exact = o.pop("_eps_x_exact")
        row.update(o)
        used, src = _used_eps_x(spec, iid)
        row["eps_x_used_in_runs"] = json.dumps(used)
        row["eps_x_used_source"] = src
        pos = [u for u in used if u > 0]
        if pos:
            row["eps_x_used_min"] = min(pos)
            row["eps_x_used_equals_code"] = bool(all(u == o["eps_x_code"] for u in pos))
            row["eps_x_used_minus_out"] = float(min(pos) - o["eps_x_out"])

            from fractions import Fraction
            row["eps_x_exact_minus_used"] = float(eps_x_exact - Fraction(min(pos)))
            row["eps_x_used_covers_outward"] = bool(min(pos) >= o["eps_x_out"])
        row["status"] = "OK"
    except Exception as e:
        row["status"] = "ERROR"
        row["error"] = repr(e)[:300]
        row["traceback"] = traceback.format_exc()[-2000:]
    row["seconds"] = time.perf_counter() - t0
    return row


def main(argv=None):
    import pandas as pd
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    um = used_eps_x_map()
    sp = [dict(s, _used=um.get(s["family"], {})) for s in specs()]
    rows = []

    par = [s for s in sp if s["family"] != "prism9"]
    seq = [s for s in sp if s["family"] == "prism9"]
    with Pool(min(a.workers, 6)) as pool:
        for r in pool.imap_unordered(worker, par, chunksize=1):
            rows.append(r)
            print(f"[rho] {r['family']} {r.get('model_id')} {r['status']} {r['seconds']:.1f}s", flush=True)
    for s in seq:
        r = worker(s)
        rows.append(r)
        print(f"[rho] {r['family']} {r.get('model_id')} {r['status']} {r['seconds']:.1f}s", flush=True)
    df = pd.DataFrame(rows)
    order = {"primary": 0, "decision6": 1, "overcooked": 2, "tables13": 3, "prism9": 4}
    df["_o"] = df["family"].map(order)
    df = df.sort_values(["_o", "model_id"]).drop(columns="_o")
    df.to_csv(OUT / "outward_bound.csv", index=False)
    ok = df[df["status"] == "OK"]
    runs = ok[ok["family"].isin(["primary", "decision6", "overcooked"])]
    summ = {
        "n_models": int(len(df)), "n_ok": int(len(ok)),
        "families": {f: int((ok["family"] == f).sum()) for f in order},
        "rho_exact_minus_1_max": float(ok["rho_exact_minus_1"].max()),
        "rho_P_minus_1_max": float(ok["rho_P_minus_1"].max()),
        "rho_w_minus_1_max": float(ok["rho_w_minus_1"].max()),
        "n_models_code_rho_below_exact": int(ok["code_rho_below_exact"].sum()),
        "max_rho_exact_minus_code": float((-ok["rho_code_minus_rho_exact"]).max()),
        "n_models_subset_max_exceeds_allslot_product": int(ok["subset_max_exceeds_allslot_product"].sum()),
        "n_models_code_eps_x_below_outward": int((~ok["code_eps_x_covers_outward"].astype(bool)).sum()),
        "max_eps_x_out_minus_code": float(ok["eps_x_out_minus_code"].max()),
        "max_eps_x_rel_excess": float(ok["eps_x_rel_excess"].max()),
        "max_eta_v_out_minus_code": float(ok["eta_v_out_minus_code"].max()),
        "max_eps_x_exact_minus_code": float(ok["eps_x_exact_minus_code"].max()),
        "max_eps_x_exact_rel_excess": float((ok["eps_x_exact_minus_code"] / ok["eps_x_code"]).max()),
        "run_models": {
            "n": int(len(runs)),
            "n_used_eps_x_equals_code": int(runs["eps_x_used_equals_code"].astype(bool).sum()),
            "n_used_eps_x_below_outward": int((~runs["eps_x_used_covers_outward"].astype(bool)).sum()),
            "max_outward_minus_used": float((-runs["eps_x_used_minus_out"]).max()),
            "max_rel_outward_minus_used": float(((-runs["eps_x_used_minus_out"]) / runs["eps_x_used_min"]).max()),
            "n_used_eps_x_below_exact": int((runs["eps_x_exact_minus_used"] > 0).sum()),
            "max_exact_minus_used": float(runs["eps_x_exact_minus_used"].max()),
            "max_rel_exact_minus_used": float((runs["eps_x_exact_minus_used"] / runs["eps_x_used_min"]).max()),
        },
        "git_commit": rb.git_commit(), "hardware": rb.hardware_tags(), "time": rb.now_iso(),
        "method": "exact rational row/slot sums of the stored doubles; rho_w = max_s prod_i max(1, s_i) (all pinned "
                  "subsets); eta_v, eps_Upsilon, eps_X exactly as cubcr/fp_error.py in exact rational arithmetic, "
                  "rounded up to FP64 (rebuttal/lib/e1_rho.py)",
    }
    rb.write_json(OUT / "outward_bound_summary.json", summ)
    print(json.dumps(summ, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
