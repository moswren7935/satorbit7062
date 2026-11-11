#!/usr/bin/env python3


from __future__ import annotations

import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "4")
import argparse
import glob
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import numpy as np
import pandas as pd

import rb
from e4_table_game import exact_shapley, monotone_violation_along_chains, n_from_table

OUT = rb.results_dir("E4")
STORED = [("e0", "results/e0/values/e0__*__gpu_batched_fp64_t0_s20260829.npy"),
          ("e0hz", "results/e0hz/values/e0hz__*__gpu_batched_fp64_t0_s20260829.npy"),
          ("e11", "results/e11/values/*.npy")]
OVERCOOKED = rb.TABLE_DIR / "overcooked_k10_step.npy"
OC_EXACT = rb.REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__exact_step.json"


def parse_iid(iid: str) -> dict:

    parts = iid.split("_")
    n = int(parts[2][1:]); S = int(parts[3][1:]); k = int(parts[4][1:]); seed = int(parts[7][1:])
    hz = float(parts[8][2:]) if len(parts) > 8 and parts[8].startswith("hz") else 0.0
    return {"n": n, "states": S, "k": k, "gen_seed": seed, "hazard": hz, "states_target": 1000 if S < 5000 else 10000}


def build_comp(info: dict):
    from experiments.grid import build_model
    comp, iid, meta, _ = build_model("ticketbench", info["n"], info["states_target"], info["k"], info["gen_seed"],
                                     "none", 0.05, hazard=info["hazard"])
    return comp, iid


def stored_phi_for(iid: str, family: str):

    pats = {"e0": f"results/e0/e0__{iid}__gpu_batched_fp64_t0_s20260829.json",
            "e0hz": f"results/e0hz/e0hz__{iid}__gpu_batched_fp64_t0_s20260829.json"}
    p = rb.REPO / pats.get(family, "__none__")
    if p.exists():
        return np.asarray(json.load(open(p))["shapley"], dtype=np.float64), str(p.relative_to(rb.REPO))
    return None, None


def main(argv=None):
    ap = argparse.ArgumentParser()
    a = ap.parse_args(argv)
    games = {}

    for fam, pat in STORED:
        for f in sorted(glob.glob(str(rb.REPO / pat))):
            base = os.path.basename(f)
            iid = base.split("__")[1] if fam != "e11" else base[:-4]
            games.setdefault(iid, {"table_id": iid, "family": fam, "path": str(Path(f).relative_to(rb.REPO)),
                                   "stored_paths": []})["stored_paths"].append(str(Path(f).relative_to(rb.REPO)))

    tasks = {t["iid"]: t for t in rb.paper_tasks() if t["n"] in (16, 20)}
    for iid, t in sorted(tasks.items()):
        tab = rb.exact_table(t)
        p = rb.table_path(iid)
        g = games.get(iid)
        if g is None:
            games[iid] = {"table_id": iid, "family": "grid", "path": str(p.relative_to(rb.REPO)), "stored_paths": []}
        else:
            st = np.load(rb.REPO / g["path"])
            g["regenerated_path"] = str(p.relative_to(rb.REPO))
            g["regenerated_bitwise_equal"] = bool(np.array_equal(st, tab))
            if not g["regenerated_bitwise_equal"]:
                raise RuntimeError(f"stored and regenerated tables differ for {iid}")
            g["family"] = g["family"] + "+grid"
    if OVERCOOKED.exists():
        games["overcooked_k10_step"] = {"table_id": "overcooked_k10_step", "family": "overcooked",
                                        "path": str(OVERCOOKED.relative_to(rb.REPO)), "stored_paths": []}
    rows = []
    fps = {}
    for iid in sorted(games, key=lambda s: (n_from_table(np.load(rb.REPO / games[s]["path"], mmap_mode="r")), s)):
        g = games[iid]
        tab = np.load(rb.REPO / g["path"])
        d = n_from_table(tab)
        t0 = time.perf_counter()
        phi = exact_shapley(tab)
        g.update(d=d, v_empty=float(tab[0]), v_full=float(tab[-1]), upsilon=float(tab[-1] - tab[0]),
                 phi_exact=phi.tolist(), shapley_s=time.perf_counter() - t0,
                 efficiency_dev=float(abs(phi.sum() - (tab[-1] - tab[0]))))
        g["tau_a_ceiling"] = float(rb.kendall_tau_a(phi, phi))
        srt = np.sort(phi)[::-1]
        g["n_tied_pairs_1e12"] = int(np.sum(np.abs(phi[:, None] - phi[None, :])[np.triu_indices(d, 1)] <= 1e-12))
        g["top1_tie"] = bool(abs(srt[0] - srt[1]) <= 1e-12)
        g["top3_boundary_tie"] = bool(abs(srt[2] - srt[3]) <= 1e-12)
        g["monotone_worst_drop_1000_chains"] = monotone_violation_along_chains(tab, 1000, seed=20260829)
        g["min_table_value"] = float(tab.min()); g["max_table_value"] = float(tab.max())
        if g["family"] == "overcooked":


            import warnings
            from generators import overcooked_csg as oc
            from cubcr.fp_error import forward_error_bound
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", 10, verbose=False)
                pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
                comp, index, meta_oc = oc.build_layered_compiled(keys, succ, layers, pol, 10)
            fp = forward_error_bound(comp, 10, dtype="fp64")
            fps[iid] = fp
            ex = json.load(open(OC_EXACT))
            sphi = np.asarray(ex["shapley"], dtype=np.float64).reshape(-1)
            g.update(k=10, n_agents=2, n_states=int(comp.n_states), gen_seed=None, hazard=None, build=None,
                     player_convention="p = i*k + t (role i, tick t)", stored_phi_source=str(OC_EXACT.relative_to(rb.REPO)),
                     stored_phi_max_dev=float(np.max(np.abs(phi - sphi))), paper_task=False,
                     stored_upsilon_equal=bool(float(ex["attributable_value"]) == g["upsilon"]),
                     fp_eps_upsilon=float(fp.eps_upsilon), fp_eps_unit2=float(fp.eps_unit(2)),
                     fp_eps_unit1=float(fp.eps_unit(1)))
            if int(comp.n_states) != int(ex["states"]):
                raise RuntimeError("overcooked state count mismatch")
        else:
            info = parse_iid(iid)
            comp, iid2 = build_comp(info)
            if iid2 != iid:
                raise RuntimeError(f"model id mismatch {iid2} != {iid}")
            from cubcr.fp_error import forward_error_bound
            fp = forward_error_bound(comp, info["k"], dtype="fp64")
            fps[iid] = fp
            g.update(k=info["k"], n_states=int(comp.n_states), gen_seed=info["gen_seed"], hazard=info["hazard"],
                     build=dict(n=info["n"], states_target=info["states_target"], k=info["k"],
                                gen_seed=info["gen_seed"], hazard=info["hazard"], pattern="none", pfail=0.05),
                     fp_eps_upsilon=float(fp.eps_upsilon), fp_eps_unit2=float(fp.eps_unit(2)),
                     fp_eps_unit1=float(fp.eps_unit(1)))

            sphi, src = stored_phi_for(iid, g["family"].split("+")[0])
            if sphi is None and iid in tasks and tasks[iid].get("phi_exact") is not None:
                sphi = tasks[iid]["phi_exact"]
                r = tasks[iid]["exact_gpu"] if tasks[iid]["has_exact_gpu"] else tasks[iid]["exact_cpu"]
                src = r["_file"]
            g["stored_phi_source"] = src
            g["stored_phi_max_dev"] = None if sphi is None else float(np.max(np.abs(phi - sphi)))
            g["paper_task"] = iid in tasks
        rows.append({k: v for k, v in g.items() if k not in ("phi_exact",)})
        print(f"{iid:50s} d={d} fam={g['family']:9s} ups={g['upsilon']:.4f} tau_ceiling={g['tau_a_ceiling']:.3f} "
              f"stored_dev={g.get('stored_phi_max_dev')}", flush=True)
    meta = {"created": rb.now_iso(), "hardware": rb.hardware_tags(), "n_games": len(games),
            "overcooked_present": OVERCOOKED.exists()}
    rb.write_json(OUT / "raw" / "games.json", {"meta": meta, "games": [games[r["table_id"]] for r in rows]})
    pd.DataFrame(rows).to_csv(OUT / "raw" / "games.csv", index=False)
    with open(OUT / "raw" / "fp_bounds.pkl", "wb") as f:
        pickle.dump(fps, f)
    print(f"{len(rows)} games -> {OUT / 'raw' / 'games.json'}")


if __name__ == "__main__":
    main()
