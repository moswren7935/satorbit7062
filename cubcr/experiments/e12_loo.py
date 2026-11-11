#!/usr/bin/env python3


from __future__ import annotations
import argparse, glob, json, math, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.grid import build_model, lib_hashes
from cubcr.shapley_fast import shapley_from_table_fast
from cubcr.shapley_sampled import exact_player_table, kendall_tau
from cubcr.screening import singleton_screening

RESULTS = os.path.join(REPO, "results", "e12")


def rank_metrics(phi: np.ndarray, base: np.ndarray) -> dict:
    N = len(phi)
    o_phi = np.argsort(-phi, kind="stable"); o_b = np.argsort(-base, kind="stable")
    r_phi = np.empty(N, int); r_phi[o_phi] = np.arange(N)
    r_b = np.empty(N, int); r_b[o_b] = np.arange(N)
    iu = np.triu_indices(N, 1)
    sp = np.sign(phi[:, None] - phi[None, :])[iu]; sb = np.sign(base[:, None] - base[None, :])[iu]
    strict = (sp != 0) & (sb != 0)
    mis = float(np.mean(sp[strict] != sb[strict])) if strict.any() else 0.0
    return {"tau": kendall_tau(base, phi), "top1_agree": bool(o_phi[0] == o_b[0]),
            "top3_overlap": float(len(set(o_phi[:3].tolist()) & set(o_b[:3].tolist())) / min(3, N)),
            "max_rank_disp": int(np.max(np.abs(r_phi - r_b))), "misordered_pair_frac": mis,
            "top1_phi": int(o_phi[0]), "top1_base": int(o_b[0])}


def make_record(iid, source, comp, n, k, phi, add, remove, extra, engine, threads, wall_s, seed, tags):
    if comp is not None:
        states = int(comp.n_states)
    else:
        states = int(iid.split("_S")[1].split("_")[0])
    rec = protocol.make_record(instance_id=iid, exp="e12", source=source, engine=engine, dtype="fp64", n=n,
                               states=states, k=k, seed=seed,
                               threads=threads, cap_s=3600, wall_s=wall_s, status="OK",
                               shapley=phi.tolist(), loo_add=add.tolist(), loo_remove=remove.tolist(),
                               remove_vs_shapley=rank_metrics(phi, remove), add_vs_shapley=rank_metrics(phi, add),
                               provenance=lib_hashes(), **extra)
    protocol.write_result(RESULTS, rec, filename=f"e12__{source}__{iid}.json", tags=tags)
    m = rec["remove_vs_shapley"]
    print(f"  [{source}] {iid}: tau_remove={m['tau']:.3f} top1={m['top1_agree']} top3={m['top3_overlap']:.2f} "
          f"maxdisp={m['max_rank_disp']} mis={m['misordered_pair_frac']:.3f}", flush=True)
    return rec


def from_table(table: np.ndarray, n: int):
    phi = shapley_from_table_fast(table, n)
    full = (1 << n) - 1
    add = np.array([table[1 << p] - table[0] for p in range(n)])
    remove = np.array([table[full] - table[full & ~(1 << p)] for p in range(n)])
    return phi, add, remove


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    tags = protocol.machine_tags()
    from cubcr.engine_gpu import GPUEngine
    shapley_from_table_fast(np.zeros(8), 3)

    for f in sorted(glob.glob(os.path.join(REPO, "results", "e0*", "values", "*.npy"))):
        table = np.load(f); n = int(round(math.log2(table.size)))
        iid = os.path.basename(f).split("__")[1]
        k = int(iid.split("_k")[1].split("_")[0])
        t0 = time.perf_counter(); phi, add, remove = from_table(table, n)
        make_record(iid, "saved_table", None, n, k, phi, add, remove,
                    {"table_file": os.path.relpath(f, REPO), "attributable_value": float(table[-1] - table[0])},
                    "gpu_batched", 0, time.perf_counter() - t0, 20260829, tags)

    for f in sorted(glob.glob(os.path.join(REPO, "results", "e9", "*__exact_gpu.json"))):
        d = json.load(open(f))
        if d.get("status") != "OK" or "shapley" not in d:
            continue
        cell = d["cell"]; n = d["n"]
        comp, iid, meta, _ = build_model("ticketbench", n, cell["states"], cell["k"], d["seed"], cell["pattern"],
                                         0.05, hazard=cell.get("hazard", 0.0))
        eng = GPUEngine(comp, dtype="fp64", use_graph=True)
        t0 = time.perf_counter(); scr = singleton_screening(eng, n, cell["k"], kind="agent"); wall = time.perf_counter() - t0
        eng.close()
        phi = np.asarray(d["shapley"])
        make_record(iid, "e9_exact", comp, n, cell["k"], phi, scr.add.reshape(-1), scr.remove.reshape(-1),
                    {"attributable_value": float(scr.attributable), "e9_record": os.path.relpath(f, REPO),
                     "pattern": cell["pattern"], "hazard": cell.get("hazard", 0.0)}, "gpu_batched", 0, wall, d["seed"], tags)

    for pattern in ("none", "symmetric", "redundant", "joint", "dummy"):
        for n in (8, 12, 16):
            for k in (25, 100):
                for seed in (20260829, 20260830, 20260831):
                    try:
                        comp, iid, meta, _ = build_model("ticketbench", n, 1000, k, seed, pattern, 0.05)
                    except Exception as e:
                        print(f"  skip {pattern} n={n}: {e}"); continue
                    if a.skip_existing and os.path.exists(os.path.join(RESULTS, f"e12__ticketbench_new__{iid}.json")):
                        continue
                    eng = GPUEngine(comp, dtype="fp64", use_graph=True)
                    t0 = time.perf_counter(); table = exact_player_table(eng, n, kind="agent", n=n, k=k)
                    wall = time.perf_counter() - t0; eng.close()
                    phi, add, remove = from_table(np.asarray(table), n)
                    make_record(iid, "ticketbench_new", comp, n, k, phi, add, remove,
                                {"attributable_value": float(table[-1] - table[0]), "pattern": pattern,
                                 "pattern_meta": {kk: v for kk, v in meta.items() if kk not in ("instance_id",)}},
                                "gpu_batched", 0, wall, seed, tags)

    from tests.helpers import random_csg
    csg = random_csg(n_agents=6, n_states=200, k=25, seed=20260829, max_active=3, max_actions=4, min_actions=4)
    comp = csg.compile()
    eng = GPUEngine(comp, dtype="fp64", use_graph=True)
    t0 = time.perf_counter(); table = exact_player_table(eng, 6, kind="agent"); wall = time.perf_counter() - t0
    eng.close()
    phi, add, remove = from_table(np.asarray(table), 6)
    make_record("rand_n6_S200_k25_act3_a4_RMAX64", "random_csg_ceiling", comp, 6, 25, phi, add, remove,
                {"attributable_value": float(table[-1] - table[0]), "max_rows_per_state": int(comp.max_rows_per_state),
                 "note": "3 simultaneous actors x 4 actions = 64 rows/state (engine RMAX); documents the concurrency ceiling"},
                "gpu_batched", 0, wall, 20260829, tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
