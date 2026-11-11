from __future__ import annotations

import json
from typing import Dict, List

import numpy as np

import rb
import e3_models as M

GEN_SEED = 20260829
SAMPLER_SEED = 20260829
REVISION_SEED = 20260829
RES = rb.RB / "results" / "E3"
RAW = RES / "raw"

DT_RECORD = rb.REPO / "results" / "e4v2" / "e4__tb_none_n12_S100009_k25_pf0.05_C3_s20260829__eps0.005__s20260829.json"
OC_RECORD = rb.REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__cert_gpu_eps_vs_exact_s20260829.json"

DT_SAMPLER_KW = dict(kind="step", batch_perms=16, alpha=0.05, max_perms=100000, time_budget=180.0, antithetic=True,
                     two_sided=True, bonferroni=True, sequential=True, value_range="monotone", ci="prpl_eb_cs",
                     batch_schedule="geometric")
DT_EPS = 0.005
OC_SAMPLER_KW = dict(kind="step", batch_perms=16, alpha=0.05, max_perms=2_000_000, time_budget=1800.0, antithetic=True,
                     value_range="monotone", ci="prpl_eb_cs", batch_schedule="geometric")
OC_EPS_DIV = 20

REVISION_GRID = [(16, 1000, 50), (16, 10000, 50), (28, 1000, 50), (28, 10000, 50), (32, 1000, 50), (32, 10000, 50)]


def grid_name(n, S, k) -> str:
    return rb.cell_label((n, S, k))


def all_configs() -> List[dict]:
    out = []
    for n in rb.GRID_N:
        for S in rb.GRID_STATES:
            for k in rb.GRID_K:
                out.append({"name": grid_name(n, S, k), "family": "grid", "n": n, "states_target": S, "k": k,
                            "kind": "agent", "players": n})
    out.append({"name": "dt_n12_S1e5_k25", "family": "dt", "n": 12, "states_target": 100000, "k": 25, "kind": "step",
                "players": 12 * 25})
    out.append({"name": "oc_k10", "family": "oc", "n": 2, "states_target": None, "k": M.OC_K, "kind": "step",
                "players": 2 * M.OC_K})
    return out


def config(name: str) -> dict:
    for c in all_configs():
        if c["name"] == name:
            return c
    raise KeyError(name)


def build(cfg):

    if cfg["family"] == "oc":
        return M.overcooked_build_phased(cfg["k"])
    return M.tb_build_phased(cfg["n"], cfg["states_target"], cfg["k"], GEN_SEED)


def endpoint_masks(cfg):

    n, k = cfg["n"], cfg["k"]
    full = (1 << n) - 1
    if cfg["kind"] == "agent":
        return np.array([0, full], dtype=np.uint64)
    m = np.zeros((k, 2), dtype=np.uint64)
    m[:, 1] = full
    return m


def random_masks(cfg, count: int, seed: int):
    rng = np.random.default_rng(seed)
    n, k = cfg["n"], cfg["k"]
    if cfg["kind"] == "agent":
        return rng.integers(0, 1 << n, size=count, dtype=np.uint64)
    return rng.integers(0, 1 << n, size=(k, count), dtype=np.uint64)


def needs_upsilon(cfg) -> bool:
    return cfg["family"] in ("grid", "oc")


def eps_from_upsilon(cfg, ups: float) -> float:
    if cfg["family"] == "grid":
        return ups / rb.PAPER_EPS_DIV
    if cfg["family"] == "oc":
        return ups / OC_EPS_DIV
    return DT_EPS


def sampler_kwargs(cfg, eps: float, seed: int = SAMPLER_SEED) -> dict:
    if cfg["family"] == "grid":
        return rb.paper_sampler_kwargs(eps, seed, cfg["n"], cfg["k"])
    kw = dict(DT_SAMPLER_KW if cfg["family"] == "dt" else OC_SAMPLER_KW)
    kw.update(n=cfg["n"], k=cfg["k"], seed=int(seed), eps=float(eps))
    return kw


def run_sampler(eng, cfg, eps: float, seed: int = SAMPLER_SEED):
    from cubcr.shapley_sampled import sample_shapley
    import time
    kw = sampler_kwargs(cfg, eps, seed)
    t0 = time.perf_counter()
    r = sample_shapley(eng, **kw)
    return r, time.perf_counter() - t0


def paper_warmup(eng, cfg) -> dict:

    n, k = cfg["n"], cfg["k"]
    if cfg["family"] == "grid":
        rb.paper_warmup(eng, n, k)
        return {"warmup": "e9v3 (rng 2, 64(n+1) agent masks)"}
    N = n * k
    B = 16 * 2 * (N + 1)
    warm = np.random.default_rng(0).integers(0, 1 << n, size=(k, B), dtype=np.uint64)
    eng.solve(warm, k)
    if cfg["family"] == "dt":
        return {"warmup": "e4_sampling.py (rng 0, 32(N+1) step masks)"}
    return {"warmup": "analogous to e4_sampling.py (rng 0, 32(N+1) step masks); E14 k=10 itself ran after its exhaustive table"}


def stored_record(cfg) -> Dict:

    if cfg["family"] == "grid":
        for t in rb.paper_tasks():
            if (t["n"], t["states_target"], t["k"], t["gen_seed"]) == (cfg["n"], cfg["states_target"], cfg["k"], GEN_SEED):
                r = t["cert_gpu"][SAMPLER_SEED]
                return {"file": r["_file"], "estimates": np.asarray(r["shapley_sampled"]),
                        "half_widths": np.asarray(r["half_widths"]), "n_perms": int(r["n_perms"]),
                        "n_solves": int(r["n_solves"]), "batch_sizes": list(r["batch_sizes"]), "wall_s": float(r["wall_s"]),
                        "upsilon": float(r["attributable_value"]), "eps": float(r["eps_abs"]),
                        "model_build_s": r.get("model_build_s"), "engine_setup_s": r.get("engine_setup_s")}
        raise KeyError(cfg["name"])
    if cfg["family"] == "dt":
        d = json.load(open(DT_RECORD))
        s = d["sampling"]
        return {"file": str(DT_RECORD.relative_to(rb.REPO)), "estimates": np.asarray(s["estimates"]).reshape(-1),
                "half_widths": np.asarray(s["half_widths"]).reshape(-1), "n_perms": int(s["n_perms"]),
                "n_solves": int(s["n_solves"]), "batch_sizes": list(d["sampler_v2"]["batch_sizes"]),
                "wall_s": float(s["wall_s"]), "upsilon": float(s["v_full"] - s["v_empty"]), "eps": DT_EPS,
                "warmup_s": d.get("warmup_s")}
    d = json.load(open(OC_RECORD))
    return {"file": str(OC_RECORD.relative_to(rb.REPO)), "estimates": np.asarray(d["estimates"]).reshape(-1),
            "half_widths": np.asarray(d["half_widths"]).reshape(-1), "n_perms": int(d["n_perms"]),
            "n_solves": int(d["n_solves"]), "batch_sizes": None, "wall_s": float(d["wall_s"]),
            "upsilon": float(d["attributable_value"]), "eps": float(d["eps_abs"])}


def compare_to_stored(r, st: dict, wall: float) -> dict:
    out = {"stored_file": st["file"],
           "est_bitwise": bool(np.array_equal(r.estimates, st["estimates"])),
           "hw_bitwise": bool(np.array_equal(r.half_widths, st["half_widths"])),
           "n_perms_equal": int(r.n_perms) == st["n_perms"], "n_solves_equal": int(r.n_solves) == st["n_solves"],
           "max_abs_est_diff": float(np.max(np.abs(r.estimates - st["estimates"]))),
           "stored_wall_s": st["wall_s"], "wall_ratio_vs_stored": float(wall / st["wall_s"])}
    if st.get("batch_sizes") is not None:
        out["batch_sizes_equal"] = list(map(int, r.batch_sizes)) == list(map(int, st["batch_sizes"]))
    out["reproduces_stored"] = bool(out["est_bitwise"] and out["hw_bitwise"] and out["n_perms_equal"]
                                    and out["n_solves_equal"] and out.get("batch_sizes_equal", True))
    return out


def same_result(a, b) -> bool:
    return bool(np.array_equal(a.estimates, b.estimates) and np.array_equal(a.half_widths, b.half_widths)
                and a.n_perms == b.n_perms and a.n_solves == b.n_solves and list(a.batch_sizes) == list(b.batch_sizes))


def p5_split(eng, wall: float) -> dict:


    c = eng.cum
    kern = c["kernel_ms"] / 1e3
    eng_total = c["total_ms"] / 1e3
    py = c["py_s"]
    return {"p5_kernel_s": kern, "p5_engine_nonkernel_s": eng_total - kern, "p5_wrapper_s": py - eng_total,
            "p5_sampler_host_s": wall - py, "p5_engine_calls": int(c["calls"]), "p5_coalitions": int(c["coalitions"]),
            "p5_waves": int(c["waves"]), "p5_max_batch": int(c["max_batch"])}


def gpu_apps() -> dict:


    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip().splitlines()
        return {"n_gpu_procs": len([l for l in q if l.strip()]), "gpu_procs": [" ".join(l.split()) for l in q]}
    except Exception as e:
        return {"n_gpu_procs": None, "gpu_procs_error": repr(e)}
