from __future__ import annotations

import contextlib
import glob
import json
import math
import os
import time
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

import rb

E2 = rb.RB / "results" / "E2"
RAW = E2 / "raw"
RAW_GRID = RAW / "partB_grid"
RAW_DT = RAW / "partB_dt"
RAW_OC = RAW / "partB_oc"
LOGS = E2 / "logs"

GRID_DIVS = (10, 20, 50, 100, 200)
DIV200_STATES = 1000


DT_INSTANCE_SEED = 20260829
DT_MODELS = [
    (3, 60, 4), (8, 1000, 25), (8, 10000, 25), (16, 1000, 25), (16, 10000, 25), (12, 100000, 25)]
DT_RADII = (0.005, 0.0025, 0.001)
E4V2_DIR = rb.REPO / "results" / "e4v2"

DT_SAMPLER_KW = dict(kind="step", batch_perms=16, alpha=0.05, max_perms=100_000, time_budget=180.0,
                     antithetic=True, two_sided=True, bonferroni=True, sequential=True,
                     value_range="monotone", ci="prpl_eb_cs", batch_schedule="geometric")
DT_TIGHT_TIME_BUDGET = 1800.0


OC_K = 10
OC_N = 2
OC_IID = "overcooked_cramped_room_k10"
E14_DIR = rb.REPO / "results" / "e14"
OC_SAMPLER_KW = dict(kind="step", batch_perms=16, alpha=0.05, max_perms=2_000_000, time_budget=1800.0,
                     antithetic=True, value_range="monotone", ci="prpl_eb_cs", batch_schedule="geometric")
OC_DIVS = (20, 100)


_HW = None


def hw_tags() -> dict:
    global _HW
    if _HW is None:
        _HW = rb.hardware_tags()
    return _HW


def gpu_mem():
    try:
        from cubcr.engine_gpu import mem_info
        f, t = mem_info(0)
        return int(f), int(t)
    except Exception:
        return None, None


@contextlib.contextmanager
def group_lock(tag: str):


    t0 = time.perf_counter()
    with rb.gpu_lock(tag) as waited:
        g = {"group_tag": tag, "group_lock_wait_s": float(waited), "t_acquired": time.perf_counter(), "first": True}
        yield g
    g["group_hold_s"] = time.perf_counter() - g["t_acquired"]
    g["group_request_to_release_s"] = time.perf_counter() - t0


class ChunkedLock:


    def __init__(self, tag: str, max_hold_s: float = 240.0):
        self.tag, self.max_hold_s = tag, float(max_hold_s)
        self._cm = None
        self.held = None
        self.n_holds = 0
        self.log = []

    def ensure(self) -> dict:
        if self._cm is None:
            self._cm = group_lock(f"{self.tag} chunk{self.n_holds}")
            self.held = self._cm.__enter__()
            self.n_holds += 1
        return self.held

    def boundary(self):
        if self._cm is not None and time.perf_counter() - self.held["t_acquired"] >= self.max_hold_s:
            self.release()

    def release(self):
        if self._cm is not None:
            cm, h = self._cm, self.held
            self._cm, self.held = None, None
            cm.__exit__(None, None, None)
            self.log.append({"tag": h["group_tag"], "wait_s": h["group_lock_wait_s"], "hold_s": h.get("group_hold_s")})
            print(f"[lock] released {h['group_tag']}: waited {h['group_lock_wait_s']:.0f}s, held {h.get('group_hold_s', 0):.0f}s",
                  flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()
        return False


def locked_timed_run(tag: str, comp, warmup: Callable, run: Callable, held: Optional[dict] = None):


    t_req = time.perf_counter()
    if held is None:
        cm = rb.gpu_lock(tag)
    else:
        cm = contextlib.nullcontext(held["group_lock_wait_s"] if held.get("first") else 0.0)
    with cm as waited:
        t_acq = time.perf_counter()
        snap = rb.load_snapshot()
        free_b, total_b = gpu_mem()
        t0 = time.perf_counter()
        eng = rb.make_gpu_engine(comp)
        t1 = time.perf_counter()
        try:
            warmup(eng)
            t2 = time.perf_counter()
            out = run(eng)
            t3 = time.perf_counter()
            stats = dict(eng.stats) if getattr(eng, "stats", None) else None
        finally:
            eng.close()
        t4 = time.perf_counter()
    info = dict(lock_wait_s=float(waited), lock_hold_s=float(t4 - t_acq), engine_create_s=float(t1 - t0),
                warmup_s=float(t2 - t1), run_block_s=float(t3 - t2), close_s=float(t4 - t3),
                request_to_release_s=float(t4 - t_req), gpu_free_bytes_before=free_b, gpu_total_bytes=total_b,
                engine_stats_last=stats, load_snapshot=snap, timestamp=rb.now_iso(),
                lock_mode="per_run" if held is None else "group", group_tag=None if held is None else held["group_tag"])
    if held is not None:
        held["first"] = False
    return out, info


def locked_upsilon(tag: str, comp, masks, k: int, held: Optional[dict] = None):

    cm = rb.gpu_lock(tag) if held is None else contextlib.nullcontext(
        held["group_lock_wait_s"] if held.get("first") else 0.0)
    with cm as waited:
        t0 = time.perf_counter()
        eng = rb.make_gpu_engine(comp)
        try:
            vv = eng.solve(masks, k)
        finally:
            eng.close()
        hold = time.perf_counter() - t0
    if held is not None:
        held["first"] = False
    vv = np.asarray(vv, dtype=np.float64)
    return float(vv[1] - vv[0]), [float(vv[0]), float(vv[1])], dict(lock_wait_s=float(waited), lock_hold_s=float(hold),
                                                                    lock_mode="per_run" if held is None else "group")


def result_fields(r, outer_wall: float) -> dict:

    row = rb.result_row(r, outer_wall)
    cert = r.certificate or {}
    row.update(wall_time_internal_s=float(r.wall_time),
               time_to_certify_s=None if not cert else float(cert.get("wall_s")),
               eps_upsilon=float(r.eps_upsilon), graph_captures=int(r.graph_captures),
               n_engine_calls=int(r.n_engine_calls), n_looks_used=int(r.n_looks_used),
               n_players=int(r.n_players), kind=r.kind, alpha=float(r.alpha),
               max_perms=int(r.max_perms), time_budget=None if r.time_budget is None else float(r.time_budget),
               batch_perms=int(r.batch_perms), budget_overshoot_s=float(r.budget_overshoot_s),
               alpha_allocation=r.alpha_allocation())
    return row


def requested_invariant(n_requested: int, m: int, d: int) -> bool:

    return int(n_requested) == 2 + 2 * int(m) * (int(d) + 1)


def grid_path(iid: str, div: int, seed: int) -> Path:
    return RAW_GRID / f"partB_grid__{iid}__div{div}__s{seed}.json"


def grid_task_path(iid: str) -> Path:
    return RAW_GRID / f"partB_grid_task__{iid}.json"


def grid_divs_for(task, divs: Iterable[int]) -> List[int]:
    return [d for d in divs if d != 200 or task["states_target"] == DIV200_STATES]


def dt_build(n: int, S: int, k: int):
    from generators.ticketbench import build_compiled
    t0 = time.perf_counter()
    comp, meta = build_compiled(n, S, k, 0.05, "none", DT_INSTANCE_SEED)
    return comp, meta, time.perf_counter() - t0


def e4v2_record(iid: str, rule: str, seed: int) -> Optional[dict]:
    p = E4V2_DIR / f"e4__{iid}__{rule}__s{seed}.json"
    return rb._read(p) if p.exists() else None


def dt_sampler_kwargs(eps: float, seed: int, n: int, k: int, time_budget: Optional[float] = None,
                      callback=None) -> dict:
    kw = dict(DT_SAMPLER_KW, eps=float(eps), seed=int(seed), n=int(n), k=int(k))
    if time_budget is not None:
        kw["time_budget"] = float(time_budget)
    if callback is not None:
        kw["callback"] = callback
    return kw


def kwargs_from_e4v2_record(rec: dict) -> dict:

    s, v2 = rec["sampling"], rec["sampler_v2"]
    return dict(kind="step", batch_perms=int(s["batch_perms"]), alpha=float(rec["alpha"]),
                max_perms=int(s["max_perms"]), time_budget=float(s["time_budget_s"]), antithetic=True,
                two_sided=True, bonferroni=True, sequential=True, value_range=v2["value_range"], ci=v2["ci"],
                batch_schedule=v2["batch_schedule"])


def e4_warmup(eng, n: int, k: int, batch_perms: int = 16):

    N = n * k
    rng = np.random.default_rng(0)
    B = batch_perms * 2 * (N + 1)
    warm = rng.integers(0, 1 << n, size=(k, B), dtype=np.uint64)
    eng.solve(warm, k)


def curve_callback(store: list, t_ref: list):

    def cb(m, est, hw):
        store.append({"m": int(m), "t_s": time.perf_counter() - t_ref[0], "hw_max": float(np.max(hw)),
                      "hw_median": float(np.median(hw)), "hw_mean": float(np.mean(hw))})
    return cb


def run_step_sampler(eng, kw: dict):

    from cubcr.shapley_sampled import sample_shapley
    t0 = time.perf_counter()
    r = sample_shapley(eng, **kw)
    return r, time.perf_counter() - t0


def dt_path(iid: str, eps: float, seed: int) -> Path:
    return RAW_DT / f"partB_dt__{iid}__eps{eps:g}__s{seed}.json"


def step_upsilon_masks(n: int, k: int) -> np.ndarray:

    m = np.zeros((k, 2), dtype=np.uint64)
    m[:, 1] = np.uint64((1 << n) - 1)
    return m


def oc_build():

    np.Inf = np.inf
    from generators import overcooked_csg as oc
    t0 = time.perf_counter()
    mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", OC_K, verbose=False)
    pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
    comp, index, meta = oc.build_layered_compiled(keys, succ, layers, pol, OC_K)
    return comp, dict(meta, policy=pmeta), time.perf_counter() - t0


def oc_sampler_kwargs(eps: float, seed: int) -> dict:
    return dict(OC_SAMPLER_KW, n=OC_N, k=OC_K, seed=int(seed), eps=float(eps))


def oc_path(div: int, seed: int) -> Path:
    return RAW_OC / f"partB_oc__{OC_IID}__div{div}__s{seed}.json"


def e14_record(name: str) -> dict:
    return rb._read(E14_DIR / f"e14__{OC_IID}__{name}.json")


def interval_LR(cs_lower, cs_upper, fp_term):

    return (np.asarray(cs_lower, dtype=np.float64) - float(fp_term),
            np.asarray(cs_upper, dtype=np.float64) + float(fp_term))


def covers_paper(phi_hat, hw, phi, tol: float = 1e-12) -> bool:

    return bool(np.all(np.abs(np.asarray(phi_hat) - np.asarray(phi)) <= np.asarray(hw) + tol))


def covers_strict(phi_hat, hw, phi) -> bool:
    return bool(np.all(np.abs(np.asarray(phi_hat) - np.asarray(phi)) <= np.asarray(hw)))


def covers_LR(L, R, phi) -> bool:
    phi = np.asarray(phi)
    return bool(np.all((np.asarray(L) <= phi) & (phi <= np.asarray(R))))


def matched_ratios(exact_wall: float, sampled_walls: Sequence[float]) -> List[float]:

    return [float(exact_wall) / float(w) for w in sampled_walls]


def fit_loglog_slope(x: Sequence[float], y: Sequence[float]) -> float:

    lx = np.log(np.asarray(x, dtype=np.float64)); ly = np.log(np.asarray(y, dtype=np.float64))
    if len(lx) < 2 or np.ptp(lx) == 0:
        return float("nan")
    A = np.vstack([lx, np.ones_like(lx)]).T
    return float(np.linalg.lstsq(A, ly, rcond=None)[0][0])


def jaccard(a: Iterable[int], b: Iterable[int]) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b) if (a | b) else 1.0


def topk_set(values, kk: int):
    return rb.top_set(values, kk).tolist()


def pairwise_topk_jaccard(estimates: Sequence[Sequence[float]], kk: int) -> List[float]:
    out = []
    for i in range(len(estimates)):
        for j in range(i + 1, len(estimates)):
            out.append(jaccard(topk_set(estimates[i], kk), topk_set(estimates[j], kk)))
    return out


def med_minmax(xs) -> tuple:
    xs = [float(x) for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return (float("nan"), float("nan"), float("nan"))
    return (float(np.median(xs)), float(np.min(xs)), float(np.max(xs)))


def load_json_glob(pattern: str) -> List[dict]:
    out = []
    for f in sorted(glob.glob(pattern)):
        with open(f) as fh:
            d = json.load(fh)
        d["_file"] = os.path.relpath(f, rb.REPO)
        out.append(d)
    return out
