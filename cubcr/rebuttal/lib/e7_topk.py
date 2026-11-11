from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb

KS = (1, 3)
E7_MAX_PERMS = 2_000_000
E7_GPU_TIME_BUDGET = 120.0
TIE_TOL = 1e-12
EXP = "E7"


def topk_kwargs(K: int, seed: int, n: int, k: int, time_budget: Optional[float],
                max_perms: int = E7_MAX_PERMS, **overrides) -> dict:


    kw = dict(rb.PAPER_SAMPLER_KW, eps=None, top_k=int(K), seed=int(seed), n=int(n), k=int(k),
              time_budget=None if time_budget is None else float(time_budget), max_perms=int(max_perms))
    kw.update(overrides)
    return kw


def eps_kwargs(eps: float, seed: int, n: int, k: int, time_budget: Optional[float] = "paper") -> dict:


    kw = rb.paper_sampler_kwargs(eps, seed, n, k)
    if time_budget != "paper":
        kw["time_budget"] = time_budget
    return kw


DT_SAMPLER_KW = dict(kind="step", batch_perms=16, alpha=0.05, antithetic=True, value_range="monotone",
                     two_sided=True, bonferroni=True, sequential=True, ci="prpl_eb_cs", batch_schedule="geometric")
DT_MODELS = [
    (3, 60, 4), (8, 1000, 25), (8, 10000, 25), (16, 1000, 25), (16, 10000, 25), (12, 100000, 25)]
DT_INSTANCE_SEED = 20260829


def dt_topk_kwargs(K: int, seed: int, n: int, k: int, time_budget: Optional[float] = E7_GPU_TIME_BUDGET,
                   max_perms: int = E7_MAX_PERMS) -> dict:
    return dict(DT_SAMPLER_KW, eps=None, top_k=int(K), seed=int(seed), n=int(n), k=int(k),
                time_budget=time_budget, max_perms=int(max_perms))


class StepTableEngine:


    dtype = "fp64"

    def __init__(self, table: np.ndarray, n: int, k: int, compiled=None):
        self.table = np.asarray(table, dtype=np.float64)
        self.n, self.k, self.compiled = int(n), int(k), compiled
        if self.table.shape[0] != (1 << (self.n * self.k)):
            raise ValueError("table size != 2^(n k)")
        self.n_requested = 0
        self.n_calls = 0

    def bitsets(self, masks) -> np.ndarray:
        M = np.asarray(masks, dtype=np.uint64)
        if M.ndim == 1:
            M = M.reshape(self.k, -1)
        if M.shape[0] != self.k:
            raise ValueError("step masks must have shape (k, B)")
        idx = np.zeros(M.shape[1], dtype=np.int64)
        for t in range(self.k):
            col = M[t]
            for i in range(self.n):
                idx |= ((col >> np.uint64(i)) & np.uint64(1)).astype(np.int64) << (i * self.k + t)
        return idx

    def solve(self, masks, k=None):
        idx = self.bitsets(masks)
        self.n_requested += idx.size
        self.n_calls += 1
        return self.table[idx]


def exact_topk(phi, K: int, tie_tol: float = TIE_TOL) -> dict:


    phi = np.asarray(phi, dtype=np.float64)
    if not 1 <= K < len(phi):
        raise ValueError("need 1 <= K < len(phi)")
    order = np.argsort(-phi, kind="stable")
    s = phi[order]
    gap = float(s[K - 1] - s[K])
    return {"top_set": sorted(int(i) for i in order[:K]), "gap": gap, "tie": bool(gap <= tie_tol),
            "phi_K": float(s[K - 1]), "phi_K1": float(s[K]),
            "n_tied_at_boundary": int(np.sum(np.abs(phi - s[K - 1]) <= tie_tol))}


def tie_aware_set_ok(players, phi, K: int, tie_tol: float = TIE_TOL) -> bool:


    phi = np.asarray(phi, dtype=np.float64)
    players = [int(p) for p in players]
    if len(set(players)) != K:
        return False
    thr = np.sort(phi)[::-1][K - 1] - tie_tol
    return bool(np.all(phi[players] >= thr))


def separation_holds(cs_lower, cs_upper, fp_term: float, K: int, estimates=None) -> dict:


    L = np.asarray(cs_lower, dtype=np.float64) - fp_term
    R = np.asarray(cs_upper, dtype=np.float64) + fp_term
    out = {"brief_rule": bool(rb.cs_separated(L, R, K))}
    if estimates is not None:
        order = np.argsort(-np.asarray(estimates, dtype=np.float64), kind="stable")
        top, rest = order[:K], order[K:]
        out["sampler_rule"] = bool(np.all(np.isfinite(L[top])) and np.min(L[top]) > np.max(R[rest]))
    return out


def run_sampler(engine, kw: dict):

    from cubcr.shapley_sampled import sample_shapley
    t0 = time.perf_counter()
    r = sample_shapley(engine, **kw)
    return r, time.perf_counter() - t0


def _scalar_cert(cert: Optional[dict]) -> Optional[dict]:
    if not cert:
        return None
    keep = ("kind", "m", "n_perms", "n_solves", "n_engine_calls", "wall_s", "engine_s", "look_index",
            "terminal_look", "terminal_reason", "max_half_width", "top_set", "batch_sizes")
    out = {}
    for key in keep:
        v = cert.get(key)
        out[key] = v.tolist() if isinstance(v, np.ndarray) else v
    return out


def kw_for_record(kw: dict) -> dict:
    return {key: (val if isinstance(val, (int, float, str, bool, type(None))) else str(val)) for key, val in kw.items()}


def result_record(r, wall: float, kw: dict, *, engine: str, task_meta: dict, phi_exact=None,
                  K: Optional[int] = None, extra: Optional[dict] = None) -> dict:

    N = int(r.n_players)
    fp = float(r.fp_term)
    cs_L = None if r.cs_lower is None else np.asarray(r.cs_lower, dtype=np.float64)
    cs_U = None if r.cs_upper is None else np.asarray(r.cs_upper, dtype=np.float64)
    rec = {
        "exp": EXP, "engine": engine, **task_meta,
        "sampler_seed": int(kw["seed"]), "rule": ("top_k" if kw.get("top_k") is not None else "eps"),
        "K": None if kw.get("top_k") is None else int(kw["top_k"]), "eps": kw.get("eps"),
        "config": kw_for_record(kw),
        "status": r.status, "certified": bool(r.certified), "wall_s": float(wall),
        "cert_wall_s": None if not r.certificate else float(r.certificate["wall_s"]),
        "n_units": int(r.n_samples), "n_perms": int(r.n_perms), "n_evals_unique": int(r.n_solves),
        "n_evals_requested": int(r.n_coalitions_requested), "cache_hits": int(r.cache_hits),
        "n_engine_calls": int(r.n_engine_calls), "n_batches": int(r.n_batches),
        "batch_sizes": [int(x) for x in r.batch_sizes], "engine_s": float(r.engine_time),
        "budget_overshoot_s": float(r.budget_overshoot_s),
        "v_empty": float(r.v_empty), "v_full": float(r.v_full),
        "value_range": None if r.value_range is None else [float(x) for x in r.value_range],
        "fp_term": fp, "eps_upsilon": float(r.eps_upsilon), "cs_anomaly": bool(r.cs_anomaly),
        "alpha_allocation": {kk: (vv if not isinstance(vv, np.ndarray) else vv.tolist())
                             for kk, vv in r.alpha_allocation().items() if kk != "look_m"},
        "n_players": N,
        "phi_hat": r.estimates.tolist(), "half_widths": r.half_widths.tolist(),
        "cs_lower": None if cs_L is None else cs_L.tolist(), "cs_upper": None if cs_U is None else cs_U.tolist(),
        "L": None if cs_L is None else (cs_L - fp).tolist(), "R": None if cs_U is None else (cs_U + fp).tolist(),
        "top_k_players": None if r.top_k_players is None else sorted(int(i) for i in r.top_k_players),
        "certificate": _scalar_cert(r.certificate),
        "graph_captures": r.graph_captures,
    }
    if K is not None:
        rec["est_top_set_at_stop"] = sorted(int(i) for i in np.argsort(-r.estimates, kind="stable")[:K])
        if cs_L is not None:
            rec["separation_recheck"] = separation_holds(cs_L, cs_U, fp, K, r.estimates)
    if phi_exact is not None:
        phi = np.asarray(phi_exact, dtype=np.float64).reshape(-1)
        rec["phi_exact"] = phi.tolist()
        err = np.abs(r.estimates - phi)
        rec["max_abs_err"] = float(err.max())
        if cs_L is not None and np.all(np.isfinite(cs_L)):
            rec["intervals_contain_exact"] = bool(np.all((cs_L - fp <= phi) & (phi <= cs_U + fp)))
        rec["covers_exact_hw"] = bool(np.all(err <= r.half_widths + 1e-12))
        if K is not None:
            ex = exact_topk(phi, K)
            rec["exact_topk"] = ex
            rec["certified_set_equals_exact"] = (None if rec["top_k_players"] is None
                                                 else rec["top_k_players"] == ex["top_set"])
            rec["est_set_equals_exact_at_stop"] = rec["est_top_set_at_stop"] == ex["top_set"]
    if extra:
        rec.update(extra)
    return rec


def task_meta(task: dict) -> dict:
    return {"iid": task["iid"], "n": int(task["n"]), "states_target": int(task["states_target"]),
            "states": int(task["states"]), "k": int(task["k"]), "gen_seed": int(task["gen_seed"]),
            "cell": rb.cell_label(task), "upsilon_stored": float(task["upsilon"])}


def reference_tasks():

    return [t for t in rb.paper_tasks() if t["has_exact_gpu"]]


def boundary_width_over_gap(phi_hat, L, R, K: int) -> float:


    ph = np.asarray(phi_hat, dtype=np.float64).reshape(-1)
    lo = np.asarray(L, dtype=np.float64).reshape(-1)
    hi = np.asarray(R, dtype=np.float64).reshape(-1)
    order = np.argsort(-ph, kind="stable")
    top, rest = order[:K], order[K:]
    if len(rest) == 0:
        return 0.0
    w = (ph[top] - lo[top])[:, None] + (hi[rest] - ph[rest])[None, :]
    gap = ph[top][:, None] - ph[rest][None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(gap > 0, w / np.where(gap > 0, gap, 1.0), np.inf)
    return float(ratio.max())


def raw_path(sub: str, name: str):
    p = rb.results_dir(EXP) / "raw" / sub
    p.mkdir(parents=True, exist_ok=True)
    return p / name


def run_name(rule: str, iid: str, seed: int) -> str:
    return f"{rule}__{iid}__s{seed}.json"
