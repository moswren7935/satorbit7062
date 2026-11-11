from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb

EXP = "E2"
OUT = rb.RB / "results" / EXP
RAW = OUT / "raw"
RUNS_DIR = RAW / "partA_runs"
NEW_SEEDS = tuple(range(20261101, 20261121))
PAPER_SEEDS = tuple(rb.PAPER_SAMPLER_SEEDS)
TIE_TOL = 1e-12
PAPER_COVER_TOL = 1e-12


OC_IID = "overcooked_cramped_room_k10"
OC_N, OC_K = 2, 10
OC_D = OC_N * OC_K
OC_EXACT_JSON = rb.REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__exact_step.json"
OC_CERT_JSON = str(rb.REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__cert_gpu_eps_vs_exact_s{seed}.json")
OC_TABLE = rb.TABLE_DIR / "overcooked_k10_step.npy"


OC_SAMPLER_KW = dict(kind="step", n=OC_N, k=OC_K, batch_perms=16, alpha=0.05, max_perms=2_000_000,
                     time_budget=1800.0, antithetic=True, value_range="monotone", ci="prpl_eb_cs",
                     batch_schedule="geometric")


def build_overcooked_k10():

    import warnings
    warnings.filterwarnings("ignore")
    from generators import overcooked_csg as oc
    mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", OC_K, verbose=False)
    pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
    comp, index, meta = oc.build_layered_compiled(keys, succ, layers, pol, OC_K)
    return comp, meta


def overcooked_task() -> dict:

    ex = rb._read(OC_EXACT_JSON)
    phi = np.asarray(ex["shapley"], dtype=np.float64).reshape(-1)
    cert = {}
    for s in PAPER_SEEDS:
        f = OC_CERT_JSON.format(seed=s)
        if os.path.exists(f):
            r = rb._read(f)
            r["_file"] = os.path.relpath(f, rb.REPO)
            cert[s] = r
    eps20 = float(cert[min(cert)]["eps_abs"]) if cert else float(ex["attributable_value"]) / 20
    return {"iid": OC_IID, "family": "overcooked", "config": "overcooked_k10", "n": OC_N, "k": OC_K, "d": OC_D,
            "states": int(ex["states"]), "states_target": int(ex["states"]), "gen_seed": None,
            "upsilon": float(ex["attributable_value"]), "eps20": eps20, "phi_exact": phi,
            "exact_wall_s": float(ex["wall_s"]), "cert_gpu": cert, "has_exact_gpu": True}


def ticketbench_reference_tasks() -> List[dict]:
    out = []
    for t in rb.paper_tasks():
        if not t["has_exact_gpu"]:
            continue
        t = dict(t)
        t["family"] = "ticketbench"
        t["config"] = rb.cell_label(t)
        t["d"] = t["n"]
        out.append(t)
    return out


def reference_tasks(include_overcooked: bool = True) -> List[dict]:
    ts = ticketbench_reference_tasks()
    if include_overcooked:
        ts.append(overcooked_task())
    return ts


class StepTableEngine:


    dtype = "fp64"

    def __init__(self, table: np.ndarray, compiled, n: int, k: int):
        self.table = np.asarray(table, dtype=np.float64)
        if self.table.shape[0] != 1 << (n * k):
            raise ValueError("table size must be 2^(n*k)")
        self.compiled = compiled
        self.n, self.k = int(n), int(k)
        self.n_requested = 0
        self.n_calls = 0

    def bitsets(self, masks) -> np.ndarray:
        m = np.asarray(masks, dtype=np.uint64)
        if m.ndim != 2 or m.shape[0] != self.k:
            raise ValueError(f"step-level masks must have shape (k={self.k}, B)")
        if np.any(m >> np.uint64(self.n)):
            raise ValueError("mask word has bits for agents >= n")
        bits = np.zeros(m.shape[1], dtype=np.int64)
        for t in range(self.k):
            for i in range(self.n):
                b = ((m[t] >> np.uint64(i)) & np.uint64(1)).astype(np.int64)
                bits |= b << np.int64(i * self.k + t)
        return bits

    def solve(self, masks, k=None):
        if k is not None and int(k) != self.k:
            raise ValueError("horizon mismatch")
        bits = self.bitsets(masks)
        self.n_requested += bits.size
        self.n_calls += 1
        return self.table[bits]


def load_overcooked_table(mmap: bool = False) -> np.ndarray:
    if not OC_TABLE.exists():
        raise FileNotFoundError(f"{OC_TABLE} missing: run rebuttal/scripts/e2a_gpu_runs.py --overcooked-table")
    return np.load(OC_TABLE, mmap_mode="r" if mmap else None)


def load_task_table(task, mmap: bool = True) -> np.ndarray:


    p = rb.table_path(task["iid"])
    if not p.exists():
        raise FileNotFoundError(str(p))
    return np.load(p, mmap_mode="r" if mmap else None)


def snap_ties(x, tol: float = TIE_TOL) -> np.ndarray:


    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    o = np.argsort(x, kind="stable")
    xs = x[o]
    out = xs.copy()
    start = 0
    for i in range(1, xs.size + 1):
        if i == xs.size or xs[i] - xs[i - 1] > tol:
            if i - start > 1:
                out[start:i] = xs[start:i].mean()
            start = i
    res = np.empty_like(x)
    res[o] = out
    return res


def boundary_well_defined(phi, kk: int, tol: float = TIE_TOL) -> bool:

    s = np.sort(np.asarray(phi, dtype=np.float64))[::-1]
    if kk >= s.size:
        return True
    return bool(s[kk - 1] - s[kk] > tol)


def concentration_metrics(phi, ups: float, eps20: float) -> dict:
    phi = np.asarray(phi, dtype=np.float64)
    s = np.sort(phi)[::-1]
    d = s.size
    cum = np.cumsum(s) / ups
    n80 = int(np.searchsorted(cum, 0.8 - 1e-15) + 1)
    out = {"d": d, "upsilon": ups, "eps20": eps20, "sum_phi_minus_upsilon": float(phi.sum() - ups),
           "share_top1": float(s[0] / ups), "share_top2": float(s[:2].sum() / ups), "share_top3": float(s[:3].sum() / ups),
           "share_top5": float(s[:5].sum() / ups), "n_cover80": n80, "gini": rb.gini(phi),
           "n_gt_eps20": int(np.sum(phi > eps20)), "n_gt_ups100": int(np.sum(phi > ups / 100)),
           "gap12_eps20": float((s[0] - s[1]) / eps20), "gap23_eps20": float((s[1] - s[2]) / eps20),
           "gap34_eps20": float((s[2] - s[3]) / eps20), "max_share": float(s[0] / ups),
           "share_rank4": float(s[3] / ups),
           "min_share": float(s[-1] / ups), "mean_share": float(1.0 / d),
           "top1_unique": boundary_well_defined(phi, 1), "top2_unique": boundary_well_defined(phi, 2),
           "top3_unique": boundary_well_defined(phi, 3), "top4_unique": boundary_well_defined(phi, 4),
           "tie_groups": ";".join("{" + ",".join(str(p) for p in g) + "}" for g in tie_groups(phi))}
    return out


def tie_groups(phi, tol: float = TIE_TOL):

    phi = np.asarray(phi, dtype=np.float64)
    o = np.argsort(phi, kind="stable")
    groups, cur = [], [int(o[0])]
    for a, b in zip(o[:-1], o[1:]):
        if phi[b] - phi[a] <= tol:
            cur.append(int(b))
        else:
            if len(cur) > 1:
                groups.append(sorted(cur))
            cur = [int(b)]
    if len(cur) > 1:
        groups.append(sorted(cur))
    return sorted(groups, key=lambda g: -phi[g[0]])


def _sep(lower, upper, kk):
    return rb.cs_separated(lower, upper, kk)


def run_metrics(phi_hat, H, L, R, phi, ups: float, eps_run: float, eps20: float, ks=(1, 2, 3)) -> dict:

    phi_hat = np.asarray(phi_hat, dtype=np.float64); H = np.asarray(H, dtype=np.float64)
    phi = np.asarray(phi, dtype=np.float64)
    L = None if L is None else np.asarray(L, dtype=np.float64)
    R = None if R is None else np.asarray(R, dtype=np.float64)
    err = np.abs(phi_hat - phi)
    phs = snap_ties(phi)
    out = {"max_abs_err": float(err.max()), "max_abs_err_over_ups": float(err.max() / ups),
           "max_abs_err_over_eps_run": float(err.max() / eps_run), "max_abs_err_over_eps20": float(err.max() / eps20),
           "mean_abs_err": float(err.mean()), "mse": float(np.mean(err ** 2)), "hw_max": float(H.max()),
           "hw_max_over_eps_run": float(H.max() / eps_run)}
    big = phi >= eps20
    out["n_phi_ge_eps20"] = int(big.sum())
    if big.any():
        rel = err[big] / phi[big]
        out["rel_err_median"] = float(np.median(rel)); out["rel_err_max"] = float(rel.max())
    else:
        out["rel_err_median"] = out["rel_err_max"] = float("nan")
    T3 = rb.top_set(phs, 3)
    out["H_top3_over_phi_median"] = float(np.median(H[T3] / phi[T3]))
    out["H_top3_over_phi_max"] = float(np.max(H[T3] / phi[T3]))
    out["H_top3_over_ups_median"] = float(np.median(H[T3]) / ups)
    out["tau_a"] = rb.kendall_tau_a(phi_hat, phs)
    out["spearman"] = rb.spearman_rho(phi_hat, phs)
    out["tau_ge_09"] = bool(out["tau_a"] >= 0.9)

    tk = rb.topk_metrics(phi_hat, phs, ks=tuple(k for k in ks if k >= 1), tie_tol=TIE_TOL)
    out["top1_correct"] = bool(tk["top1_correct"])
    for kk in ks:
        wd = boundary_well_defined(phi, kk)
        T = set(rb.top_set(phs, kk).tolist())
        for name in ("exact", "recall", "jaccard"):
            v = tk.get(f"top{kk}_{name}")
            out[f"top{kk}_{name}"] = (float(v) if v is not None else float("nan")) if wd else float("nan")

        if L is not None and R is not None and wd:
            sp = _sep(L, R, kk)
            S = set(rb.top_set(L, kk).tolist())
            out[f"sep{kk}"] = float(sp)
            out[f"sep{kk}_and_correct"] = float(sp and S == T)
            out[f"sep{kk}_wrong_set"] = float(sp and S != T)
        else:
            out[f"sep{kk}"] = out[f"sep{kk}_and_correct"] = out[f"sep{kk}_wrong_set"] = float("nan")

        out[f"sep{kk}_reported"] = float(_sep(phi_hat - H, phi_hat + H, kk)) if wd else float("nan")

    cov_rep_strict = err <= H
    cov_rep_paper = err <= H + PAPER_COVER_TOL
    out["n_players"] = int(phi.size)
    out["n_cover_reported_strict"] = int(cov_rep_strict.sum())
    out["n_cover_reported_paper_tol"] = int(cov_rep_paper.sum())
    out["covers_reported_strict"] = bool(cov_rep_strict.all())
    out["covers_reported_paper_tol"] = bool(cov_rep_paper.all())
    if L is not None and R is not None:
        cov_cs = (L <= phi) & (phi <= R)
        out["n_cover_cs"] = int(cov_cs.sum()); out["covers_cs"] = bool(cov_cs.all())
    else:
        out["n_cover_cs"] = -1; out["covers_cs"] = None
    return out


def run_path(iid: str, div: int, seed: int) -> Path:
    return RUNS_DIR / f"{iid}__div{int(div)}__s{int(seed)}.json"


def result_payload(r, wall: float) -> dict:

    row = rb.result_row(r, wall)
    fp = float(r.fp_term)
    row["L"] = None if r.cs_lower is None else (np.asarray(r.cs_lower) - fp).tolist()
    row["R"] = None if r.cs_upper is None else (np.asarray(r.cs_upper) + fp).tolist()
    row["n_evals_unique"] = int(r.n_solves)
    row["n_evals_total"] = int(r.n_coalitions_requested)
    row["n_pairs"] = int(r.n_samples)
    return row


def stored_payload(rec: dict) -> dict:

    cert = rec["certificate"]
    fp = float(rec["fp_term"])
    n = int(rec["n"])
    return {"status": rec["sampling_status"], "certified": bool(rec["certified"]), "wall_s": float(rec["wall_s"]),
            "n_units": int(rec["n_units"]), "n_pairs": int(rec["n_units"]), "n_perms": int(rec["n_perms"]),
            "n_solves": int(rec["n_solves"]), "n_evals_unique": int(rec["n_solves"]),

            "n_evals_total": 2 + int(rec["n_perms"]) * (n + 1), "n_batches": int(rec["n_batches"]),
            "batch_sizes": list(rec["batch_sizes"]), "eps": float(rec["eps_abs"]), "fp_term": fp,
            "phi_hat": list(rec["shapley_sampled"]), "half_widths": list(rec["half_widths"]),
            "cs_lower": list(cert["cs_lower"]), "cs_upper": list(cert["cs_upper"]),
            "L": (np.asarray(cert["cs_lower"]) - fp).tolist(), "R": (np.asarray(cert["cs_upper"]) + fp).tolist(),
            "hw_max": float(rec["hw_max"])}
