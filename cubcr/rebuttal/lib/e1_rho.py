from __future__ import annotations

import math
from fractions import Fraction
from typing import Dict

import numpy as np

U64 = Fraction(1, 2 ** 53)


def fr(x: float) -> Fraction:
    return Fraction(float(x))


def round_up_float(q: Fraction) -> float:

    f = float(q)
    if Fraction(f) < q:
        f = float(np.nextafter(f, np.inf))
    return f


def round_down_float(q: Fraction) -> float:
    f = float(q)
    if Fraction(f) > q:
        f = float(np.nextafter(f, -np.inf))
    return f


def gamma_exact(n: int, u: Fraction = U64) -> Fraction:
    nu = n * u
    if nu >= 1:
        raise ValueError("n u >= 1")
    return nu / (1 - nu)


def _exact_sum(vals) -> Fraction:
    s = Fraction(0)
    for v in vals:
        s += Fraction(abs(float(v)))
    return s


def exact_max_row_sum(row_ptr: np.ndarray, val: np.ndarray) -> Fraction:

    row_ptr = np.asarray(row_ptr, dtype=np.int64)
    val = np.abs(np.asarray(val, dtype=np.float64))
    if row_ptr.size <= 1:
        return Fraction(1)
    nnz_row = np.diff(row_ptr)
    best = Fraction(0)
    seen = set()

    for L in np.unique(nnz_row):
        L = int(L)
        if L == 0:
            continue
        starts = row_ptr[:-1][nnz_row == L]
        idx = starts[:, None] + np.arange(L)[None, :]
        block = np.sort(val[idx], axis=1)
        uniq = np.unique(block, axis=0)


        approx_hi = uniq.sum(axis=1) * (1 + 4 * L * 2.0 ** -53)
        best_rd = round_down_float(best) if best else -1.0
        for row, hi in zip(uniq, approx_hi):
            if hi < best_rd:
                continue
            key = row.tobytes()
            if key in seen:
                continue
            seen.add(key)
            s = _exact_sum(row)
            if s > best:
                best = s
                best_rd = round_down_float(best)
    return best


def exact_slot_sums(weights: np.ndarray):

    w = np.abs(np.asarray(weights, dtype=np.float64))
    S, D, A = w.shape
    flat = w.reshape(S * D, A)
    uniq, inv = np.unique(flat, axis=0, return_inverse=True)
    sums = [_exact_sum(row) for row in uniq]
    inv = np.asarray(inv).reshape(-1)
    return [[sums[inv[s * D + i]] for i in range(D)] for s in range(S)], uniq.shape[0]


def exact_rho(comp) -> Dict:

    rho_P = exact_max_row_sum(comp.row_ptr, comp.val)
    slot, n_unique_slots = exact_slot_sums(comp.weights)
    one = Fraction(1)
    rho_w_subsets = Fraction(0)
    rho_w_allslots = Fraction(0)
    n_slot_below_one = 0
    n_slot_above_one = 0
    seen = set()
    for row in slot:
        key = tuple(row)
        if key in seen:
            continue
        seen.add(key)
        p_sub = one
        p_all = one
        for s in row:
            p_all *= s
            p_sub *= (s if s > one else one)
            n_slot_below_one += int(s < one)
            n_slot_above_one += int(s > one)
        rho_w_subsets = max(rho_w_subsets, p_sub)
        rho_w_allslots = max(rho_w_allslots, p_all)
    return dict(rho_P=rho_P, rho_w=rho_w_subsets, rho_w_allslots=rho_w_allslots, rho=rho_P * rho_w_subsets,
                n_unique_slot_vectors=int(n_unique_slots), n_distinct_slot_sum_vectors_below1=n_slot_below_one,
                n_distinct_slot_sum_vectors_above1=n_slot_above_one)


def outward_bound(comp, k: int, perms_per_unit: int = 2) -> Dict:


    from cubcr.fp_error import forward_error_bound
    from cubcr.model import DMAX
    code = forward_error_bound(comp, k, dtype="fp64")
    nnz_row = np.diff(comp.row_ptr)
    max_nnz = int(nnz_row.max()) if nnz_row.size else 0
    max_rows = int(comp.max_rows_per_state) if comp.n_states else 0
    M = 2 * max_nnz + 2 * max_rows + DMAX + 2
    if M != code.M:
        raise RuntimeError("flop-count constant differs from the code's")
    r = exact_rho(comp)
    rho = r["rho"]
    g = gamma_exact(M)
    k = int(k)
    eta = (k * g * rho ** k * (1 + g) ** max(k - 1, 0)) if k > 0 else Fraction(0)
    eps_ups = 2 * eta + U64
    em = 2 * eta + U64
    eps_x = em + gamma_exact(max(1, int(perms_per_unit))) * (1 + em) + U64
    out = dict(
        k=k, M=M, perms_per_unit=int(perms_per_unit),
        rho_code=float(code.rho), rho_exact_up=round_up_float(rho),
        rho_exact_minus_1=float(rho - 1), rho_P_minus_1=float(r["rho_P"] - 1), rho_w_minus_1=float(r["rho_w"] - 1),
        rho_w_allslots_minus_1=float(r["rho_w_allslots"] - 1),
        rho_code_minus_rho_exact=float(Fraction(code.rho) - rho),
        code_rho_below_exact=bool(Fraction(code.rho) < rho),
        subset_max_exceeds_allslot_product=bool(r["rho_w"] > r["rho_w_allslots"]),
        n_unique_slot_vectors=r["n_unique_slot_vectors"],
        eta_v_code=float(code.eta_v), eta_v_out=round_up_float(eta),
        eps_upsilon_code=float(code.eps_upsilon), eps_upsilon_out=round_up_float(eps_ups),
        eps_x_code=float(code.eps_unit(perms_per_unit)), eps_x_out=round_up_float(eps_x),
    )
    out["eps_x_out_minus_code"] = float(Fraction(out["eps_x_out"]) - Fraction(out["eps_x_code"]))

    out["eps_x_exact_minus_code"] = float(eps_x - Fraction(out["eps_x_code"]))
    out["_eps_x_exact"] = eps_x
    out["eps_x_rel_excess"] = out["eps_x_out_minus_code"] / out["eps_x_code"] if out["eps_x_code"] else 0.0
    out["eta_v_out_minus_code"] = float(Fraction(out["eta_v_out"]) - Fraction(out["eta_v_code"]))
    out["code_eps_x_covers_outward"] = bool(Fraction(out["eps_x_code"]) >= eps_x)
    return out
