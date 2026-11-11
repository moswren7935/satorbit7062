from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import DMAX, CompiledCSG

UNIT_ROUNDOFF = {"fp64": 2.0 ** -53, "fp32": 2.0 ** -24}


def gamma(n: float, u: float) -> float:
    if n * u >= 1.0:
        raise ValueError("n u >= 1: the rounding bound is void")
    return n * u / (1.0 - n * u)


@dataclass(frozen=True)
class FPErrorBound:
    eta_v: float
    M: int
    u: float
    gamma_M: float
    rho: float
    k: int
    max_nnz_row: int
    max_rows_per_state: int
    dtype: str

    @property
    def eps_upsilon(self) -> float:

        return 2.0 * self.eta_v + UNIT_ROUNDOFF["fp64"]

    @property
    def eps_marginal(self) -> float:
        return 2.0 * self.eta_v + UNIT_ROUNDOFF["fp64"]

    def eps_unit(self, perms_per_unit: int) -> float:

        em = self.eps_marginal
        u64 = UNIT_ROUNDOFF["fp64"]
        return em + gamma(max(1, int(perms_per_unit)), u64) * (1.0 + em) + u64

    def statistical_range(self, upsilon_fp: float, perms_per_unit: int):

        ex = self.eps_unit(perms_per_unit)
        return (-ex, float(upsilon_fp) + self.eps_upsilon + ex)

    def as_dict(self) -> dict:
        return {"eta_v": self.eta_v, "M": self.M, "u": self.u, "gamma_M": self.gamma_M, "rho": self.rho,
                "k": self.k, "max_nnz_row": self.max_nnz_row, "max_rows_per_state": self.max_rows_per_state,
                "dtype": self.dtype, "eps_upsilon": self.eps_upsilon, "eps_marginal": self.eps_marginal}


def lipschitz_rho(comp: CompiledCSG) -> float:


    if comp.n_rows == 0:
        return 1.0
    row_abs = np.add.reduceat(np.abs(comp.val), comp.row_ptr[:-1]) if comp.nnz else np.zeros(comp.n_rows)
    empty = np.diff(comp.row_ptr) == 0
    row_abs = np.where(empty, 0.0, row_abs)
    rho_p = float(row_abs.max()) if row_abs.size else 1.0
    w_abs = np.abs(comp.weights).sum(axis=2)
    rho_w = float(np.prod(w_abs, axis=1).max()) if w_abs.size else 1.0
    return max(rho_p, 0.0) * max(rho_w, 0.0)


def forward_error_bound(comp: CompiledCSG, k: int | None = None, dtype: str = "fp64") -> FPErrorBound:
    if dtype not in UNIT_ROUNDOFF:
        raise ValueError("dtype must be 'fp64' or 'fp32'")
    if k is None:
        k = int(comp.horizon_k)
    u = UNIT_ROUNDOFF[dtype]
    nnz_row = np.diff(comp.row_ptr)
    max_nnz = int(nnz_row.max()) if nnz_row.size else 0
    max_rows = int(comp.max_rows_per_state) if comp.n_states else 0
    M = 2 * max_nnz + 2 * max_rows + DMAX + 2
    g = gamma(M, u)
    rho = lipschitz_rho(comp)
    k = int(k)
    eta = k * g * (rho ** k) * ((1.0 + g) ** max(k - 1, 0)) if k > 0 else 0.0
    return FPErrorBound(eta_v=float(eta), M=int(M), u=u, gamma_M=g, rho=rho, k=k,
                        max_nnz_row=max_nnz, max_rows_per_state=max_rows, dtype=dtype)
