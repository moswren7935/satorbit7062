from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from . import vi_ref
from .masks import step_mask_matrix
from .shapley import shapley_from_table
from .shapley_sampled import CoalitionOracle, exact_player_table, heatmap, kendall_tau


@dataclass
class ScreeningResult:
    add: np.ndarray
    remove: np.ndarray
    v_empty: float
    v_full: float
    n_solves: int
    n_engine_calls: int
    kind: str
    n: int
    k: Optional[int]
    wall_time: float

    @property
    def n_players(self) -> int:
        return self.add.size

    @property
    def attributable(self) -> float:
        return self.v_full - self.v_empty

    def flat(self):
        return self.add.reshape(-1), self.remove.reshape(-1)

    def top_players(self, K: int, by: str = "add") -> np.ndarray:
        arr = (self.add if by == "add" else self.remove).reshape(-1)
        return np.argsort(-arr, kind="stable")[:K]


def singleton_screening(engine, n: Optional[int] = None, k: Optional[int] = None,
                        kind: str = "step") -> ScreeningResult:

    import time
    comp = getattr(engine, "compiled", None)
    if n is None:
        n = getattr(comp, "n_agents")
    if k is None:
        k = getattr(comp, "horizon_k")
    n_players = n if kind == "agent" else n * k
    orc = CoalitionOracle(engine, n_players, kind, n, k, use_cache=False)
    t0 = time.perf_counter()
    empty = orc.empty
    grand = orc.grand

    rows1 = np.concatenate([empty[None, :], orc.onehot], axis=0)
    vals1 = orc.values(rows1)

    rows2 = np.concatenate([grand[None, :], grand[None, :] & ~orc.onehot], axis=0)
    vals2 = orc.values(rows2)
    wall = time.perf_counter() - t0
    v_empty, v_full = float(vals1[0]), float(vals2[0])
    add = vals1[1:] - v_empty
    remove = v_full - vals2[1:]
    return ScreeningResult(add=heatmap(add, n, k, kind), remove=heatmap(remove, n, k, kind),
                           v_empty=v_empty, v_full=v_full, n_solves=orc.n_solves,
                           n_engine_calls=orc.n_engine_calls, kind=kind, n=n, k=k, wall_time=wall)


@dataclass
class ScreeningCrossCheck:
    add_exact: np.ndarray
    remove_exact: np.ndarray
    max_abs_diff_add: float
    max_abs_diff_remove: float
    shapley_exact: Optional[np.ndarray] = None
    tau_add_vs_shapley: Optional[float] = None
    tau_remove_vs_shapley: Optional[float] = None
    table: Optional[np.ndarray] = None
    method: str = ""


def exact_marginal_crosscheck(screen: ScreeningResult, engine=None, csg=None) -> ScreeningCrossCheck:


    n, k, kind = screen.n, screen.k, screen.kind
    N = screen.n_players
    add_s, rem_s = screen.flat()
    if csg is not None:
        def val(players):
            if kind == "agent":
                return vi_ref.solve(csg, int(sum(1 << p for p in players)), k)
            return vi_ref.solve(csg, step_mask_matrix(players, n, k).tolist(), k)
        v0 = val([])
        vN = val(range(N))
        add_e = np.array([val([p]) - v0 for p in range(N)])
        rem_e = np.array([vN - val([q for q in range(N) if q != p]) for p in range(N)])
        return ScreeningCrossCheck(add_exact=heatmap(add_e, n, k, kind), remove_exact=heatmap(rem_e, n, k, kind),
                                   max_abs_diff_add=float(np.max(np.abs(add_e - add_s))),
                                   max_abs_diff_remove=float(np.max(np.abs(rem_e - rem_s))),
                                   method="vi_ref")
    if engine is None:
        raise ValueError("pass either csg or engine")
    table = exact_player_table(engine, N, kind, n, k)
    full = (1 << N) - 1
    add_e = np.array([table[1 << p] - table[0] for p in range(N)])
    rem_e = np.array([table[full] - table[full ^ (1 << p)] for p in range(N)])
    phi = shapley_from_table(table, N)
    return ScreeningCrossCheck(add_exact=heatmap(add_e, n, k, kind), remove_exact=heatmap(rem_e, n, k, kind),
                               max_abs_diff_add=float(np.max(np.abs(add_e - add_s))),
                               max_abs_diff_remove=float(np.max(np.abs(rem_e - rem_s))),
                               shapley_exact=heatmap(phi, n, k, kind),
                               tau_add_vs_shapley=kendall_tau(add_s, phi),
                               tau_remove_vs_shapley=kendall_tau(rem_s, phi),
                               table=table, method="table")
