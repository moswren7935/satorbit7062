from __future__ import annotations

import math
import time
import warnings

import numpy as np

import rb
from e4_table_game import TableGame, EvalCounter, n_from_table

ESTIMATORS = ["cubcr_paired_perm", "plain_perm", "shapiq_PermutationSamplingSV", "shapiq_SVARM", "StratifiedSVARM",
              "shapiq_UnbiasedKernelSHAP", "shapiq_KernelSHAP", "LeverageSHAP", "shapiq_kADDSHAP",
              "shapiq_OwenSamplingSV", "shapiq_StratifiedSamplingSV", "Adalina"]
LABELS = {"cubcr_paired_perm": "Paired perm. (cuBCR)", "plain_perm": "Plain perm.",
          "shapiq_PermutationSamplingSV": "PermutationSamplingSV", "shapiq_SVARM": "SVARM (shapiq)",
          "StratifiedSVARM": "Stratified SVARM", "shapiq_UnbiasedKernelSHAP": "Unbiased KernelSHAP",
          "shapiq_KernelSHAP": "KernelSHAP", "LeverageSHAP": "Leverage SHAP", "shapiq_kADDSHAP": "2-ADD-SHAP",
          "shapiq_OwenSamplingSV": "Owen sampling", "shapiq_StratifiedSamplingSV": "Stratified sampling",
          "Adalina": "Adalina"}
SOURCE = {"cubcr_paired_perm": "cubcr (submission code)", "plain_perm": "cubcr (submission code)",
          "StratifiedSVARM": "our port of the reference implementation (validated)",
          "LeverageSHAP": "authors' reference code @9702c2cc", "Adalina": "authors' reference code @19554d00"}
for _k in ESTIMATORS:
    SOURCE.setdefault(_k, "shapiq 1.3.0")


def min_feasible_budget(est: str, d: int) -> int:

    if est == "cubcr_paired_perm":
        return 2 + 2 * (d + 1)
    if est == "plain_perm":
        return 2 + (d + 1)
    if est == "shapiq_PermutationSamplingSV":
        return 2 + (d - 1)
    if est == "StratifiedSVARM":
        from e4_stratified_svarm import min_budget
        return min_budget(d)
    if est == "shapiq_kADDSHAP":
        return 1 + d + d * (d - 1) // 2
    if est in ("shapiq_KernelSHAP", "LeverageSHAP"):
        return d + 2
    if est == "Adalina":
        return d + 2
    if est == "shapiq_OwenSamplingSV":
        return 1 + 2 * d
    if est == "shapiq_StratifiedSamplingSV":
        return 2 + 2 * d
    if est in ("shapiq_SVARM", "shapiq_UnbiasedKernelSHAP"):
        return 3
    raise KeyError(est)


class TableContext:


    def __init__(self, table_id: str, table: np.ndarray, phi_exact: np.ndarray, k: int, comp=None, fp=None):
        self.table_id = table_id
        self.table = np.ascontiguousarray(np.asarray(table, dtype=np.float64))
        self.d = n_from_table(self.table)
        self.phi = np.asarray(phi_exact, dtype=np.float64)
        self.upsilon = float(self.table[-1] - self.table[0])
        self.k = int(k)
        self.comp = comp
        self.fp = fp


def _cubcr(ctx: TableContext, T: int, seed: int, antithetic: bool) -> dict:
    from cubcr.shapley_sampled import sample_shapley
    d = ctx.d
    if antithetic:
        m = (T - 2) // (2 * (d + 1)); max_perms = 2 * m
    else:
        m = (T - 2) // (d + 1); max_perms = m
    if m < 1:
        raise ValueError("budget below one sampling unit")
    eng = rb.TableEngine(ctx.table, compiled=ctx.comp)

    kw = dict(rb.PAPER_SAMPLER_KW, eps=None, seed=int(seed), n=d, k=ctx.k, max_perms=int(max_perms),
              antithetic=bool(antithetic), time_budget=None)
    if ctx.fp is not None:
        kw["fp_bound"] = ctx.fp
    r = sample_shapley(eng, **kw)
    if r.n_samples != m:
        raise RuntimeError(f"sampler drew {r.n_samples} units, expected {m}")
    out = {"phi_hat": np.asarray(r.estimates, dtype=np.float64), "n_total": int(r.n_coalitions_requested),
           "n_unique": int(r.n_solves), "n_units": int(r.n_samples), "n_batches": int(r.n_batches),
           "L": None if r.cs_lower is None else np.asarray(r.cs_lower), "R": None if r.cs_upper is None else np.asarray(r.cs_upper),
           "hw_max": float(np.max(r.half_widths)) if np.all(np.isfinite(r.half_widths)) else float("nan")}
    expect = 2 + m * (d + 1) * (2 if antithetic else 1)
    if out["n_total"] != expect:
        raise RuntimeError(f"requested count {out['n_total']} != {expect}")
    return out


def _shapiq(ctx: TableContext, T: int, seed: int, name: str) -> dict:
    import shapiq
    game = TableGame(ctx.table)
    ap = getattr(shapiq.approximator, name)(n=ctx.d, random_state=int(seed))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iv = ap.approximate(budget=int(T), game=game)
    phi = np.array([iv[(p,)] for p in range(ctx.d)], dtype=np.float64)
    return {"phi_hat": phi, "n_total": game.n_total, "n_unique": game.n_unique,
            "n_units": int(getattr(iv, "estimation_budget", -1) or -1)}


def _counting_oracle(ctx: TableContext):
    cnt = EvalCounter(ctx.d)
    tab = ctx.table
    v0 = float(tab[0])

    def value_of_masks(masks):
        m = np.asarray(masks, dtype=np.int64).reshape(-1)
        cnt.add(m)
        return tab[m] - v0
    return cnt, value_of_masks


def run_estimator(est: str, ctx: TableContext, T: int, seed: int) -> dict:

    T = int(T)
    if T < min_feasible_budget(est, ctx.d):
        return {"status": "INFEASIBLE"}
    t0 = time.perf_counter()
    if est == "cubcr_paired_perm":
        out = _cubcr(ctx, T, seed, True)
    elif est == "plain_perm":
        out = _cubcr(ctx, T, seed, False)
    elif est.startswith("shapiq_"):
        out = _shapiq(ctx, T, seed, est[len("shapiq_"):])
    elif est == "StratifiedSVARM":
        from e4_stratified_svarm import stratified_svarm
        cnt, f = _counting_oracle(ctx)
        res = stratified_svarm(f, ctx.d, T, seed, mode="fast", count_empty=True)
        out = {"phi_hat": res["phi"], "n_total": cnt.n_total, "n_unique": cnt.n_unique, "n_units": res["n_main"]}
    elif est == "LeverageSHAP":
        from e4_external import run_leverage_shap
        cnt, f = _counting_oracle(ctx)
        Te = T - (T % 2)
        out = {"phi_hat": run_leverage_shap(f, ctx.d, Te, seed), "n_total": cnt.n_total, "n_unique": cnt.n_unique,
               "n_units": Te - 2}
    elif est == "Adalina":
        from e4_external import run_adalina
        cnt, f = _counting_oracle(ctx)
        out = {"phi_hat": run_adalina(f, ctx.d, T, seed), "n_total": cnt.n_total, "n_unique": cnt.n_unique,
               "n_units": ctx.d * ((T - 2) // ctx.d)}
    else:
        raise KeyError(est)
    out["wall_s"] = time.perf_counter() - t0
    out["status"] = "OK" if np.all(np.isfinite(out["phi_hat"])) else "NONFINITE"
    if out["n_total"] > T:
        raise RuntimeError(f"{est} used {out['n_total']} > budget {T} evaluations")
    return out


TIE_TOL = 1e-12


def snap_ties(x, tol: float = TIE_TOL) -> np.ndarray:


    x = np.asarray(x, dtype=np.float64)
    o = np.argsort(x, kind="stable"); xs = x[o]; out = xs.copy(); start = 0
    for i in range(1, xs.size + 1):
        if i == xs.size or xs[i] - xs[i - 1] > tol:
            if i - start > 1:
                out[start:i] = xs[start:i].mean()
            start = i
    res = np.empty_like(x); res[o] = out
    return res


def metrics(phi_hat, phi, upsilon: float) -> dict:
    phi_hat = np.asarray(phi_hat, dtype=np.float64); phi = np.asarray(phi, dtype=np.float64)
    err = phi_hat - phi
    tk = rb.topk_metrics(phi_hat, phi, ks=(1, 3), tie_tol=TIE_TOL)
    return {"mse": float(np.mean(err ** 2)), "max_abs": float(np.max(np.abs(err))),
            "max_abs_rel": float(np.max(np.abs(err)) / upsilon), "top1": float(tk["top1_correct"]),
            "top3_recall": float(tk.get("top3_recall", np.nan)), "tau_a": float(rb.kendall_tau_a(phi_hat, phi)),
            "tau_a_snap": float(rb.kendall_tau_a(phi_hat, snap_ties(phi))),
            "sum_err": float(np.sum(phi_hat) - upsilon)}


def fmt_vec(x) -> str:
    if x is None:
        return ""
    return " ".join(repr(float(v)) for v in np.asarray(x, dtype=np.float64).reshape(-1))


def parse_vec(s) -> np.ndarray:
    if s is None or (isinstance(s, float) and math.isnan(s)) or s == "":
        return np.array([])
    return np.array([float(v) for v in str(s).split()], dtype=np.float64)


def budget_grid(d: int, n_points: int = 12) -> list:

    hi = min((1 << d) // 2, 1 << 16)
    lo = 4 * d
    g = np.geomspace(lo, hi, n_points)
    g = np.unique((np.round(g / 2) * 2).astype(np.int64))
    return [int(x) for x in g]
