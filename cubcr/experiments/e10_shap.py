#!/usr/bin/env python3


from __future__ import annotations
import argparse, os, sys, time
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.grid import build_model, lib_hashes
from cubcr.shapley import shapley_from_table
from cubcr.shapley_sampled import sample_shapley

RESULTS = os.path.join(REPO, "results", "e10")
SEEDS = [20260829, 20260830, 20260831]
GEN_SEED = 20260829


class CountingEngine:


    def __init__(self, engine):
        self.engine = engine
        self.compiled = getattr(engine, "compiled", None)
        self.n_rows = 0
        self.n_calls = 0

    def solve(self, masks, k=None):
        arr = np.asarray(masks)
        b = int(arr.shape[-1]) if arr.ndim == 2 else (int(arr.shape[0]) if arr.ndim == 1 else 1)
        self.n_rows += b
        self.n_calls += 1
        return self.engine.solve(masks, k=k)


def masks_from_rows(X: np.ndarray, n: int) -> np.ndarray:


    X = np.asarray(X)
    if X.ndim == 1:
        X = X[None, :]
    if X.shape[1] != n:
        raise ValueError(f"expected rows with {n} columns, got shape {X.shape}")
    bits = (X > 0.5).astype(np.uint64)
    return (bits << np.arange(n, dtype=np.uint64)[None, :]).sum(axis=1).astype(np.uint64)


def make_value_fn(engine, n: int, k: int):

    def f(X):
        return engine.solve(masks_from_rows(X, n), k=k)
    return f


_SHAP_WARM = {"done": False}


def shap_explain(engine, n: int, k: int, npermutations: int, seed: int = 0,
                 warmup: bool = True) -> dict:


    import shap
    if npermutations < 1:
        raise ValueError("npermutations must be >= 1")
    ce = engine if isinstance(engine, CountingEngine) else CountingEngine(engine)
    f = make_value_fn(ce, n, k)
    masker = shap.maskers.Independent(np.zeros((1, n)), max_samples=1)
    x = np.ones((1, n))
    evals_per_perm = 2 * n + 1
    t0 = time.perf_counter()
    explainer = shap.PermutationExplainer(f, masker, seed=seed)
    setup_s = time.perf_counter() - t0
    warm_rows = 0
    if warmup and not _SHAP_WARM["done"]:
        r0 = ce.n_rows
        explainer(x, max_evals=evals_per_perm, batch_size=evals_per_perm, silent=True)
        warm_rows = ce.n_rows - r0
        _SHAP_WARM["done"] = True
        explainer = shap.PermutationExplainer(f, masker, seed=seed)
    rows0, calls0 = ce.n_rows, ce.n_calls
    max_evals = npermutations * evals_per_perm
    t0 = time.perf_counter()
    res = explainer(x, max_evals=max_evals, batch_size=evals_per_perm, silent=True)
    wall = time.perf_counter() - t0
    return dict(phi=np.asarray(res.values[0], dtype=np.float64),
                base=float(res.base_values[0]),
                wall_s=wall, setup_s=setup_s,
                n_oracle_rows=ce.n_rows - rows0, n_engine_calls=ce.n_calls - calls0,
                warmup_rows=warm_rows, npermutations=npermutations, max_evals=max_evals,
                shap_version=shap.__version__)


def npermutations_for_budget(budget_rows: int, n: int) -> int:

    return max(1, int(round(budget_rows / (2 * n + 1))))


def _kendall(a, b):
    try:
        from scipy.stats import kendalltau
        t = kendalltau(a, b)
        return float(getattr(t, "statistic", getattr(t, "correlation", np.nan)))
    except Exception:
        return None


def tool_record(iid, cell, comp, tool, seed, threads, cap_s, wall_s, n_rows, n_calls,
                phi_hat, phi_exact, ups, exact_ref, shap_version, **extra) -> dict:
    err = float(np.max(np.abs(phi_hat - phi_exact)))
    rec = protocol.make_record(
        instance_id=iid, engine="cpu_multicore", dtype="fp64", n=cell["n"],
        states=int(comp.n_states), k=cell["k"], seed=seed, threads=threads, exp="e10",
        tool=tool, cap_s=cap_s, wall_s=wall_s, censored=False, status="OK",
        n_oracle_rows=int(n_rows), n_engine_calls=int(n_calls),
        coalition_solves_per_s=(n_rows / wall_s) if wall_s > 0 else None,
        attributable_value=ups, max_abs_err_vs_exact=err,
        kendall_tau=_kendall(phi_hat, phi_exact),
        shapley_estimate=np.asarray(phi_hat, dtype=np.float64).tolist(),
        shapley_exact=np.asarray(phi_exact, dtype=np.float64).tolist(),
        exact_reference=exact_ref, shap_version=shap_version,
        pattern=cell.get("pattern", "none"), hazard=cell.get("hazard", 0.0),
        nnz=int(comp.nnz), provenance=lib_hashes(), cell=dict(cell), **extra)
    return rec


def run_exact(eng, n: int, k: int):

    masks = np.arange(1 << n, dtype=np.uint64)
    eng.solve(masks[: min(len(masks), 4096)], k)
    t0 = time.perf_counter()
    v = eng.solve(masks, k)
    t_solve = time.perf_counter() - t0
    t0 = time.perf_counter()
    phi = shapley_from_table(v, n)
    t_shap = time.perf_counter() - t0
    return v, phi, float(v[-1] - v[0]), t_solve + t_shap


def run_sampler_arm(eng, cell, seed, eps_div, cap_s):

    n, k = cell["n"], cell["k"]
    ce = CountingEngine(eng)


    vv = ce.solve(np.array([0, (1 << n) - 1], dtype=np.uint64), k)
    ups = float(vv[1] - vv[0])
    eps = ups / eps_div if ups > 0 else None
    t0 = time.perf_counter()
    r = sample_shapley(ce, kind="agent", n=n, k=k, seed=seed, batch_perms=32,
                       eps=eps, alpha=0.05, max_perms=2_000_000, time_budget=cap_s,
                       antithetic=True, value_range="monotone",
                       two_sided=True, bonferroni=True, sequential=True)
    wall = time.perf_counter() - t0
    fields = dict(certified=bool(r.certified), sampling_status=r.status, eps_abs=eps,
                  eps_div=eps_div, alpha=0.05, n_perms=int(r.n_perms), n_units=int(r.n_samples),
                  n_oracle_rows_requested=int(r.n_coalitions_requested),
                  hw_max=float(np.max(r.half_widths)) if np.all(np.isfinite(r.half_widths)) else None,
                  half_widths=r.half_widths.tolist(),
                  efficiency_gap=float(abs(r.estimates.sum() - (r.v_full - r.v_empty))),
                  alpha_allocation={"alpha": r.alpha, "alpha_player": r.alpha_player,
                                    "alpha_look": r.alpha_look, "n_looks_max": r.n_looks_max,
                                    "n_looks_used": r.n_looks_used},
                  accuracy_statement="CERTIFIED: anytime-valid empirical-Bernstein intervals "
                                     "(two-sided, Bonferroni over players, union bound over looks)")
    return r, wall, ce, ups, fields


def run_cell(cell, args, tags):
    from cubcr.engine_cpu import CPUEngine
    n, k = cell["n"], cell["k"]
    comp, iid, meta, build_s = build_model("ticketbench", n, cell["states"], k,
                                           GEN_SEED, cell.get("pattern", "none"), 0.05,
                                           hazard=cell.get("hazard", 0.0))
    print(f"[e10] {iid} (n={n} S={comp.n_states} k={k} threads={args.threads})", flush=True)
    eng = CPUEngine(comp, n_threads=args.threads, block=8, dtype="fp64")
    import shap as _shap


    v, phi_exact, ups, exact_wall = run_exact(eng, n, k)
    exact_ref = {"n_coalitions": 1 << n, "wall_s": exact_wall,
                 "source": "CPUEngine full enumeration + shapley_from_table (verification only, "
                           "not part of either tool's budget)"}
    print(f"  exact: 2^{n} solves in {exact_wall:.3f}s  Y={ups:.6f}", flush=True)


    if n <= 6:
        rows = ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)
        ce = CountingEngine(eng)
        fv = make_value_fn(ce, n, k)(rows)
        assert np.array_equal(masks_from_rows(rows, n), np.arange(1 << n, dtype=np.uint64))
        assert np.max(np.abs(fv - v)) == 0.0, "row->mask correspondence violated"
        assert ce.n_calls == 1, "f must call engine.solve exactly once per batch"
        print("  correspondence check (all 2^n rows through f): PASS", flush=True)

    for seed in args.seeds:

        r, wall, ce, ups2, fields = run_sampler_arm(eng, cell, seed, args.eps_div, args.cap)
        cov = bool(np.all(np.abs(r.estimates - phi_exact) <= r.half_widths + 1e-12))
        rec = tool_record(iid, cell, comp, "cubcr_sampler", seed, args.threads, args.cap,
                          wall, ce.n_rows, ce.n_calls, r.estimates, phi_exact, ups2,
                          exact_ref, _shap.__version__, covers_exact=cov, **fields)
        if r.status == "time":
            rec.update(status="TO", censored=True, wall_s=max(wall, args.cap),
                       coalition_solves_per_s=None, speedup_vs_cpu=None)
        protocol.write_result(args.out, rec, filename=f"e10__{iid}__cubcr_sampler_s{seed}.json",
                              tags=tags)
        budget_rows = ce.n_rows
        print(f"  [cubcr_sampler s{seed}] {r.status} certified={r.certified} wall={wall:.3f}s "
              f"rows={ce.n_rows} calls={ce.n_calls} hw_max={np.max(r.half_widths):.2e} "
              f"err={rec['max_abs_err_vs_exact']:.2e} covers={cov}", flush=True)


        for factor in (1, 2):
            nperm = npermutations_for_budget(factor * budget_rows, n)
            sh = shap_explain(eng, n, k, npermutations=nperm, seed=seed)
            add_gap = float(abs(sh["base"] + sh["phi"].sum() - v[-1]))
            base_gap = float(abs(sh["base"] - v[0]))
            rec = tool_record(iid, cell, comp, "shap_permutation", seed, args.threads, args.cap,
                              sh["wall_s"], sh["n_oracle_rows"], sh["n_engine_calls"],
                              sh["phi"], phi_exact, ups, exact_ref, sh["shap_version"],
                              certified=False,
                              accuracy_statement="NOT certified: shap provides no error "
                                                 "guarantee; max_abs_err_vs_exact is empirical only",
                              budget_factor=factor, budget_rows_target=int(factor * budget_rows),
                              budget_matched_to=f"cubcr_sampler seed {seed} oracle rows",
                              npermutations=sh["npermutations"], max_evals=sh["max_evals"],
                              explainer_setup_s=sh["setup_s"], warmup_rows=sh["warmup_rows"],
                              base_value=sh["base"], base_value_gap_vs_v_empty=base_gap,
                              additivity_gap=add_gap,
                              cap_note="cap_s is nominal: shap has no anytime stopping rule")
            protocol.write_result(args.out, rec,
                                  filename=f"e10__{iid}__shap_permutation_s{seed}_x{factor}.json",
                                  tags=tags)
            print(f"  [shap_permutation s{seed} x{factor}] wall={sh['wall_s']:.3f}s "
                  f"rows={sh['n_oracle_rows']} (target {factor * budget_rows}) "
                  f"perms={nperm} err={rec['max_abs_err_vs_exact']:.2e} "
                  f"addgap={add_gap:.1e} NO certificate", flush=True)
    eng.close()


def cells(scope):
    if scope == "smoke":
        return [dict(pattern="none", hazard=0.0, n=6, states=100, k=25)]
    return [dict(pattern="none", hazard=0.0, n=n, states=1000, k=25) for n in (12, 16)]


def main(argv=None):
    ap = argparse.ArgumentParser(description="Compare cuBCR and SHAP estimators")
    ap.add_argument("--scope", default="smoke", choices=["smoke", "full"],
                    help="default smoke: full must only run on an idle machine")
    ap.add_argument("--eps-div", type=int, default=20, help="eps = Y / eps_div (default Y/20)")
    ap.add_argument("--cap", type=float, default=900, help="sampler time budget / nominal cap [s]")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default: 32 full / 2 smoke)")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="sampler seeds (default: 3 preregistered seeds; smoke: first only)")
    ap.add_argument("--out", default=RESULTS)
    a = ap.parse_args(argv)
    if a.threads is None:
        a.threads = 32 if a.scope == "full" else 2
    if a.seeds is None:
        a.seeds = SEEDS if a.scope == "full" else SEEDS[:1]
    tags = protocol.machine_tags()
    os.makedirs(a.out, exist_ok=True)
    for cell in cells(a.scope):
        run_cell(cell, a, tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
