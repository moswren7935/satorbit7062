#!/usr/bin/env python3


from __future__ import annotations
import argparse, glob, json, math, os, sys, time, warnings
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.grid import lib_hashes
from cubcr.shapley_fast import shapley_from_table_fast
from cubcr.shapley_sampled import sample_shapley, kendall_tau

RESULTS = os.path.join(REPO, "results", "e11")
SEEDS = [20260829, 20260830, 20260831]
BUDGETS = [2 ** b for b in range(9, 17)]
SHAPIQ_ARMS = ["PermutationSamplingSV", "KernelSHAP", "UnbiasedKernelSHAP", "kADDSHAP", "SVARM",
               "StratifiedSamplingSV", "OwenSamplingSV"]


class TableOracle:

    compiled = None
    dtype = "fp64"

    def __init__(self, table, n):
        self.table = np.asarray(table, dtype=np.float64); self.n = n
        self.n_requested = 0; self.seen = set(); self.n_calls = 0

    def solve(self, masks, k=None):
        m = np.asarray(masks, dtype=np.uint64).astype(np.int64).reshape(-1)
        self.n_requested += m.size; self.n_calls += 1
        self.seen.update(np.unique(m).tolist())
        return self.table[m]

    def reset(self):
        self.n_requested = 0; self.seen = set(); self.n_calls = 0

    @property
    def n_distinct(self):
        return len(self.seen)


def metrics(phi_hat, phi):
    err = phi_hat - phi
    o = np.argsort(-phi, kind="stable")[:3]; oh = np.argsort(-phi_hat, kind="stable")[:3]
    return {"max_abs_err": float(np.max(np.abs(err))), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "kendall_tau": kendall_tau(phi_hat, phi), "top3_overlap": float(len(set(o.tolist()) & set(oh.tolist())) / 3),
            "sum_est": float(phi_hat.sum())}


def run_ours(orc, n, phi, ups, budget, seed, variant):
    N = n
    perms = max(2, budget // (N + 1))
    kw = dict(ci="prpl_eb_cs", batch_schedule="geometric", stratify=(variant == "ours_cs_strat")) if variant != "ours_eb" \
        else dict(ci="eb_looks")
    if variant == "ours_cs_strat":
        perms = max(1, budget // ((N + 1) * N))
    orc.reset()
    t0 = time.perf_counter()
    res = sample_shapley(orc, n, kind="agent", seed=seed, eps=None, alpha=0.05, value_range="monotone",
                         max_perms=perms, batch_perms=8, two_sided=True, bonferroni=True, sequential=True, **kw)
    wall = time.perf_counter() - t0
    m = metrics(res.estimates, phi)
    cov = bool(np.all(np.abs(res.estimates - phi) <= res.half_widths)) if np.all(np.isfinite(res.half_widths)) else None
    return res.estimates, dict(m, wall_host_s=wall, n_requested=orc.n_requested, n_distinct=orc.n_distinct,
                               n_perms=res.n_perms, hw_max=float(res.half_widths.max()) if np.all(np.isfinite(res.half_widths)) else None,
                               covers_exact=cov, certified_interval=True, ci_method=res.ci_method)


def run_shapiq(orc, n, phi, budget, seed, arm):
    import shapiq
    from shapiq import approximator as A

    class G(shapiq.Game):
        def __init__(self):
            super().__init__(n_players=n, normalize=False)

        def value_function(self, coalitions):
            C = np.asarray(coalitions, dtype=bool)
            masks = np.zeros(C.shape[0], dtype=np.uint64)
            for p in range(n):
                masks |= (C[:, p].astype(np.uint64) << np.uint64(p))
            return orc.solve(masks)
    game = G()
    ap = getattr(A, arm)(n=n, random_state=seed)
    orc.reset()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iv = ap.approximate(budget=int(budget), game=game)
    wall = time.perf_counter() - t0
    est = np.array([iv[(p,)] for p in range(n)], dtype=np.float64)
    m = metrics(est, phi)
    return est, dict(m, wall_host_s=wall, n_requested=orc.n_requested, n_distinct=orc.n_distinct,
                     estimation_budget=int(getattr(iv, "estimation_budget", -1)), certified_interval=False,
                     shapiq_version=shapiq.__version__)


def run_shap(orc, n, phi, budget, seed):
    from experiments.e10_shap import shap_explain, npermutations_for_budget
    nperm = npermutations_for_budget(budget, n)
    orc.reset()
    sh = shap_explain(orc, n, k=None, npermutations=nperm, seed=seed)
    m = metrics(sh["phi"], phi)
    return sh["phi"], dict(m, wall_host_s=sh["wall_s"], n_requested=orc.n_requested, n_distinct=orc.n_distinct,
                           npermutations=nperm, certified_interval=False, shap_version=sh["shap_version"])


def instances(extra_n20: bool):
    out = []
    for f in sorted(glob.glob(os.path.join(REPO, "results", "e0*", "values", "*.npy"))):
        iid = os.path.basename(f).split("__")[1]
        out.append((iid, f))
    if extra_n20:
        f20 = os.path.join(RESULTS, "values", "tb_none_n20_S961_k100_pf0.05_C3_s20260829.npy")
        if not os.path.exists(f20):
            from experiments.grid import build_model
            from cubcr.engine_gpu import GPUEngine
            comp, iid, meta, _ = build_model("ticketbench", 20, 1000, 100, 20260829, "none", 0.05)
            eng = GPUEngine(comp, dtype="fp64", use_graph=True)
            v = eng.solve(np.arange(1 << 20, dtype=np.uint64), 100); eng.close()
            os.makedirs(os.path.dirname(f20), exist_ok=True); np.save(f20, v)
        out.append(("tb_none_n20_S961_k100_pf0.05_C3_s20260829", f20))
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=int, nargs="*", default=BUDGETS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--arms", nargs="*", default=["ours_cs", "ours_cs_strat", "ours_eb"] + SHAPIQ_ARMS + ["shap"])
    ap.add_argument("--no-n20", action="store_true")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--tables-only", action="store_true", help="generate exact tables and exit")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    tags = protocol.machine_tags(use_cubcr_gpu=False)
    shapley_from_table_fast(np.zeros(8), 3)
    inputs = instances(not a.no_n20)
    if a.tables_only:
        print(f"{len(inputs)} exact tables available")
        return 0
    for iid, f in inputs:
        table = np.load(f); n = int(round(math.log2(table.size)))
        phi = shapley_from_table_fast(table, n); ups = float(table[-1] - table[0])
        orc = TableOracle(table, n)
        k = int(iid.split("_k")[1].split("_")[0]); states = int(iid.split("_S")[1].split("_")[0])
        for budget in a.budgets:
            for arm in a.arms:
                for seed in a.seeds:
                    name = f"e11__{iid}__{arm}__b{budget}__s{seed}.json"
                    if a.skip_existing and os.path.exists(os.path.join(RESULTS, name)):
                        continue
                    try:
                        if arm.startswith("ours"):
                            est, info = run_ours(orc, n, phi, ups, budget, seed, arm)
                        elif arm == "shap":
                            est, info = run_shap(orc, n, phi, budget, seed)
                        else:
                            est, info = run_shapiq(orc, n, phi, budget, seed, arm)
                        status = "OK"
                    except Exception as e:
                        est, info, status = None, {"error": repr(e)[:300]}, "NOT RUN"
                    rec = protocol.make_record(instance_id=iid, exp="e11", arm=arm, budget=int(budget), seed=seed,
                                               engine="cpu_multicore", dtype="fp64", n=n, states=states, k=k, threads=1,
                                               cap_s=3600, wall_s=info.get("wall_host_s"), status=status,
                                               attributable_value=ups, shapley_exact=phi.tolist(),
                                               estimates=None if est is None else np.asarray(est).tolist(),
                                               table_file=os.path.relpath(f, REPO), provenance=lib_hashes(), **info)
                    protocol.write_result(RESULTS, rec, filename=name, tags=tags)
                    if status == "OK":
                        print(f"  {iid[:34]} {arm:22s} b={budget:6d} s={seed} maxerr={info['max_abs_err']:.4f} "
                              f"tau={info['kendall_tau']:.3f} req={info['n_requested']} dist={info['n_distinct']} "
                              f"{info['wall_host_s']:.2f}s", flush=True)
                    else:
                        print(f"  {iid[:34]} {arm:22s} b={budget:6d} s={seed} NOT RUN {info.get('error')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
