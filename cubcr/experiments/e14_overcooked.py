#!/usr/bin/env python3


from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.grid import lib_hashes
from generators.ticketbench import load_compiled_npz
from cubcr.shapley_fast import shapley_from_table_fast
from cubcr.shapley_sampled import sample_shapley, exact_player_table, kendall_tau
from cubcr.screening import singleton_screening

RESULTS = os.path.join(REPO, "results", "e14")
SEEDS = [20260829, 20260830, 20260831]


def rec_base(iid, comp, k, engine, threads, **kw):
    return protocol.make_record(instance_id=iid, exp="e14", engine=engine, dtype="fp64", n=2, states=int(comp.n_states),
                                k=k, threads=threads, provenance=lib_hashes(), **kw)


def run_model(path, a, tags):
    comp, meta = load_compiled_npz(path)
    k = int(comp.horizon_k); iid = f"overcooked_cramped_room_k{k}"
    from cubcr.engine_gpu import GPUEngine
    from cubcr.engine_cpu import CPUEngine
    t0 = time.perf_counter(); eng = GPUEngine(comp, dtype="fp64", use_graph=True); setup_s = time.perf_counter() - t0

    t0 = time.perf_counter(); v = eng.solve(np.arange(4, dtype=np.uint64), k); wall = time.perf_counter() - t0
    phi_agent = shapley_from_table_fast(np.asarray(v), 2)
    ups = float(v[3] - v[0])
    rec = rec_base(iid, comp, k, "gpu_batched", 0, seed=SEEDS[0], cap_s=1800, wall_s=wall, arm="agent_exact",
                   values={"empty": float(v[0]), "P0": float(v[1]), "P1": float(v[2]), "N": float(v[3])},
                   attributable_value=ups, shapley_agent=phi_agent.tolist(), model_meta=meta, engine_setup_s=setup_s)
    protocol.write_result(RESULTS, rec, filename=f"e14__{iid}__agent_exact.json", tags=tags)
    print(f"[e14] {iid}: S={comp.n_states} v={np.round(v, 5).tolist()} Υ={ups:.5f} φ_agent={np.round(phi_agent, 5).tolist()}", flush=True)

    t0 = time.perf_counter(); scr = singleton_screening(eng, 2, k, kind="step"); wall = time.perf_counter() - t0
    rec = rec_base(iid, comp, k, "gpu_batched", 0, seed=SEEDS[0], cap_s=1800, wall_s=wall, arm="screening",
                   add=scr.add.tolist(), remove=scr.remove.tolist(), n_solves=scr.n_solves, attributable_value=ups,
                   top5_by_remove=scr.top_players(5, by="remove").tolist(), top5_by_add=scr.top_players(5, by="add").tolist())
    protocol.write_result(RESULTS, rec, filename=f"e14__{iid}__screening.json", tags=tags)

    N = 2 * k; eps = ups / a.eps_div
    for seed in a.seeds:
        for rule, kw in (("eps", dict(eps=eps)), ("top5", dict(top_k=5))):
            for which in a.engines:
                if which == "cpu" and (seed != a.seeds[0] or rule != "eps"):
                    continue
                name = f"e14__{iid}__cert_{which}_{rule}_s{seed}.json"
                if os.path.exists(os.path.join(RESULTS, name)) and not a.force:
                    continue
                if which == "gpu":
                    e, thr = eng, 0
                else:
                    e, thr = CPUEngine(comp, n_threads=32, block=8, dtype="fp64"), 32
                budget = a.budget if rule == "eps" else min(a.budget, 600.0)
                t0 = time.perf_counter()
                r = sample_shapley(e, kind="step", n=2, k=k, seed=seed, batch_perms=16, alpha=0.05, max_perms=2_000_000,
                                   time_budget=budget, antithetic=True, value_range="monotone", ci="prpl_eb_cs",
                                   batch_schedule="geometric", **kw)
                wall = time.perf_counter() - t0
                est = r.estimates.reshape(2, k); hw = r.half_widths.reshape(2, k)
                top5 = np.argsort(-r.estimates, kind="stable")[:5]
                rec = rec_base(iid, comp, k, "gpu_batched" if which == "gpu" else "cpu_multicore", thr, seed=seed,
                               cap_s=budget, wall_s=max(wall, budget) if r.status == "time" else wall,
                               censored=(r.status == "time"), status="TO" if r.status == "time" else "OK",
                               arm=f"cert_{which}_{rule}", rule=rule, rule_params=kw, certified=r.certified,
                               sampling_status=r.status, n_players=N, attributable_value=ups,
                               eps_abs=None if r.eps is None else float(r.eps), eps_div=None if r.eps is None else int(a.eps_div),
                               top_k_requested=None if r.top_k is None else int(r.top_k),
                               n_perms=r.n_perms, n_solves=r.n_solves, n_batches=r.n_batches, batch_sizes=r.batch_sizes,
                               engine_s=r.engine_time, hw_max=float(hw.max()) if np.all(np.isfinite(hw)) else None,
                               estimates=est.tolist(), half_widths=hw.tolist(), top5_players=[(int(p // k), int(p % k)) for p in top5],
                               top_k_certified=None if r.top_k_players is None else [(int(p // k), int(p % k)) for p in r.top_k_players],
                               alpha_allocation=r.alpha_allocation(), fp_error=r.fp_error, fp_term=r.fp_term,
                               tau_remove_vs_est=kendall_tau(scr.remove.reshape(-1), r.estimates),
                               tau_add_vs_est=kendall_tau(scr.add.reshape(-1), r.estimates),
                               efficiency_gap=float(abs(r.estimates.sum() - ups)), engine_setup_s=setup_s if which == "gpu" else None,
                               coalition_solves_per_s=(r.n_solves / wall if r.status != "time" else None))
                protocol.write_result(RESULTS, rec, filename=name, tags=tags)
                print(f"  [cert_{which} {rule} s{seed}] {r.status} certified={r.certified} wall={wall:.1f}s perms={r.n_perms} "
                      f"solves={r.n_solves} hw_max={hw.max():.4f} top5={rec['top5_players']}", flush=True)
                if which != "gpu":
                    e.close()
    eng.close()


def run_exact_k10(a, tags):

    sys.path.insert(0, REPO)
    from generators import overcooked_csg as oc
    from cubcr.engine_gpu import GPUEngine
    k = 10
    mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", k, verbose=False)
    pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
    comp, index, meta = oc.build_layered_compiled(keys, succ, layers, pol, k)
    iid = f"overcooked_cramped_room_k{k}"
    eng = GPUEngine(comp, dtype="fp64", use_graph=True)
    N = 2 * k
    t0 = time.perf_counter(); table = np.asarray(exact_player_table(eng, N, kind="step", n=2, k=k)); wall = time.perf_counter() - t0
    phi = shapley_from_table_fast(table, N); ups = float(table[-1] - table[0])
    if not ups > 0.0:
        raise RuntimeError(f"exact verification model is a null game at k={k}: Upsilon={ups}")
    scr = singleton_screening(eng, 2, k, kind="step")
    rec = rec_base(iid, comp, k, "gpu_batched", 0, seed=SEEDS[0], cap_s=1800, wall_s=wall, arm="exact_step",
                   n_coalitions=1 << N, shapley=phi.reshape(2, k).tolist(), attributable_value=ups,
                   loo_remove=scr.remove.tolist(), loo_add=scr.add.tolist(),
                   tau_remove_vs_shapley=kendall_tau(scr.remove.reshape(-1), phi), tau_add_vs_shapley=kendall_tau(scr.add.reshape(-1), phi),
                   top5_shapley=[(int(p // k), int(p % k)) for p in np.argsort(-phi, kind="stable")[:5]],
                   top5_remove=[(int(p // k), int(p % k)) for p in np.argsort(-scr.remove.reshape(-1), kind="stable")[:5]],
                   model_meta=dict(meta, policy=pmeta))
    protocol.write_result(RESULTS, rec, filename=f"e14__{iid}__exact_step.json", tags=tags)
    print(f"[e14] k={k} exact: S={comp.n_states} Υ={ups:.5f} tau_remove={rec['tau_remove_vs_shapley']:.3f} top5={rec['top5_shapley']} ({wall:.1f}s)", flush=True)
    for seed in a.seeds:
        r = sample_shapley(eng, kind="step", n=2, k=k, seed=seed, batch_perms=16, alpha=0.05, max_perms=2_000_000,
                           time_budget=a.budget, antithetic=True, value_range="monotone", ci="prpl_eb_cs",
                           batch_schedule="geometric", eps=ups / a.eps_div)
        cov = bool(np.all(np.abs(r.estimates - phi) <= r.half_widths))
        rec = rec_base(iid, comp, k, "gpu_batched", 0, seed=seed, cap_s=a.budget, wall_s=r.wall_time, arm="cert_gpu_eps_vs_exact",
                       certified=r.certified, covers_exact=cov, max_abs_err_vs_exact=float(np.abs(r.estimates - phi).max()),
                       n_perms=r.n_perms, n_solves=r.n_solves, hw_max=float(r.half_widths.max()), estimates=r.estimates.reshape(2, k).tolist(),
                       half_widths=r.half_widths.reshape(2, k).tolist(), attributable_value=ups, eps_abs=ups / a.eps_div,
                       coalition_solves_per_s=r.n_solves / r.wall_time if r.wall_time > 0 else None)
        protocol.write_result(RESULTS, rec, filename=f"e14__{iid}__cert_gpu_eps_vs_exact_s{seed}.json", tags=tags)
        print(f"  [k={k} cert s{seed}] certified={r.certified} covers_exact={cov} perms={r.n_perms} wall={r.wall_time:.2f}s", flush=True)
    eng.close()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["datasets/overcooked/overcooked_cramped_room_k40.npz",
                                                     "datasets/overcooked/overcooked_cramped_room_k30.npz",
                                                     "datasets/overcooked/overcooked_cramped_room_k20.npz"])
    ap.add_argument("--engines", nargs="*", default=["gpu", "cpu"])
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--budget", type=float, default=1800)
    ap.add_argument("--eps-div", type=int, default=20)
    ap.add_argument("--skip-exact", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    os.makedirs(RESULTS, exist_ok=True)
    tags = protocol.machine_tags()
    shapley_from_table_fast(np.zeros(8), 3)
    if not a.skip_exact:
        run_exact_k10(a, tags)
    for m in a.models:
        run_model(os.path.join(REPO, m), a, tags)
    return 0


if __name__ == "__main__":
    sys.exit(main())
