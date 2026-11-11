#!/usr/bin/env python3


from __future__ import annotations
import argparse, json, os, shutil, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.grid import build_model, lib_hashes
from cubcr.shapley_fast import shapley_from_table_fast
from cubcr.shapley_sampled import sample_shapley

RESULTS = os.path.join(REPO, "results", "e9v3")
OLD_RESULTS = os.path.join(REPO, "results", "e9")
GEN_SEEDS = [20260829, 20260830, 20260831]
SEEDS = [20260829, 20260830, 20260831]
UPSILON_FLOOR = 0.05


def make_engine(which, comp):
    t0 = time.perf_counter()
    if which == "gpu":
        from cubcr.engine_gpu import GPUEngine
        eng, thr = GPUEngine(comp, dtype="fp64", use_graph=True), 0
    else:
        from cubcr.engine_cpu import CPUEngine
        eng, thr = CPUEngine(comp, n_threads=32, block=8, dtype="fp64"), 32
    return eng, thr, time.perf_counter() - t0


def base_record(a, iid, meta, comp, engine, threads, gen_seed, **kw):
    kw.setdefault("seed", SEEDS[0])
    return protocol.make_record(instance_id=iid, engine=engine, dtype="fp64", n=a["n"], states=int(comp.n_states),
                                k=a["k"], threads=threads, exp="e9v3", pattern=a["pattern"], hazard=a.get("hazard", 0.0),
                                nnz=int(comp.nnz), provenance=lib_hashes(), cell=dict(a), gen_seed=gen_seed,
                                concurrent_load=bool(a.get("_concurrent", False)), model_build_s=a.get("_build_s"), **kw)


def run_exact(a, iid, meta, comp, which, cap_s, screen_cap_s, tags, gen_seed):
    eng, threads, setup0 = make_engine(which, comp)
    n = a["n"]; total = 1 << n; B0 = min(total, 65536)
    probe = np.random.default_rng(1).integers(0, total, size=B0, dtype=np.uint64)
    t0 = time.perf_counter(); eng.solve(probe, a["k"]); warm_s = time.perf_counter() - t0
    t0 = time.perf_counter(); eng.solve(probe, a["k"]); probe_s = time.perf_counter() - t0
    eng.close() if hasattr(eng, "close") else None
    rate = B0 / probe_s; projected = total / rate
    screen = {"screen_probe_B": B0, "screen_probe_s": probe_s, "screen_warm_s": warm_s, "screen_rate_coal_per_s": rate,
              "projected_full_s": projected, "screen_cap_s": screen_cap_s, "screen_isolated": True}
    engine_name = "gpu_batched" if which == "gpu" else "cpu_multicore"
    name = f"e9v3__{iid}__exact_{which}.json"
    if projected > cap_s:
        rec = base_record(a, iid, meta, comp, engine_name, threads, gen_seed, cap_s=cap_s, wall_s=None, censored=False,
                          status="NOT RUN", arm=f"exact_{which}", n_coalitions=total, projected_full_s=projected,
                          adaptive_screen=screen, note="screen projection exceeded the cap; NOT RUN (projection is not a measurement)")
        protocol.write_result(RESULTS, rec, filename=name, tags=tags)
        print(f"  [exact_{which}] SCREEN-ONLY: projected {projected:.0f}s > cap {cap_s}s -> NOT RUN", flush=True)
        return None, None, None
    eng, threads, setup_s = make_engine(which, comp)
    t0 = time.perf_counter(); eng.solve(probe, a["k"]); setup_s += time.perf_counter() - t0
    n_rep = 3 if projected <= 300 else 1
    masks = np.arange(total, dtype=np.uint64)
    walls = []
    for _ in range(n_rep):
        t0 = time.perf_counter(); v = eng.solve(masks, a["k"]); walls.append(time.perf_counter() - t0)
    solve_s = float(np.median(walls))
    t0 = time.perf_counter(); phi = shapley_from_table_fast(v, n); shap_s = time.perf_counter() - t0
    ups = float(v[-1] - v[0])
    rec = base_record(a, iid, meta, comp, engine_name, threads, gen_seed, cap_s=cap_s, wall_s=solve_s + shap_s,
                      censored=False, status="OK", arm=f"exact_{which}", n_coalitions=total, solve_s=solve_s,
                      shapley_s=shap_s, coalition_solves_per_s=total / solve_s, attributable_value=ups,
                      shapley=phi.tolist(), adaptive_screen=screen, n_repeat=n_rep, all_s=walls,
                      engine_setup_s=setup_s, pipeline_setup_inclusive_s=a.get("_build_s", 0.0) + setup_s + solve_s + shap_s)
    protocol.write_result(RESULTS, rec, filename=name, tags=tags)
    print(f"  [exact_{which}] OK solve {solve_s:.1f}s + shapley {shap_s:.2f}s (projected {projected:.0f}s, setup {setup_s:.2f}s)", flush=True)
    eng.close() if hasattr(eng, "close") else None
    return phi, ups, solve_s + shap_s


def run_certified(a, iid, meta, comp, which, seed, ups, eps_div, budget_s, tags, phi_exact, gen_seed, sampler="cs"):
    eng, threads, setup_s = make_engine(which, comp)
    n = a["n"]; eps = ups / eps_div

    t0 = time.perf_counter(); eng.solve(np.random.default_rng(2).integers(0, 1 << n, size=32 * 2 * (n + 1), dtype=np.uint64), a["k"])
    setup_s += time.perf_counter() - t0
    kw = dict(ci="prpl_eb_cs", batch_schedule="geometric") if sampler == "cs" else dict(ci="eb_looks")
    t0 = time.perf_counter()
    r = sample_shapley(eng, kind="agent", n=n, k=a["k"], seed=seed, batch_perms=32, eps=eps, alpha=0.05,
                       max_perms=2_000_000, time_budget=budget_s, antithetic=True, value_range="monotone",
                       two_sided=True, bonferroni=True, sequential=True, **kw)
    wall = time.perf_counter() - t0
    cov = eff = None
    if phi_exact is not None:
        cov = bool(np.all(np.abs(r.estimates - phi_exact) <= r.half_widths + 1e-12))
        eff = float(np.max(np.abs(r.estimates - phi_exact)))
    arm = f"cert_{which}" + ("" if sampler == "cs" else "_eblooks")
    rec = base_record(a, iid, meta, comp, "gpu_batched" if which == "gpu" else "cpu_multicore", threads, gen_seed,
                      seed=seed, cap_s=budget_s, wall_s=wall, censored=(r.status == "time"),
                      status="TO" if r.status == "time" else "OK", arm=arm, sampler=sampler,
                      n_coalitions=r.n_solves, coalition_solves_per_s=(r.n_solves / wall if r.status != "time" else None),
                      certified=r.certified, sampling_status=r.status, eps_abs=eps, eps_div=eps_div, attributable_value=ups,
                      n_perms=r.n_perms, n_solves=r.n_solves, n_units=r.n_samples, n_batches=r.n_batches,
                      batch_sizes=r.batch_sizes, engine_s=r.engine_time,
                      hw_max=float(np.max(r.half_widths)) if np.all(np.isfinite(r.half_widths)) else None,
                      value_range=list(r.value_range) if r.value_range else None, alpha_allocation=r.alpha_allocation(),
                      fp_error=r.fp_error, fp_term=r.fp_term, cs_anomaly=r.cs_anomaly, graph_captures=r.graph_captures,
                      certificate={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in (r.certificate or {}).items()},
                      shapley_sampled=r.estimates.tolist(), half_widths=r.half_widths.tolist(), covers_exact=cov,
                      max_abs_err_vs_exact=eff, efficiency_gap=float(abs(r.estimates.sum() - (r.v_full - r.v_empty))) if r.n_samples else 0.0,
                      engine_setup_s=setup_s, pipeline_setup_inclusive_s=a.get("_build_s", 0.0) + setup_s + wall)
    if rec["censored"]:
        rec["wall_s"] = max(wall, budget_s); rec["coalition_solves_per_s"] = None
    protocol.write_result(RESULTS, rec, filename=f"e9v3__{iid}__{arm}_s{seed}.json", tags=tags)
    print(f"  [{arm} s{seed}] {r.status} certified={r.certified} wall={wall:.2f}s perms={r.n_perms} solves={r.n_solves} "
          f"batches={r.n_batches} hw_max={np.max(r.half_widths):.2e} covers={cov} setup={setup_s:.2f}s", flush=True)
    eng.close() if hasattr(eng, "close") else None
    return wall, r.certified


def cells(scope):
    out = []
    for n in (16, 20, 24, 28, 32):
        for S in (1000, 10000):
            for k in (50, 100):
                out.append(dict(pattern="none", hazard=0.0, n=n, states=S, k=k))
    if scope == "smoke":
        out = [dict(pattern="none", hazard=0.0, n=12, states=1000, k=50)]
    elif scope == "quick":
        out = [c for c in out if c["n"] <= 20]
    return out


def try_reuse(iid, which, tags):

    src = os.path.join(OLD_RESULTS, f"e9__{iid}__exact_{which}.json")
    dst = os.path.join(RESULTS, f"e9v3__{iid}__exact_{which}.json")
    if os.path.exists(dst) or not os.path.exists(src):
        return None
    d = json.load(open(src))
    cur = lib_hashes()
    key = "libcubcr_gpu.so" if which == "gpu" else "libcubcr_cpu.so"
    prov = d.get("provenance") or {}
    old_hash = prov.get(key) or prov.get(key.replace(".so", "_sha256"))
    if d.get("status") != "OK" or old_hash is None or old_hash != cur.get(key):
        return None
    d["reused_from"] = os.path.relpath(src, REPO); d["exp"] = "e9v3"
    d["gen_seed"] = GEN_SEEDS[0]; d["concurrent_load"] = False
    json.dump(d, open(dst, "w"), indent=1, default=str)
    print(f"  [exact_{which}] reused {os.path.basename(src)} (provenance match)", flush=True)
    return d


def genuine_to(cell, gen_seed, cap_cpu, tags, which_list=("cpu",)):
    import subprocess
    comp, iid, meta, _ = build_model("ticketbench", cell["n"], cell["states"], cell["k"], gen_seed, cell["pattern"], 0.05)
    for which in which_list:
        cap = cap_cpu
        name = f"e9v3__{iid}__exact_{which}_genuineTO.json"
        if os.path.exists(os.path.join(RESULTS, name)):
            previous = json.load(open(os.path.join(RESULTS, name)))
            if previous.get("status") in ("OK", "TO"):
                continue
        code = (f"import sys,numpy as np; sys.path.insert(0,'{REPO}');"
                f"from experiments.e9v3_certified import make_engine, build_model;"
                f"comp,_,_,_ = build_model('ticketbench',{cell['n']},{cell['states']},{cell['k']},{gen_seed},'{cell['pattern']}',0.05);"
                f"eng,_,_ = make_engine('{which}',comp); masks=np.arange(1<<{cell['n']},dtype=np.uint64);"
                f"eng.solve(masks[:65536],{cell['k']}); v=eng.solve(masks,{cell['k']}); print('DONE',float(v[-1]-v[0]))")
        t0 = time.perf_counter()
        p = subprocess.Popen([sys.executable, "-c", code], cwd=REPO, start_new_session=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            stdout, stderr = p.communicate(timeout=cap)
            status = "OK" if p.returncode == 0 and any(line.startswith("DONE ") for line in stdout.splitlines()) else "CRASH"
            wall = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, 9); stdout, stderr = p.communicate(); status, wall = "TO", time.perf_counter() - t0
        rec = base_record(cell, iid, meta, comp, "gpu_batched" if which == "gpu" else "cpu_multicore",
                          32 if which == "cpu" else 0, gen_seed, cap_s=cap,
                          wall_s=None if status == "CRASH" else max(wall, cap) if status == "TO" else wall,
                          censored=(status == "TO"), status=status, arm=f"exact_{which}", n_coalitions=1 << cell["n"],
                          genuine_capped_run=True, child_returncode=p.returncode, process_elapsed_s=wall,
                          child_stdout=stdout[-2000:], child_stderr=stderr[-2000:],
                          note="full enumeration with a hard subprocess cap: a TO is a genuine lower bound")
        if status == "TO":
            rec["coalition_solves_per_s"] = None
        protocol.write_result(RESULTS, rec, filename=name, tags=tags)
        print(f"[e9v3-genuineTO] {iid} exact_{which}: {status} wall={wall:.0f}s (cap {cap}s)", flush=True)
        if status == "CRASH":
            raise RuntimeError(f"{iid} exact_{which} subprocess failed or did not finish; see {os.path.join(RESULTS, name)}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", default="full", choices=["full", "quick", "smoke"])
    ap.add_argument("--arms", default="exact_gpu,cert_gpu,cert_gpu_eblooks,exact_cpu,cert_cpu")
    ap.add_argument("--gen-seeds", type=int, nargs="*", default=GEN_SEEDS)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--eps-div", type=int, default=20)
    ap.add_argument("--cap-exact-gpu", type=float, default=1800)
    ap.add_argument("--cap-exact-cpu", type=float, default=3600)
    ap.add_argument("--cap-cert", type=float, default=900)
    ap.add_argument("--screen-cap", type=float, default=90)
    ap.add_argument("--reuse-from-e9", action="store_true")
    ap.add_argument("--concurrent-load", action="store_true")
    ap.add_argument("--genuine-to", action="store_true", help="genuine capped CPU enumerations for n=32,S=1e4,k=100 of every gen seed")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only-n", type=int, nargs="*", default=None, help="restrict the grid to these n")
    a = ap.parse_args(argv)
    arms = a.arms.split(",")
    tags = protocol.machine_tags()
    os.makedirs(RESULTS, exist_ok=True)
    shapley_from_table_fast(np.zeros(8), 3)
    if a.genuine_to:
        for gs in a.gen_seeds:
            genuine_to(dict(pattern="none", hazard=0.0, n=32, states=10000, k=100), gs, a.cap_exact_cpu, tags)
        return 0
    for gen_seed in a.gen_seeds:
        for cell in cells(a.scope):
            if a.only_n and cell["n"] not in a.only_n:
                continue
            cell = dict(cell, _concurrent=a.concurrent_load)
            comp, iid, meta, build_s = build_model("ticketbench", cell["n"], cell["states"], cell["k"], gen_seed, cell["pattern"], 0.05)
            cell["_build_s"] = build_s
            print(f"[e9v3] {iid} (n={cell['n']} S={comp.n_states} k={cell['k']} gen_seed={gen_seed} build {build_s:.2f}s)", flush=True)

            from cubcr.engine_gpu import GPUEngine
            eng = GPUEngine(comp, dtype="fp64", use_graph=True)
            vv = eng.solve(np.array([0, (1 << cell["n"]) - 1], dtype=np.uint64), cell["k"]); eng.close()
            ups = float(vv[1] - vv[0])
            if ups < UPSILON_FLOOR:
                rec = base_record(cell, iid, meta, comp, "gpu_batched", 0, gen_seed, cap_s=1, wall_s=None, status="NOT RUN",
                                  arm="cell", attributable_value=ups, note=f"Υ={ups:.4f} below the preregistered floor {UPSILON_FLOOR}")
                protocol.write_result(RESULTS, rec, filename=f"e9v3__{iid}__cell_below_floor.json", tags=tags)
                print(f"  Υ={ups:.4f} < floor -> cell NOT RUN", flush=True); continue
            phi_exact = None
            for which in ("gpu", "cpu"):
                arm = f"exact_{which}"
                if arm not in arms:
                    continue
                dst = os.path.join(RESULTS, f"e9v3__{iid}__{arm}.json")
                rec = None
                if a.reuse_from_e9 and gen_seed == GEN_SEEDS[0]:
                    rec = try_reuse(iid, which, tags)
                if rec is None and os.path.exists(dst) and not a.force:
                    rec = json.load(open(dst)); print(f"  [{arm}] exists ({rec.get('status')})", flush=True)
                if rec is None:
                    phi, u, _ = run_exact(cell, iid, meta, comp, which, a.cap_exact_gpu if which == "gpu" else a.cap_exact_cpu,
                                          a.screen_cap, tags, gen_seed)
                    if phi is not None and phi_exact is None:
                        phi_exact = phi
                elif rec.get("status") == "OK" and phi_exact is None and "shapley" in rec:
                    phi_exact = np.asarray(rec["shapley"])
            if phi_exact is None:
                for which in ("gpu", "cpu"):
                    dst = os.path.join(RESULTS, f"e9v3__{iid}__exact_{which}.json")
                    if os.path.exists(dst):
                        d = json.load(open(dst))
                        if d.get("status") == "OK" and "shapley" in d:
                            phi_exact = np.asarray(d["shapley"]); break
            for seed in a.seeds:
                for arm, which, sampler in (("cert_gpu", "gpu", "cs"), ("cert_gpu_eblooks", "gpu", "eb"), ("cert_cpu", "cpu", "cs")):
                    if arm not in arms:
                        continue
                    if os.path.exists(os.path.join(RESULTS, f"e9v3__{iid}__{arm}_s{seed}.json")) and not a.force:
                        continue
                    run_certified(cell, iid, meta, comp, which, seed, ups, a.eps_div, a.cap_cert, tags, phi_exact, gen_seed, sampler)
    return 0


if __name__ == "__main__":
    sys.exit(main())
