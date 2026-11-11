#!/usr/bin/env python3


from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def _repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(4):
        if os.path.isdir(os.path.join(d, "cubcr")) and os.path.isdir(os.path.join(d, "csrc")):
            return d
        d = os.path.dirname(d)
    raise SystemExit("smoke_test: cannot locate the repository root (directory holding cubcr/ and csrc/)")


ROOT = _repo_root()
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


N_AGENTS, STATES, HORIZON, PFAIL, GEN_SEED = 12, 1000, 50, 0.05, 20260829
SAMPLER_SEED, ALPHA, EPS_DIV = 20260829, 0.05, 20
SAMPLER_KW = dict(kind="agent", batch_perms=32, alpha=ALPHA, max_perms=2_000_000, time_budget=900.0, antithetic=True,
                  value_range="monotone", two_sided=True, bonferroni=True, sequential=True, ci="prpl_eb_cs",
                  batch_schedule="geometric")
TOL = 1e-12


ARCH_HINT = ("build the CUDA engine for this GPU's compute capability as shown in README.md "
             "(section 'Install and build', GPU with compute capability below 8.0)")


def gpu_status(comp=None):


    try:
        from cubcr.engine_gpu import device_count, device_name
        c = device_count()
        if c <= 0:
            return False, "no CUDA device visible to the CUDA engine"
        name = device_name(0)
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        return False, f"CUDA engine unavailable ({type(e).__name__}: {str(e).strip()[:200]})"
    if comp is None:
        return True, name
    try:
        eng = make_engine("gpu", comp, 0)
        try:
            v = np.asarray(eng.solve(np.array([0, (1 << comp.n_agents) - 1], dtype=np.uint64), 2), dtype=np.float64)
        finally:
            eng.close()
        if v.shape != (2,) or not np.all(np.isfinite(v)):
            return False, f"CUDA engine on {name} returned invalid values in the probe solve; {ARCH_HINT}"
        return True, name
    except KeyboardInterrupt:
        raise
    except BaseException as e:
        return False, (f"CUDA engine cannot run on {name} ({type(e).__name__}: {str(e).strip()[:200]}); "
                       f"{ARCH_HINT}")


def engine_library(which: str) -> dict:

    import hashlib
    try:
        if which == "gpu":
            from cubcr.engine_cpu import CSRC
            path = os.environ.get("CUBCR_GPU_LIB") or os.path.join(CSRC, "build", "libcubcr_gpu.so")
        else:
            from cubcr.engine_cpu import LIB_PATH as path
        path = os.path.abspath(path)
        with open(path, "rb") as fh:
            sha = hashlib.sha256(fh.read()).hexdigest()
        rel = os.path.relpath(path, ROOT)
        shown = rel if not rel.startswith("..") else "<outside the release>/" + os.path.basename(path)
        return {"engine_lib": shown, "engine_lib_sha256": sha}
    except Exception as e:
        return {"engine_lib": None, "engine_lib_sha256": None, "engine_lib_error": f"{type(e).__name__}: {e}"}


def make_engine(which: str, comp, threads: int):
    if which == "gpu":
        from cubcr.engine_gpu import GPUEngine
        return GPUEngine(comp, dtype="fp64", use_graph=True)
    from cubcr.engine_cpu import CPUEngine
    return CPUEngine(comp, n_threads=threads, block=8, dtype="fp64")


def run_engine(which: str, comp, csg, threads: int) -> dict:
    from cubcr import vi_ref
    from cubcr.shapley import shapley_from_table
    from cubcr.shapley_sampled import sample_shapley

    n, k = N_AGENTS, HORIZON
    out = {"engine": which, **engine_library(which)}
    t0 = time.perf_counter()
    eng = make_engine(which, comp, threads)
    out["engine_setup_s"] = time.perf_counter() - t0
    if which == "cpu":
        out["cpu_threads"] = int(getattr(eng, "max_threads", 0)) if threads <= 0 else threads


    masks = np.arange(1 << n, dtype=np.uint64)
    t0 = time.perf_counter(); v = eng.solve(masks, k); out["exhaustive_first_call_s"] = time.perf_counter() - t0
    t0 = time.perf_counter(); v2 = eng.solve(masks, k); out["exhaustive_s"] = time.perf_counter() - t0
    out["repeat_identical"] = bool(np.array_equal(v, v2))
    v = np.asarray(v, dtype=np.float64)
    ups = float(v[-1] - v[0])
    out["v_empty"], out["v_full"], out["upsilon"] = float(v[0]), float(v[-1]), ups


    worst = 0.0
    for i in range(n):
        bit = np.uint64(1) << np.uint64(i)
        without = masks[(masks & bit) == 0]
        worst = min(worst, float(np.min(v[(without | bit).astype(np.int64)] - v[without.astype(np.int64)])))
    out["min_marginal"] = worst
    phi = shapley_from_table(v, n)
    out["phi_exact"] = phi.tolist()
    out["efficiency_gap"] = float(abs(phi.sum() - ups))


    rng = np.random.default_rng(GEN_SEED)
    check = [0, (1 << n) - 1, int(rng.integers(1, (1 << n) - 1))]
    t0 = time.perf_counter()
    ref = [float(vi_ref.solve(csg, int(c), k)) for c in check]
    out["vi_ref_s"] = time.perf_counter() - t0
    out["vi_ref_masks"] = check
    out["vi_ref_max_abs_diff"] = float(max(abs(ref[j] - v[c]) for j, c in enumerate(check)))


    if hasattr(eng, "close"):
        eng.close()
    eng = make_engine(which, comp, threads)
    eps = ups / EPS_DIV
    eng.solve(np.random.default_rng(2).integers(0, 1 << n, size=32 * 2 * (n + 1), dtype=np.uint64), k)
    t0 = time.perf_counter()
    r = sample_shapley(eng, n=n, k=k, seed=SAMPLER_SEED, eps=eps, **SAMPLER_KW)
    out["sampler_s"] = time.perf_counter() - t0
    hw = np.asarray(r.half_widths, dtype=np.float64)
    est = np.asarray(r.estimates, dtype=np.float64)
    out.update(eps=eps, sampler_status=r.status, certified=bool(r.certified), n_perms=int(r.n_perms),
               n_pairs=int(r.n_samples), unique_solves=int(r.n_solves), requested=int(r.n_coalitions_requested),
               n_batches=int(r.n_batches), max_half_width=float(np.max(hw)), phi_hat=est.tolist(),
               half_widths=hw.tolist(), max_abs_err=float(np.max(np.abs(est - phi))),
               covered=int(np.sum(np.abs(est - phi) <= hw + TOL)))
    out["values"] = v
    if hasattr(eng, "close"):
        eng.close()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Check the cuBCR engine and sampler")
    ap.add_argument("--engine", choices=["auto", "gpu", "cpu", "both"], default="auto")
    ap.add_argument("--threads", type=int, default=0, help="CPU engine threads (0 = all OpenMP threads)")
    ap.add_argument("--json", default=None, help="optional path for a JSON record of every number printed")
    a = ap.parse_args(argv)

    t_start = time.perf_counter()
    from generators.ticketbench import build_compiled, build_instance
    t0 = time.perf_counter()
    comp, meta = build_compiled(N_AGENTS, STATES, HORIZON, PFAIL, "none", GEN_SEED)
    build_s = time.perf_counter() - t0
    csg = build_instance(N_AGENTS, STATES, HORIZON, PFAIL, "none", GEN_SEED)
    print(f"[smoke] model {meta['instance_id']}: {N_AGENTS} agents, {comp.n_states} states, horizon {HORIZON}, "
          f"{comp.n_rows} joint-action rows, {comp.nnz} transitions (built in {build_s:.3f} s)")

    have_gpu, gpu_desc = gpu_status(comp) if a.engine in ("auto", "gpu", "both") else (False, "not requested")
    fallback = None
    if a.engine == "auto":
        engines = ["gpu"] if have_gpu else ["cpu"]
        if not have_gpu:
            fallback = gpu_desc
            print(f"[smoke] {gpu_desc}; falling back to the OpenMP CPU engine")
    elif a.engine == "both":
        engines = ["gpu", "cpu"] if have_gpu else ["cpu"]
        if not have_gpu:
            fallback = gpu_desc
            print(f"[smoke] {gpu_desc}; running the CPU engine only")
    elif a.engine == "gpu":
        if not have_gpu:
            print(f"[smoke] FAILED: --engine gpu requested but {gpu_desc}")
            if a.json:
                with open(a.json, "w") as fh:
                    json.dump({"engine_requested": "gpu", "gpu": None, "gpu_status": gpu_desc, "engines": [],
                               "fallback": None, "failures": [f"gpu: {gpu_desc}"], "passed": False,
                               "exit_code": 2, "total_s": time.perf_counter() - t_start}, fh, indent=1)
            return 2
        engines = ["gpu"]
    else:
        engines = ["cpu"]
    if have_gpu and "gpu" in engines:
        print(f"[smoke] CUDA device: {gpu_desc}")

    results, failures = [], []
    queue = list(engines)
    while queue:
        which = queue.pop(0)
        try:
            r = run_engine(which, comp, csg, a.threads)
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if which != "gpu" or a.engine == "gpu":
                raise

            fallback = (f"CUDA engine failed during the run ({type(e).__name__}: {str(e).strip()[:200]}); "
                        f"{ARCH_HINT}")
            print(f"[smoke] {fallback}; " + ("falling back to the OpenMP CPU engine" if a.engine == "auto"
                                              else "running the CPU engine only"))
            have_gpu = False
            if "cpu" not in queue:
                queue.append("cpu")
            continue
        results.append(r)
        tag = which.upper() + (f" ({r['cpu_threads']} threads)" if which == "cpu" else "")
        print(f"\n[smoke] engine {tag}")
        print(f"  exhaustive: 4096 coalitions in {r['exhaustive_s']:.4f} s (first call {r['exhaustive_first_call_s']:.4f} s;"
              f" engine setup {r['engine_setup_s']:.4f} s)")
        print(f"  v(empty) = {r['v_empty']:.6f}, v(P) = {r['v_full']:.6f}, Upsilon = {r['upsilon']:.6f}")
        print(f"  Proposition 1: min marginal = {r['min_marginal']:.3e}, |sum phi - Upsilon| = {r['efficiency_gap']:.3e}")
        print(f"  reference value iteration (3 coalitions, {r['vi_ref_s']:.2f} s): max |diff| = {r['vi_ref_max_abs_diff']:.3e}")
        print(f"  certified sampler: status={r['sampler_status']} certified={r['certified']} pairs={r['n_pairs']} "
              f"permutations={r['n_perms']} unique coalition solves={r['unique_solves']} batches={r['n_batches']} "
              f"({r['sampler_s']:.3f} s)")
        print(f"  max half-width {r['max_half_width']:.5f} <= eps {r['eps']:.5f}; max |phi_hat - phi| = {r['max_abs_err']:.5f};"
              f" intervals containing the exact value: {r['covered']}/{N_AGENTS}")
        order = np.argsort(-np.asarray(r["phi_exact"]))[:3]
        print("  top-3 agents (exact phi | estimate +- H): " + "; ".join(
            f"agent {int(p)}: {r['phi_exact'][p]:.4f} | {r['phi_hat'][p]:.4f} +- {r['half_widths'][p]:.4f}" for p in order))
        checks = {"repeat call bitwise identical": r["repeat_identical"],
                  "monotone (min marginal >= -1e-12)": r["min_marginal"] >= -TOL,
                  "efficiency (|sum phi - Upsilon| <= 1e-12)": r["efficiency_gap"] <= TOL,
                  "reference value iteration (<= 1e-12)": r["vi_ref_max_abs_diff"] <= TOL,
                  "sampler stopped on eps and certified": r["sampler_status"] == "eps" and r["certified"],
                  "max half-width <= eps": r["max_half_width"] <= r["eps"],
                  "every interval contains the exact value": r["covered"] == N_AGENTS}
        r["checks"] = checks
        for name, ok in checks.items():
            if not ok:
                failures.append(f"{which}: {name}")

    cross = None
    if len(results) == 2:
        cross = float(np.max(np.abs(results[0]["values"] - results[1]["values"])))
        print(f"\n[smoke] GPU vs CPU: max |v_gpu - v_cpu| over 4096 coalitions = {cross:.3e}")
        if not cross <= TOL:
            failures.append("GPU and CPU coalition values differ by more than 1e-12")

    total = time.perf_counter() - t_start
    engines = [r["engine"] for r in results]
    if not results:
        failures.append("no engine ran")
    if a.json:
        rec = {"model": meta["instance_id"], "n_agents": N_AGENTS, "n_states": int(comp.n_states), "horizon": HORIZON,
               "gen_seed": GEN_SEED, "sampler_seed": SAMPLER_SEED, "sampler_config": SAMPLER_KW, "eps_div": EPS_DIV,
               "engine_requested": a.engine, "gpu": gpu_desc if have_gpu else None,
               "gpu_status": gpu_desc, "fallback": fallback,
               "engines": [{k: v for k, v in r.items() if k != "values"} for r in results],
               "gpu_vs_cpu_max_abs_diff": cross, "failures": failures, "passed": not failures,
               "exit_code": 1 if failures else 0, "total_s": total}
        with open(a.json, "w") as fh:
            json.dump(rec, fh, indent=1)
    if failures:
        print("\n[smoke] SMOKE TEST FAILED: " + "; ".join(failures))
        return 1
    print(f"\n[smoke] SMOKE TEST PASSED on {', '.join(e.upper() for e in engines)} ({total:.1f} s in total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
