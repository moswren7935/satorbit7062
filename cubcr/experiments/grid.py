#!/usr/bin/env python3


from __future__ import annotations

import hashlib

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol

RESULTS = os.path.join(REPO, "results")
DEFAULT_SEED = 20260829
SEEDS = [20260829, 20260830, 20260831]
PY = sys.executable


def build_model(instance, n, states, k, seed, pattern="none", pfail=0.05, hazard=0.0):
    t0 = time.perf_counter()
    if instance == "ticketbench":
        from generators.ticketbench import build_compiled
        comp, meta = build_compiled(n, states, k, pfail, pattern, seed, hazard=hazard)
        iid = meta["instance_id"]
    elif instance == "ticketbench_pruned":
        from generators.ticketbench import build_compiled
        from cubcr.model import restrict_to_horizon
        comp, meta = build_compiled(n, states, k, pfail, pattern, seed, hazard=hazard)
        comp = restrict_to_horizon(comp, k)
        meta = dict(meta, instance_id=meta["instance_id"] + "_pruned", pruned_states=int(comp.n_states))
        iid = meta["instance_id"]
    elif instance == "ticketbench_perm":
        from generators.ticketbench import build_compiled
        comp, meta = build_compiled(n, states, k, pfail, pattern, seed, permute="random")
        iid = meta["instance_id"]
    elif instance == "random":
        from tests.helpers import random_csg
        csg = random_csg(n_agents=n, n_states=states, k=k, seed=seed, max_active=2, max_actions=3,
                         n_fail=max(1, states // 100))
        comp = csg.compile()
        iid = f"rand_n{n}_S{states}_k{k}_s{seed}"
        meta = {"instance_id": iid, "seed": seed}
    else:
        raise ValueError(instance)
    return comp, iid, meta, time.perf_counter() - t0


def lib_hashes():
    out = {}
    for name in ("libcubcr_cpu.so", "libcubcr_gpu.so"):
        p = os.path.join(REPO, "csrc", "build", name)
        if os.path.exists(p):
            out[name] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    mk = open(os.path.join(REPO, "csrc", "Makefile")).read()
    nvcc = next((l.split("=", 1)[1].strip() for l in mk.splitlines() if l.startswith("NVCC ?=")), "nvcc")
    try:
        ver = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=20).stdout
        out["nvcc_used"] = next((l.strip() for l in ver.splitlines() if "release" in l), nvcc)
    except Exception:
        out["nvcc_used"] = nvcc
    out["cxxflags"] = next((l.split("=", 1)[1].strip() for l in mk.splitlines() if l.startswith("CXXFLAGS ?=")), None)
    out["nvccflags"] = next((l.split("=", 1)[1].strip() for l in mk.splitlines() if l.startswith("NVCCFLAGS ?=")), None)
    return out


def cell_id(c):
    hz = f"_hz{c['hazard']}" if c.get("hazard") else ""
    return f"{c['instance']}_n{c['n']}_S{c['states']}_k{c['k']}_s{c['seed']}_B{c['B']}{'_full' if c.get('full') else ''}{hz}{c.get('tag', '')}"


def coalition_masks(n, B, seed, full=False):
    if full or B >= (1 << n):
        return np.arange(1 << n, dtype=np.uint64), True
    rng = np.random.default_rng((seed * 1000003) ^ 0x5EED)
    return rng.integers(0, 1 << n, size=B, dtype=np.uint64), False


def make_engine(engine, comp, dtype, threads):
    if engine == "gpu_batched":
        from cubcr.engine_gpu import GPUEngine
        return GPUEngine(comp, dtype=dtype, use_graph=True)
    if engine == "gpu_unbatched":
        from cubcr.engine_gpu import GPUEngine
        return GPUEngine(comp, dtype=dtype, use_graph=False, max_batch=1)
    if engine == "cpu_multicore":
        from cubcr.engine_cpu import CPUEngine
        return CPUEngine(comp, n_threads=threads, block=8 if dtype == "fp64" else 16, dtype=dtype)
    raise ValueError(engine)


def result_name(exp, iid, engine, dtype, threads, seed, tag=""):
    return f"{exp}__{iid}__{engine}_{dtype}_t{threads}_s{seed}{tag}.json"


def cmd_cell(a):
    tags = protocol.machine_tags()
    t_all = time.perf_counter()
    comp, iid, meta, build_s = build_model(a.instance, a.n, a.states, a.k, a.seed, a.pattern, a.pfail, a.hazard)
    masks, full = coalition_masks(a.n, a.B, a.seed, full=a.full)
    B = int(masks.shape[0])
    work = B * a.k * comp.nnz
    t0 = time.perf_counter()
    eng = make_engine(a.engine, comp, a.dtype, a.threads)
    setup_s = time.perf_counter() - t0
    threads = a.threads if a.engine == "cpu_multicore" else 0
    t0 = time.perf_counter()
    eng.solve(masks[:min(B, 1024)] if a.warmup_small else masks, a.k)
    warm_s = time.perf_counter() - t0
    walls = []
    vals = None
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        vals = eng.solve(masks, a.k)
        walls.append(time.perf_counter() - t0)
    wall = float(np.median(walls))
    gpu_stats = getattr(eng, "stats", None)
    cid = cell_id(dict(instance=a.instance, n=a.n, states=a.states, k=a.k, seed=a.seed, B=a.B, full=a.full, tag=a.tag, hazard=a.hazard))
    rec = protocol.make_record(
        instance_id=iid, cell_id=cid, engine=a.engine, dtype=a.dtype, n=a.n, states=int(comp.n_states), k=a.k,
        seed=a.seed, threads=threads, cap_s=a.cap_s, hard_timeout_s=a.cap_s + a.cap_slack_s, timeout_scope="subprocess",
        wall_s=wall, censored=False, coalition_solves_per_s=B / wall, exp=a.exp,
        n_coalitions=B, full_enumeration=bool(full), mask_sha256=hashlib.sha256(masks.tobytes()).hexdigest(),
        n_rows=int(comp.n_rows), nnz=int(comp.nnz), rmax=int(comp.max_rows_per_state),
        work_fma=int(work), gfma_per_s=work / wall / 1e9, all_s=walls, min_s=float(min(walls)), max_s=float(max(walls)),
        n_repeat=a.repeat, warmup_s=warm_s, warmup_full_batch=(not a.warmup_small), model_build_s=build_s,
        engine_setup_s=setup_s, instance=a.instance, pattern=a.pattern, pfail=a.pfail, gpu_stats=gpu_stats,
        block=(8 if a.dtype == "fp64" else 16) if a.engine == "cpu_multicore" else None,
        v_summary={"min": float(vals.min()), "max": float(vals.max()), "mean": float(vals.mean()),
                   "frac_zero": float(np.mean(vals == 0.0)), "n_distinct": int(np.unique(vals).size),
                   "v_empty": float(vals[0]) if full else None, "v_full": float(vals[-1]) if full else None},
        stile=os.environ.get("CUBCR_STILE"), provenance=lib_hashes(), hazard=a.hazard,
        instance_meta={k: v for k, v in meta.items() if k not in ("params",)},
        frac_reachable_k=meta.get("frac_reachable_k"), n_reachable_k=meta.get("n_reachable_k"),
    )
    if a.ref_fp64 and a.dtype == "fp32":
        ref_eng = make_engine("gpu_batched" if a.engine.startswith("gpu") else a.engine, comp, "fp64", a.threads)
        ref = ref_eng.solve(masks, a.k)
        rec["max_abs_err_vs_fp64"] = float(np.max(np.abs(vals - ref)))
        rec["mean_abs_err_vs_fp64"] = float(np.mean(np.abs(vals - ref)))
        if full:
            from cubcr.shapley import shapley_from_table
            from cubcr.shapley_sampled import kendall_tau
            phi32 = shapley_from_table(vals, a.n)
            phi64 = shapley_from_table(ref, a.n)
            rec["kendall_tau"] = float(kendall_tau(phi32, phi64))
            rec["shapley_max_abs_err_vs_fp64"] = float(np.max(np.abs(phi32 - phi64)))
    if full and a.shapley:
        from cubcr.shapley import shapley_from_table, attributable_value
        t0 = time.perf_counter()
        phi = shapley_from_table(vals, a.n)
        rec["shapley_s"] = time.perf_counter() - t0
        rec["shapley"] = phi.tolist()
        rec["attributable_value"] = attributable_value(vals, a.n)
        rec["efficiency_gap"] = float(abs(phi.sum() - attributable_value(vals, a.n)))
        rec["end_to_end_s"] = build_s + setup_s + wall + rec["shapley_s"]
        rec["end_to_end_solve_only_s"] = wall
    if a.save_values:
        vpath = os.path.join(RESULTS, a.exp, "values", result_name(a.exp, iid, a.engine, a.dtype, threads, a.seed, a.tag)[:-5] + ".npy")
        os.makedirs(os.path.dirname(vpath), exist_ok=True)
        np.save(vpath, vals)
        rec["values_path"] = os.path.relpath(vpath, REPO)
    rec["process_elapsed_s"] = time.perf_counter() - t_all
    path = protocol.write_result(os.path.join(RESULTS, a.exp), rec,
                                 filename=result_name(a.exp, iid, a.engine, a.dtype, threads, a.seed, a.tag), tags=tags)
    print(f"[cell] {os.path.basename(path)} wall={wall:.3f}s B={B} {work / wall / 1e9:.1f} GFMA/s")
    return 0


def run_cells(cells, dry=False, skip_existing=True):


    tags = None
    done = 0
    prov = lib_hashes()
    for c in cells:
        threads = c["threads"] if c["engine"] == "cpu_multicore" else 0
        stub = cell_id(c)
        out_dir = os.path.join(RESULTS, c["exp"])
        os.makedirs(out_dir, exist_ok=True)
        marker = os.path.join(out_dir, ".done_" + result_name(c["exp"], stub, c["engine"], c["dtype"], threads, c["seed"], "")[:-5])
        if skip_existing and os.path.exists(marker):
            st = open(marker).read().split()
            if st and st[0] == "OK" and (len(st) < 2 or st[1] == prov.get("libcubcr_gpu.so" if c["engine"].startswith("gpu") else "libcubcr_cpu.so")):
                continue
        cmd = [PY, os.path.abspath(__file__), "cell", "--exp", c["exp"], "--instance", c["instance"],
               "--n", str(c["n"]), "--states", str(c["states"]), "--k", str(c["k"]), "--seed", str(c["seed"]),
               "--engine", c["engine"], "--dtype", c["dtype"], "--threads", str(c["threads"]), "--B", str(c["B"]),
               "--repeat", str(c["repeat"]), "--cap-s", str(c["cap_s"]), "--pattern", c.get("pattern", "none"),
               "--pfail", str(c.get("pfail", 0.05)), "--tag", c.get("tag", ""), "--cap-slack-s", str(c.get("cap_slack_s", 300)),
               "--hazard", str(c.get("hazard", 0.0))]
        for flag in ("full", "warmup_small", "ref_fp64", "shapley", "save_values"):
            if c.get(flag):
                cmd.append("--" + flag.replace("_", "-"))
        print("[plan]", " ".join(cmd[2:]), flush=True)
        if dry:
            continue
        env = dict(os.environ)
        env.setdefault("OMP_PROC_BIND", "close")
        env.setdefault("OMP_PLACES", "cores")
        t0 = time.perf_counter()
        status = "OK"
        try:
            p = subprocess.run(cmd, cwd=REPO, env=env, timeout=c["cap_s"] + c.get("cap_slack_s", 300),
                               capture_output=True, text=True)
            elapsed = time.perf_counter() - t0
            if p.returncode != 0:
                status = "CRASH"
                log = os.path.join(out_dir, "crash_" + stub + f"_{c['engine']}_{c['dtype']}_t{threads}_s{c['seed']}.log")
                with open(log, "w") as fh:
                    fh.write(p.stdout + "\n" + p.stderr)
                print(f"[plan] CRASH rc={p.returncode} log={log}", flush=True)
            else:
                print(p.stdout.strip().splitlines()[-1] if p.stdout.strip() else "[cell] (no output)", flush=True)
        except subprocess.TimeoutExpired as e:
            elapsed = time.perf_counter() - t0
            status = "TO"
            print(f"[plan] TIMEOUT after {elapsed:.0f}s (cap {c['cap_s']}s)", flush=True)
        if status != "OK":
            if tags is None:
                tags = protocol.machine_tags()
            rec = protocol.make_record(instance_id=stub, cell_id=stub, engine=c["engine"], dtype=c["dtype"], n=c["n"], states=c["states"],
                                       k=c["k"], seed=c["seed"], threads=threads, cap_s=c["cap_s"], hard_timeout_s=c["cap_s"] + c.get("cap_slack_s", 300),
                                       timeout_scope="subprocess (model build + set-up + warm-up + all repeats)",
                                       wall_s=max(elapsed, c["cap_s"]) if status == "TO" else None, process_elapsed_s=elapsed,
                                       censored=(status == "TO"), status=status, exp=c["exp"], n_coalitions=c["B"], n_repeat=c["repeat"],
                                       instance=c["instance"], provenance=prov,
                                       note="planner-level cap/crash: wall_s is the whole-subprocess elapsed time (a lower bound for the solve), instance_id is the cell_id stand-in")
            protocol.write_result(out_dir, rec, filename=result_name(c["exp"], stub, c["engine"], c["dtype"], threads, c["seed"], c.get("tag", "")), tags=tags)
        open(marker, "w").write(status + " " + prov.get("libcubcr_gpu.so" if c["engine"].startswith("gpu") else "libcubcr_cpu.so", "?") + "\n")
        done += 1
    return done


def est_nnz(instance, states):
    return 7.3 * states if instance == "ticketbench" else 6.4 * states


def clamp(x, lo, hi):
    return int(max(lo, min(hi, x)))


W_GPU = 5e11
CAL_S = (1000, 10000, 100000, 1000000)


def plan_e1(a):

    cells = []
    for n in a.n_list:
        for S in a.states_list:
            for k in a.k_list:
                nnz = est_nnz(a.instance, S)
                B = clamp(W_GPU / (k * nnz), 1024, 1 << n)
                for seed in a.seeds:
                    base = dict(exp="e1", instance=a.instance, n=n, states=S, k=k, seed=seed, pattern=a.pattern, pfail=a.pfail, B=B, repeat=5)
                    cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=900))
                    cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, cap_s=1800))
                    if k == 25 and 64 in a.cpu_threads:
                        cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=64, cap_s=1800))
    cells.sort(key=lambda c: (c["B"] * c["k"] * est_nnz(c["instance"], c["states"]) * (1 if c["engine"] == "gpu_batched" else 6), c["seed"]))
    return cells


def plan_e1cal(a):

    cells = []
    for S in CAL_S:
        for B in (256, 1024, 4096, 16384, 65536):
            if S >= 100000 and B > 16384:
                continue
            base = dict(exp="e1cal", instance=a.instance, n=16, states=S, k=25, seed=DEFAULT_SEED, tag=f"_B{B}", pattern=a.pattern, pfail=a.pfail, B=B, repeat=5)
            cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=900))
            for thr in a.cpu_threads:
                cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=thr, cap_s=1800))
    return cells


def plan_e1perm(a):

    cells = []
    for S in (10000, 100000):
        for k in (25, 100):
            nnz = est_nnz(a.instance, S)
            B = clamp(W_GPU / (k * nnz), 1024, 1 << 16)
            base = dict(exp="e1perm", instance="ticketbench_perm", n=16, states=S, k=k, seed=DEFAULT_SEED, pattern=a.pattern, pfail=a.pfail, B=B, repeat=5)
            cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=900))
            cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, cap_s=1800))
    return cells


def plan_e1prune(a):

    cells = []
    for n in (16, 24):
        for S in (100000, 1000000):
            for k in (25, 100, 400):
                nnz = est_nnz(a.instance, S)
                B = clamp(W_GPU / (k * nnz), 1024, 1 << n)
                base = dict(exp="e1prune", instance="ticketbench_pruned", n=n, states=S, k=k, seed=DEFAULT_SEED, pattern=a.pattern, pfail=a.pfail, B=B, repeat=5)
                cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=900))
                cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, cap_s=1800))
    return cells


def plan_e2(a):
    cells = []
    for n in (8, 12):
        for S in (1000, 10000):
            for k in (25, 100):
                B = 1 << n
                base = dict(exp="e2", instance=a.instance, n=n, states=S, k=k, seed=DEFAULT_SEED, full=True, pattern=a.pattern, pfail=a.pfail, B=B)
                cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, repeat=5, cap_s=900))
                cells.append(dict(base, engine="gpu_unbatched", dtype="fp64", threads=0, repeat=5 if n == 8 else 3, cap_s=1800))
    return cells


def plan_e6(a):
    cells = []
    for S in a.states_list:
        for k in a.k_list:
            nnz = est_nnz(a.instance, S)
            B = clamp(W_GPU / (k * nnz), 1024, 1 << 16)
            base = dict(exp="e6", instance=a.instance, n=16, states=S, k=k, seed=DEFAULT_SEED, pattern=a.pattern, pfail=a.pfail, B=B, repeat=5)
            cells.append(dict(base, engine="gpu_batched", dtype="fp32", threads=0, cap_s=900, ref_fp64=True))
            cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=900))
            cells.append(dict(base, engine="cpu_multicore", dtype="fp32", threads=32, cap_s=1800))
            cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, cap_s=1800))
    return cells


def plan_e0hz(a):

    cells = []
    for n in (12, 16):
        for S in (1000, 10000):
            base = dict(exp="e0hz", instance="ticketbench", n=n, states=S, k=100, seed=DEFAULT_SEED, full=True, shapley=True, pattern="none", pfail=0.05, B=1 << n, hazard=0.02, repeat=5)
            cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=1800, save_values=True))
            cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, cap_s=5400))
    return cells


def plan_e0(a):

    cells = []
    for n in a.n_list:
        for S in a.states_list:
            for k in a.k_list:
                if n == 24 and k > 25:
                    continue
                base = dict(exp="e0", instance=a.instance, n=n, states=S, k=k, seed=DEFAULT_SEED, full=True, shapley=True, pattern=a.pattern, pfail=a.pfail, B=1 << n)
                rep = 5 if n <= 20 else 3
                cells.append(dict(base, engine="gpu_batched", dtype="fp64", threads=0, repeat=rep, cap_s=1800, save_values=(n <= 16)))
                if n <= 16:
                    cells.append(dict(base, engine="gpu_batched", dtype="fp32", threads=0, repeat=rep, cap_s=1800, ref_fp64=True, tag="_fp32tau"))
                cells.append(dict(base, engine="cpu_multicore", dtype="fp64", threads=32, repeat=rep, cap_s=5400))
    cells.sort(key=lambda c: (1 << c["n"]) * c["k"] * c["states"] * (1 if c["engine"] == "gpu_batched" else 6))
    return cells


def main(argv=None):
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("cell")
    c.add_argument("--exp", required=True); c.add_argument("--instance", default="ticketbench")
    c.add_argument("--n", type=int, required=True); c.add_argument("--states", type=int, required=True)
    c.add_argument("--k", type=int, required=True); c.add_argument("--seed", type=int, default=DEFAULT_SEED)
    c.add_argument("--engine", required=True); c.add_argument("--dtype", default="fp64")
    c.add_argument("--threads", type=int, default=32); c.add_argument("--B", type=int, default=1024)
    c.add_argument("--full", action="store_true"); c.add_argument("--repeat", type=int, default=3)
    c.add_argument("--cap-s", type=float, default=900); c.add_argument("--pattern", default="none")
    c.add_argument("--pfail", type=float, default=0.05); c.add_argument("--warmup-small", action="store_true")
    c.add_argument("--ref-fp64", action="store_true"); c.add_argument("--shapley", action="store_true")
    c.add_argument("--save-values", action="store_true"); c.add_argument("--tag", default="")
    c.add_argument("--cap-slack-s", type=float, default=300); c.add_argument("--hazard", type=float, default=0.0)
    for name in ("e0", "e0hz", "e1", "e1cal", "e1perm", "e1prune", "e2", "e6"):
        p = sub.add_parser(name)
        p.add_argument("--instance", default="ticketbench"); p.add_argument("--pattern", default="none")
        p.add_argument("--pfail", type=float, default=0.05); p.add_argument("--dry", action="store_true")
        p.add_argument("--n-list", type=int, nargs="*", default=[8, 12, 16, 20, 24])
        p.add_argument("--states-list", type=int, nargs="*", default=[1000, 10000, 100000, 1000000])
        p.add_argument("--k-list", type=int, nargs="*", default=[25, 100, 400])
        p.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
        p.add_argument("--cpu-threads", type=int, nargs="*", default=[32, 64])
    a = ap.parse_args(argv)
    if a.cmd == "cell":
        return cmd_cell(a)
    plan = {"e0": plan_e0, "e0hz": plan_e0hz, "e1": plan_e1, "e1cal": plan_e1cal, "e1perm": plan_e1perm, "e1prune": plan_e1prune, "e2": plan_e2, "e6": plan_e6}[a.cmd](a)
    print(f"[plan] {a.cmd}: {len(plan)} cells")
    n = run_cells(plan, dry=a.dry)
    print(f"[plan] {a.cmd}: ran {n} cells")
    return 0


if __name__ == "__main__":
    sys.exit(main())
