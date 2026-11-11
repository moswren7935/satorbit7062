#!/usr/bin/env python3


from __future__ import annotations
import argparse, glob, hashlib, json, os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from generators.ticketbench import build_instance
from cubcr import io_prism
from cubcr import engine_cpu as _engine_cpu
from cubcr.engine_gpu import GPUEngine
from cubcr.engine_cpu import CPUEngine
from cubcr.shapley import shapley_from_table

TOL_PRISM = 1e-6
TOL_GPU_CPU = 1e-12
CPU_THREADS = 32
N_REPEAT = 5
WARMUP = 1
SEED = 20260829
SPEEDUP_FIELDS = ("speedup_gpu_vs_prism", "speedup_gpu_vs_prism_solver_only", "speedup_gpu_vs_prism_lower_bound",
                  "speedup_cpu_vs_prism", "speedup_cpu_vs_prism_lower_bound")


def sha256_file(path):
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def engine_binaries() -> dict:

    gpu = os.environ.get("CUBCR_GPU_LIB") or os.path.join(REPO, "csrc", "build", "libcubcr_gpu.so")
    cpu = _engine_cpu.LIB_PATH
    out = {}
    for name, p in (("libcubcr_gpu.so", gpu), ("libcubcr_cpu.so", cpu)):
        ok = os.path.isfile(p)
        out[name] = {"path": p, "sha256": sha256_file(p) if ok else None,
                     "size": os.path.getsize(p) if ok else None,
                     "mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p))) if ok else None}
    return out


def cpu_threads_label(tags: dict, threads: int) -> str:
    pc = tags.get("physical_cores")
    if pc == threads:
        return f"{threads} OpenMP threads = {pc} physical cores (SMT unused)"
    return f"{threads} OpenMP threads (physical cores: {pc}, logical CPUs: {tags.get('logical_cpus')})"


def time_engines(csg, k, n_repeat: int = N_REPEAT, threads: int = CPU_THREADS, warmup: int = WARMUP) -> dict:


    t0 = time.perf_counter()
    comp = csg.compile()
    build_s = time.perf_counter() - t0
    n = csg.n_agents
    masks = np.arange(1 << n, dtype=np.uint64)
    g = GPUEngine(comp, dtype="fp64", use_graph=True)
    try:
        tg = protocol.timed_run(lambda: g.solve(masks, k), n_repeat=n_repeat, warmup=warmup)
    finally:
        g.close()
    vg = np.asarray(tg.pop("result"), dtype=np.float64)
    c = CPUEngine(comp, n_threads=threads, block=8)
    try:
        tc = protocol.timed_run(lambda: c.solve(masks, k), n_repeat=n_repeat, warmup=warmup)
    finally:
        c.close()
    vc = np.asarray(tc.pop("result"), dtype=np.float64)
    finite = bool(np.all(np.isfinite(vg)) and np.all(np.isfinite(vc)))
    in_unit = bool(finite and np.all(vg >= -TOL_GPU_CPU) and np.all(vg <= 1 + TOL_GPU_CPU))
    max_abs = float(np.max(np.abs(vg - vc))) if finite else float("inf")
    return {"vg": vg, "vc": vc, "gpu": tg, "cpu": tc, "build_s": build_s, "threads": threads,
            "n_repeat": n_repeat, "warmup": warmup, "max_abs_err_gpu_vs_cpu": max_abs,
            "gpu_cpu_gate_ok": bool(finite and in_unit and max_abs <= TOL_GPU_CPU),
            "gpu_values_finite_in_unit": in_unit}


def _not_run_row(mask, reason):
    return {"mask": int(mask), "status": "NOT RUN", "value": None, "wall_s": None, "prism_construction_s": None,
            "prism_checking_s": None, "prism_solver_s": None, "prism_states": None, "prism_transitions": None,
            "peak_rss_kb": None, "cmd": None, "rc": None, "log": None, "model_path": None, "props_path": None,
            "results_path": None, "emit_s": None, "error": None, "timed_out": False, "reason": reason}


PRISM_EXTRA = []


def prism_all_coalitions(csg, k, iid, timeout_s, cap_total, workdir, javamaxmem=None,
                         maxiters=io_prism.DEFAULT_MAXITERS, stop_on_err=True, verbose=True,
                         masks_to_run=None):


    n = csg.n_agents
    rows, total, emit_total, aborted = [], 0.0, 0.0, None
    for mask in range(1 << n):
        if aborted is not None:
            rows.append(_not_run_row(mask, f"aborted after {aborted}"))
            continue
        if masks_to_run is not None and mask not in masks_to_run:
            rows.append(_not_run_row(mask, "excluded by --prism-masks (validation subset)"))
            continue
        if total > cap_total:
            rows.append(_not_run_row(mask, f"aggregate cap: cumulative PRISM wall {total:.1f}s > cap_total {cap_total}s"))
            continue
        t0 = time.perf_counter()
        model, props = io_prism.emit_coalition_model(csg, mask, k)
        emit_s = time.perf_counter() - t0
        emit_total += emit_s
        r = io_prism.run_prism(model, props, timeout_s=timeout_s, workdir=workdir, tag=f"{iid}_mask{mask}",
                               javamaxmem=javamaxmem, maxiters=maxiters, include_identity=False,
                               extra_args=tuple(PRISM_EXTRA))
        row = {"mask": int(mask), "status": r["status"], "value": r["value"], "wall_s": r["wall_s"],
               "prism_construction_s": r["prism_construction_s"], "prism_checking_s": r["prism_checking_s"],
               "prism_solver_s": r["prism_solver_s"], "prism_states": r["prism_states"],
               "prism_transitions": r["prism_transitions"], "peak_rss_kb": r["peak_rss_kb"],
               "time_elapsed_s": r["time_elapsed_s"], "cmd": r["cmd"], "rc": r["returncode"], "log": r["stdout_path"],
               "model_path": r["model_path"], "props_path": r["props_path"], "results_path": r["results_path"],
               "emit_s": emit_s, "error": r["error"], "timed_out": r["timed_out"], "reason": None}
        rows.append(row)
        total += r["wall_s"]
        if verbose:
            print(f"  [prism] {iid} mask={mask} {r['status']} v={r['value']} wall={r['wall_s']:.3f}s "
                  f"solver={r['prism_solver_s']} rss={r['peak_rss_kb']}kB" + (f" error={r['error']}" if r["error"] else ""), flush=True)
        if r["status"] == "ERR" and stop_on_err:
            aborted = f"ERR on mask {mask}: {r['error']} (log: {r['stdout_path']})"
            print(f"[e3] ERROR: PRISM-games returned ERR for {iid} mask={mask}: {r['error']} -> see {r['stdout_path']}",
                  file=sys.stderr, flush=True)
    return rows, total, emit_total


def _stats(xs):
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return {"median": None, "min": None, "max": None, "sum": None, "n": 0}
    return {"median": float(np.median(xs)), "min": float(min(xs)), "max": float(max(xs)), "sum": float(sum(xs)), "n": len(xs)}


def summarize(csg, k, iid, rows, total, eng, tags, source, timeout_s, cap_total, javamaxmem=None,
              maxiters=io_prism.DEFAULT_MAXITERS, emit_total=None, tool_identity=None):

    n = csg.n_agents
    vg = eng["vg"]
    for r in rows:
        if r["status"] == "OK":
            v = r["value"]
            finite = v is not None and np.isfinite(v) and (0.0 <= v <= 1.0)
            r["gpu_value"] = float(vg[r["mask"]])
            r["abs_err"] = float(abs(v - vg[r["mask"]])) if finite else None
            r["mismatch"] = bool((not finite) or r["abs_err"] > TOL_PRISM)
        else:
            r["gpu_value"] = float(vg[r["mask"]])
            r["abs_err"] = None
            r["mismatch"] = None
    ok = [r for r in rows if r["status"] == "OK"]
    n_ok, n_to = len(ok), sum(r["status"] == "TO" for r in rows)
    n_mo, n_err = sum(r["status"] == "MO" for r in rows), sum(r["status"] == "ERR" for r in rows)
    n_nr = sum(r["status"] == "NOT RUN" for r in rows)
    mism = [r for r in ok if r["mismatch"]]
    complete = n_ok == len(rows)
    reasons = []
    if mism:
        reasons.append(f"prism_value_mismatch: {len(mism)} coalition(s) with |v_PRISM - v_GPU| > {TOL_PRISM:g} or "
                       f"non-finite/out-of-[0,1] value (masks {[r['mask'] for r in mism][:16]})")
    if not eng["gpu_cpu_gate_ok"]:
        reasons.append(f"gpu_cpu_mismatch: max|v_GPU - v_CPU| = {eng['max_abs_err_gpu_vs_cpu']:.3e} > {TOL_GPU_CPU:g}"
                       + ("" if eng["gpu_values_finite_in_unit"] else " (or non-finite / outside [0,1])"))
    if n_err:
        first = next(r for r in rows if r["status"] == "ERR")
        reasons.append(f"prism_err: {n_err} coalition(s) ERR (first: mask {first['mask']}: {first['error']}; log {first['log']})")

    values_match = not reasons
    gate_failed = "; ".join(reasons) if reasons else None
    walls = _stats([r["wall_s"] for r in ok])


    solver_times_complete = bool(ok) and all(r["prism_solver_s"] is not None for r in ok)
    if solver_times_complete:
        solver = _stats([r["prism_solver_s"] for r in ok])
        constr = _stats([r["prism_construction_s"] for r in ok])
        check = _stats([r["prism_checking_s"] for r in ok])
    else:
        solver = constr = check = _stats([])
    errs = [r["abs_err"] for r in ok if r["abs_err"] is not None]
    tg, tc = eng["gpu"]["median_s"], eng["cpu"]["median_s"]
    prism = {"n_ok": n_ok, "n_to": n_to, "n_mo": n_mo, "n_err": n_err, "n_not_run": n_nr, "n_mismatch": len(mism),
             "complete": complete,
             "total_wall_s": float(total),
             "total_wall_ok_s": walls["sum"],
             "solver_total_s": solver["sum"],
             "construction_total_s": constr["sum"], "checking_total_s": check["sum"],
             "solver_times_complete": solver_times_complete,
             "jvm_overhead_s": (walls["sum"] - solver["sum"]) if (walls["sum"] is not None and solver["sum"] is not None) else None,
             "median_wall_s": walls["median"], "min_wall_s": walls["min"], "max_wall_s": walls["max"],
             "median_solver_s": solver["median"], "min_solver_s": solver["min"], "max_solver_s": solver["max"],
             "max_peak_rss_kb": max([r["peak_rss_kb"] or 0 for r in rows] or [0]) or None,
             "max_abs_err_vs_gpu": float(max(errs)) if errs else None, "tol": TOL_PRISM,
             "timeout_s": float(timeout_s), "cap_total_s": float(cap_total),
             "javamaxmem": javamaxmem or io_prism.DEFAULT_JAVAMAXMEM, "maxiters": maxiters,
             "emit_total_s": emit_total,
             "incomplete_statuses": sorted({r["status"] for r in rows if r["status"] != "OK"}),
             "rows": rows}
    rec = {"instance_id": iid, "source": source, "n": n, "states": csg.n_states, "k": k, "coalitions": len(rows),
           "values_match": values_match, "gate_failed": gate_failed, "prism": prism,
           "gpu_wall_s": tg, "gpu_timing": eng["gpu"], "cpu32_wall_s": tc, "cpu_timing": eng["cpu"],
           "cpu_threads": eng["threads"], "cpu_threads_label": cpu_threads_label(tags, eng["threads"]),
           "build_s": eng["build_s"], "max_abs_err_gpu_vs_cpu": eng["max_abs_err_gpu_vs_cpu"],
           "gpu_cpu_gate_ok": eng["gpu_cpu_gate_ok"], "tol_gpu_cpu": TOL_GPU_CPU,
           "timing_protocol": {"warmup": eng["warmup"], "n_repeat": eng["n_repeat"], "statistic": "median",
                               "gpu": "one batched solve of all 2^n coalitions (mask staging + k sweeps + D2H)",
                               "cpu": f"same batch, {eng['threads']} OpenMP threads, block=8",
                               "prism": "one CLI process per coalition under /usr/bin/time -v; wall_s = whole process (JVM start-up included); "
                                        "solver-only = PRISM-reported 'Time for model construction' + 'Time for model checking'"},
           "shapley_gpu": shapley_from_table(vg, n).tolist(), "gpu": tags.get("gpu"), "git": tags.get("git"),
           "git_dirty": tags.get("git_dirty"), "engine_binaries": engine_binaries(),
           "tool_identity": tool_identity if tool_identity is not None else io_prism.tool_identity(),
           "timestamp": protocol.now_iso()}
    rec["prism_version"] = rec["tool_identity"].get("prism_version")
    rec["java_version"] = rec["tool_identity"].get("java_version")
    rec["javamaxmem"] = prism["javamaxmem"]
    rec["maxiters"] = maxiters
    for f in SPEEDUP_FIELDS:
        rec[f] = None
    if gate_failed is None and complete:
        rec["prism_outcome"] = "OK"
        rec["speedup_gpu_vs_prism"] = float(total / tg)
        rec["speedup_gpu_vs_prism_solver_only"] = float(solver["sum"] / tg) if (solver_times_complete and solver["sum"]) else None
        rec["speedup_cpu_vs_prism"] = float(total / tc)
        vp = np.array([r["value"] for r in sorted(ok, key=lambda r: r["mask"])])
        rec["shapley_prism"] = shapley_from_table(vp, n).tolist()
        rec["shapley_max_abs_err"] = float(np.max(np.abs(np.array(rec["shapley_prism"]) - np.array(rec["shapley_gpu"]))))
    elif gate_failed is None and n_mo == 0 and n_err == 0:

        rec["prism_outcome"] = "TO"
        rec["speedup_gpu_vs_prism_lower_bound"] = float(total / tg)
        rec["speedup_cpu_vs_prism_lower_bound"] = float(total / tc)
    elif gate_failed is None:
        rec["prism_outcome"] = "MO"
    else:
        rec["prism_outcome"] = "ERR" if n_err else "GATE_FAILED"
    return rec


def _prism_record_status(rec):

    p = rec["prism"]
    if p["n_err"] > 0:
        return "CRASH", False, float(p["cap_total_s"]), None
    if p["n_mo"] > 0:
        return "MO", False, float(p["cap_total_s"]), None
    if p["complete"]:
        return "OK", False, float(p["cap_total_s"]), float(p["total_wall_s"])

    cap = float(p["cap_total_s"]) if p["n_not_run"] > 0 else float(p["timeout_s"])
    return "TO", True, cap, max(float(p["total_wall_s"]), cap)


def write(rec, tags, out_dir, cap_s=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"e3__{rec['instance_id']}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=protocol._json_default)
    p = rec["prism"]
    cap_total = float(cap_s if cap_s is not None else p["cap_total_s"])
    common = dict(values_match=rec["values_match"], gate_failed=rec["gate_failed"], tool_identity=rec["tool_identity"],
                  prism_version=rec["prism_version"], java_version=rec["java_version"], javamaxmem=rec["javamaxmem"],
                  maxiters=rec["maxiters"], engine_binaries=rec["engine_binaries"], prism_outcome=rec["prism_outcome"])
    base = dict(instance_id=rec["instance_id"], n=rec["n"], states=rec["states"], k=rec["k"], seed=SEED, exp="e3",
                coalitions=rec["coalitions"], full_enumeration=True, source=rec["source"], **common)
    status, censored, cap, wall = _prism_record_status(rec)
    ok_for_rate = status == "OK" and rec["values_match"]
    paths = []
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="prism_games", dtype="n/a", threads=1,
        threads_label="PRISM-games explicit engine (single-threaded value iteration; JVM JIT/GC helper threads not restricted)",
        cap_s=cap, cap_per_call_s=p["timeout_s"], cap_total_s=cap_total,
        wall_s=wall, censored=censored, status=status,
        coalition_solves_per_s=(rec["coalitions"] / p["total_wall_s"]) if ok_for_rate else None,
        max_abs_err_vs_fp64=p["max_abs_err_vs_gpu"], prism_n_ok=p["n_ok"], prism_n_to=p["n_to"], prism_n_mo=p["n_mo"],
        prism_n_err=p["n_err"], prism_n_not_run=p["n_not_run"], prism_n_mismatch=p["n_mismatch"],
        prism_total_wall_s=p["total_wall_s"], prism_solver_total_s=p["solver_total_s"], prism_jvm_overhead_s=p["jvm_overhead_s"],
        prism_median_wall_s=p["median_wall_s"], prism_min_wall_s=p["min_wall_s"], prism_max_wall_s=p["max_wall_s"],
        prism_median_solver_s=p["median_solver_s"], emit_s=p["emit_total_s"],
        notes="wall_s = sum of PRISM CLI walls over coalitions (JVM start-up included); status TO/censored only when the "
              "incomplete coalitions are all per-call timeouts or aggregate-cap NOT RUN; MO/CRASH carry no wall"),
        filename=f"e3__{rec['instance_id']}__prism_games.json", tags=tags))
    eng_note = "engines ran in-process without a cap; cap_s is the instance's aggregate PRISM cap"
    g = rec["gpu_timing"]
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=cap_total, wall_s=rec["gpu_wall_s"], censored=False,
        coalition_solves_per_s=rec["coalitions"] / rec["gpu_wall_s"], n_repeat=g["n_repeat"], warmup=g["warmup"],
        all_s=g["all_s"], min_s=g["min_s"], max_s=g["max_s"], warmup_s=g["warmup_s"], load_s=rec["build_s"],
        max_abs_err_gpu_vs_cpu=rec["max_abs_err_gpu_vs_cpu"], use_graph=True,
        speedup_vs_prism=rec["speedup_gpu_vs_prism"], speedup_vs_prism_solver_only=rec["speedup_gpu_vs_prism_solver_only"],
        speedup_vs_prism_lower_bound=rec["speedup_gpu_vs_prism_lower_bound"],
        speedup_is_lower_bound=rec["speedup_gpu_vs_prism_lower_bound"] is not None, notes=eng_note),
        filename=f"e3__{rec['instance_id']}__gpu_batched.json", tags=tags))
    c = rec["cpu_timing"]
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="cpu_multicore", dtype="fp64", threads=rec["cpu_threads"], threads_label=rec["cpu_threads_label"],
        cap_s=cap_total, wall_s=rec["cpu32_wall_s"], censored=False, coalition_solves_per_s=rec["coalitions"] / rec["cpu32_wall_s"],
        n_repeat=c["n_repeat"], warmup=c["warmup"], all_s=c["all_s"], min_s=c["min_s"], max_s=c["max_s"], warmup_s=c["warmup_s"],
        load_s=rec["build_s"], block=8, max_abs_err_gpu_vs_cpu=rec["max_abs_err_gpu_vs_cpu"],
        speedup_vs_prism=rec["speedup_cpu_vs_prism"], speedup_vs_prism_lower_bound=rec["speedup_cpu_vs_prism_lower_bound"],
        speedup_is_lower_bound=rec["speedup_cpu_vs_prism_lower_bound"] is not None, notes=eng_note),
        filename=f"e3__{rec['instance_id']}__cpu_multicore.json", tags=tags))
    return paths


def parse_prism_masks(spec: str, n: int):


    if not spec:
        return None
    if spec == "extremes":
        return {0, (1 << n) - 1}
    masks = {int(t) for t in spec.split(",") if t}
    bad = [m for m in masks if not 0 <= m < (1 << n)]
    if bad:
        raise ValueError(f"--prism-masks {spec!r}: masks {bad} out of range for n={n}")
    return masks


def run_instance(csg, k, iid, source, a, tags, out, work, extra=None):

    eng = time_engines(csg, k, n_repeat=a.n_repeat, threads=a.threads)
    print(f"[e3] {iid}: GPU median {eng['gpu']['median_s']:.4f}s (all {['%.4f' % x for x in eng['gpu']['all_s']]}) | "
          f"CPU-{eng['threads']} median {eng['cpu']['median_s']:.4f}s | max|GPU-CPU|={eng['max_abs_err_gpu_vs_cpu']:.2e}", flush=True)
    masks_to_run = parse_prism_masks(getattr(a, "prism_masks", ""), csg.n_agents)
    rows, total, emit_total = prism_all_coalitions(csg, k, iid, a.timeout, a.cap_total, work, javamaxmem=a.javamaxmem,
                                                   maxiters=a.maxiters, masks_to_run=masks_to_run)
    rec = summarize(csg, k, iid, rows, total, eng, tags, source, a.timeout, a.cap_total, javamaxmem=a.javamaxmem,
                    maxiters=a.maxiters, emit_total=emit_total)
    if extra:
        rec.update(extra)
    write(rec, tags, out, a.cap_total)
    p = rec["prism"]
    sp = rec["speedup_gpu_vs_prism"]
    spl = rec["speedup_gpu_vs_prism_lower_bound"]
    sps = f"{sp:.0f}x" if sp else (f">= {spl:.0f}x (lower bound)" if spl else "n/a")
    print(f"[e3] {iid}: PRISM ok={p['n_ok']}/{rec['coalitions']} to={p['n_to']} mo={p['n_mo']} err={p['n_err']} not_run={p['n_not_run']} "
          f"total={total:.1f}s solver-only={p['solver_total_s']} median={p['median_wall_s']} maxerr={p['max_abs_err_vs_gpu']} | "
          f"GPU {rec['gpu_wall_s']:.4f}s CPU {rec['cpu32_wall_s']:.4f}s | gate={'PASS' if rec['values_match'] and not rec['gate_failed'] else 'FAIL'} "
          f"| GPU/PRISM={sps}", flush=True)
    if rec["gate_failed"]:
        print(f"[e3] GATE FAILED for {iid}: {rec['gate_failed']}", file=sys.stderr, flush=True)
    return rec


def find_example(name):
    hits = glob.glob(os.path.join(REPO, "external", "prism-games", "prism-examples", "csgs", "**", name), recursive=True)
    return hits[0] if hits else None


BENCH = [
    ("robot_coordination2.prism", {"l": 3, "q": 0.1}, 3, {"fail_label": "crash"}),
    ("robot_coordination2.prism", {"l": 6, "q": 0.1}, 6, {"fail_label": "crash"}),
    ("medium_access2.prism", {"emax": 2, "q1": 0.9, "q2": 0.75}, 3, {"fail_expr": "s1=1&s2=1"}),
    ("rps2.prism", {}, 2, {"fail_expr": "win=1"}),


    ("aloha_backoff2.prism", {"bcmax": 2, "D": 8, "q": 0.9}, 20, {"fail_expr": "t>D & (s1!=3 | s2!=3)"}),
]
SMOKE_BENCH = [("rps2.prism", {}, 2, {"fail_expr": "win=1"})]


def bench_iid(name, consts, k):
    return f"prism_{name[:-6]}_" + "_".join(f"{kk}{v}" for kk, v in consts.items()) + f"_k{k}"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=600.0, help="cap per PRISM call [s]")
    ap.add_argument("--cap-total", type=float, default=1800.0, help="cap on cumulative PRISM wall per instance [s]")
    ap.add_argument("--configs", default="default")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--bench-file", default="", choices=["", "e3b"],
                    help="'e3b': run the PRISM-games benchmark-suite extension (experiments/e3b_bench.py BENCH3B) "
                         "instead of the E3 TicketBench+BENCH grid; default out/work move to results/e3b")
    ap.add_argument("--prism-masks", default="",
                    help="restrict the PRISM arm to these coalition masks ('' = all 2^n; 'extremes' = all-free + "
                         "all-pinned; or comma-separated ints). Skipped masks are NOT RUN -> lower-bound speedups only "
                         "(validation use, not for the full sweep)")
    ap.add_argument("--prism-extra", default="", help="extra PRISM CLI args for every call, e.g. '-javastack 1g' (recorded in each row's cmd)")
    ap.add_argument("--only", default="", help="comma-separated substrings: only run instances whose id contains one of them")
    ap.add_argument("--javamaxmem", default=None, help=f"PRISM -javamaxmem (default {io_prism.DEFAULT_JAVAMAXMEM})")
    ap.add_argument("--maxiters", type=int, default=io_prism.DEFAULT_MAXITERS)
    ap.add_argument("--n-repeat", type=int, default=N_REPEAT, help="timed engine solves after one warm-up")
    ap.add_argument("--threads", type=int, default=CPU_THREADS, help="CPU OpenMP threads (32 = physical cores)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "e3"))
    ap.add_argument("--work", default=os.path.join(REPO, "results", "prism_work", "e3"))
    a = ap.parse_args(argv)
    PRISM_EXTRA[:] = a.prism_extra.split()
    only = [t for t in a.only.split(",") if t]
    tags = protocol.machine_tags()
    if a.bench_file == "e3b":
        if a.out == os.path.join(REPO, "results", "e3"):
            a.out = os.path.join(REPO, "results", "e3b")
        if a.work == os.path.join(REPO, "results", "prism_work", "e3"):
            a.work = os.path.join(REPO, "results", "prism_work", "e3b")
    out, work = a.out, a.work
    os.makedirs(work, exist_ok=True)
    ident = io_prism.tool_identity()
    print(f"[e3] {ident.get('prism_version')} | {ident.get('java_version')} | javamaxmem={a.javamaxmem or io_prism.DEFAULT_JAVAMAXMEM} "
          f"maxiters={a.maxiters} | launcher sha256 {str(ident.get('prism_launcher_sha256'))[:12]} classes sha256 "
          f"{str(ident.get('prism_classes_sha256'))[:12]} | git {tags.get('git')}", flush=True)
    if a.bench_file == "e3b":


        from experiments.e3b_bench import active_bench, RECOMMENDED_PRISM_EXTRA
        if not PRISM_EXTRA:
            print(f"[e3b] note: consider --prism-extra '{RECOMMENDED_PRISM_EXTRA}' for the full sweep "
                  f"(the largest imports overflow the default JVM parser stack, see e3b_bench.py)", flush=True)
        failed = []
        for iid, path, consts, k, fkw, entry in active_bench():
            if only and not any(t in iid for t in only):
                continue
            if os.path.exists(os.path.join(out, f"e3__{iid}.json")):
                print(f"[e3b] skip {iid} (exists)"); continue
            csg = io_prism.import_prism_csg(path, consts, policy="uniform", k=k,
                                            out_base=os.path.join(work, "import", iid), **fkw)
            rec = run_instance(csg, k, iid, f"prism_csg:{entry['path']}", a, tags, out, work,
                               extra={"consts": consts, "fail": fkw, "bench_path": path,
                                      "bench_source": entry["source"], "bench_paper": entry["paper"]})
            if rec["gate_failed"]:
                failed.append(iid)
        if failed:
            print(f"[e3b] GATE FAILED on {len(failed)} instance(s): {failed}", file=sys.stderr, flush=True)
            return 2
        return 0
    if a.configs == "smoke":
        cfgs, bench = [(2, 20, 5)], SMOKE_BENCH
    else:
        cfgs = [(4, 100, 25), (6, 100, 25), (8, 100, 25), (4, 1000, 25), (6, 1000, 25), (8, 1000, 25), (4, 1000, 100), (4, 5000, 25), (4, 20000, 25)]
        bench = BENCH
    failed = []
    for n, S, k in cfgs:
        csg = build_instance(n, S, k, 0.05, "none", SEED)
        iid = csg.meta["instance_id"] if csg.meta.get("instance_id") else f"tb_n{n}_S{S}_k{k}"
        if only and not any(t in iid for t in only):
            continue
        if os.path.exists(os.path.join(out, f"e3__{iid}.json")):
            print(f"[e3] skip {iid} (exists)"); continue
        rec = run_instance(csg, k, iid, "ticketbench", a, tags, out, work)
        if rec["gate_failed"]:
            failed.append(iid)
    if not a.skip_bench:
        for name, consts, k, fkw in bench:
            path = find_example(name)
            if path is None:
                print(f"[e3] benchmark {name} not found under external/prism-games/prism-examples/csgs -> skipped"); continue
            iid = bench_iid(name, consts, k)
            if os.path.exists(os.path.join(out, f"e3__{iid}.json")):
                print(f"[e3] skip {iid} (exists)"); continue
            try:
                csg = io_prism.import_prism_csg(path, consts, policy="uniform", k=k,
                                                out_base=os.path.join(work, "import", iid), **fkw)
            except Exception as e:
                print(f"[e3] import of {name} {consts} not possible ({e!r}) -> skipped"); continue
            rec = run_instance(csg, k, iid, f"prism-games:{name}", a, tags, out, work,
                               extra={"consts": consts, "fail": fkw, "bench_path": path})
            if rec["gate_failed"]:
                failed.append(iid)
    if failed:
        print(f"[e3] GATE FAILED on {len(failed)} instance(s): {failed}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
