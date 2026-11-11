#!/usr/bin/env python3


from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from experiments.e3_prism import (time_engines, engine_binaries, cpu_threads_label, _stats,
                                  TOL_GPU_CPU, CPU_THREADS, N_REPEAT, SEED)
from generators.ticketbench import build_instance
from cubcr import io_prism, io_storm
from cubcr.shapley import shapley_from_table

TOL_EXT = 1e-6
SPEEDUP_FIELDS = ("speedup_gpu_vs_storm", "speedup_gpu_vs_storm_solver_only", "speedup_gpu_vs_storm_lower_bound",
                  "speedup_cpu_vs_storm", "speedup_cpu_vs_storm_lower_bound")
STORM_EXTRA = []


def _not_run_row(mask, reason, status="NOT RUN"):
    return {"mask": int(mask), "status": status, "value": None, "wall_s": None,
            "storm_reported_time_s": None, "storm_construction_s": None, "storm_checking_s": None,
            "storm_states": None, "storm_transitions": None, "peak_rss_kb": None,
            "cmd": None, "rc": None, "log": None, "model_path": None, "props_path": None,
            "emit_s": None, "error": None, "timed_out": False, "reason": reason}


def storm_all_coalitions(csg, k, iid, timeout_s, cap_total, workdir, storm_bin=None,
                         stop_on_err=True, verbose=True):


    n = csg.n_agents
    rows, total, emit_total, aborted = [], 0.0, 0.0, None
    if not io_storm.storm_available(storm_bin):
        b = io_storm.resolve_bin(storm_bin)
        if verbose:
            print(f"  [storm] {iid}: binary missing at {b} -> all {1 << n} coalitions SKIP", flush=True)
        for mask in range(1 << n):
            rows.append(_not_run_row(mask, f"storm binary missing at {b}", status="SKIP"))
        return rows, 0.0, 0.0
    for mask in range(1 << n):
        if aborted is not None:
            rows.append(_not_run_row(mask, f"aborted after {aborted}"))
            continue
        if total > cap_total:
            rows.append(_not_run_row(mask, f"aggregate cap: cumulative Storm wall {total:.1f}s > cap_total {cap_total}s"))
            continue
        t0 = time.perf_counter()
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, k)
        emit_s = time.perf_counter() - t0
        emit_total += emit_s
        r = io_storm.run_storm(mdp, props, timeout_s=timeout_s, workdir=workdir, tag=f"{iid}_mask{mask}",
                               storm_bin=storm_bin, extra_args=tuple(STORM_EXTRA), include_identity=False)
        row = {"mask": int(mask), "status": r["status"], "value": r["value"], "wall_s": r["wall_s"],
               "storm_reported_time_s": r["storm_reported_time_s"],
               "storm_construction_s": r["storm_construction_s"], "storm_checking_s": r["storm_checking_s"],
               "storm_states": r["storm_states"], "storm_transitions": r["storm_transitions"],
               "peak_rss_kb": r["peak_rss_kb"], "time_elapsed_s": r["time_elapsed_s"],
               "cmd": r["cmd"], "rc": r["rc"], "log": r["log_path"],
               "model_path": r["model_path"], "props_path": r["props_path"],
               "emit_s": emit_s, "error": r["error"], "timed_out": r["timed_out"], "reason": None}
        rows.append(row)
        if r["wall_s"] is not None:
            total += r["wall_s"]
        if verbose:
            w = f"{r['wall_s']:.3f}s" if r["wall_s"] is not None else "n/a"
            print(f"  [storm] {iid} mask={mask} {r['status']} v={r['value']} wall={w} "
                  f"solver={r['storm_reported_time_s']}" + (f" error={r['error']}" if r["error"] else ""), flush=True)
        if r["status"] == "ERR" and stop_on_err:
            aborted = f"ERR on mask {mask}: {r['error']} (log: {r['log_path']})"
            print(f"[e3storm] ERROR: Storm returned ERR for {iid} mask={mask}: {r['error']} -> see {r['log_path']}",
                  file=sys.stderr, flush=True)
    return rows, total, emit_total


def prism_mdp_rows(csg, k, iid, masks_to_run, timeout_s, workdir, verbose=True):

    if masks_to_run is None:
        return None
    rows = []
    for mask in sorted(masks_to_run):
        t0 = time.perf_counter()
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, k)
        emit_s = time.perf_counter() - t0
        r = io_prism.run_prism(mdp, props, timeout_s=timeout_s, workdir=workdir,
                               tag=f"{iid}_mdp_mask{mask}", include_identity=False)
        rows.append({"mask": int(mask), "status": r["status"], "value": r["value"], "wall_s": r["wall_s"],
                     "cmd": r["cmd"], "rc": r["returncode"], "log": r["stdout_path"], "emit_s": emit_s,
                     "error": r["error"]})
        if verbose:
            print(f"  [prism-mdp] {iid} mask={mask} {r['status']} v={r['value']} wall={r['wall_s']:.3f}s"
                  + (f" error={r['error']}" if r["error"] else ""), flush=True)
    return rows


def summarize(csg, k, iid, rows, total, eng, tags, source, timeout_s, cap_total,
              emit_total=None, prism_rows=None, storm_identity=None):


    n = csg.n_agents
    vg = eng["vg"]
    for r in rows:
        r["gpu_value"] = float(vg[r["mask"]])
        if r["status"] == "OK":
            v = r["value"]
            finite = v is not None and np.isfinite(v) and (0.0 <= v <= 1.0)
            r["abs_err"] = float(abs(v - vg[r["mask"]])) if finite else None
            r["mismatch"] = bool((not finite) or r["abs_err"] > TOL_EXT)
        else:
            r["abs_err"] = None
            r["mismatch"] = None
    ok = [r for r in rows if r["status"] == "OK"]
    n_ok, n_to = len(ok), sum(r["status"] == "TO" for r in rows)
    n_mo, n_err = sum(r["status"] == "MO" for r in rows), sum(r["status"] == "ERR" for r in rows)
    n_nr = sum(r["status"] == "NOT RUN" for r in rows)
    n_skip = sum(r["status"] == "SKIP" for r in rows)
    mism = [r for r in ok if r["mismatch"]]
    complete = n_ok == len(rows)
    skipped = n_skip == len(rows)


    pm = None
    if prism_rows is not None:
        for r in prism_rows:
            r["gpu_value"] = float(vg[r["mask"]])
            if r["status"] == "OK" and r["value"] is not None and np.isfinite(r["value"]):
                r["abs_err"] = float(abs(r["value"] - vg[r["mask"]]))
                r["mismatch"] = bool(r["abs_err"] > TOL_EXT)
            else:
                r["abs_err"], r["mismatch"] = None, None
        p_ok = [r for r in prism_rows if r["status"] == "OK"]
        p_mism = [r for r in p_ok if r["mismatch"]]
        pm = {"ran": True, "n_rows": len(prism_rows), "n_ok": len(p_ok),
              "n_fail": len(prism_rows) - len(p_ok), "n_mismatch": len(p_mism),
              "max_abs_err_vs_gpu": float(max([r["abs_err"] for r in p_ok], default=0.0)) if p_ok else None,
              "tol": TOL_EXT, "rows": prism_rows,
              "note": "PRISM-games CLI on the SAME plain-mdp emission Storm gets; validates the "
                      "emission semantics end-to-end (docs/e3storm.md)"}

    reasons = []
    if mism:
        reasons.append(f"storm_value_mismatch: {len(mism)} coalition(s) with |v_Storm - v_GPU| > {TOL_EXT:g} or "
                       f"non-finite/out-of-[0,1] value (masks {[r['mask'] for r in mism][:16]})")
    if not eng["gpu_cpu_gate_ok"]:
        reasons.append(f"gpu_cpu_mismatch: max|v_GPU - v_CPU| = {eng['max_abs_err_gpu_vs_cpu']:.3e} > {TOL_GPU_CPU:g}"
                       + ("" if eng["gpu_values_finite_in_unit"] else " (or non-finite / outside [0,1])"))
    if n_err:
        first = next(r for r in rows if r["status"] == "ERR")
        reasons.append(f"storm_err: {n_err} coalition(s) ERR (first: mask {first['mask']}: {first['error']}; log {first['log']})")
    if pm is not None and (pm["n_mismatch"] or pm["n_fail"]):
        reasons.append(f"prism_mdp_gate: {pm['n_mismatch']} mismatch(es) / {pm['n_fail']} failure(s) on the "
                       f"PRISM-on-MDP validation arm (emission suspect)")
    values_match = not reasons
    gate_failed = "; ".join(reasons) if reasons else None

    walls = _stats([r["wall_s"] for r in ok])
    solver_times_complete = bool(ok) and all(r["storm_reported_time_s"] is not None for r in ok)
    solver = _stats([r["storm_reported_time_s"] for r in ok]) if solver_times_complete else _stats([])
    errs = [r["abs_err"] for r in ok if r["abs_err"] is not None]
    tg, tc = eng["gpu"]["median_s"], eng["cpu"]["median_s"]
    storm = {"n_ok": n_ok, "n_to": n_to, "n_mo": n_mo, "n_err": n_err, "n_not_run": n_nr, "n_skip": n_skip,
             "n_mismatch": len(mism), "complete": complete, "skipped": skipped,
             "total_wall_s": float(total),
             "total_wall_ok_s": walls["sum"],
             "solver_total_s": solver["sum"],
             "solver_times_complete": solver_times_complete,
             "startup_overhead_s": (walls["sum"] - solver["sum"]) if (walls["sum"] is not None and solver["sum"] is not None) else None,
             "median_wall_s": walls["median"], "min_wall_s": walls["min"], "max_wall_s": walls["max"],
             "median_solver_s": solver["median"], "min_solver_s": solver["min"], "max_solver_s": solver["max"],
             "max_peak_rss_kb": max([r["peak_rss_kb"] or 0 for r in rows] or [0]) or None,
             "max_abs_err_vs_gpu": float(max(errs)) if errs else None, "tol": TOL_EXT,
             "timeout_s": float(timeout_s), "cap_total_s": float(cap_total), "emit_total_s": emit_total,
             "incomplete_statuses": sorted({r["status"] for r in rows if r["status"] != "OK"}),
             "rows": rows}
    rec = {"instance_id": iid, "source": source, "n": n, "states": csg.n_states, "k": k, "coalitions": len(rows),
           "values_match": values_match, "gate_failed": gate_failed, "storm": storm, "prism_mdp": pm,
           "gpu_wall_s": tg, "gpu_timing": eng["gpu"], "cpu32_wall_s": tc, "cpu_timing": eng["cpu"],
           "cpu_threads": eng["threads"], "cpu_threads_label": cpu_threads_label(tags, eng["threads"]),
           "build_s": eng["build_s"], "max_abs_err_gpu_vs_cpu": eng["max_abs_err_gpu_vs_cpu"],
           "gpu_cpu_gate_ok": eng["gpu_cpu_gate_ok"], "tol_gpu_cpu": TOL_GPU_CPU,
           "timing_protocol": {"warmup": eng["warmup"], "n_repeat": eng["n_repeat"], "statistic": "median",
                               "gpu": "one batched solve of all 2^n coalitions (mask staging + k sweeps + D2H)",
                               "cpu": f"same batch, {eng['threads']} OpenMP threads, block=8",
                               "storm": "one CLI process per coalition MDP under /usr/bin/time -v; wall_s = whole "
                                        "process (start-up included); solver-only = Storm-reported "
                                        "'Time for model construction' + 'Time for model checking'"},
           "shapley_gpu": shapley_from_table(vg, n).tolist(), "gpu": tags.get("gpu"), "git": tags.get("git"),
           "git_dirty": tags.get("git_dirty"), "engine_binaries": engine_binaries(),
           "tool_identity": storm_identity if storm_identity is not None else io_storm.tool_identity(),
           "storm_version": None, "timestamp": protocol.now_iso()}
    rec["storm_version"] = rec["tool_identity"].get("storm_version")
    for f in SPEEDUP_FIELDS:
        rec[f] = None
    if skipped:
        rec["storm_outcome"] = "SKIP"
    elif gate_failed is None and complete:
        rec["storm_outcome"] = "OK"
        rec["speedup_gpu_vs_storm"] = float(total / tg)
        rec["speedup_gpu_vs_storm_solver_only"] = float(solver["sum"] / tg) if (solver_times_complete and solver["sum"]) else None
        rec["speedup_cpu_vs_storm"] = float(total / tc)
        vs = np.array([r["value"] for r in sorted(ok, key=lambda r: r["mask"])])
        rec["shapley_storm"] = shapley_from_table(vs, n).tolist()
        rec["shapley_max_abs_err"] = float(np.max(np.abs(np.array(rec["shapley_storm"]) - np.array(rec["shapley_gpu"]))))
    elif gate_failed is None and n_mo == 0 and n_err == 0 and n_skip == 0:
        rec["storm_outcome"] = "TO"
        rec["speedup_gpu_vs_storm_lower_bound"] = float(total / tg)
        rec["speedup_cpu_vs_storm_lower_bound"] = float(total / tc)
    elif gate_failed is None:
        rec["storm_outcome"] = "MO" if n_mo else "SKIP_PARTIAL"
    else:
        rec["storm_outcome"] = "ERR" if n_err else "GATE_FAILED"
    return rec


def _storm_record_status(rec):

    s = rec["storm"]
    if s["skipped"]:
        return "NOT RUN", False, float(s["cap_total_s"]), None
    if s["n_err"] > 0:
        return "CRASH", False, float(s["cap_total_s"]), None
    if s["n_mo"] > 0:
        return "MO", False, float(s["cap_total_s"]), None
    if s["complete"]:
        return "OK", False, float(s["cap_total_s"]), float(s["total_wall_s"])
    cap = float(s["cap_total_s"]) if s["n_not_run"] > 0 else float(s["timeout_s"])
    return "TO", True, cap, max(float(s["total_wall_s"]), cap)


def write(rec, tags, out_dir, cap_s=None):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"e3storm__{rec['instance_id']}.json"), "w") as fh:
        json.dump(rec, fh, indent=1, default=protocol._json_default)
    s = rec["storm"]
    cap_total = float(cap_s if cap_s is not None else s["cap_total_s"])
    common = dict(values_match=rec["values_match"], gate_failed=rec["gate_failed"], tool_identity=rec["tool_identity"],
                  storm_version=rec["storm_version"], engine_binaries=rec["engine_binaries"],
                  storm_outcome=rec["storm_outcome"])
    base = dict(instance_id=rec["instance_id"], n=rec["n"], states=rec["states"], k=rec["k"], seed=SEED,
                exp="e3storm", coalitions=rec["coalitions"], full_enumeration=True, source=rec["source"], **common)
    status, censored, cap, wall = _storm_record_status(rec)
    ok_for_rate = status == "OK" and rec["values_match"] and s["total_wall_s"]
    paths = []
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="storm", dtype="n/a", threads=1,
        threads_label="Storm CLI, one collapsed-MDP process per coalition (sparse engine defaults)",
        cap_s=cap, cap_per_call_s=s["timeout_s"], cap_total_s=cap_total,
        wall_s=wall, censored=censored, status=status,
        coalition_solves_per_s=(rec["coalitions"] / s["total_wall_s"]) if ok_for_rate else None,
        max_abs_err_vs_fp64=s["max_abs_err_vs_gpu"], storm_n_ok=s["n_ok"], storm_n_to=s["n_to"],
        storm_n_mo=s["n_mo"], storm_n_err=s["n_err"], storm_n_not_run=s["n_not_run"], storm_n_skip=s["n_skip"],
        storm_n_mismatch=s["n_mismatch"], storm_total_wall_s=s["total_wall_s"],
        storm_solver_total_s=s["solver_total_s"], storm_startup_overhead_s=s["startup_overhead_s"],
        storm_median_wall_s=s["median_wall_s"], storm_min_wall_s=s["min_wall_s"], storm_max_wall_s=s["max_wall_s"],
        storm_median_solver_s=s["median_solver_s"], emit_s=s["emit_total_s"],
        notes="wall_s = sum of Storm CLI walls over coalition MDPs (process start-up included); each coalition is a "
              "collapsed plain MDP (io_prism.emit_coalition_mdp), the fair best-single-solve-tool arm; status "
              "NOT RUN = storm binary missing (SKIP rows); TO/censored only when the incomplete coalitions are all "
              "per-call timeouts or aggregate-cap NOT RUN; MO/CRASH carry no wall"),
        filename=f"e3storm__{rec['instance_id']}__storm.json", tags=tags))
    eng_note = "engines ran in-process without a cap; cap_s is the instance's aggregate Storm cap"
    g = rec["gpu_timing"]
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="gpu_batched", dtype="fp64", threads=0, cap_s=cap_total, wall_s=rec["gpu_wall_s"], censored=False,
        coalition_solves_per_s=rec["coalitions"] / rec["gpu_wall_s"], n_repeat=g["n_repeat"], warmup=g["warmup"],
        all_s=g["all_s"], min_s=g["min_s"], max_s=g["max_s"], warmup_s=g["warmup_s"], load_s=rec["build_s"],
        max_abs_err_gpu_vs_cpu=rec["max_abs_err_gpu_vs_cpu"], use_graph=True,
        speedup_vs_storm=rec["speedup_gpu_vs_storm"], speedup_vs_storm_solver_only=rec["speedup_gpu_vs_storm_solver_only"],
        speedup_vs_storm_lower_bound=rec["speedup_gpu_vs_storm_lower_bound"],
        speedup_is_lower_bound=rec["speedup_gpu_vs_storm_lower_bound"] is not None, notes=eng_note),
        filename=f"e3storm__{rec['instance_id']}__gpu_batched.json", tags=tags))
    c = rec["cpu_timing"]
    paths.append(protocol.write_result(out_dir, protocol.make_record(
        **base, engine="cpu_multicore", dtype="fp64", threads=rec["cpu_threads"], threads_label=rec["cpu_threads_label"],
        cap_s=cap_total, wall_s=rec["cpu32_wall_s"], censored=False,
        coalition_solves_per_s=rec["coalitions"] / rec["cpu32_wall_s"],
        n_repeat=c["n_repeat"], warmup=c["warmup"], all_s=c["all_s"], min_s=c["min_s"], max_s=c["max_s"],
        warmup_s=c["warmup_s"], load_s=rec["build_s"], block=8, max_abs_err_gpu_vs_cpu=rec["max_abs_err_gpu_vs_cpu"],
        speedup_vs_storm=rec["speedup_cpu_vs_storm"], speedup_vs_storm_lower_bound=rec["speedup_cpu_vs_storm_lower_bound"],
        speedup_is_lower_bound=rec["speedup_cpu_vs_storm_lower_bound"] is not None, notes=eng_note),
        filename=f"e3storm__{rec['instance_id']}__cpu_multicore.json", tags=tags))
    return paths


def parse_prism_mdp(spec: str, n: int, configs: str):


    if spec == "auto":
        spec = "all" if configs == "smoke" else "extremes"
    if spec == "none":
        return None
    if spec == "all":
        return set(range(1 << n))
    if spec == "extremes":
        return {0, (1 << n) - 1}
    raise ValueError(f"--prism-mdp {spec!r}: expected auto|all|extremes|none")


def run_instance(csg, k, iid, source, a, tags, out, work):
    eng = time_engines(csg, k, n_repeat=a.n_repeat, threads=a.threads)
    print(f"[e3storm] {iid}: GPU median {eng['gpu']['median_s']:.4f}s | CPU-{eng['threads']} median "
          f"{eng['cpu']['median_s']:.4f}s | max|GPU-CPU|={eng['max_abs_err_gpu_vs_cpu']:.2e}", flush=True)
    pmasks = parse_prism_mdp(a.prism_mdp, csg.n_agents, a.configs) if io_prism.prism_available() else None
    if a.prism_mdp != "none" and not io_prism.prism_available():
        print(f"[e3storm] {iid}: PRISM binary missing at {io_prism.PRISM_BIN} -> prism-mdp validation arm skipped", flush=True)
    prows = prism_mdp_rows(csg, k, iid, pmasks, a.timeout, work)
    rows, total, emit_total = storm_all_coalitions(csg, k, iid, a.timeout, a.cap_total, work,
                                                   storm_bin=a.storm_bin or None)
    rec = summarize(csg, k, iid, rows, total, eng, tags, source, a.timeout, a.cap_total,
                    emit_total=emit_total, prism_rows=prows,
                    storm_identity=io_storm.tool_identity(a.storm_bin or None))
    write(rec, tags, out, a.cap_total)
    st = rec["storm"]
    sp, spl = rec["speedup_gpu_vs_storm"], rec["speedup_gpu_vs_storm_lower_bound"]
    sps = f"{sp:.0f}x" if sp else (f">= {spl:.0f}x (lower bound)" if spl else "n/a")
    print(f"[e3storm] {iid}: storm ok={st['n_ok']}/{rec['coalitions']} to={st['n_to']} mo={st['n_mo']} err={st['n_err']} "
          f"skip={st['n_skip']} not_run={st['n_not_run']} total={total:.1f}s | GPU {rec['gpu_wall_s']:.4f}s "
          f"CPU {rec['cpu32_wall_s']:.4f}s | outcome={rec['storm_outcome']} "
          f"gate={'PASS' if rec['values_match'] else 'FAIL'} | GPU/Storm={sps}", flush=True)
    if rec["gate_failed"]:
        print(f"[e3storm] GATE FAILED for {iid}: {rec['gate_failed']}", file=sys.stderr, flush=True)
    return rec


FULL_CFGS = [(4, 100, 25), (6, 100, 25), (8, 100, 25), (4, 1000, 25), (6, 1000, 25), (8, 1000, 25)]
SMOKE_CFGS = [(2, 25, 5)]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=600.0, help="cap per Storm (and prism-mdp) call [s]")
    ap.add_argument("--cap-total", type=float, default=1800.0, help="cap on cumulative Storm wall per instance [s]")
    ap.add_argument("--configs", default="smoke", choices=["smoke", "full"],
                    help="smoke: n=2 |S|~25 k=5 (prism-mdp on all coalitions + Storm if built); "
                         "full: the E3 TicketBench cells n in {4,6,8} x |S| in {100,1000}, k=25")
    ap.add_argument("--prism-mdp", default="auto", choices=["auto", "all", "extremes", "none"],
                    help="PRISM-games on the SAME mdp emission (validation arm): auto = all coalitions in smoke, "
                         "extreme masks {0, 2^n-1} in full")
    ap.add_argument("--storm-bin", default="", help=f"Storm CLI binary (default {io_storm.STORM_BIN}; "
                                                    "missing binary -> SKIP rows, never an error)")
    ap.add_argument("--storm-extra", default="", help="extra Storm CLI args for every call (recorded in each row's cmd)")
    ap.add_argument("--only", default="", help="comma-separated substrings: only run instances whose id contains one")
    ap.add_argument("--n-repeat", type=int, default=N_REPEAT, help="timed engine solves after one warm-up")
    ap.add_argument("--threads", type=int, default=CPU_THREADS, help="CPU OpenMP threads (32 = physical cores)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "e3storm"))
    ap.add_argument("--work", default=os.path.join(REPO, "results", "storm_work", "e3storm"))
    a = ap.parse_args(argv)
    STORM_EXTRA[:] = a.storm_extra.split()
    only = [t for t in a.only.split(",") if t]
    tags = protocol.machine_tags()
    out, work = a.out, a.work
    os.makedirs(work, exist_ok=True)
    ident = io_storm.tool_identity(a.storm_bin or None, refresh=True)
    print(f"[e3storm] storm={'MISSING (' + ident['storm_bin'] + ')' if not ident['available'] else ident['storm_version']} "
          f"| bin sha256 {str(ident['storm_bin_sha256'])[:12]} | prism={'present' if io_prism.prism_available() else 'MISSING'} "
          f"| git {tags.get('git')}", flush=True)
    cfgs = SMOKE_CFGS if a.configs == "smoke" else FULL_CFGS
    failed = []
    for n, S, k in cfgs:
        csg = build_instance(n, S, k, 0.05, "none", SEED)
        iid = csg.meta["instance_id"] if csg.meta.get("instance_id") else f"tb_n{n}_S{S}_k{k}"
        if only and not any(t in iid for t in only):
            continue
        if os.path.exists(os.path.join(out, f"e3storm__{iid}.json")):
            print(f"[e3storm] skip {iid} (exists)"); continue
        rec = run_instance(csg, k, iid, "ticketbench", a, tags, out, work)
        if rec["gate_failed"]:
            failed.append(iid)
    if failed:
        print(f"[e3storm] GATE FAILED on {len(failed)} instance(s): {failed}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
