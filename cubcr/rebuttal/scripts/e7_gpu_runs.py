#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import numpy as np

import rb
import e7_topk as e7


def _table_twin(rule, iid, seed):
    p = e7.raw_path("table", e7.run_name(rule, iid, seed))
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def _compare(rec, twin):
    if twin is None:
        return None
    keys = ("status", "n_units", "n_perms", "n_evals_unique", "n_evals_requested", "batch_sizes", "top_k_players")
    same = {k: rec.get(k) == twin.get(k) for k in keys}
    same["phi_hat_bitwise"] = rec["phi_hat"] == twin["phi_hat"]
    same["half_widths_bitwise"] = rec["half_widths"] == twin["half_widths"]
    same["all"] = all(same.values())
    same["gpu_censored"] = rec["status"] in ("time",)
    return same


def _predicted_cost(t, rule, seed):


    rate = (1 << t["n"]) / float(t["exact_gpu"]["solve_s"])
    tw = _table_twin(rule, t["iid"], seed)
    if tw is not None:
        uniq, req = tw["n_evals_unique"], tw["n_evals_requested"]
    elif rule == "topk3":
        req = 2 + 2 * 2016 * (t["n"] + 1); uniq = req
    elif rule == "eps100":
        req = 2 + 2 * 12000 * (t["n"] + 1); uniq = req
    elif rule == "eps20":
        req = 2 + 2 * 480 * (t["n"] + 1); uniq = req
    else:
        return 1e9 if not rule.startswith("topk") else e7.E7_GPU_TIME_BUDGET + 10.0
    pred = 0.5 + uniq / rate + 1e-6 * req
    if rule.startswith("topk"):
        pred = min(pred, e7.E7_GPU_TIME_BUDGET + 10.0)
    return pred


def _gpu_body(t, comp, kw):

    n, k = t["n"], t["k"]
    t_hold = time.perf_counter()
    snap = rb.load_snapshot()
    t0 = time.perf_counter()
    eng = rb.make_gpu_engine(comp)
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rb.paper_warmup(eng, n, k)
    warm_s = time.perf_counter() - t0
    try:
        r, wall = e7.run_sampler(eng, kw)
        stats = getattr(eng, "stats", None)
    finally:
        eng.close()
    return r, wall, {"load": snap, "engine_setup_s": setup_s, "warmup_s": warm_s, "engine_stats_last": stats,
                     "lock_hold_s": time.perf_counter() - t_hold}


def _kw(t, rule, seed):
    n, k = t["n"], t["k"]
    if rule.startswith("topk"):
        return e7.topk_kwargs(int(rule[4:]), seed, n, k, time_budget=e7.E7_GPU_TIME_BUDGET), int(rule[4:])
    if rule == "eps100":
        return e7.eps_kwargs(t["upsilon"] / 100.0, seed, n, k), None
    if rule == "eps20":
        return e7.eps_kwargs(t["eps20"], seed, n, k), None
    raise ValueError(rule)


def _write(t, rule, seed, r, wall, kw, K, info, tags, waited, group):
    extra = dict(info, hardware=tags, timestamp=rb.now_iso(), lock_wait_s=float(waited), lock_group=group,
                 timing_convention="perf_counter around sample_shapley only (engine creation and warm-up excluded, "
                                   "as in the paper)",
                 exhaustive_gpu_wall_s_stored=float(t["exact_gpu"]["wall_s"]),
                 exhaustive_gpu_solve_s_stored=float(t["exact_gpu"]["solve_s"]))
    if rule == "eps100":
        extra["eps_div"] = 100
    if rule == "eps20":
        st = t["cert_gpu"][seed]
        extra["eps_div"] = 20
        extra["stored_cert_gpu_match"] = {
            "estimates": bool(np.array_equal(r.estimates, np.asarray(st["shapley_sampled"]))),
            "half_widths": bool(np.array_equal(r.half_widths, np.asarray(st["half_widths"]))),
            "n_perms": r.n_perms == st["n_perms"], "n_solves": r.n_solves == st["n_solves"],
            "batch_sizes": list(r.batch_sizes) == list(st["batch_sizes"]),
            "stored_wall_s": float(st["wall_s"]), "stored_file": st.get("_file")}
    rec = e7.result_record(r, wall, kw, engine="gpu", task_meta=e7.task_meta(t), phi_exact=t["phi_exact"],
                           K=K, extra=extra)
    if t["n"] <= 24:
        rec["table_twin_match"] = _compare(rec, _table_twin(rule, t["iid"], seed))
    rb.write_json(e7.raw_path("gpu", e7.run_name(rule, t["iid"], seed)), rec)
    tm = rec.get("table_twin_match")
    print(f"[{rb.now_iso()}] {t['iid']} {rule} s{seed}: {r.status} wall={wall:.2f}s units={r.n_samples} "
          f"uniq={r.n_solves} hold={info['lock_hold_s']:.1f}s wait={waited:.1f}s group={group}"
          + (f" set_ok={rec.get('certified_set_equals_exact')}" if K else "")
          + ("" if tm is None else f" twin_all={tm['all']}"), flush=True)


def run_group(jobs, tags, comps, hold_budget_s=None):


    group = f"{jobs[0][1]}:{jobs[0][0]['iid']}:s{jobs[0][2]}+{len(jobs) - 1}"
    outs, rest = [], []
    with rb.gpu_lock(f"E7 group {group}") as waited:
        t_lock = time.perf_counter()
        for i, (t, rule, seed) in enumerate(jobs):
            if hold_budget_s is not None and i > 0 and time.perf_counter() - t_lock > hold_budget_s:
                rest = list(jobs[i:])
                break
            kw, K = _kw(t, rule, seed)
            r, wall, info = _gpu_body(t, comps[t["iid"]], kw)
            outs.append((t, rule, seed, r, wall, kw, K, info))
    for t, rule, seed, r, wall, kw, K, info in outs:
        _write(t, rule, seed, r, wall, kw, K, info, tags, waited, group)
    if rest:
        print(f"[{rb.now_iso()}] hold budget {hold_budget_s:.0f} s reached after {len(outs)} runs; "
              f"{len(rest)} runs re-queued for a later hold", flush=True)
    return rest


def run_one(t, rule, seed, tags, comp):
    kw, K = _kw(t, rule, seed)
    with rb.gpu_lock(f"E7 {rule} {t['iid']} s{seed}") as waited:
        r, wall, info = _gpu_body(t, comp, kw)
    _write(t, rule, seed, r, wall, kw, K, info, tags, waited, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="topk3,eps100_28,topk1")
    ap.add_argument("--max-runs", type=int, default=10 ** 9)
    ap.add_argument("--max-n", type=int, default=28)
    ap.add_argument("--min-n", type=int, default=0)
    ap.add_argument("--max-pred-s", type=float, default=None, help="topk1: skip runs predicted above this (s)")
    ap.add_argument("--group-max-pred-s", type=float, default=600.0,
                    help="pack consecutive runs into one lock hold while their predicted total is below this (0: never)")
    ap.add_argument("--hold-budget-s", type=float, default=None,
                    help="optional: stop starting new runs once a lock hold has lasted this long and re-queue the rest "
                         "(bounds a hold at this budget + one run; default: off)")
    ap.add_argument("--dry-run", action="store_true", help="print the planned lock holds and exit (no GPU work)")
    a = ap.parse_args()
    tags = rb.hardware_tags()
    tasks = [t for t in e7.reference_tasks() if a.min_n <= t["n"] <= a.max_n]
    tasks.sort(key=lambda t: (t["n"], t["states_target"], t["k"], t["gen_seed"]))
    n_runs = 0
    comp_cache = {}
    all_jobs = []
    for phase in a.phase.split(","):
        if phase in ("topk3", "topk1"):
            jobs = [(t, phase, s) for t in tasks for s in rb.PAPER_SAMPLER_SEEDS]
        elif phase == "eps100_28":
            jobs = [(t, "eps100", s) for t in tasks if t["n"] == 28 for s in rb.PAPER_SAMPLER_SEEDS]
        elif phase == "eps100":
            jobs = [(t, "eps100", s) for t in tasks if t["n"] <= 24 for s in rb.PAPER_SAMPLER_SEEDS]
        elif phase == "eps20":
            jobs = [(t, "eps20", s) for t in tasks for s in rb.PAPER_SAMPLER_SEEDS]
        else:
            raise SystemExit(f"unknown phase {phase}")
        jobs = [j for j in jobs if not e7.raw_path("gpu", e7.run_name(j[1], j[0]["iid"], j[2])).exists()]
        if phase in ("topk1", "eps100"):
            jobs.sort(key=lambda j: _predicted_cost(j[0], j[1], j[2]))
        if phase == "topk1" and a.max_pred_s is not None:
            jobs = [j for j in jobs if _predicted_cost(j[0], j[1], j[2]) <= a.max_pred_s]
        print(f"[{rb.now_iso()}] phase {phase}: {len(jobs)} runs to do", flush=True)
        all_jobs.extend(jobs)


    units, cur, cur_s = [], [], 0.0
    for j in all_jobs:
        c = _predicted_cost(*j)
        if c > a.group_max_pred_s or a.group_max_pred_s <= 0:
            if cur:
                units.append(cur); cur, cur_s = [], 0.0
            units.append([j])
            continue
        if cur and cur_s + c > a.group_max_pred_s:
            units.append(cur); cur, cur_s = [], 0.0
        cur.append(j); cur_s += c
    if cur:
        units.append(cur)
    print(f"[{rb.now_iso()}] {len(all_jobs)} runs in {len(units)} lock holds"
          + ("" if a.hold_budget_s is None else f" (hold budget {a.hold_budget_s:.0f} s)"), flush=True)
    if a.dry_run:
        for u in units:
            print(f"  hold: {len(u)} runs, predicted {sum(_predicted_cost(*j) for j in u):.0f} s: "
                  + ", ".join(f"{j[1]}/{j[0]['iid'][8:30]}/s{j[2]}" for j in u), flush=True)
        return
    queue = list(units)
    while queue:
        unit = queue.pop(0)
        if n_runs >= a.max_runs:
            print("max-runs reached", flush=True)
            return
        rest = []
        try:
            comps = {}
            for t, _, _ in unit:
                if t["iid"] not in comps:
                    comps[t["iid"]] = comp_cache[t["iid"]] if t["iid"] in comp_cache else rb.build_task(t)[0]
            comp_cache.clear(); comp_cache.update({unit[-1][0]["iid"]: comps[unit[-1][0]["iid"]]})
            if len(unit) > 1:
                rest = run_group(unit, tags, comps, a.hold_budget_s) or []
            else:
                t, rule, seed = unit[0]
                run_one(t, rule, seed, tags, comps[t["iid"]])
        except Exception:
            print(f"ERROR {[(u[0]['iid'], u[1], u[2]) for u in unit]}\n{traceback.format_exc()}", flush=True)
        n_runs += len(unit) - len(rest)
        if rest:
            queue.insert(0, rest)
    print(f"[{rb.now_iso()}] ALL DONE ({n_runs} runs)", flush=True)


if __name__ == "__main__":
    main()
