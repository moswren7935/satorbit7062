#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import glob
import json
import math
import os
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e1_cs

OUT = rb.results_dir("E1")
RAW = OUT / "raw"
UNITS = RAW / "units"
RUNS = RAW / "interval_runs"
REPO = rb.REPO
ALPHA = 0.05
EXT_FACTOR = 4
EXT_FACTOR_54 = 2
OUTWARD = OUT / "outward_bound.csv"


def paper_runs():
    out = []
    for t in rb.paper_tasks():
        if t["phi_exact"] is None:
            continue
        for seed in rb.PAPER_SAMPLER_SEEDS:
            rec = t["cert_gpu"][seed]
            out.append(dict(family="paper126", run_id=f"paper__{t['iid']}__s{seed}", iid=t["iid"], n=t["n"],
                            states_target=t["states_target"], k=t["k"], gen_seed=t["gen_seed"], seed=seed,
                            backend="table" if t["n"] <= 24 else "gpu", record=rec["_file"]))
    return out


def paper54_runs():
    out = []
    for t in rb.paper_tasks():
        if t["phi_exact"] is not None:
            continue
        for seed in rb.PAPER_SAMPLER_SEEDS:
            rec = t["cert_gpu"][seed]
            out.append(dict(family="paper54", run_id=f"paper54__{t['iid']}__s{seed}", iid=t["iid"], n=t["n"],
                            states_target=t["states_target"], k=t["k"], gen_seed=t["gen_seed"], seed=seed,
                            backend="gpu", record=rec["_file"]))
    return out


def e11_runs():
    out = []
    for f in sorted(glob.glob(str(REPO / "results" / "e11" / "*__ours_cs__*.json"))):
        d = json.load(open(f))
        out.append(dict(family="e11_312", run_id=os.path.basename(f)[:-5], iid=d["instance_id"], n=int(d["n"]),
                        k=int(d["k"]), seed=int(d["seed"]), budget=int(d["budget"]), backend="table",
                        record=os.path.relpath(f, REPO), table_file=d["table_file"]))
    return out


def overcooked_runs():
    out = []
    for seed in (20260829, 20260830, 20260831):
        f = REPO / "results" / "e14" / f"e14__overcooked_cramped_room_k10__cert_gpu_eps_vs_exact_s{seed}.json"
        out.append(dict(family="overcooked3", run_id=f"overcooked_k10__s{seed}", iid="overcooked_cramped_room_k10",
                        n=2, k=10, seed=seed, backend="gpu", record=os.path.relpath(f, REPO)))
    return out


_TASKS = None


def task_of(iid):
    global _TASKS
    if _TASKS is None:
        _TASKS = {t["iid"]: t for t in rb.paper_tasks()}
    return _TASKS[iid]


def _result_fields(r):
    return dict(estimates=r.estimates, half_widths=r.half_widths, cs_lower=r.cs_lower, cs_upper=r.cs_upper,
                batch_sizes=np.asarray(r.batch_sizes, dtype=np.int64), n_samples=r.n_samples, n_perms=r.n_perms,
                n_solves=r.n_solves, n_requested=r.n_coalitions_requested, n_batches=r.n_batches,
                value_range=np.asarray(r.value_range, dtype=np.float64), fp_term=r.fp_term,
                eps_upsilon=r.eps_upsilon, eps=(np.nan if r.eps is None else r.eps), v_empty=r.v_empty,
                v_full=r.v_full, status=r.status, certified=r.certified, cs_anomaly=r.cs_anomaly)


def gpu_phase_paper(run, lock=True, ext_factor=EXT_FACTOR, comp=None):
    p = UNITS / f"{run['run_id']}.npz"
    if p.exists():
        return
    t = task_of(run["iid"])
    rec = t["cert_gpu"][run["seed"]]
    if comp is None:
        comp, _, _, _ = rb.build_task(t)
    n, k = t["n"], t["k"]
    with (rb.gpu_lock(f"E1 conservative {run['run_id']}") if lock else _null()):
        load = rb.load_snapshot()
        eng = rb.make_gpu_engine(comp)
        rb.paper_warmup(eng, n, k)
        reng = rb.RecordingEngine(eng)
        r, wall, _ = rb.run_paper_sampler(reng, n, k, rec["eps_abs"], run["seed"], record_units=False)
        m = r.n_samples
        ext = e1_cs.units_from_engine(eng, n, run["seed"], ext_factor * m, kind="agent", n=n, k=k)
        eng.close()
    rep = rb.replay_units(r, reng.calls, n, run["seed"], k)
    call_sizes = np.asarray([int(mk.size) for mk, _ in reng.calls], dtype=np.int64)
    if not np.array_equal(ext[:m], rep["units"]):
        raise RuntimeError("regenerated GPU unit stream differs from the run's units")
    np.savez(p, units=rep["units"], units_ext=ext, wall_s=wall, load=json.dumps(load), call_sizes=call_sizes,
             hardware=json.dumps(rb.hardware_tags()), git_commit=rb.git_commit(), gen_seed=run.get("gen_seed", -1),
             sampler_seed=run["seed"], ext_factor=ext_factor, **_result_fields(r))
    print(f"[gpu] {run['run_id']}: m={m} wall={wall:.3f}s", flush=True)


def gpu_phase_overcooked(run, lock=True):
    p = UNITS / f"{run['run_id']}.npz"
    if p.exists():
        return
    import e1_models
    from cubcr.engine_gpu import GPUEngine
    from cubcr.shapley_sampled import sample_shapley
    comp, meta = e1_models.overcooked_k10()
    rec = json.load(open(REPO / run["record"]))
    n, k = 2, 10
    with (rb.gpu_lock(f"E1 conservative {run['run_id']}") if lock else _null()):
        load = rb.load_snapshot()
        eng = GPUEngine(comp, dtype="fp64", use_graph=True)
        m_ends = np.zeros((k, 2), dtype=np.uint64)
        m_ends[:, 1] = 3
        vv = eng.solve(m_ends, k)
        ups = float(vv[1] - vv[0])
        reng = e1_cs.StepRecordingEngine(eng)
        t0 = time.perf_counter()

        r = sample_shapley(reng, kind="step", n=2, k=k, seed=run["seed"], batch_perms=16, alpha=0.05,
                           max_perms=2_000_000, time_budget=1800.0, antithetic=True, value_range="monotone",
                           ci="prpl_eb_cs", batch_schedule="geometric", eps=ups / 20)
        wall = time.perf_counter() - t0
        rep = e1_cs.replay_units_step(r, reng.calls, n, k, run["seed"])
        m = r.n_samples
        ext = e1_cs.units_from_engine(eng, n * k, run["seed"], EXT_FACTOR * m, kind="step", n=n, k=k)
        eng.close()
    if not np.array_equal(ext[:m], rep["units"]):
        raise RuntimeError("regenerated GPU step unit stream differs from the run's units")
    np.savez(p, units=rep["units"], units_ext=ext, wall_s=wall, ups_solve=ups, load=json.dumps(load),
             hardware=json.dumps(rb.hardware_tags()), **_result_fields(r))
    print(f"[gpu] {run['run_id']}: m={m} wall={wall:.3f}s ups={ups}", flush=True)


import contextlib


@contextlib.contextmanager
def _null():
    yield 0.0


def _lock_log(entry):
    RAW.mkdir(parents=True, exist_ok=True)
    with open(RAW / "gpu_lock_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")


def gpu_phase_per_task(runs, tasks_per_lock=6):


    todo = [r for r in runs if r["family"] == "paper54" and not (UNITS / f"{r['run_id']}.npz").exists()]
    by_task = {}
    for r in todo:
        by_task.setdefault(r["iid"], []).append(r)
    iids = list(by_task)
    for g0 in range(0, len(iids), max(1, int(tasks_per_lock))):
        group = iids[g0:g0 + max(1, int(tasks_per_lock))]
        tb = time.perf_counter()
        comps = {iid: rb.build_task(task_of(iid))[0] for iid in group}
        build_s = time.perf_counter() - tb
        n_runs = sum(len(by_task[i]) for i in group)
        t0 = time.perf_counter()
        with rb.gpu_lock(f"E1 conservative paper54 group {g0 // max(1, int(tasks_per_lock)) + 1} "
                         f"({len(group)} tasks, {n_runs} runs)"):
            t_lock = time.perf_counter()
            for iid in group:
                for r in by_task[iid]:
                    gpu_phase_paper(r, lock=False, ext_factor=EXT_FACTOR_54, comp=comps[iid])
            held = time.perf_counter() - t_lock
        _lock_log(dict(phase="intervals_paper54", tasks=group, n_runs=n_runs, build_s_outside_lock=build_s,
                       waited_s=t_lock - t0, held_s=held, time=rb.now_iso()))
        print(f"[gpu] paper54 group of {len(group)} tasks: {n_runs} re-runs, lock held {held:.1f}s "
              f"(waited {t_lock - t0:.1f}s; models built outside the lock in {build_s:.1f}s)", flush=True)
        del comps


def cpu_engine_phase_paper54(runs, threads=6, tail=128):


    from cubcr.engine_cpu import CPUEngine
    todo = [r for r in runs if r["family"] == "paper54" and not (UNITS / f"{r['run_id']}.npz").exists()]
    by_task = {}
    for r in todo:
        by_task.setdefault(r["iid"], []).append(r)
    for iid, rs in by_task.items():
        if all((UNITS / f"{r['run_id']}.npz").exists() for r in rs):
            continue
        t = task_of(iid)
        comp, _, _, _ = rb.build_task(t)
        with rb.host_lock(f"E1 paper54 CPU-engine re-runs {iid}", shared=True):
            eng = CPUEngine(comp, n_threads=threads, block=8, dtype="fp64")
            for r in rs:
                p = UNITS / f"{r['run_id']}.npz"
                if p.exists():
                    continue
                rec = t["cert_gpu"][r["seed"]]
                load = rb.load_snapshot()
                reng = rb.RecordingEngine(eng)
                res, wall, _ = rb.run_paper_sampler(reng, t["n"], t["k"], rec["eps_abs"], r["seed"], record_units=False)
                rep = rb.replay_units(res, reng.calls, t["n"], r["seed"], t["k"])
                m = res.n_samples
                same = (np.array_equal(res.estimates, np.asarray(rec["shapley_sampled"])) and
                        np.array_equal(res.half_widths, np.asarray(rec["half_widths"])) and
                        np.array_equal(res.cs_lower, np.asarray(rec["certificate"]["cs_lower"])) and
                        np.array_equal(res.cs_upper, np.asarray(rec["certificate"]["cs_upper"])) and
                        list(res.batch_sizes) == list(rec["batch_sizes"]) and int(rec["n_perms"]) == 2 * m)
                if not same:
                    raise RuntimeError(f"CPU-engine re-run of {r['run_id']} does not reproduce the stored GPU record")
                tail_u = e1_cs.units_from_engine_tail(eng, t["n"], r["seed"], m, tail, kind="agent", n=t["n"], k=t["k"])
                ext = np.concatenate([rep["units"], tail_u], axis=0)
                call_sizes = np.asarray([int(mk.size) for mk, _ in reng.calls], dtype=np.int64)
                tmp = UNITS / f"{r['run_id']}.tmp.npz"
                np.savez(tmp, units=rep["units"], units_ext=ext, wall_s=wall, load=json.dumps(load),
                         call_sizes=call_sizes, hardware=json.dumps(rb.hardware_tags()), git_commit=rb.git_commit(),
                         gen_seed=run_gen_seed(r), sampler_seed=r["seed"], ext_factor=-1, ext_tail=tail,
                         engine="cpu", cpu_threads=threads, timing_note="not a timing measurement",
                         **_result_fields(res))
                if p.exists():
                    tmp.unlink()
                    continue
                os.replace(tmp, p)
                print(f"[cpu-engine] {r['run_id']}: m={m} (reproduces the stored GPU record bitwise) "
                      f"wall={wall:.1f}s (not a timing measurement)", flush=True)
            eng.close()


def run_gen_seed(r):
    return int(r.get("gen_seed") or -1)


def gpu_phase_batched(runs):


    todo = [r for r in runs if r["backend"] == "gpu" and r["family"] != "paper54"
            and not (UNITS / f"{r['run_id']}.npz").exists()]
    if not todo:
        return
    for r in todo:
        if r["family"] == "paper126":
            rb.build_task(task_of(r["iid"]))
        else:
            import e1_models
            e1_models.overcooked_k10()
    t0 = time.perf_counter()
    with rb.gpu_lock(f"E1 conservative intervals: {len(todo)} GPU re-runs"):
        t_lock = time.perf_counter()
        for r in todo:
            (gpu_phase_paper if r["family"] == "paper126" else gpu_phase_overcooked)(r, lock=False)
        held = time.perf_counter() - t_lock
    _lock_log(dict(phase="intervals_batched", n_runs=len(todo), waited_s=t_lock - t0, held_s=held, time=rb.now_iso()))
    print(f"[gpu] {len(todo)} re-runs under one lock: held {held:.1f}s (waited {t_lock - t0:.1f}s)", flush=True)


def _load_units_npz(run):
    z = np.load(UNITS / f"{run['run_id']}.npz", allow_pickle=False)
    return {kk: z[kk] for kk in z.files}


def _table_run_paper(run):
    t = task_of(run["iid"])
    rec = t["cert_gpu"][run["seed"]]
    comp, _, _, _ = rb.build_task(t)
    table = rb.exact_table(t)
    eng = rb.RecordingEngine(rb.TableEngine(table, compiled=comp))
    r, wall, _ = rb.run_paper_sampler(eng, t["n"], t["k"], rec["eps_abs"], run["seed"], record_units=False)
    rep = rb.replay_units(r, eng.calls, t["n"], run["seed"], t["k"])
    call_sizes = [int(mk.size) for mk, _ in eng.calls]
    _table_run_paper.last_wall = wall
    return r, rep["units"], table, comp, call_sizes


_OUTW = None


def outward_eps_x(iid):


    global _OUTW
    if _OUTW is None:
        import pandas as pd
        df = pd.read_csv(OUTWARD)
        df = df[df["status"] == "OK"]
        _OUTW = {r.model_id: (float(r.eps_x_code), float(r.eps_x_out)) for r in df.itertuples()}
    if iid not in _OUTW:
        raise RuntimeError(f"no outward bound for model {iid}; run e1_outward_bound.py first")
    return _OUTW[iid]


def _table_run_e11(run):
    from cubcr.shapley_sampled import sample_shapley
    table = np.load(REPO / run["table_file"])
    n = run["n"]
    perms = max(2, run["budget"] // (n + 1))
    eng = rb.RecordingEngine(rb.TableEngine(table, compiled=None))

    t0 = time.perf_counter()
    r = sample_shapley(eng, n, kind="agent", seed=run["seed"], eps=None, alpha=0.05, value_range="monotone",
                       max_perms=perms, batch_perms=8, two_sided=True, bonferroni=True, sequential=True,
                       ci="prpl_eb_cs", batch_schedule="geometric", stratify=False)
    _table_run_e11.last_wall = time.perf_counter() - t0
    _table_run_e11.last_calls = [int(mk.size) for mk, _ in eng.calls]
    rep = rb.replay_units(r, eng.calls, n, run["seed"], None)
    distinct = len(set(int(x) for masks, _ in eng.calls for x in masks.tolist()))
    return r, rep["units"], table, distinct


STORED_FIELDS = {
    "paper126": "estimates;half_widths;cs_lower;cs_upper;batch_sizes;n_solves;n_perms",
    "paper54": "estimates;half_widths;cs_lower;cs_upper;batch_sizes;n_solves;n_perms",
    "overcooked3": "estimates;half_widths;n_solves;n_perms (CS bounds not stored)",
    "e11_312": "estimates;hw_max;n_perms;n_requested;n_distinct (per-player half-widths and CS bounds not stored)",
}


def recompute(args):
    run, beta_reb, prov = args
    try:
        os.nice(19)
    except Exception:
        pass
    t_start = time.perf_counter()
    row = dict(family=run["family"], run_id=run["run_id"], iid=run["iid"], n=run["n"], k=run["k"],
               seed=run["seed"], backend=run["backend"], budget=run.get("budget"), gen_seed=run.get("gen_seed"),
               states_target=run.get("states_target"), record=run["record"])
    players = []
    try:
        rec = json.load(open(REPO / run["record"]))
        fam = run["family"]
        stored_hw = stored_csL = stored_csU = None
        if fam in ("paper126", "paper54"):
            N = run["n"]
            if run["backend"] == "table":
                r, units, table, comp, call_sizes = _table_run_paper(run)
                row["engine_call_sizes"] = json.dumps(call_sizes)
                row["wall_s_rerun_table_host"] = float(_table_run_paper.last_wall)
                est, hw, csL, csU = r.estimates, r.half_widths, r.cs_lower, r.cs_upper
                bs = list(r.batch_sizes); m = r.n_samples; lo, hi = r.value_range; eps_x = r.fp_term
                eps = r.eps; ups = r.v_full - r.v_empty; n_solves, n_req = r.n_solves, r.n_coalitions_requested
                provider = _table_provider(table, N, run["seed"], "agent", N, run["k"], units)
                UNITS.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(UNITS / f"{run['run_id']}.npz", units=units, batch_sizes=np.asarray(bs))
            else:
                z = _load_units_npz(run)
                units = z["units"]; est, hw, csL, csU = z["estimates"], z["half_widths"], z["cs_lower"], z["cs_upper"]
                bs = z["batch_sizes"].tolist(); m = int(z["n_samples"]); lo, hi = z["value_range"].tolist()
                eps_x = float(z["fp_term"]); eps = float(z["eps"]); ups = float(z["v_full"] - z["v_empty"])
                n_solves, n_req = int(z["n_solves"]), int(z["n_requested"])
                ext = z["units_ext"]
                provider = _array_provider(ext)
                row["engine_call_sizes"] = json.dumps(z["call_sizes"].tolist())
                eng_kind = str(z["engine"]) if "engine" in z else "gpu"
                row["rerun_engine"] = eng_kind
                if eng_kind == "gpu":
                    row["wall_s_rerun_gpu"] = float(z["wall_s"])
                else:
                    row["wall_s_rerun_cpu_engine_not_timing"] = float(z["wall_s"])
                row["units_pregenerated"] = int(ext.shape[0])
            phi = np.asarray(task_of(run["iid"])["phi_exact"]) if fam == "paper126" else None
            stored_hw = np.asarray(rec["half_widths"])
            stored_csL = np.asarray(rec["certificate"]["cs_lower"])
            stored_csU = np.asarray(rec["certificate"]["cs_upper"])
            stored_ok = (np.array_equal(est, np.asarray(rec["shapley_sampled"])) and
                         np.array_equal(hw, stored_hw) and
                         np.array_equal(csL, stored_csL) and np.array_equal(csU, stored_csU) and
                         bs == list(rec["batch_sizes"]) and n_solves == rec["n_solves"] and
                         int(rec["n_perms"]) == 2 * m)
            row.update(stored_covers_exact=(bool(rec["covers_exact"]) if fam == "paper126" else None),
                       wall_s_stored=float(rec["wall_s"]), eps_div=20, batch_perms=32)
            model_id = run["iid"]
        elif fam == "e11_312":
            N = run["n"]
            r, units, table, distinct = _table_run_e11(run)
            est, hw, csL, csU = r.estimates, r.half_widths, r.cs_lower, r.cs_upper
            bs = list(r.batch_sizes); m = r.n_samples; lo, hi = r.value_range; eps_x = r.fp_term
            eps = None; ups = r.v_full - r.v_empty; n_solves, n_req = r.n_solves, r.n_coalitions_requested
            provider = _table_provider(table, N, run["seed"], "agent", N, None, units)
            phi = np.asarray(rec["shapley_exact"])
            stored_ok = (np.array_equal(est, np.asarray(rec["estimates"])) and float(np.max(hw)) == rec["hw_max"]
                         and r.n_perms == rec["n_perms"] and n_solves == rec["n_requested"]
                         and distinct == rec["n_distinct"])
            row.update(stored_covers_exact=bool(rec["covers_exact"]), wall_s_stored=float(rec["wall_s"]),
                       eps_div=None, batch_perms=8, gen_seed=20260829,
                       wall_s_rerun_table_host=float(_table_run_e11.last_wall),
                       engine_call_sizes=json.dumps(_table_run_e11.last_calls))
            model_id = run["iid"]
        else:
            N = 20
            z = _load_units_npz(run)
            units = z["units"]; est, hw, csL, csU = z["estimates"], z["half_widths"], z["cs_lower"], z["cs_upper"]
            bs = z["batch_sizes"].tolist(); m = int(z["n_samples"]); lo, hi = z["value_range"].tolist()
            eps_x = float(z["fp_term"]); eps = float(z["eps"]); ups = float(z["v_full"] - z["v_empty"])
            n_solves, n_req = int(z["n_solves"]), int(z["n_requested"])
            provider = _array_provider(z["units_ext"])
            row["units_pregenerated"] = int(z["units_ext"].shape[0])
            ex = json.load(open(REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__exact_step.json"))
            phi = np.asarray(ex["shapley"], dtype=np.float64).reshape(-1)
            stored_hw = np.asarray(rec["half_widths"]).reshape(-1)
            stored_ok = (np.array_equal(est, np.asarray(rec["estimates"]).reshape(-1)) and
                         np.array_equal(hw, stored_hw) and
                         n_solves == rec["n_solves"] and int(rec["n_perms"]) == 2 * m)
            row.update(stored_covers_exact=bool(rec["covers_exact"]), wall_s_stored=float(rec["wall_s"]),
                       eps_div=20, batch_perms=16, wall_s_rerun_gpu=float(z["wall_s"]), gen_seed=None,
                       model_note="Overcooked k=10 built as experiments/e14_overcooked.py::run_exact_k10 (no generator seed)")
            model_id = "overcooked_cramped_room_k10"
        eps_x_code_model, eps_x_out_model = outward_eps_x(model_id)
        beta_b2 = max(float(eps_x), eps_x_code_model, eps_x_out_model, float(beta_reb))
        row.update(eps_x_code_model=eps_x_code_model, eps_x_out_model=eps_x_out_model,
                   beta_analytic_model=max(eps_x_code_model, eps_x_out_model), has_reference=phi is not None,
                   stored_fields_compared=STORED_FIELDS[fam])
        row.update(d=N, m_stop_a=m, n_batches_a=len(bs), batch_sizes_a=json.dumps(bs), n_solves=n_solves,
                   n_requested=n_req, upsilon_fp=ups, eps_a=eps, beta_a=eps_x, a_a=lo, b_a=hi,
                   U_a=hi - eps_x, reproduces_stored_bitwise=bool(stored_ok))

        looks_a, _ = e1_cs.fp64_replay(units, bs, lo, hi, eps_x, ALPHA, eps=eps)
        La = looks_a[-1]
        replay_ok = (np.array_equal(La.hw, hw) and np.array_equal(La.cs_L, csL) and np.array_equal(La.cs_U, csU)
                     and np.array_equal(La.est, est))
        stops = [lk.stop for lk in looks_a]
        stop_ok = (eps is None) or (stops[-1] and not any(stops[:-1]))
        row.update(replay_a_bitwise=bool(replay_ok), replay_a_equals_rerun_bitwise=bool(replay_ok),
                   replay_a_equals_stored_hw=(None if stored_hw is None else bool(np.array_equal(La.hw, stored_hw))),
                   replay_a_equals_stored_cs=(None if stored_csL is None else
                                              bool(np.array_equal(La.cs_L, stored_csL) and
                                                   np.array_equal(La.cs_U, stored_csU))),
                   replay_a_equals_stored_hwmax=(bool(float(np.max(La.hw)) == rec["hw_max"]) if fam == "e11_312" else None),
                   stop_rule_a_consistent=bool(stop_ok), hwmax_a=float(np.max(hw)))
        if phi is not None:
            ok_hw, mar_hw = e1_cs.contained(phi, est - hw, est + hw)
            ok_I, mar_I = e1_cs.contained(phi, La.lower, La.upper)
            ok_tol, _ = e1_cs.contained(phi, est - hw, est + hw, tol=1e-12)
            row.update(contained_a_hw=bool(ok_hw.all()), contained_a_I=bool(ok_I.all()),
                       contained_a_hw_tol1e12=bool(ok_tol.all()),
                       relies_on_1e12_slack=bool(ok_tol.all() and not ok_hw.all()),
                       min_margin_a_I=float(mar_I.min()),
                       contained_a_all_looks=bool(all(e1_cs.contained(phi, lk.lower, lk.upper)[0].all()
                                                      for lk in looks_a)))
        phi_col = phi if phi is not None else np.full(N, np.nan)
        player_cols = {"phi_exact": phi_col, "phi_hat": est, "L_a": La.lower, "R_a": La.upper, "H_a": hw}

        bp = row["batch_perms"]
        for tag, beta, xmode in (("b1", str(beta_reb), "strict"), ("b2", beta_b2, "strict"),
                                 ("b1p", str(beta_reb), "paper")):
            prm = e1_cs.make_params(ups, beta, row["eps_div"])
            out = e1_cs.conservative_replay(provider, bs, N, prm, lo, hi, ALPHA, bp, 16 * bp, xhat_mode=xmode)
            lk_stop = out["looks"][out["stop_index"]] if out["stop_index"] is not None else out["looks"][-1]
            lk_orig = out["looks"][len(bs) - 1]
            m_b = out["m_stop"]
            row.update({f"beta_{tag}": prm.beta, f"a_{tag}": prm.a, f"b_{tag}": prm.b, f"U_{tag}": prm.b - prm.beta,
                        f"eps_{tag}": prm.eps, f"m_stop_{tag}": m_b, f"stopped_{tag}": out["stopped"],
                        f"continued_{tag}": out["continued"], f"n_batches_{tag}": out["n_batches"],
                        f"batch_sizes_{tag}": json.dumps(out["batch_sizes"]), f"xhat_mode_{tag}": out["xhat_mode"],
                        f"xhat_n_differ_{tag}": out["xhat_n_differ"], f"xhat_n_total_{tag}": out["xhat_n_total"],
                        f"xhat_max_abs_diff_{tag}": out["xhat_max_abs_diff"],
                        f"delta_m_{tag}": (None if (m_b is None or eps is None) else int(m_b - m)),
                        f"delta_m_pct_{tag}": (None if (m_b is None or eps is None) else 100.0 * (m_b - m) / m),
                        f"Hmax_{tag}_at_stop": float(np.max(lk_stop.hw)),
                        f"Hmax_{tag}_at_orig_stop": float(np.max(lk_orig.hw)),
                        f"max_abs_H_{tag}_minus_H_a_at_orig_stop": float(np.max(np.abs(lk_orig.hw - hw))),
                        f"max_H_{tag}_minus_H_a_at_orig_stop": float(np.max(lk_orig.hw - hw)),
                        f"min_H_{tag}_minus_H_a_at_orig_stop": float(np.min(lk_orig.hw - hw)),
                        f"implied_runtime_change_s_{tag}": (None if (m_b is None or eps is None)
                                                            else (m_b - m) * row["wall_s_stored"] / m)})
            if phi is not None:
                okI, marI = e1_cs.contained(phi, lk_stop.lower, lk_stop.upper)
                okH, _ = e1_cs.contained(phi, lk_stop.est - lk_stop.hw, lk_stop.est + lk_stop.hw)
                row.update({f"contained_{tag}_I": bool(okI.all()), f"contained_{tag}_hw": bool(okH.all()),
                            f"min_margin_{tag}_I": float(marI.min()),
                            f"contained_{tag}_all_looks": bool(all(e1_cs.contained(phi, lk.lower, lk.upper)[0].all()
                                                                  for lk in out["looks"]))})
            player_cols[f"L_{tag}"] = lk_stop.lower
            player_cols[f"R_{tag}"] = lk_stop.upper
            player_cols[f"H_{tag}"] = lk_stop.hw
            if tag == "b1p":

                row["max_abs_L_b1_minus_L_b1p"] = float(np.max(np.abs(player_cols["L_b1"] - lk_stop.lower)))
                row["max_abs_R_b1_minus_R_b1p"] = float(np.max(np.abs(player_cols["R_b1"] - lk_stop.upper)))
                row["max_abs_H_b1_minus_H_b1p"] = float(np.max(np.abs(player_cols["H_b1"] - lk_stop.hw)))
                row["same_stop_b1_b1p"] = bool(row["m_stop_b1"] == m_b)
        for p in range(N):
            players.append(dict(run_id=run["run_id"], family=run["family"], player=p,
                                **{kk: float(v[p]) for kk, v in player_cols.items()}))
        row["status"] = "OK"
    except Exception as e:
        row["status"] = "ERROR"
        row["error"] = repr(e)[:400]
        row["traceback"] = traceback.format_exc()[-3000:]
    row["recompute_s"] = time.perf_counter() - t_start
    RUNS.mkdir(parents=True, exist_ok=True)
    cfg = {"alpha": ALPHA, "beta_rebuttal": beta_reb,
           "modes": {"a": "FP64 replay (paper)", "b1": "outward, beta_rebuttal, strictly predictable x_hat",
                     "b2": "outward, max(eps_X used, eps_X code, eps_X outward, beta_rebuttal), strictly predictable x_hat",
                     "b1p": "outward, beta_rebuttal, the sampler's FP64 x_hat (1-ulp dependence on the current unit)"},
           "iv_dps": 40, "U": "Upsilon~ + 2 beta", "eps": "(Upsilon~ - 2 beta)/eps_div", "version": 2}
    rb.write_json(RUNS / f"{run['run_id']}.json", {"row": row, "players": players, "config": cfg,
                                                     "provenance": dict(prov, time=rb.now_iso())})
    return row


class _Prov:
    def __init__(self, fn):
        self.fn = fn
        self.cache = None

    def __call__(self, m):
        if self.cache is None or self.cache.shape[0] < m:
            self.cache = self.fn(m)
        return self.cache[:m]


def _table_provider(table, N, seed, kind, n, k, units_logged):
    def gen(m):
        if m <= units_logged.shape[0]:
            return units_logged[:m]
        u = e1_cs.units_from_engine(rb.TableEngine(table), N, seed, max(m, 2 * units_logged.shape[0]),
                                    kind=kind, n=n, k=k)
        if not np.array_equal(u[:units_logged.shape[0]], units_logged):
            raise RuntimeError("regenerated table unit stream differs from the logged units")
        return u
    return _Prov(gen)


def _array_provider(ext):
    def gen(m):
        if m > ext.shape[0]:
            raise RuntimeError(f"continuation needs {m} units > pre-generated {ext.shape[0]} (GPU)")
        return ext[:m]
    return gen


def cpu_gpu_interval_identity():


    g = {(r["instance_id"], r["seed"]): r for r in rb.load_e9v3(arms=["cert_gpu"], paper_only=True)}
    c = {(r["instance_id"], r["seed"]): r for r in rb.load_e9v3(arms=["cert_cpu"], paper_only=True)}
    keys = sorted(set(g) & set(c))
    same = 0
    for kk in keys:
        a, b = g[kk], c[kk]
        same += int(np.array_equal(a["shapley_sampled"], b["shapley_sampled"]) and
                    np.array_equal(a["half_widths"], b["half_widths"]) and
                    np.array_equal(a["certificate"]["cs_lower"], b["certificate"]["cs_lower"]) and
                    np.array_equal(a["certificate"]["cs_upper"], b["certificate"]["cs_upper"]) and
                    list(a["batch_sizes"]) == list(b["batch_sizes"]))
    return {"n_pairs": len(keys), "n_identical": same, "n_gpu": len(g), "n_cpu": len(c),
            "fields": "shapley_sampled, half_widths, certificate.cs_lower/cs_upper, batch_sizes",
            "source": "results/e9v3 cert_gpu vs cert_cpu records (paper seeds, rb.load_e9v3)"}


REF_FAMS = ("paper126", "e11_312", "overcooked3")
EPS_FAMS = ("paper126", "paper54", "overcooked3")


def _b(series):
    return series.astype(object).map(lambda v: bool(v) if v is not None and v == v else False)


def summarize():
    import pandas as pd
    rows, players = [], []
    for f in sorted(glob.glob(str(RUNS / "*.json"))):
        d = json.load(open(f))
        if d.get("config", {}).get("version") != 2:
            continue
        rows.append(d["row"])
        players.extend(d["players"])
    df = pd.DataFrame(rows)
    order = ["family", "run_id", "status", "has_reference", "stored_fields_compared", "reproduces_stored_bitwise",
             "replay_a_bitwise", "replay_a_equals_stored_hw", "replay_a_equals_stored_cs", "replay_a_equals_stored_hwmax",
             "stop_rule_a_consistent", "d", "m_stop_a", "m_stop_b1", "m_stop_b2", "m_stop_b1p", "delta_m_pct_b1",
             "delta_m_pct_b2", "contained_a_hw", "contained_a_I", "contained_b1_I", "contained_b2_I", "contained_b1p_I",
             "relies_on_1e12_slack", "eps_a", "eps_b1", "eps_b2", "beta_a", "beta_b1", "beta_b2", "U_a", "U_b1", "U_b2",
             "hwmax_a", "Hmax_b1_at_stop", "Hmax_b2_at_stop"]
    df = df[[c for c in order if c in df.columns] + [c for c in df.columns if c not in order]]
    df.to_csv(OUT / "interval_recompute.csv", index=False)
    pd.DataFrame(players).to_csv(RAW / "interval_players.csv.gz", index=False, compression="gzip")
    ok = df[df["status"] == "OK"]
    summ = {"n_runs": int(len(df)), "n_ok": int(len(ok)), "n_error": int((df["status"] != "OK").sum()),
            "version": 2}
    for fam, g in ok.groupby("family"):
        e = {"n_runs": int(len(g)), "stored_fields_compared": sorted(set(g["stored_fields_compared"]))[0],
             "reproduce_stored_bitwise": int(_b(g["reproduces_stored_bitwise"]).sum()),
             "replay_a_equals_rerun_bitwise": int(_b(g["replay_a_bitwise"]).sum())}
        if g["replay_a_equals_stored_hw"].notna().any():
            e["replay_a_equals_stored_half_widths"] = int(_b(g["replay_a_equals_stored_hw"]).sum())
        if g["replay_a_equals_stored_cs"].notna().any():
            e["replay_a_equals_stored_cs_bounds"] = int(_b(g["replay_a_equals_stored_cs"]).sum())
        if g["replay_a_equals_stored_hwmax"].notna().any():
            e["replay_a_equals_stored_hw_max"] = int(_b(g["replay_a_equals_stored_hwmax"]).sum())
        if fam in REF_FAMS:
            for c in ("contained_a_hw", "contained_a_I", "relies_on_1e12_slack", "contained_b1_I", "contained_b2_I",
                      "contained_b1p_I", "contained_b1_all_looks", "contained_b2_all_looks", "contained_b1p_all_looks"):
                e[c] = int(_b(g[c]).sum())
            e["stored_covers_exact"] = int(_b(g["stored_covers_exact"]).sum())
            for tag in ("a", "b1", "b2", "b1p"):
                e[f"min_margin_{tag}_I"] = float(g[f"min_margin_{tag}_I"].min())
        if fam in EPS_FAMS:
            for tag in ("b1", "b2", "b1p"):
                dm = g[f"delta_m_{tag}"].astype(float)
                e[f"n_m_stop_changed_{tag}"] = int((dm != 0).sum())
                e[f"max_abs_delta_m_pct_{tag}"] = float(np.nanmax(np.abs(g[f"delta_m_pct_{tag}"].astype(float))))
                e[f"n_continued_{tag}"] = int(_b(g[f"continued_{tag}"]).sum())
                e[f"sum_implied_runtime_change_s_{tag}"] = float(np.nansum(g[f"implied_runtime_change_s_{tag}"].astype(float)))
        for tag in ("b1", "b2", "b1p"):
            e[f"max_abs_H_minus_H_a_{tag}"] = float(np.nanmax(g[f"max_abs_H_{tag}_minus_H_a_at_orig_stop"].astype(float)))
            e[f"max_H_minus_H_a_{tag}"] = float(np.nanmax(g[f"max_H_{tag}_minus_H_a_at_orig_stop"].astype(float)))
            e[f"min_H_minus_H_a_{tag}"] = float(np.nanmin(g[f"min_H_{tag}_minus_H_a_at_orig_stop"].astype(float)))
        e["xhat_n_differ"] = int(g["xhat_n_differ_b1"].sum())
        e["xhat_n_total"] = int(g["xhat_n_total_b1"].sum())
        e["xhat_runs_with_difference"] = int((g["xhat_n_differ_b1"] > 0).sum())
        e["xhat_max_abs_diff"] = float(g["xhat_max_abs_diff_b1"].max())
        e["max_abs_L_b1_minus_L_b1p"] = float(g["max_abs_L_b1_minus_L_b1p"].max())
        e["max_abs_R_b1_minus_R_b1p"] = float(g["max_abs_R_b1_minus_R_b1p"].max())
        e["max_abs_H_b1_minus_H_b1p"] = float(g["max_abs_H_b1_minus_H_b1p"].max())
        e["n_same_stop_b1_b1p"] = int(_b(g["same_stop_b1_b1p"]).sum())
        summ[fam] = e
    ref = ok[ok["family"].isin(REF_FAMS)]
    summ["all_reference_checks"] = {
        "families": list(REF_FAMS), "n": int(len(ref)),
        "contained_a_hw": int(_b(ref["contained_a_hw"]).sum()), "contained_a_I": int(_b(ref["contained_a_I"]).sum()),
        "contained_b1_I": int(_b(ref["contained_b1_I"]).sum()), "contained_b2_I": int(_b(ref["contained_b2_I"]).sum()),
        "contained_b1p_I": int(_b(ref["contained_b1p_I"]).sum()),
        "relies_on_1e12_slack": int(_b(ref["relies_on_1e12_slack"]).sum()),
        "min_margin_b1_I": float(ref["min_margin_b1_I"].min()), "min_margin_b2_I": float(ref["min_margin_b2_I"].min())}
    eps_runs = ok[ok["family"].isin(EPS_FAMS)]
    summ["all_eps_stopped"] = {

        "families": [f for f in EPS_FAMS if f in set(eps_runs["family"])], "n": int(len(eps_runs)),
        "n_per_family": {f: int((eps_runs["family"] == f).sum()) for f in EPS_FAMS if f in set(eps_runs["family"])},
        **{f"n_m_stop_changed_{t}": int((eps_runs[f"delta_m_{t}"].astype(float) != 0).sum()) for t in ("b1", "b2", "b1p")},
        **{f"max_abs_delta_m_pct_{t}": float(np.nanmax(np.abs(eps_runs[f"delta_m_pct_{t}"].astype(float))))
           for t in ("b1", "b2", "b1p")},
        **{f"n_continued_{t}": int(_b(eps_runs[f"continued_{t}"]).sum()) for t in ("b1", "b2", "b1p")},
        **{f"max_abs_H_minus_H_a_{t}": float(eps_runs[f"max_abs_H_{t}_minus_H_a_at_orig_stop"].max()) for t in ("b1", "b2", "b1p")},
        **{f"max_H_minus_H_a_{t}": float(eps_runs[f"max_H_{t}_minus_H_a_at_orig_stop"].max()) for t in ("b1", "b2", "b1p")},
        **{f"min_H_minus_H_a_{t}": float(eps_runs[f"min_H_{t}_minus_H_a_at_orig_stop"].min()) for t in ("b1", "b2", "b1p")},
        "n_replay_a_equals_stored_cs_bounds": int(_b(eps_runs["replay_a_equals_stored_cs"]).sum()),
        "n_reproduce_stored_bitwise": int(_b(eps_runs["reproduces_stored_bitwise"]).sum())}
    summ["xhat_all"] = {"n_runs": int(len(ok)), "n_runs_with_difference": int((ok["xhat_n_differ_b1"] > 0).sum()),
                        "n_differ": int(ok["xhat_n_differ_b1"].sum()), "n_total": int(ok["xhat_n_total_b1"].sum()),
                        "max_abs_diff": float(ok["xhat_max_abs_diff_b1"].max()),
                        "max_abs_bound_change_b1_vs_b1p": float(max(ok["max_abs_L_b1_minus_L_b1p"].max(),
                                                                   ok["max_abs_R_b1_minus_R_b1p"].max())),
                        "max_abs_H_change_b1_vs_b1p": float(ok["max_abs_H_b1_minus_H_b1p"].max()),
                        "n_same_stop_b1_b1p": int(_b(ok["same_stop_b1_b1p"]).sum())}
    summ["beta_b2_minus_beta_a_max"] = float((ok["beta_b2"] - ok["beta_a"]).max())
    summ["n_runs_b2_beta_raised_by_outward_bound"] = int((ok["eps_x_out_model"] > np.maximum(ok["beta_a"], ok["eps_x_code_model"])).sum())
    summ["cert_cpu_vs_cert_gpu_stored"] = cpu_gpu_interval_identity()
    summ["git_commit"] = rb.git_commit()
    summ["time"] = rb.now_iso()
    rb.write_json(OUT / "interval_recompute_summary.json", summ)
    print(json.dumps(summ, indent=1))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["gpu", "cpuengine54", "cpu", "summary", "all"])
    ap.add_argument("--cpu-threads", type=int, default=6, help="cpuengine54 phase: CPU-engine threads (<= 6)")
    ap.add_argument("--families", default="paper126,paper54,e11_312,overcooked3")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--beta-rebuttal", type=float, default=None,
                    help="default: beta_rebuttal_brief_rule from precision_audit_summary.json")
    ap.add_argument("--only", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--tasks-per-lock", type=int, default=6, help="paper54 GPU phase: tasks per lock acquisition")
    a = ap.parse_args(argv)
    fams = a.families.split(",")
    runs = []
    if "paper126" in fams:
        runs += paper_runs()
    if "paper54" in fams:
        runs += paper54_runs()
    if "e11_312" in fams:
        runs += e11_runs()
    if "overcooked3" in fams:
        runs += overcooked_runs()
    if a.only:
        runs = [r for r in runs if a.only in r["run_id"]]
    UNITS.mkdir(parents=True, exist_ok=True)
    if a.phase in ("gpu", "all"):
        gpu_phase_batched(runs)
        gpu_phase_per_task(runs, tasks_per_lock=a.tasks_per_lock)
    if a.phase == "cpuengine54":
        try:
            os.nice(19)
        except Exception:
            pass
        cpu_engine_phase_paper54(runs, threads=min(int(a.cpu_threads), 6))
    if a.phase in ("cpu", "all"):
        beta = a.beta_rebuttal
        if beta is None:
            beta = json.load(open(OUT / "precision_audit_summary.json"))["beta_rebuttal_brief_rule"]
        def _done(r):
            p = RUNS / f"{r['run_id']}.json"
            if not p.exists():
                return False
            d = json.load(open(p))
            return d.get("config", {}).get("version") == 2 and d["row"].get("status") == "OK"
        todo = [r for r in runs if a.force or not _done(r)]

        missing = [r for r in todo if r["backend"] == "gpu" and not (UNITS / f"{r['run_id']}.npz").exists()]
        if missing:
            print(f"[cpu] skipping {len(missing)} GPU-backed runs without a units file (not re-run yet)", flush=True)
            todo = [r for r in todo if r not in missing]

        todo.sort(key=lambda r: (r.get("n", 0) >= 24 and (r.get("states_target") or 0) >= 10000, r["run_id"]))
        print(f"[cpu] beta_rebuttal = {beta:g}; {len(todo)} runs", flush=True)
        prov = {"git_commit": rb.git_commit(), "hardware": rb.hardware_tags(), "load_at_start": rb.load_snapshot()}
        with Pool(min(a.workers, 6)) as pool:
            for row in pool.imap_unordered(recompute, [(r, beta, prov) for r in todo], chunksize=1):
                if row["status"] != "OK":
                    print(f"[cpu] ERROR {row['run_id']}: {row.get('error')}", flush=True)
    if a.phase in ("summary", "all"):
        summarize()
    return 0


if __name__ == "__main__":
    sys.exit(main())
