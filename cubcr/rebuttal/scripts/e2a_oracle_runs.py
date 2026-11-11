#!/usr/bin/env python3


from __future__ import annotations

import argparse
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[_v] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import numpy as np

import rb
import e2a_lib as L


def plan_for(divs, seedsets):
    plan = []
    for div in divs:
        for ss in seedsets:
            for s in (L.PAPER_SEEDS if ss == "paper" else L.NEW_SEEDS):
                if (div, s) not in plan:
                    plan.append((div, s))
    return plan


def _check_tb(r, rec, eps):
    c = {"stored_file": rec["_file"], "eps_equal": float(eps) == float(rec["eps_abs"]),
         "estimates_bitwise": bool(np.array_equal(r.estimates, np.asarray(rec["shapley_sampled"]))),
         "half_widths_bitwise": bool(np.array_equal(r.half_widths, np.asarray(rec["half_widths"]))),
         "cs_lower_bitwise": bool(np.array_equal(r.cs_lower, np.asarray(rec["certificate"]["cs_lower"]))),
         "cs_upper_bitwise": bool(np.array_equal(r.cs_upper, np.asarray(rec["certificate"]["cs_upper"]))),
         "n_perms_equal": int(r.n_perms) == int(rec["n_perms"]), "n_solves_equal": int(r.n_solves) == int(rec["n_solves"]),
         "batch_sizes_equal": list(r.batch_sizes) == list(rec["batch_sizes"])}
    c["all_bitwise"] = all(v for kk, v in c.items() if kk != "stored_file")
    return c


def _check_oc(r, rec, eps):
    c = {"stored_file": rec["_file"], "eps_equal": float(eps) == float(rec["eps_abs"]),
         "estimates_bitwise": bool(np.array_equal(r.estimates, np.asarray(rec["estimates"]).reshape(-1))),
         "half_widths_bitwise": bool(np.array_equal(r.half_widths, np.asarray(rec["half_widths"]).reshape(-1))),
         "n_perms_equal": int(r.n_perms) == int(rec["n_perms"]), "n_solves_equal": int(r.n_solves) == int(rec["n_solves"]),
         "note": "stored E14 records carry no cs_lower/cs_upper/batch_sizes; those fields are taken from this re-run"}
    c["all_bitwise"] = all(v for kk, v in c.items() if kk not in ("stored_file", "note"))
    return c


def job(task, plan, tags):

    try:
        os.nice(19)
    except Exception:
        pass
    from cubcr.shapley_sampled import sample_shapley
    todo = [(d, s) for d, s in plan if not L.run_path(task["iid"], d, s).exists()]
    if not todo:
        return task["iid"], 0, 0.0, []
    t0 = time.perf_counter()
    if task["family"] == "overcooked":
        comp, _ = L.build_overcooked_k10()
        table = L.load_overcooked_table(mmap=False)
        engine_name = "step_table_oracle"
        ups_oracle = float(table[-1] - table[0])
        if ups_oracle != task["upsilon"]:
            raise RuntimeError("Overcooked table Υ differs from the stored attributable value")
    else:
        comp, _, _, _ = rb.build_task(task)
        table = L.load_task_table(task, mmap=True)
        engine_name = "table_oracle"
        if float(table[-1] - table[0]) != task["upsilon"]:
            raise RuntimeError(f"{task['iid']}: table Υ differs from the stored attributable value")
    prep_s = time.perf_counter() - t0
    snap = rb.load_snapshot()
    msgs = []
    for div, seed in todo:
        if task["family"] == "overcooked":
            eps = ups_oracle / div
            eng = L.StepTableEngine(table, compiled=comp, n=L.OC_N, k=L.OC_K)
            kw = dict(L.OC_SAMPLER_KW, seed=int(seed), eps=float(eps))
            t1 = time.perf_counter()
            r = sample_shapley(eng, **kw)
            wall = time.perf_counter() - t1
        else:
            eps = task["upsilon"] / div
            eng = rb.TableEngine(table, compiled=comp)
            kw = rb.paper_sampler_kwargs(eps, seed, task["n"], task["k"])
            r, wall, _ = rb.run_paper_sampler(eng, task["n"], task["k"], eps, seed)
        check = None
        if div == 20 and seed in L.PAPER_SEEDS and seed in task["cert_gpu"]:
            rec = task["cert_gpu"][seed]
            check = _check_oc(r, rec, eps) if task["family"] == "overcooked" else _check_tb(r, rec, eps)
            if not check["all_bitwise"]:
                raise RuntimeError(f"oracle re-run of {rec['_file']} is NOT bitwise identical: {check}")
        out = {"exp": "E2", "part": "A", "iid": task["iid"], "family": task["family"], "config": task["config"],
               "n": task["n"], "k": task["k"], "d": task["d"], "states": task["states"], "gen_seed": task["gen_seed"],
               "sampler_seed": int(seed), "div": int(div), "eps": float(eps), "upsilon": task["upsilon"],
               "eps20": task["eps20"], "engine": engine_name, "sampler_kwargs": kw,
               "table": os.path.relpath(L.OC_TABLE if task["family"] == "overcooked" else rb.table_path(task["iid"]), rb.REPO),
               "git_commit": tags.get("git_commit"), "hardware": tags, "timestamp": rb.now_iso(),
               "wall_s": float(wall), "wall_is_gpu_timing": False,
               "wall_convention": "CPU table lookup around sample_shapley; not comparable to GPU time",
               "worker_prep_s": prep_s, "load_snapshot": snap, "phi_exact": task["phi_exact"].tolist(),
               "result": L.result_payload(r, wall), "check_vs_stored": check}
        rb.write_json(L.run_path(task["iid"], div, seed), out)
        msgs.append(f"{task['iid']} div{div} s{seed} {r.status} units={r.n_samples} wall={wall:.2f}s"
                    + ("" if check is None else f" bitwise={check['all_bitwise']}"))
    return task["iid"], len(todo), time.perf_counter() - t0, msgs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--divs", type=int, nargs="*", default=[20, 100, 50])
    ap.add_argument("--seedsets", nargs="*", default=["paper", "new"])
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-overcooked", action="store_true")
    ap.add_argument("--only-overcooked", action="store_true")
    ap.add_argument("--max-n", type=int, default=24)
    a = ap.parse_args(argv)
    if a.workers > 8:
        raise SystemExit("at most 8 CPU workers")
    L.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    plan = plan_for(a.divs, a.seedsets)
    tasks = [] if a.only_overcooked else [t for t in L.ticketbench_reference_tasks() if t["n"] <= a.max_n]
    ready, missing = [], []
    for t in tasks:
        (ready if rb.table_path(t["iid"]).exists() else missing).append(t)
    if not a.no_overcooked:
        oc = L.overcooked_task()
        (ready if L.OC_TABLE.exists() else missing).append(oc)

    ready.sort(key=lambda t: -(t["d"] * 1000 + t["k"]) if t["family"] == "ticketbench" else -23500)
    print(f"[{rb.now_iso()}] plan {len(plan)} runs/task; ready {len(ready)} tasks; table missing for "
          f"{[t['iid'] for t in missing]}", flush=True)
    tags = rb.hardware_tags()
    from joblib import Parallel, delayed
    t0 = time.perf_counter()
    res = Parallel(n_jobs=a.workers, backend="loky", verbose=0)(delayed(job)(t, plan, tags) for t in ready)
    for iid, nrun, sec, msgs in res:
        print(f"  {iid}: {nrun} new runs in {sec:.1f}s", flush=True)
    print(f"[{rb.now_iso()}] done in {time.perf_counter() - t0:.1f}s; missing tables: {len(missing)}", flush=True)


if __name__ == "__main__":
    main()
