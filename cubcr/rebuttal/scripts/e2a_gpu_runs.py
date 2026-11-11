#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time

os.environ.setdefault("NUMBA_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "4")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "lib"))
import numpy as np

import rb
import e2a_lib as L


HOLD_BUDGET_S = 45.0


def log(msg):
    print(f"[{rb.now_iso()}] {msg}", flush=True)


def overcooked_table(force: bool = False):
    out_json = L.RAW / "partA_overcooked_table.json"
    if L.OC_TABLE.exists() and out_json.exists() and not force:
        log(f"overcooked table exists: {L.OC_TABLE}")
        return
    from cubcr.shapley_fast import shapley_from_table_fast
    from cubcr.shapley_sampled import exact_player_table
    ex = rb._read(L.OC_EXACT_JSON)
    phi_ref = np.asarray(ex["shapley"], dtype=np.float64).reshape(-1)
    t0 = time.perf_counter()
    comp, meta = L.build_overcooked_k10()
    build_s = time.perf_counter() - t0
    log(f"overcooked k=10 model built in {build_s:.1f}s: S={comp.n_states} rows={comp.n_rows}")
    if comp.n_states != int(ex["states"]):
        raise RuntimeError(f"state count {comp.n_states} != stored {ex['states']}")
    t_req = time.perf_counter()
    with rb.gpu_lock("E2A overcooked k10 table") as waited:
        t_acq = time.perf_counter()
        snap = rb.load_snapshot()
        generated = False
        if L.OC_TABLE.exists() and not force:
            table = np.load(L.OC_TABLE)
            solve_s = None
        else:
            eng = rb.make_gpu_engine(comp)
            t0 = time.perf_counter()
            table = np.asarray(exact_player_table(eng, L.OC_D, kind="step", n=L.OC_N, k=L.OC_K), dtype=np.float64)
            solve_s = time.perf_counter() - t0
            eng.close()
            generated = True
        hold_s = time.perf_counter() - t_acq
    log(f"table {'generated' if generated else 'loaded'}; lock wait {t_acq - t_req:.1f}s hold {hold_s:.1f}s")
    ups = float(table[-1] - table[0])
    t0 = time.perf_counter()
    phi = shapley_from_table_fast(table, L.OC_D)
    shap_s = time.perf_counter() - t0
    dev = float(np.max(np.abs(phi - phi_ref)))
    ok_ups = ups == float(ex["attributable_value"])
    if not (dev <= 1e-12 and ok_ups):
        raise RuntimeError(f"overcooked table check FAILED: max|phi-phi_ref|={dev}, ups={ups} vs {ex['attributable_value']}")
    if generated:
        tmp = L.OC_TABLE.with_name(L.OC_TABLE.name + f".tmp{os.getpid()}.npy")
        L.OC_TABLE.parent.mkdir(parents=True, exist_ok=True)
        np.save(tmp, table)
        os.replace(tmp, L.OC_TABLE)
    sha = hashlib.sha256(open(L.OC_TABLE, "rb").read()).hexdigest()
    rec = {"exp": "E2", "part": "A", "what": "Overcooked cramped_room k=10 step-level exhaustive table (2^20)",
           "path": os.path.relpath(L.OC_TABLE, rb.REPO), "sha256": sha, "generated_now": generated,
           "n_players": L.OC_D, "player_index": "p = i*k + t (cubcr.masks.step_mask_matrix); bit p set = pinned",
           "n_states": int(comp.n_states), "n_rows": int(comp.n_rows), "model_build_s": build_s,
           "gpu_solve_s": solve_s, "stored_exact_wall_s": float(ex["wall_s"]), "shapley_s": shap_s,
           "upsilon": ups, "upsilon_stored": float(ex["attributable_value"]), "upsilon_bitwise_equal": ok_ups,
           "v_empty": float(table[0]), "v_full": float(table[-1]),
           "max_abs_dev_phi_vs_stored": dev, "check_tol": 1e-12, "check_passed": True,
           "lock_wait_s": t_acq - t_req, "lock_hold_s": hold_s, "load_snapshot": snap,
           "hardware": rb.hardware_tags(), "timestamp": rb.now_iso(),
           "source_of_model": "generators/overcooked_csg.py enumerate_layers + fit_human_policy(min_obs=3) + "
                              "build_layered_compiled (as experiments/e14_overcooked.py::run_exact_k10)"}
    rb.write_json(L.RAW / "partA_overcooked_table.json", rec)
    log(f"overcooked table OK: max|dphi|={dev:.2e}, ups={ups}, sha256={sha[:12]}")


def n28_plan(divs, seedsets):
    plan = []
    if "paper" in seedsets and 20 in divs:
        plan += [(20, s) for s in L.PAPER_SEEDS]
    for div in divs:
        for ss in seedsets:
            seeds = L.PAPER_SEEDS if ss == "paper" else L.NEW_SEEDS
            for s in seeds:
                if (div, s) not in plan:
                    plan.append((div, s))

    order = {20: 1, 100: 0, 50: 2}
    head = [p for p in plan if p[0] == 20 and p[1] in L.PAPER_SEEDS]
    rest = sorted([p for p in plan if p not in head], key=lambda p: (order.get(p[0], 3), p[1]))
    return head + rest


def run_gpu_tasks(tasks, plan, tags):


    queue = []
    built = {}
    for task in tasks:
        todo = [(d, s) for d, s in plan if not L.run_path(task["iid"], d, s).exists()]
        if todo:
            built[task["iid"]] = rb.build_task(task)
            queue += [(task, d, s) for d, s in todo]

    xq = []
    all_tasks = {t["iid"]: t for t in L.ticketbench_reference_tasks()}
    for iid, div, seed in XCHECKS:
        out = L.RAW / "partA_xcheck" / f"{iid}__div{div}__s{seed}.json"
        if not out.exists() and L.run_path(iid, div, seed).exists():
            t = all_tasks[iid]
            built[iid] = rb.build_task(t)
            xq.append((t, div, seed, "xcheck"))
    queue = xq + [(t, d, s_, "run") for t, d, s_ in queue]
    gpu_s = 0.0
    i = 0
    while i < len(queue):
        t_req = time.perf_counter()
        with rb.gpu_lock(f"E2A n28 batch from {queue[i][0]['iid']} {queue[i][1:3]}"):
            t_acq = time.perf_counter()
            batch_wait = t_acq - t_req
            first = True
            while i < len(queue) and (first or time.perf_counter() - t_acq < HOLD_BUDGET_S):
                task, div, seed, kind = queue[i]
                comp, iid, meta, build_s = built[task["iid"]]
                n, k = task["n"], task["k"]
                eps = task["upsilon"] / div
                t_run0 = time.perf_counter()
                snap = rb.load_snapshot()
                t0 = time.perf_counter()
                eng = rb.make_gpu_engine(comp)
                rb.paper_warmup(eng, n, k)
                setup_s = time.perf_counter() - t0
                r, wall, _ = rb.run_paper_sampler(eng, n, k, eps, seed)
                eng.close()
                snap_after = rb.load_snapshot()
                hold_s = time.perf_counter() - t_run0
                gpu_s += hold_s
                if kind == "xcheck":
                    _write_xcheck(task, iid, div, seed, r, wall, hold_s, snap, tags)
                else:
                    _write_gpu_run(task, comp, iid, n, k, div, seed, eps, r, wall, setup_s, build_s,
                                   batch_wait if first else 0.0, hold_s, snap, snap_after, tags)
                first = False
                i += 1
    return gpu_s


XCHECKS = [("tb_none_n16_S961_k50_pf0.05_C3_s20260829", 100, 20261101),
           ("tb_none_n24_S1009_k100_pf0.05_C3_s20260829", 100, 20261101)]


def _write_xcheck(task, iid, div, seed, r, wall, hold_s, snap, tags):
    o = rb._read(L.run_path(iid, div, seed))["result"]
    c = {"estimates_bitwise": bool(np.array_equal(r.estimates, np.asarray(o["phi_hat"]))),
         "half_widths_bitwise": bool(np.array_equal(r.half_widths, np.asarray(o["half_widths"]))),
         "cs_lower_bitwise": bool(np.array_equal(r.cs_lower, np.asarray(o["cs_lower"]))),
         "cs_upper_bitwise": bool(np.array_equal(r.cs_upper, np.asarray(o["cs_upper"]))),
         "n_perms_equal": int(r.n_perms) == int(o["n_perms"]), "n_solves_equal": int(r.n_solves) == int(o["n_solves"]),
         "batch_sizes_equal": list(map(int, r.batch_sizes)) == list(map(int, o["batch_sizes"]))}
    c["all_bitwise"] = all(c.values())
    rb.write_json(L.RAW / "partA_xcheck" / f"{iid}__div{div}__s{seed}.json",
                  {"what": "GPU re-run (paper engine + warm-up) of a table-oracle run at the same eps/seed",
                   "iid": iid, "div": div, "sampler_seed": seed, "oracle_file": os.path.relpath(L.run_path(iid, div, seed), rb.REPO),
                   "check": c, "gpu_wall_s": wall, "lock_hold_s": hold_s, "n_units": int(r.n_samples),
                   "load_snapshot": snap, "hardware": tags, "timestamp": rb.now_iso()})
    log(f"XCHECK {iid} div{div} s{seed}: GPU vs table oracle bitwise={c['all_bitwise']} {c}")


def _write_gpu_run(task, comp, iid, n, k, div, seed, eps, r, wall, setup_s, build_s, wait_s, hold_s, snap, snap_after,
                   tags):
    payload = L.result_payload(r, wall)
    check = None
    if div == 20 and seed in L.PAPER_SEEDS:
        rec = task["cert_gpu"][seed]
        check = {"stored_file": rec["_file"],
                 "eps_equal": float(eps) == float(rec["eps_abs"]),
                 "estimates_bitwise": bool(np.array_equal(r.estimates, np.asarray(rec["shapley_sampled"]))),
                 "half_widths_bitwise": bool(np.array_equal(r.half_widths, np.asarray(rec["half_widths"]))),
                 "cs_lower_bitwise": bool(np.array_equal(r.cs_lower, np.asarray(rec["certificate"]["cs_lower"]))),
                 "cs_upper_bitwise": bool(np.array_equal(r.cs_upper, np.asarray(rec["certificate"]["cs_upper"]))),
                 "n_perms_equal": int(r.n_perms) == int(rec["n_perms"]),
                 "n_solves_equal": int(r.n_solves) == int(rec["n_solves"]),
                 "batch_sizes_equal": list(r.batch_sizes) == list(rec["batch_sizes"]),
                 "stored_wall_s": float(rec["wall_s"])}
        check["all_bitwise"] = all(v for kk, v in check.items() if kk not in ("stored_file", "stored_wall_s"))
        if not check["all_bitwise"]:
            raise RuntimeError(f"GPU re-run of the paper run {rec['_file']} is NOT bitwise identical: {check}")
    out = {"exp": "E2", "part": "A", "iid": iid, "family": "ticketbench", "config": task["config"], "n": n, "k": k,
           "d": n, "states": task["states"], "gen_seed": task["gen_seed"], "sampler_seed": int(seed),
           "div": int(div), "eps": float(eps), "upsilon": task["upsilon"], "eps20": task["eps20"],
           "engine": "gpu", "sampler_kwargs": rb.paper_sampler_kwargs(eps, seed, n, k),
           "git_commit": tags.get("git_commit"), "hardware": tags, "timestamp": rb.now_iso(),
           "wall_s": float(wall), "wall_is_gpu_timing": True,
           "wall_convention": "perf_counter around sample_shapley only (paper convention: excludes model "
                              "build, engine creation and the warm-up batch)",
           "engine_setup_plus_warmup_s": setup_s, "model_build_s": build_s,
           "lock_wait_s": wait_s, "lock_hold_s": hold_s,
           "lock_note": "runs are batched under one lock acquisition of <= HOLD_BUDGET_S; lock_hold_s is this run's "
                        "own time inside the lock (snapshot, engine creation, warm-up, sampling, close, snapshot)",
           "load_snapshot": snap, "load_snapshot_after": snap_after, "phi_exact": task["phi_exact"].tolist(),
           "result": payload, "check_vs_stored": check}
    rb.write_json(L.run_path(iid, div, seed), out)
    log(f"{iid} div{div} s{seed}: {r.status} units={r.n_samples} solves={r.n_solves} wall={wall:.3f}s "
        f"hold={hold_s:.2f}s" + ("" if check is None else f" bitwise={check['all_bitwise']}"))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--overcooked-table", action="store_true")
    ap.add_argument("--n28", action="store_true")
    ap.add_argument("--divs", type=int, nargs="*", default=[20, 100, 50])
    ap.add_argument("--seedsets", nargs="*", default=["paper", "new"])
    ap.add_argument("--force-table", action="store_true")
    ap.add_argument("--hold-budget", type=float, default=HOLD_BUDGET_S,
                    help="max seconds of consecutive runs per GPU-lock acquisition (one run always completes)")
    a = ap.parse_args(argv)
    globals()["HOLD_BUDGET_S"] = float(a.hold_budget)
    L.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    rb.check_engine_provenance(strict=False)
    if a.overcooked_table:
        overcooked_table(force=a.force_table)
    if a.n28:
        tags = rb.hardware_tags()
        plan = n28_plan(a.divs, a.seedsets)
        tasks = [t for t in L.ticketbench_reference_tasks() if t["n"] == 28]
        tot = run_gpu_tasks(tasks, plan, tags)
        log(f"n28 done; {len(tasks)} tasks with exact GPU references; lock hold this invocation {tot:.1f}s")
    log("ALL DONE")


if __name__ == "__main__":
    main()
