#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import time as _time
_T0 = _time.perf_counter()
_EPOCH0 = _time.time()
import argparse
import json
import os
import sys
import fcntl
import contextlib

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e3_engine as E
import e3_models as M
import e3_common as C
import cubcr.shapley_sampled
import generators.ticketbench


def _args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--mode", choices=["cold", "warm"], required=True)
    ap.add_argument("--rep", type=int, default=1, help="cold: repetition index of this process")
    ap.add_argument("--reps", type=int, default=5, help="warm: number of timed repetitions (after rep 0)")
    ap.add_argument("--lock", choices=["parent", "rep", "all"], default="rep")
    ap.add_argument("--spawn-epoch", type=float, default=None)
    ap.add_argument("--replicate", type=int, default=1, help="run the paper-protocol replication (1/0)")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


A = _args()
CFG = C.config(A.config)
if CFG["family"] == "oc":
    M.overcooked_import()
E.lib()
_T_IMPORTED = _time.perf_counter()
OUT_DIR = C.RAW / "phases"


def out_path(rep):
    return OUT_DIR / f"{CFG['name']}__{A.mode}__rep{rep}.json"


@contextlib.contextmanager
def try_host_lock(tag, tries=1, sleep=0.0):


    path = rb.LOCK_DIR / "host.lock"
    f = open(path, "a+")
    held = False
    for _ in range(tries):
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
            break
        except BlockingIOError:
            _time.sleep(sleep)
    try:
        yield held
    finally:
        if held:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def gpu_lock_ctx(scope):
    if A.lock == scope:
        return rb.gpu_lock(f"E3 phases {CFG['name']} {A.mode}")
    return contextlib.nullcontext(0.0)


def exact_phi(cfg):
    if cfg["family"] == "grid":
        for t in rb.paper_tasks():
            if (t["n"], t["states_target"], t["k"], t["gen_seed"]) == (cfg["n"], cfg["states_target"], cfg["k"], C.GEN_SEED):
                return None if t["phi_exact"] is None else t["phi_exact"].tolist()
    if cfg["family"] == "oc":
        d = json.load(open(rb.REPO / "results" / "e14" / "e14__overcooked_cramped_room_k10__exact_step.json"))
        return np.asarray(d["shapley"]).reshape(-1).tolist()
    return None


def one_rep(rep: int, first_in_process: bool) -> dict:
    rec = {"config": CFG["name"], "family": CFG["family"], "mode": A.mode, "rep": rep, "pid": os.getpid(),
           "lock_scope": A.lock, "first_in_process": first_in_process, "gen_seed": C.GEN_SEED,
           "sampler_seed": C.SAMPLER_SEED, "n": CFG["n"], "k": CFG["k"], "players": CFG["players"]}
    if first_in_process and A.mode == "cold":
        rec["interp_start_s"] = (_EPOCH0 - A.spawn_epoch) if A.spawn_epoch else None
        rec["imports_s"] = _T_IMPORTED - _T0
    else:
        rec["interp_start_s"] = 0.0
        rec["imports_s"] = 0.0
    if A.mode == "warm" or A.lock != "parent":
        rec["load_before"] = rb.load_snapshot()

    with try_host_lock(f"E3 host phases {CFG['name']}") as held:
        rec["host_lock_held"] = bool(held)
        comp, meta, tb = C.build(CFG)
        rec.update({f"build_{k_}": v for k_, v in tb.items()})
        rec["iid"] = meta.get("instance_id")
        rec["states"] = int(comp.n_states); rec["nnz"] = int(comp.nnz); rec["rows"] = int(comp.n_rows)
        eng = E.E3Engine(comp, defer_device=True)
    ph = eng.phase_s
    rec.update({"marshal_s": ph["marshal_s"], "p2_s": ph["p2_host_s"], "p2_c_tables_s": ph["p2_c_tables_s"],
                "p2_c_nrow_s": ph["p2_c_nrow_s"], "p2_c_val32_s": ph["p2_c_val32_s"],
                "p2_c_classes_s": ph["p2_c_classes_s"], "table_entries": ph["table_entries"],
                "table_bytes": ph["table_bytes"], "rmax": ph["rmax"]})

    with gpu_lock_ctx("rep") as waited:
        rec["gpu_lock_wait_s"] = float(waited or 0.0)
        t0 = _time.perf_counter()
        ctx_c = E.context_init(0)
        rec["p3a_context_s"] = _time.perf_counter() - t0
        rec["p3a_context_c_s"] = ctx_c
        eng.attach_device()
        rec.update({"p3b_alloc_s": ph["p3_alloc_s"], "p4_upload_s": ph["p4_upload_s"],
                    "p4_upload_event_s": ph["p4_event_s"], "h2d_bytes": ph["h2d_bytes"]})

        st = STORED
        if C.needs_upsilon(CFG):
            t0 = _time.perf_counter()
            vv = eng.solve(C.endpoint_masks(CFG), CFG["k"])
            rec["ups_solve_s"] = _time.perf_counter() - t0
            ups = float(vv[1] - vv[0])
            rec["upsilon"] = ups
            rec["upsilon_equals_stored"] = bool(ups == st["upsilon"])
            rec["ups_solve_engine"] = dict(eng.stats)
        else:
            rec["ups_solve_s"] = 0.0
            ups = None
            rec["upsilon"] = None
        eps = C.eps_from_upsilon(CFG, ups)
        rec["eps"] = eps
        rec["eps_equals_stored"] = bool(eps == st["eps"])

        rec["mem_free_before_p5"], rec["mem_total"] = E.mem_info(0)
        gc0 = int(eng.stats["graph_captures"]) if eng.stats else 0
        eng.reset_cum()
        r, wall = C.run_sampler(eng, CFG, eps)
        rec["p5_s"] = wall
        rec.update(C.p5_split(eng, wall))
        rec["p5_graph_captures"] = int(eng.stats["graph_captures"]) - gc0
        rec["p5_calls"] = list(eng.calls)
        rec["p5_status"] = r.status; rec["p5_certified"] = bool(r.certified)
        rec["p5_n_perms"] = int(r.n_perms); rec["p5_n_solves"] = int(r.n_solves); rec["p5_n_units"] = int(r.n_samples)
        rec["p5_n_batches"] = int(r.n_batches); rec["p5_batch_sizes"] = list(map(int, r.batch_sizes))
        rec["p5_engine_time_s"] = float(r.engine_time)

        t0 = _time.perf_counter()
        row = rb.result_row(r, wall)
        blob = json.dumps(row)
        rec["p6_assembly_s"] = _time.perf_counter() - t0
        rec["p6_result_bytes"] = len(blob)
        rec["p5_result"] = row
        rec["p5_vs_stored"] = C.compare_to_stored(r, st, wall)
        eng.close()

        if A.replicate:
            eng2 = E.E3Engine(comp)
            rec["rep_create_s"] = eng2.phase_s["p2_host_s"] + eng2.phase_s["p3_alloc_s"] + eng2.phase_s["p4_upload_s"]
            t0 = _time.perf_counter()
            winfo = C.paper_warmup(eng2, CFG)
            rec["rep_warmup_s"] = _time.perf_counter() - t0
            rec["rep_warmup"] = winfo["warmup"]
            gc0 = int(eng2.stats["graph_captures"])
            eng2.reset_cum()
            r2, wall2 = C.run_sampler(eng2, CFG, eps)
            rec["rep_p5_s"] = wall2
            rec.update({"rep_" + k_: v for k_, v in C.p5_split(eng2, wall2).items()})
            rec["rep_p5_graph_captures"] = int(eng2.stats["graph_captures"]) - gc0
            rec["rep_p5_calls"] = list(eng2.calls)
            rec["rep_p5_vs_stored"] = C.compare_to_stored(r2, st, wall2)
            rec["rep_same_as_deployed"] = C.same_result(r, r2)
            rec["rep_p5_result"] = rb.result_row(r2, wall2)
            eng2.close()

    parts = ["interp_start_s", "imports_s", "build_p0_s", "build_p0m_s", "build_p1_s", "marshal_s", "p2_s",
             "p3a_context_s", "p3b_alloc_s", "p4_upload_s", "ups_solve_s", "p5_s", "p6_assembly_s"]
    rec["total_s"] = float(sum((rec.get(p) or 0.0) for p in parts))
    rec["total_parts"] = parts
    rec["construction_s"] = float(sum((rec.get(p) or 0.0) for p in parts[2:10]))
    rec["exact_phi"] = EXACT_PHI
    if A.mode == "warm" or A.lock != "parent":
        rec["load_after"] = rb.load_snapshot()
        rec["gpu_apps_after"] = C.gpu_apps()
    rec["env"] = {k_: os.environ.get(k_) for k_ in ("CUDA_MODULE_LOADING", "OMP_NUM_THREADS", "CUDA_VISIBLE_DEVICES")}
    rec["omp_max_threads"] = E.omp_max_threads()
    return rec


STORED = None
EXACT_PHI = None


def main():
    global STORED, EXACT_PHI
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    STORED = C.stored_record(CFG)
    EXACT_PHI = exact_phi(CFG)
    reps = [A.rep] if A.mode == "cold" else list(range(0, A.reps + 1))
    todo = [r for r in reps if A.force or not out_path(r).exists()]
    if A.mode == "warm" and todo and 0 not in todo:
        todo = [0] + todo
    if not todo:
        print(f"[E3 worker] {CFG['name']} {A.mode}: all reps present", flush=True)
        return 0
    prov = E.provenance()
    with gpu_lock_ctx("all") as waited_all:
        for i, rep in enumerate(todo):
            rec = one_rep(rep, first_in_process=(i == 0))
            if A.lock == "all":
                rec["gpu_lock_wait_all_s"] = float(waited_all or 0.0)
            rec["e3_provenance"] = prov
            rec["hardware"] = rb.hardware_tags()
            rec["timestamp"] = rb.now_iso()
            rb.write_json(out_path(rep), rec)
            print(f"[E3 worker] {CFG['name']} {A.mode} rep{rep}: total {rec['total_s']:.3f}s "
                  f"(imports {rec['imports_s']:.2f} ctx {rec['p3a_context_s']:.3f} P5 {rec['p5_s']:.3f}) "
                  f"stored={rec['p5_vs_stored']['reproduces_stored']} "
                  f"rep_stored={rec.get('rep_p5_vs_stored', {}).get('reproduces_stored')}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
