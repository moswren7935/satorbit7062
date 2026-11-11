#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "6")
sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))

import argparse
import json
import time
import traceback

import numpy as np

import rb
import e9_gpu_estimators as e9

E9 = rb.results_dir("E9")
RAW = E9 / "raw" / "gpu"
LOG = E9 / "logs" / "run.log"
LOCKLOG = E9 / "raw" / "lock_log.jsonl"
LEVELS = ("B4k", "Bstop", "Bhi")


def log(msg):
    line = f"[{rb.now_iso()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run_name(iid, level, arm, seed):
    return f"{iid}__{level}__{arm}__s{seed}.json"


def load_plan():
    plan = json.load(open(E9 / "raw" / "plan.json"))
    out = {}
    for r in plan["rows"]:
        out[(r["iid"], r["level"], r["arm"])] = r
    return plan, out


def nominal_of(row):
    return {"m": int(row["nominal_m"])} if row["arm"].startswith("paired") else {"T": int(row["nominal_T"])}


def _ser(x):
    if x is None:
        return None
    return np.asarray(x, dtype=np.float64).tolist()


def one_run(t, comp, level, arm, seed, prow, pos, hw, build_s):
    n, k = t["n"], t["k"]
    nominal = nominal_of(prow)
    snap0 = rb.load_snapshot()
    started = rb.now_iso()
    t0 = time.perf_counter()
    eng = rb.make_gpu_engine(comp)
    setup_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    rb.paper_warmup(eng, n, k)
    warm_s = time.perf_counter() - t0
    try:
        te = e9.TimedEngine(eng)
        out = e9.run_arm(arm, te, n, k, nominal, seed)
        rs = e9.resolve_union(te, k, n_rep=2)
    finally:
        eng.close()
    snap1 = rb.load_snapshot()
    phi = t["phi_exact"]
    met = e9.error_metrics(out["phi_hat"], phi, t["upsilon"])
    ivc = e9.interval_checks(out, phi)
    rec = {"exp": "E9", "kind": "gpu", "iid": t["iid"], "n": n, "states": t["states"], "states_target": t["states_target"],
           "k": k, "gen_seed": t["gen_seed"], "upsilon": t["upsilon"], "eps20": t["eps20"], "level": level,
           "B": int(prow["B"]), "B_source": prow["B_source"], "arm": arm, "label": e9.LABELS[arm], "seed": int(seed),
           "nominal": nominal, "nominal_how": prow["how"], "order_pos": pos, "started": started,
           "engine_setup_s": setup_s, "warmup_s": warm_s, "model_build_s": build_s,
           "config": {"engine": "rb.make_gpu_engine (GPUEngine fp64, use_graph=True)", "warmup": "rb.paper_warmup",
                      "paired_sampler_kw": dict(rb.PAPER_SAMPLER_KW, eps=None) if arm.startswith("paired") else None,
                      "shapiq": __import__("shapiq").__version__ if arm == "kadd2" else None,
                      "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS"), "omp_threads": os.environ.get("OMP_NUM_THREADS")},
           "resolve": rs, "metrics": met, "intervals": ivc, "phi_exact": _ser(phi),
           "load_before": snap0, "load_after": snap1, "hardware": hw}
    for key, val in out.items():
        if key in ("phi_hat", "half_widths", "cs_lower", "cs_upper"):
            rec[key] = _ser(val)
        else:
            rec[key] = val
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", default=None, help="substrings of instance ids (default: all planned)")
    ap.add_argument("--levels", nargs="*", default=list(LEVELS))
    ap.add_argument("--arms", nargs="*", default=list(e9.ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--max-groups", type=int, default=0)
    ap.add_argument("--max-hold", type=float, default=420.0,
                    help="maximum seconds per GPU-lock hold")
    a = ap.parse_args()
    import shapiq
    from e4_external import leverage_module
    leverage_module()
    plan, P = load_plan()
    iids = []
    for r in plan["rows"]:
        if r["iid"] not in iids:
            iids.append(r["iid"])
    if a.tasks:
        iids = [i for i in iids if any(s in i for s in a.tasks)]
    tasks = {t["iid"]: t for t in rb.paper_tasks(include_records=False)}
    hw = rb.hardware_tags()
    RAW.mkdir(parents=True, exist_ok=True)

    groups = []
    for iid in iids:
        t = tasks[iid]
        if t["phi_exact"] is None:
            raise RuntimeError(f"{iid}: no exact reference")
        for level in a.levels:
            for si, seed in enumerate(a.seeds):
                todo = [arm for arm in a.arms if not (RAW / run_name(iid, level, arm, seed)).exists()]
                if todo:
                    rot = (si + LEVELS.index(level)) % len(e9.ARMS)
                    order = [x for x in (list(e9.ARMS)[rot:] + list(e9.ARMS)[:rot]) if x in todo]
                    groups.append((iid, level, seed, order))
    if a.max_groups:
        groups = groups[:a.max_groups]
    models = {}
    for iid in dict.fromkeys(g[0] for g in groups):
        comp, _, _, build_s = rb.build_task(tasks[iid])
        models[iid] = (comp, build_s)
    log(f"{len(groups)} groups to run; models built for {len(models)} tasks; max hold {a.max_hold:.0f}s")

    gi = 0
    while gi < len(groups):
        t_req = time.perf_counter()
        with rb.gpu_lock(f"E9 hold from group {gi} ({len(groups) - gi} left)") as waited:
            t_hold0 = time.perf_counter()
            done_tags = []
            log(f"lock after {waited:.1f}s wait; groups from #{gi}")
            while gi < len(groups) and (time.perf_counter() - t_hold0) < a.max_hold:
                iid, level, seed, order = groups[gi]
                t = tasks[iid]
                comp, build_s = models[iid]
                tag = f"E9 {iid} {level} s{seed}"
                for pos, arm in enumerate(order):
                    prow = P[(iid, level, arm)]
                    try:
                        rec = one_run(t, comp, level, arm, seed, prow, pos, hw, build_s)
                        rec["status_run"] = "OK"
                    except Exception as ex:
                        rec = {"exp": "E9", "kind": "gpu", "iid": iid, "level": level, "arm": arm, "seed": int(seed),
                               "status_run": "ERROR", "error": repr(ex), "traceback": traceback.format_exc()}
                        log(f"{tag} {arm}: ERROR {ex!r}")
                    rec["git_commit"] = hw.get("git_commit")
                    rb.write_json(RAW / run_name(iid, level, arm, seed), rec)
                    if rec["status_run"] == "OK":
                        log(f"{tag} {arm}: wall {rec['wall_s']:.3f}s engine {rec['engine_wall_s']:.3f}s "
                            f"host {rec['host_s']:.3f}s calls {rec['engine_calls']} distinct {rec['n_engine']} "
                            f"req {rec['n_requested']} max_err/Ups {rec['metrics']['max_abs_rel']:.4f} "
                            f"resolve2 {rec['resolve'][-1]['kernel_ms']:.1f}ms same {rec['resolve'][-1]['same_values']}")
                done_tags.append(tag)
                gi += 1
            hold = time.perf_counter() - t_hold0
        with open(LOCKLOG, "a") as f:
            f.write(json.dumps({"groups": done_tags, "wait_s": float(waited), "hold_s": hold,
                                "requested_at_s": t_req, "time": rb.now_iso()}) + "\n")
        log(f"released after {hold:.1f}s ({len(done_tags)} groups)")
    log("all planned groups done")


if __name__ == "__main__":
    main()
