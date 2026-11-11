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
sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/scripts"))

import argparse
import contextlib
import fcntl
import json
import time
import traceback

import rb
import e9_gpu_estimators as e9
import e9_run

E9 = rb.results_dir("E9")
RAW = E9 / "raw" / "gpu_hostlock"
LOG = E9 / "logs" / "run_hostlock.log"
LOCKLOG = E9 / "raw" / "lock_log_hostlock.jsonl"
SESSION = "hostlock_rerun"


def log(msg):
    line = f"[{rb.now_iso()}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


@contextlib.contextmanager
def poll_host_lock(tag, max_wait_s, every_s=1.0, path=None):


    if path is None:
        rb.LOCK_DIR.mkdir(parents=True, exist_ok=True)
        path = rb.LOCK_DIR / "host.lock"
    f = open(path, "a+")
    held, attempts = False, 0
    t0 = time.perf_counter()
    while True:
        attempts += 1
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            held = True
            f.seek(0); f.truncate(); f.write(f"{os.getpid()} {tag} {rb.now_iso()}\n"); f.flush()
            break
        except BlockingIOError:
            if time.perf_counter() - t0 + every_s > max_wait_s:
                break
            time.sleep(every_s)
    waited = time.perf_counter() - t0
    try:
        yield held, waited, attempts
    finally:
        if held:
            fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def run_gpu_test_in_hold():


    import subprocess
    out = E9 / "logs" / "pytest_gpu.log"
    env = dict(os.environ, E9_GPU_LOCK_HELD_BY_PARENT=str(os.getpid()), PYTHONDONTWRITEBYTECODE="1")
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider",
                        "rebuttal/tests/test_e9_gpu_estimators.py::test_gpu_twin_bitwise"],
                       cwd=str(rb.REPO), env=env, capture_output=True, text=True, timeout=900)
    el = time.perf_counter() - t0
    with open(out, "a") as f:
        f.write(f"\n[{rb.now_iso()}] run inside the GPU-lock hold of e9_run_hostlock.py (pid {os.getpid()}), "
                f"exit code {p.returncode}, {el:.1f}s:\n[stderr]\n{p.stderr.strip()}\n[stdout]\n{p.stdout.strip()}\n")
    log(f"GPU unit test exit code {p.returncode} ({el:.1f}s): {p.stdout.strip().splitlines()[-1] if p.stdout.strip() else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", nargs="*", default=list(e9_run.LEVELS))
    ap.add_argument("--arms", nargs="*", default=list(e9.ARMS))
    ap.add_argument("--seeds", nargs="*", type=int, default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--max-hold", type=float, default=420.0)
    ap.add_argument("--host-lock-max-wait", type=float, default=90.0)
    ap.add_argument("--no-gpu-test", dest="run_gpu_test", action="store_false",
                    help="do not run test_gpu_twin_bitwise at the end of the last hold")
    a = ap.parse_args()
    import shapiq
    from e4_external import leverage_module
    leverage_module()
    plan, P = e9_run.load_plan()
    iids = list(dict.fromkeys(r["iid"] for r in plan["rows"]))
    tasks = {t["iid"]: t for t in rb.paper_tasks(include_records=False)}
    hw = rb.hardware_tags()
    RAW.mkdir(parents=True, exist_ok=True)
    groups = []
    for iid in iids:
        if tasks[iid]["phi_exact"] is None:
            raise RuntimeError(f"{iid}: no exact reference")
        for level in a.levels:
            for si, seed in enumerate(a.seeds):
                todo = [arm for arm in a.arms if not (RAW / e9_run.run_name(iid, level, arm, seed)).exists()]
                if todo:

                    rot = (si + e9_run.LEVELS.index(level)) % len(e9.ARMS)
                    order = [x for x in (list(e9.ARMS)[rot:] + list(e9.ARMS)[:rot]) if x in todo]
                    groups.append((iid, level, seed, order))
    models = {}
    for iid in dict.fromkeys(g[0] for g in groups):
        comp, _, _, build_s = rb.build_task(tasks[iid])
        models[iid] = (comp, build_s)
    log(f"{len(groups)} groups to run; models built for {len(models)} tasks; max hold {a.max_hold:.0f}s; "
        f"host-lock max wait {a.host_lock_max_wait:.0f}s; BLAS threads {os.environ.get('OPENBLAS_NUM_THREADS')}, "
        f"OMP {os.environ.get('OMP_NUM_THREADS')}")
    gi = 0
    test_pending = bool(a.run_gpu_test)
    while gi < len(groups) or test_pending:
        t_req = time.perf_counter()
        with rb.gpu_lock(f"E9 host-lock re-run from group {gi} ({len(groups) - gi} left)") as waited:
            t_hold0 = time.perf_counter()
            log(f"GPU lock after {waited:.1f}s wait; groups from #{gi}")
            done_tags = []
            with poll_host_lock(f"E9 host-lock re-run (pid {os.getpid()})", a.host_lock_max_wait) as (h_held, h_wait, h_att):
                log(f"exclusive host lock held={h_held} after {h_wait:.1f}s ({h_att} non-blocking attempts)")
                while gi < len(groups) and (time.perf_counter() - t_hold0) < a.max_hold:
                    iid, level, seed, order = groups[gi]
                    t = tasks[iid]
                    comp, build_s = models[iid]
                    tag = f"E9-HL {iid} {level} s{seed}"
                    for pos, arm in enumerate(order):
                        prow = P[(iid, level, arm)]
                        try:
                            rec = e9_run.one_run(t, comp, level, arm, seed, prow, pos, hw, build_s)
                            rec["status_run"] = "OK"
                        except Exception as ex:
                            rec = {"exp": "E9", "kind": "gpu", "iid": iid, "level": level, "arm": arm, "seed": int(seed),
                                   "status_run": "ERROR", "error": repr(ex), "traceback": traceback.format_exc()}
                            log(f"{tag} {arm}: ERROR {ex!r}")
                        rec["git_commit"] = hw.get("git_commit")
                        rec["session"] = SESSION
                        rec["host_lock_held"] = bool(h_held)
                        rec["host_lock_wait_s"] = float(h_wait)
                        rec["host_lock_attempts"] = int(h_att)
                        rec["gpu_lock_wait_s"] = float(waited)
                        rec["niceness"] = os.nice(0)
                        rec["pid"] = os.getpid()
                        rb.write_json(RAW / e9_run.run_name(iid, level, arm, seed), rec)
                        if rec["status_run"] == "OK":
                            log(f"{tag} {arm}: wall {rec['wall_s']:.3f}s engine {rec['engine_wall_s']:.3f}s "
                                f"host {rec['host_s']:.3f}s distinct {rec['n_engine']} load1 "
                                f"{rec['load_before']['loadavg'][0]:.1f} hostlock {h_held}")
                    done_tags.append(tag)
                    gi += 1
                if gi >= len(groups) and test_pending:


                    run_gpu_test_in_hold()
                    test_pending = False
            hold = time.perf_counter() - t_hold0
        with open(LOCKLOG, "a") as f:
            f.write(json.dumps({"groups": done_tags, "wait_s": float(waited), "hold_s": hold,
                                "host_lock_held": bool(h_held), "host_lock_wait_s": float(h_wait),
                                "host_lock_attempts": int(h_att), "requested_at_s": t_req, "time": rb.now_iso()}) + "\n")
        log(f"released after {hold:.1f}s ({len(done_tags)} groups)")
    log("all planned groups done")


if __name__ == "__main__":
    main()
