#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import json
import subprocess
import sys
import time

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e3_common as C

WORKER = rb.RB / "scripts" / "e3_phase_worker.py"
OUT = C.RAW / "phases"


def out_path(cfg, mode, rep):
    return OUT / f"{cfg['name']}__{mode}__rep{rep}.json"


def run_worker(cfg, mode, lock, rep=None, reps=5, replicate=1):
    cmd = [sys.executable, str(WORKER), "--config", cfg["name"], "--mode", mode, "--lock", lock,
           "--replicate", str(int(replicate))]
    cmd += ["--rep", str(rep)] if mode == "cold" else ["--reps", str(reps)]
    snap = {}
    if mode == "cold":
        snap["parent_load_before"] = rb.load_snapshot()
        snap["parent_gpu_apps_before"] = C.gpu_apps()
    epoch = time.time()
    cmd += ["--spawn-epoch", repr(epoch)]
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=str(rb.REPO), capture_output=True, text=True, timeout=3600)
    wall = time.perf_counter() - t0
    for line in p.stdout.splitlines():
        if line.startswith("[E3 worker]"):
            print(f"[{rb.now_iso()}] {line}", flush=True)
    if p.returncode != 0:
        print(f"[{rb.now_iso()}] WORKER FAILED {cfg['name']} {mode} rep{rep}: rc={p.returncode}\n{p.stderr[-4000:]}", flush=True)
        return False
    if mode == "cold":
        snap["parent_load_after"] = rb.load_snapshot()
        snap["parent_gpu_apps_after"] = C.gpu_apps()
        snap["parent_process_wall_s"] = wall
        f = out_path(cfg, "cold", rep)
        d = json.load(open(f))
        d.update(snap)
        rb.write_json(f, d)
    return True


def missing(cfg, mode, reps):
    rr = range(1, reps + 1)
    return [r for r in rr if not out_path(cfg, mode, r).exists()]


def do_config(cfg, reps, cold_lock, warm_lock):
    rep_flag = 0 if cfg["family"] == "dt" else 1
    for r in missing(cfg, "cold", reps):
        run_worker(cfg, "cold", cold_lock, rep=r, replicate=rep_flag)
    if missing(cfg, "warm", reps):
        run_worker(cfg, "warm", warm_lock, reps=reps, replicate=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=5)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in C.all_configs() if a.only is None or c["name"] in a.only]
    grid = [c for c in cfgs if c["family"] == "grid"]
    print(f"[{rb.now_iso()}] E3 phases: {len(cfgs)} configurations, reps={a.reps}", flush=True)
    groups = {}
    for c in grid:
        groups.setdefault(c["n"], []).append(c)
    for n, group in sorted(groups.items()):
        todo = [c for c in group if missing(c, "cold", a.reps) or missing(c, "warm", a.reps)]
        if not todo:
            continue
        with rb.gpu_lock(f"E3 phases grid n{n}") as waited:
            print(f"[{rb.now_iso()}] grid n={n}: GPU lock after {waited:.0f}s", flush=True)
            t0 = time.perf_counter()
            for c in todo:
                do_config(c, a.reps, "parent", "parent")
            print(f"[{rb.now_iso()}] grid n={n}: held the GPU lock {time.perf_counter() - t0:.0f}s", flush=True)
    for c in cfgs:
        if c["family"] == "grid":
            continue
        if not (missing(c, "cold", a.reps) or missing(c, "warm", a.reps)):
            continue


        with rb.gpu_lock(f"E3 phases {c['name']} cold+warm") as waited:
            print(f"[{rb.now_iso()}] {c['name']}: GPU lock after {waited:.0f}s", flush=True)
            t0 = time.perf_counter()
            do_config(c, a.reps, "parent", "parent")
            print(f"[{rb.now_iso()}] {c['name']}: held the GPU lock {time.perf_counter() - t0:.0f}s", flush=True)
    print(f"[{rb.now_iso()}] E3 phases: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
