#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import importlib.util
import os
import sys

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e2b_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def load_e2b_dt():
    spec = importlib.util.spec_from_file_location("e2b_dt", os.path.join(HERE, "e2b_dt.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_remaining(comp, meta, n: int, k: int, eps: float, seeds, hw, dt) -> dict:

    iid = meta["instance_id"]
    cut = C.RAW_DT / f"partB_dt_scopecut__{iid}__eps{eps:g}.json"
    out = {"ran": [], "skipped_on_disk": [], "scope_cut": False, "n_acquisitions": 0}
    with C.group_lock(f"E2B dt big {iid} eps{eps:g}") as held:
        out["n_acquisitions"] += 1
        todo = [s for s in seeds if not C.dt_path(iid, eps, s).exists()]
        out["skipped_on_disk"] = [s for s in seeds if s not in todo]
        if cut.exists():
            print(f"[e2b-dt-big] {iid} eps{eps:g}: scope cut recorded earlier, nothing to run", flush=True)
            return out
        for i, seed in enumerate(todo):
            rec = dt.run_one(comp, meta, n, k, eps, seed, hw, None, held=held)
            out["ran"].append(seed)
            if rec["time_to_certify_s"] is not None and rec["wall_s"] > dt.SCOPE_CUT_S and len(todo) > i + 1:
                rb.write_json(cut, {"instance_id": iid, "eps_abs": eps, "ran_seed": seed, "wall_s": rec["wall_s"],
                                    "skipped_seeds": todo[i + 1:], "rule": f"one run > {dt.SCOPE_CUT_S:.0f} s",
                                    "timestamp": rb.now_iso()})
                print(f"[e2b-dt-big] SCOPE CUT {iid} eps{eps:g}: run took {rec['wall_s']:.0f}s", flush=True)
                out["scope_cut"] = True
                break
    print(f"[lock] released E2B dt big {iid} eps{eps:g}: waited {held['group_lock_wait_s']:.0f}s, "
          f"held {held.get('group_hold_s', 0):.0f}s", flush=True)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.001)
    ap.add_argument("--seeds", type=int, nargs="*", default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--allow-rebuilt-engine", action="store_true",
                    help="accepted for compatibility; local rebuilds are always supported")
    a = ap.parse_args(argv)
    for s in a.seeds:
        if s not in rb.PAPER_SAMPLER_SEEDS:
            raise SystemExit(f"seed {s} is not a paper sampler seed")
    dt = load_e2b_dt()
    hw = C.hw_tags()
    if not hw.get("local_build_complete"):
        raise SystemExit("engine libraries are missing; run make -C csrc all")
    n, S, k = dt.BIG
    comp, meta, build_s = C.dt_build(n, S, k)
    iid = meta["instance_id"]
    if C.e4v2_record(iid, "eps0.005", rb.PAPER_SAMPLER_SEEDS[0]) is None:
        raise RuntimeError(f"{iid} not among the stored e4v2 instances")
    if all(C.dt_path(iid, a.eps, s).exists() for s in a.seeds):
        print(f"[e2b-dt-big] {iid} eps{a.eps:g}: nothing to do", flush=True)
        return 0
    print(f"[e2b-dt-big] {iid}: built in {build_s:.1f}s; waiting for the GPU lock", flush=True)
    out = run_remaining(comp, meta, n, k, a.eps, a.seeds, hw, dt)
    print(f"[e2b-dt-big] done: ran {out['ran']}, already on disk {out['skipped_on_disk']}, "
          f"scope cut {out['scope_cut']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
