#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import contextlib
import sys
import time

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import numpy as np

import rb
import e2b_common as C


def bitwise_vs_stored(r, rec) -> dict:
    cert = rec.get("certificate") or {}
    chk = {
        "estimates": bool(np.array_equal(r.estimates, np.asarray(rec["shapley_sampled"], dtype=np.float64))),
        "half_widths": bool(np.array_equal(r.half_widths, np.asarray(rec["half_widths"], dtype=np.float64))),
        "n_perms": int(r.n_perms) == int(rec["n_perms"]),
        "n_solves": int(r.n_solves) == int(rec["n_solves"]),
        "n_units": int(r.n_samples) == int(rec["n_units"]),
        "n_batches": int(r.n_batches) == int(rec["n_batches"]),
        "batch_sizes": list(map(int, r.batch_sizes)) == list(map(int, rec["batch_sizes"])),
    }
    if cert.get("cs_lower") is not None and r.cs_lower is not None:
        chk["cs_lower"] = bool(np.array_equal(r.cs_lower, np.asarray(cert["cs_lower"], dtype=np.float64)))
        chk["cs_upper"] = bool(np.array_equal(r.cs_upper, np.asarray(cert["cs_upper"], dtype=np.float64)))
    chk["all"] = all(chk.values())
    return chk


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--divs", type=int, nargs="+", default=[20])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--only-iid", nargs="*", default=None)
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many new runs (0 = no limit)")
    ap.add_argument("--lock-mode", choices=["chunk", "run"], default="chunk",
                    help="chunk: hold the GPU lock across consecutive (task, divisor) groups for up to --max-hold seconds "
                         "(each run still: fresh engine + warm-up + timed run); run: one hold per run")
    ap.add_argument("--max-hold", type=float, default=600.0, help="seconds per lock hold in chunk mode")
    ap.add_argument("--allow-rebuilt-engine", action="store_true",
                    help="accepted for compatibility; local rebuilds are always supported")
    a = ap.parse_args(argv)
    for s in a.seeds:
        if s not in rb.PAPER_SAMPLER_SEEDS:
            raise SystemExit(f"seed {s} is not a paper sampler seed")
    C.RAW_GRID.mkdir(parents=True, exist_ok=True)
    hw = C.hw_tags()
    if not hw.get("local_build_complete"):
        raise SystemExit("engine libraries are missing; run make -C csrc all")
    tasks = rb.paper_tasks()
    assert len(tasks) == 60
    if a.only_iid:
        tasks = [t for t in tasks if t["iid"] in set(a.only_iid)]
    cache = {}
    n_new = 0
    t_start = time.perf_counter()
    with C.ChunkedLock("E2B grid", a.max_hold) as lock:
      for div in a.divs:
        for t in tasks:
            if div not in C.grid_divs_for(t, [div]):
                continue
            todo = [s for s in a.seeds if not C.grid_path(t["iid"], div, s).exists()]
            if not todo:
                continue
            iid, n, k = t["iid"], t["n"], t["k"]
            if iid not in cache:
                comp, iid2, meta, build_s = rb.build_task(t)
                cache[iid] = {"comp": comp, "build_s": build_s, "ups": None}
            ent = cache[iid]
            comp, build_s = ent["comp"], ent["build_s"]
            held = lock.ensure() if a.lock_mode == "chunk" else None
            if ent["ups"] is None:
                masks = np.array([0, (1 << n) - 1], dtype=np.uint64)
                ups, vv, linfo = C.locked_upsilon(f"E2B ups {iid}", comp, masks, k, held=held)
                ups_ok = ups == t["upsilon"]
                eps20_ok = (ups / 20) == t["eps20"]
                tp = C.grid_task_path(iid)
                trec = rb._read(tp) if tp.exists() else {"iid": iid, "checks": []}
                trec["checks"].append({"timestamp": rb.now_iso(), "build_s": build_s, "upsilon_today": ups,
                                       "v_empty_full": vv, "upsilon_stored": t["upsilon"], "upsilon_bitwise": ups_ok,
                                       "eps20_bitwise": eps20_ok, "lock": linfo, "load_snapshot": rb.load_snapshot(),
                                       "git_commit": hw["git_commit"]})
                rb.write_json(tp, trec)
                if not ups_ok:
                    raise RuntimeError(f"{iid}: Upsilon {ups!r} != stored {t['upsilon']!r}")
                ent["ups"] = ups
                print(f"[e2b-grid] {iid}: build {build_s:.3f}s ups={ups:.6g} (bitwise={ups_ok}, "
                      f"eps20 bitwise={eps20_ok})", flush=True)
            ups = ent["ups"]
            eps = ups / div
            phi = t["phi_exact"]
            phi_src = (t["exact_gpu"] or {}).get("_file") if t["has_exact_gpu"] else \
                ((t["exact_cpu"] or {}).get("_file") if t["has_exact_cpu"] else None)
            for seed in todo:
                def warm(eng):
                    rb.paper_warmup(eng, n, k)

                def run(eng):
                    r, wall, _ = rb.run_paper_sampler(eng, n, k, eps, seed)
                    return r, wall

                (r, wall), info = C.locked_timed_run(f"E2B grid {iid} div{div} s{seed}", comp, warm, run, held=held)
                row = C.result_fields(r, wall)
                L, R = C.interval_LR(r.cs_lower, r.cs_upper, r.fp_term)
                stored = t["cert_gpu"].get(seed)
                rec = {"exp": "E2-partB", "arm": "grid_sweep", "instance_id": iid, "n": n, "states": t["states"],
                       "states_target": t["states_target"], "k": k, "gen_seed": t["gen_seed"], "seed": seed,
                       "cell": rb.cell_label(t), "eps_div": div, "eps_abs": eps, "upsilon_today": ups,
                       "upsilon_stored": t["upsilon"], "eps_formula": "Upsilon_tilde / eps_div",
                       "config": {k_: v for k_, v in rb.paper_sampler_kwargs(eps, seed, n, k).items()},
                       "protocol": "e9v3: separate Upsilon engine; fresh engine + rb.paper_warmup + timed "
                                   "sample_shapley (outer perf_counter); GPU lock held per run or across "
                                   "consecutive (task, divisor) groups for <= ~10 min (timing.lock_mode)",
                       "model_build_s": build_s, **row, "L": L.tolist(), "R": R.tolist(),
                       "unique_frac": float(r.n_solves) / float(1 << n),
                       "requested_invariant": C.requested_invariant(r.n_coalitions_requested, r.n_samples, n),
                       "phi_exact": None if phi is None else phi.tolist(), "phi_exact_source": phi_src,
                       "stored_cert_gpu": None if stored is None else {
                           "file": stored["_file"], "wall_s": stored["wall_s"], "n_units": stored["n_units"],
                           "n_solves": stored["n_solves"], "n_perms": stored["n_perms"],
                           "timestamp": stored.get("timestamp")},
                       "timing": info, "hardware": hw}
                if div == 20 and stored is not None:
                    rec["bitwise_vs_stored"] = bitwise_vs_stored(r, stored)
                    rec["eps_bitwise_vs_stored"] = eps == stored["eps_abs"]
                rb.write_json(C.grid_path(iid, div, seed), rec)
                n_new += 1
                bw = rec.get("bitwise_vs_stored", {}).get("all")
                print(f"[e2b-grid] {iid} div{div} s{seed}: {r.status} wall={wall:.3f}s m={r.n_samples} "
                      f"solves={r.n_solves} req={r.n_coalitions_requested} batches={r.n_batches} "
                      f"hw_max={np.max(r.half_widths):.3g} eps={eps:.3g}"
                      + ("" if bw is None else f" bitwise={bw}")
                      + f" load1={info['load_snapshot']['loadavg'][0]:.1f} lock_wait={info['lock_wait_s']:.1f}s",
                      flush=True)
                if a.max_runs and n_new >= a.max_runs:
                    print(f"[e2b-grid] max-runs reached ({n_new})", flush=True)
                    return 0
            lock.boundary()
    print(f"[e2b-grid] done: {n_new} new runs in {time.perf_counter() - t_start:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
