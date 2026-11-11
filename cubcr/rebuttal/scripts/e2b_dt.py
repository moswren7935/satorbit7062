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

SCOPE_CUT_S = 900.0
BIG = (12, 100000, 25)


def bitwise_vs_e4v2(r, rec) -> dict:
    s, v2, cert = rec["sampling"], rec["sampler_v2"], rec.get("certificate") or {}
    chk = {
        "estimates": bool(np.array_equal(r.estimates, np.asarray(s["estimates"], dtype=np.float64).reshape(-1))),
        "half_widths": bool(np.array_equal(r.half_widths, np.asarray(s["half_widths"], dtype=np.float64).reshape(-1))),
        "n_perms": int(r.n_perms) == int(s["n_perms"]),
        "n_solves": int(r.n_solves) == int(s["n_solves"]),
        "n_requested": int(r.n_coalitions_requested) == int(s["n_requested"]),
        "n_batches": int(r.n_batches) == int(s["n_batches"]),
        "batch_sizes": list(map(int, r.batch_sizes)) == list(map(int, v2["batch_sizes"])),
    }
    if cert.get("cs_lower") is not None and r.cs_lower is not None:
        chk["cs_lower"] = bool(np.array_equal(r.cs_lower, np.asarray(cert["cs_lower"], dtype=np.float64).reshape(-1)))
        chk["cs_upper"] = bool(np.array_equal(r.cs_upper, np.asarray(cert["cs_upper"], dtype=np.float64).reshape(-1)))
    chk["all"] = all(chk.values())
    return chk


def exact_12(comp, meta, n, k, hw, held=None):

    from cubcr.shapley_sampled import exact_player_table
    from cubcr.shapley import shapley_from_table
    iid = meta["instance_id"]
    p = C.RAW_DT / f"partB_dt_exact__{iid}.json"
    if p.exists():
        return rb._read(p)
    N = n * k
    cm = rb.gpu_lock(f"E2B dt exact {iid}") if held is None else contextlib.nullcontext(held["group_lock_wait_s"])
    with cm as waited:
        t0 = time.perf_counter()
        eng = rb.make_gpu_engine(comp)
        try:
            table = np.asarray(exact_player_table(eng, N, kind="step", n=n, k=k), dtype=np.float64)
        finally:
            eng.close()
        hold = time.perf_counter() - t0
    phi = shapley_from_table(table, N)
    stored = C.e4v2_record(iid, "eps0.005", rb.PAPER_SAMPLER_SEEDS[0])
    sphi = np.asarray(stored["exact"]["phi"], dtype=np.float64).reshape(-1)
    rec = {"exp": "E2-partB", "arm": "dt_exact", "instance_id": iid, "n": n, "k": k, "n_players": N,
           "n_coalitions": int(table.size), "table": table.tolist(), "phi": phi.tolist(),
           "upsilon": float(table[-1] - table[0]), "stored_phi_source": stored["_file"] if "_file" in stored else
           str(C.E4V2_DIR / f"e4__{iid}__eps0.005__s{rb.PAPER_SAMPLER_SEEDS[0]}.json"),
           "max_abs_dev_vs_stored_phi": float(np.max(np.abs(phi - sphi))),
           "bitwise_vs_stored_phi": bool(np.array_equal(phi, sphi)), "lock_wait_s": float(waited),
           "lock_hold_s": float(hold), "hardware": hw, "timestamp": rb.now_iso()}
    rb.write_json(p, rec)
    print(f"[e2b-dt] exact {iid}: Upsilon={rec['upsilon']:.6g} max|phi-stored|={rec['max_abs_dev_vs_stored_phi']:.3g} "
          f"bitwise={rec['bitwise_vs_stored_phi']}", flush=True)
    return rec


def run_one(comp, meta, n, k, eps, seed, hw, exact_phi=None, held=None):
    iid = meta["instance_id"]
    N = n * k
    tb = None if eps == 0.005 else C.DT_TIGHT_TIME_BUDGET
    curve, t_ref = [], [0.0]
    kw = C.dt_sampler_kwargs(eps, seed, n, k, time_budget=tb, callback=C.curve_callback(curve, t_ref))

    def warm(eng):
        C.e4_warmup(eng, n, k, kw["batch_perms"])

    def run(eng):
        t_ref[0] = time.perf_counter()
        return C.run_step_sampler(eng, kw)

    (r, wall), info = C.locked_timed_run(f"E2B dt {iid} eps{eps:g} s{seed}", comp, warm, run, held=held)
    row = C.result_fields(r, wall)
    L, R = C.interval_LR(r.cs_lower, r.cs_upper, r.fp_term)
    cfg = {k_: v for k_, v in kw.items() if k_ != "callback"}
    rec = {"exp": "E2-partB", "arm": "dt_sweep", "instance_id": iid, "n": n, "k": k, "n_players": N,
           "states": int(comp.n_states), "states_requested": meta.get("states_requested"),
           "n_reachable_k": meta.get("n_reachable_k"), "instance_seed": C.DT_INSTANCE_SEED, "seed": seed,
           "eps_abs": eps, "config": cfg, "time_budget_note": ("stored e4v2 value" if tb is None else
                                                              "raised from 180 s (non-binding cap; deviation)"),
           "protocol": "e4: fresh engine + e4 warm-up (k, 16*2*(N+1)) + timed sample_shapley; GPU lock per run; "
                       "reported time = time_to_certify_s (Table 1)",
           **row, "L": L.tolist(), "R": R.tolist(), "curve": curve,
           "requested_invariant": C.requested_invariant(r.n_coalitions_requested, r.n_samples, N),
           "unique_frac": float(r.n_solves) / float(2 ** N) if N <= 60 else None,
           "phi_exact": None if exact_phi is None else list(map(float, exact_phi)),
           "timing": info, "hardware": hw}
    stored = C.e4v2_record(iid, f"eps{eps:g}", seed)
    if stored is not None:
        rec["stored_e4v2"] = {"file": str((C.E4V2_DIR / f"e4__{iid}__eps{eps:g}__s{seed}.json").relative_to(rb.REPO)),
                              "time_to_certify_s": stored.get("time_to_certify_s"),
                              "wall_s": stored["sampling"]["wall_s"], "n_solves": stored["sampling"]["n_solves"],
                              "n_perms": stored["sampling"]["n_perms"], "timestamp": stored.get("timestamp")}
        if C.kwargs_from_e4v2_record(stored) == {k_: v for k_, v in cfg.items() if k_ not in ("eps", "seed", "n", "k")}:
            rec["bitwise_vs_stored"] = bitwise_vs_e4v2(r, stored)
        else:
            rec["bitwise_vs_stored"] = {"all": None, "note": "configuration differs from the stored record"}
    rb.write_json(C.dt_path(iid, eps, seed), rec)
    bw = (rec.get("bitwise_vs_stored") or {}).get("all")
    print(f"[e2b-dt] {iid} eps{eps:g} s{seed}: {r.status} ttc={row['time_to_certify_s']} wall={wall:.3f}s "
          f"perms={r.n_perms} solves={r.n_solves} batches={r.n_batches} hw_max={np.max(r.half_widths):.3g}"
          + ("" if bw is None else f" bitwise={bw}") + f" load1={info['load_snapshot']['loadavg'][0]:.1f}", flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", type=float, nargs="+", default=[0.005])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--models", type=int, nargs="*", default=None, help="indices into e2b_common.DT_MODELS")
    ap.add_argument("--skip-big", action="store_true", help="skip the 100,009-state model")
    ap.add_argument("--lock-mode", choices=["chunk", "run"], default="chunk",
                    help="chunk: the lock is held across consecutive (model, radius) groups of the small models for up to "
                         "--max-hold seconds (the 300-player model at radius < 0.0025: one hold per run); run: one hold per run")
    ap.add_argument("--max-hold", type=float, default=600.0)
    ap.add_argument("--allow-rebuilt-engine", action="store_true",
                    help="accepted for compatibility; local rebuilds are always supported")
    a = ap.parse_args(argv)
    C.RAW_DT.mkdir(parents=True, exist_ok=True)
    hw = C.hw_tags()
    if not hw.get("local_build_complete"):
        raise SystemExit("engine libraries are missing; run make -C csrc all")
    models = [C.DT_MODELS[i] for i in a.models] if a.models is not None else list(C.DT_MODELS)
    models = [m for m in models if m != BIG] + ([BIG] if (BIG in models and not a.skip_big) else [])
    built = {}
    with C.ChunkedLock("E2B dt", a.max_hold) as lock:
        for eps in a.radii:
            for (n, S, k) in models:
                if (n, S, k) not in built:
                    comp, meta, build_s = C.dt_build(n, S, k)
                    iid = meta["instance_id"]
                    if C.e4v2_record(iid, "eps0.005", rb.PAPER_SAMPLER_SEEDS[0]) is None:
                        raise RuntimeError(f"{iid} not among the stored e4v2 instances")
                    built[(n, S, k)] = [comp, meta, build_s, None]
                ent = built[(n, S, k)]
                comp, meta, build_s = ent[0], ent[1], ent[2]
                iid = meta["instance_id"]
                todo = [s for s in a.seeds if not C.dt_path(iid, eps, s).exists()]
                need_exact = n * k <= 16 and ent[3] is None
                if not todo and not need_exact:
                    continue
                cut = C.RAW_DT / f"partB_dt_scopecut__{iid}__eps{eps:g}.json"
                if cut.exists():
                    print(f"[e2b-dt] {iid} eps{eps:g}: scope cut recorded earlier, skipping", flush=True)
                    continue
                big = (n, S, k) == BIG


                if (big and eps < 0.0025) or a.lock_mode == "run":
                    lock.release()
                    held = None
                else:
                    held = lock.ensure()
                if need_exact:
                    ent[3] = exact_12(comp, meta, n, k, hw, held=held)["phi"]
                    if held is not None:
                        held["first"] = False
                for i, seed in enumerate(todo):
                    rec = run_one(comp, meta, n, k, eps, seed, hw, ent[3], held=held)
                    if big and eps < 0.005 and rec["time_to_certify_s"] is not None and \
                            rec["wall_s"] > SCOPE_CUT_S and len(todo) > i + 1:
                        rb.write_json(cut, {"instance_id": iid, "eps_abs": eps, "ran_seed": seed, "wall_s": rec["wall_s"],
                                            "skipped_seeds": todo[i + 1:], "rule": f"one run > {SCOPE_CUT_S:.0f} s",
                                            "timestamp": rb.now_iso()})
                        print(f"[e2b-dt] SCOPE CUT {iid} eps{eps:g}: run took {rec['wall_s']:.0f}s > {SCOPE_CUT_S:.0f}s",
                              flush=True)
                        break
                lock.boundary()
    print("[e2b-dt] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
