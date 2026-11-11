#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import sys
import time

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import numpy as np

import rb
import e2b_common as C


def bitwise_vs_e14(r, rec) -> dict:
    chk = {
        "estimates": bool(np.array_equal(r.estimates, np.asarray(rec["estimates"], dtype=np.float64).reshape(-1))),
        "half_widths": bool(np.array_equal(r.half_widths, np.asarray(rec["half_widths"], dtype=np.float64).reshape(-1))),
        "n_perms": int(r.n_perms) == int(rec["n_perms"]),
        "n_solves": int(r.n_solves) == int(rec["n_solves"]),
    }
    chk["all"] = all(chk.values())
    return chk


def one_run(comp, build_s, phi, ups, div, seed, hw, held):
    n, k, N = C.OC_N, C.OC_K, C.OC_N * C.OC_K
    eps = ups / div
    kw = C.oc_sampler_kwargs(eps, seed)

    def warm(eng):
        C.e4_warmup(eng, n, k, kw["batch_perms"])

    def run(eng):
        return C.run_step_sampler(eng, kw)

    (r, wall), info = C.locked_timed_run(f"E2B oc div{div} s{seed}", comp, warm, run, held=held)
    row = C.result_fields(r, wall)
    L, R = C.interval_LR(r.cs_lower, r.cs_upper, r.fp_term)
    rec = {"exp": "E2-partB", "arm": "oc_sweep", "instance_id": C.OC_IID, "n": n, "k": k, "n_players": N,
           "states": int(comp.n_states), "seed": seed, "eps_div": div, "eps_abs": eps, "upsilon_today": ups,
           "config": {k_: v for k_, v in kw.items()}, "model_build_s": build_s,
           "protocol": "fresh engine + warm-up (k, 16*2*(N+1)) + timed sample_shapley; GPU lock per divisor group; "
                       "stored E14 wall_s = r.wall_time on an engine warm from the 2^20 table",
           **row, "L": L.tolist(), "R": R.tolist(), "phi_exact": phi.tolist(),
           "phi_exact_source": "results/e14/e14__overcooked_cramped_room_k10__exact_step.json",
           "requested_invariant": C.requested_invariant(r.n_coalitions_requested, r.n_samples, N),
           "unique_frac": float(r.n_solves) / float(1 << N),
           "covers_exact_paper": C.covers_paper(r.estimates, r.half_widths, phi),
           "max_abs_err_vs_exact": float(np.max(np.abs(r.estimates - phi))),
           "timing": info, "hardware": hw}
    if div == 20:
        st = C.e14_record(f"cert_gpu_eps_vs_exact_s{seed}")
        rec["stored_e14"] = {"file": f"results/e14/e14__{C.OC_IID}__cert_gpu_eps_vs_exact_s{seed}.json",
                             "wall_s": st["wall_s"], "n_perms": st["n_perms"], "n_solves": st["n_solves"],
                             "eps_abs": st["eps_abs"], "timestamp": st.get("timestamp")}
        rec["bitwise_vs_stored"] = bitwise_vs_e14(r, st)
        rec["eps_bitwise_vs_stored"] = eps == st["eps_abs"]
    rb.write_json(C.oc_path(div, seed), rec)
    bw = (rec.get("bitwise_vs_stored") or {}).get("all")
    print(f"[e2b-oc] div{div} s{seed}: {r.status} wall_internal={r.wall_time:.3f}s outer={wall:.3f}s "
          f"perms={r.n_perms} solves={r.n_solves} ({rec['unique_frac']:.3f} of 2^20) batches={r.n_batches} "
          f"hw_max={np.max(r.half_widths):.3g} covers={rec['covers_exact_paper']}"
          + ("" if bw is None else f" bitwise={bw}"), flush=True)
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--divs", type=int, nargs="+", default=[20])
    ap.add_argument("--seeds", type=int, nargs="*", default=list(rb.PAPER_SAMPLER_SEEDS))
    ap.add_argument("--allow-rebuilt-engine", action="store_true",
                    help="accepted for compatibility; local rebuilds are always supported")
    a = ap.parse_args(argv)
    C.RAW_OC.mkdir(parents=True, exist_ok=True)
    hw = C.hw_tags()
    if not hw.get("local_build_complete"):
        raise SystemExit("engine libraries are missing; run make -C csrc all")
    todo = [(d, s) for d in a.divs for s in a.seeds if not C.oc_path(d, s).exists()]
    if not todo:
        print("[e2b-oc] nothing to do", flush=True)
        return 0
    comp, meta, build_s = C.oc_build()
    exact = C.e14_record("exact_step")
    phi = np.asarray(exact["shapley"], dtype=np.float64).reshape(-1)
    n, k, N = C.OC_N, C.OC_K, C.OC_N * C.OC_K
    if int(comp.n_states) != int(exact["states"]):
        raise RuntimeError(f"Overcooked k=10 rebuilt with {comp.n_states} states != stored {exact['states']}")
    tp = C.RAW_OC / "partB_oc_task.json"
    ups = None
    for div in sorted({d for d, _ in todo}):
        seeds = [s_ for d, s_ in todo if d == div]
        with C.group_lock(f"E2B oc div{div}") as held:
            if ups is None:
                ups, vv, linfo = C.locked_upsilon("E2B oc ups", comp, C.step_upsilon_masks(n, k), k, held=held)
                trec = rb._read(tp) if tp.exists() else {"instance_id": C.OC_IID, "checks": []}
                trec["checks"].append({"timestamp": rb.now_iso(), "build_s": build_s, "n_states": int(comp.n_states),
                                       "upsilon_today": ups, "v_empty_full": vv,
                                       "upsilon_stored": exact["attributable_value"],
                                       "upsilon_bitwise": ups == exact["attributable_value"], "lock": linfo,
                                       "load_snapshot": rb.load_snapshot(), "git_commit": hw["git_commit"]})
                rb.write_json(tp, trec)
                if ups != exact["attributable_value"]:
                    raise RuntimeError(f"Upsilon {ups!r} != stored {exact['attributable_value']!r}")
                print(f"[e2b-oc] model built in {build_s:.1f}s, S={comp.n_states}, Upsilon={ups!r}", flush=True)
            for seed in seeds:
                one_run(comp, build_s, phi, ups, div, seed, hw, held)
    print("[e2b-oc] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
