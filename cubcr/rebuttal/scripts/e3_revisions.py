#!/usr/bin/env python3


import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import sys
import time

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e3_engine as E
import e3_models as M
import e3_common as C

OUT = C.RAW / "revisions"
N_RANDOM = 10000


def rpath(cfg, R, rep):
    return OUT / f"{cfg['name']}__{R}__rep{rep}.json"


def types_for(cfg):
    if cfg["family"] == "oc":
        return ["R1", "R2", "R3"]
    if cfg["family"] == "grid" and (cfg["n"], cfg["states_target"], cfg["k"]) in C.REVISION_GRID:
        return ["R1", "R2", "R3"]
    return ["R1"]


def attribute(eng, cfg):

    out = {}
    if C.needs_upsilon(cfg):
        t0 = time.perf_counter()
        vv = eng.solve(C.endpoint_masks(cfg), cfg["k"])
        out["ups_solve_s"] = time.perf_counter() - t0
        ups = float(vv[1] - vv[0])
    else:
        out["ups_solve_s"] = 0.0
        ups = None
    eps = C.eps_from_upsilon(cfg, ups)
    gc0 = int(eng.stats["graph_captures"]) if eng.stats else 0
    eng.reset_cum()
    r, wall = C.run_sampler(eng, cfg, eps)
    out.update({"upsilon": ups, "eps": eps, "p5_s": wall, "p5_graph_captures": int(eng.stats["graph_captures"]) - gc0})
    out.update(C.p5_split(eng, wall))
    out["p5_calls"] = list(eng.calls)
    return out, r


def make_revision(cfg, comp, R):

    t0 = time.perf_counter()
    if R == "R3":
        tr, info = M.revise_structure_triplets(comp, C.REVISION_SEED)
        edit_s = time.perf_counter() - t0
        c2 = M.assemble_csr(comp, tr["rows"], tr["cols"], tr["vals"])
    else:
        c2, info = M.REVISERS[R](comp, C.REVISION_SEED)
        edit_s = time.perf_counter() - t0
        tr = None
    return c2, info, edit_s, tr


def apply_update(eng, R, c2):
    t0 = time.perf_counter()
    if R == "R1":
        u = eng.update_policy(c2)
    elif R == "R2":
        u = eng.update_values(c2)
    else:
        u = eng.rebuild(c2)
    return time.perf_counter() - t0, u


def prepare(cfg, reps, force=False):

    types = types_for(cfg)
    todo = {R: [r for r in range(1, reps + 1) if force or not rpath(cfg, R, r).exists()] for R in types}
    if not any(todo.values()):
        print(f"[{rb.now_iso()}] {cfg['name']}: all revision records present", flush=True)
        return None
    comp, meta, _ = C.build(cfg)
    revs = {}
    for R in types:
        c2, info, edit_s, tr = make_revision(cfg, comp, R)
        info["validity"] = M.validate_compiled(c2)
        revs[R] = (c2, info, edit_s, tr)
        if info.get("degenerate"):
            rec = {"config": cfg["name"], "revision": R, "status": "NOT APPLICABLE", "info": info,
                   "reason": "every transition row of this model is deterministic (one successor with probability 1); "
                             "a sparsity-preserving change of P(s'|s,a) followed by row renormalisation is the identity, "
                             "so R2 cannot be exercised without changing the sparsity (that is R3)",
                   "timestamp": rb.now_iso(), "hardware": None}
            rb.write_json(OUT / f"{cfg['name']}__{R}__NA.json", rec)
            todo[R] = []
            print(f"[{rb.now_iso()}] {cfg['name']} {R}: NOT APPLICABLE (degenerate revision)", flush=True)
    if not any(todo.values()):
        return None
    return {"cfg": cfg, "types": types, "todo": todo, "comp": comp, "meta": meta, "revs": revs}


def run_prepared(P, waited):

    cfg, types, todo, comp, meta, revs = P["cfg"], P["types"], P["todo"], P["comp"], P["meta"], P["revs"]
    prov = E.provenance()
    rng_masks = {}
    if True:
        t_lock = time.perf_counter()
        print(f"[{rb.now_iso()}] {cfg['name']}: GPU work starts (lock wait {waited:.0f}s); types {[R for R in types if todo[R]]}", flush=True)
        E.context_init(0)
        hw = rb.hardware_tags()
        eng = E.E3Engine(comp)
        base, rb0 = attribute(eng, cfg)
        base_vals = eng.solve(C.endpoint_masks(cfg), cfg["k"])
        for R in types:
            c2, info, edit_s, tr = revs[R]
            for rep in todo[R]:
                p1_s = 0.0
                if R == "R3":
                    t0 = time.perf_counter()
                    c2b = M.assemble_csr(comp, tr["rows"], tr["cols"], tr["vals"])
                    p1_s = time.perf_counter() - t0
                    assert c2b.fingerprint() == c2.fingerprint()
                    c2 = c2b
                rec = {"config": cfg["name"], "family": cfg["family"], "revision": R, "rep": rep, "iid": meta.get("instance_id"),
                       "gen_seed": C.GEN_SEED, "sampler_seed": C.SAMPLER_SEED, "revision_seed": C.REVISION_SEED,
                       "n": cfg["n"], "k": cfg["k"], "players": cfg["players"], "states": int(comp.n_states),
                       "nnz_before": int(comp.nnz), "nnz_after": int(c2.nnz), "info": info,
                       "edit_s": edit_s, "p1_reassembly_s": p1_s, "gpu_lock_wait_s": float(waited)}
                rec["load_before"] = rb.load_snapshot()
                rec["mem_free_before"] = E.mem_info(0)[0]

                upd_s, u = apply_update(eng, R, c2)
                rec["update_s"] = upd_s
                rec["update"] = u
                att, r = attribute(eng, cfg)
                rec.update({k_: v for k_, v in att.items()})
                rec["reattr_s"] = upd_s + att["ups_solve_s"] + att["p5_s"]
                rec["reattr_incl_p1_s"] = rec["reattr_s"] + p1_s
                rec["p5_status"] = r.status; rec["p5_certified"] = bool(r.certified)
                rec["p5_n_perms"] = int(r.n_perms); rec["p5_n_solves"] = int(r.n_solves); rec["p5_n_units"] = int(r.n_samples)
                rec["p5_n_batches"] = int(r.n_batches); rec["p5_batch_sizes"] = list(map(int, r.batch_sizes))
                rec["p5_result"] = rb.result_row(r, att["p5_s"])
                rec["upsilon_base"] = base["upsilon"]

                t0 = time.perf_counter()
                fresh = rb.make_gpu_engine(c2)
                rec["fresh_create_submitted_s"] = time.perf_counter() - t0
                chk = {}
                ev = C.endpoint_masks(cfg)
                a_end, f_end = eng.solve(ev, cfg["k"]), fresh.solve(ev, cfg["k"])
                chk["endpoints_bitwise"] = bool(np.array_equal(a_end, f_end))
                if att["upsilon"] is not None:
                    chk["upsilon_bitwise"] = bool(float(f_end[1] - f_end[0]) == att["upsilon"])
                key = (rep,)
                if key not in rng_masks:
                    rng_masks[key] = C.random_masks(cfg, N_RANDOM, 7000 + rep)
                rm = rng_masks[key]
                a_v, f_v = eng.solve(rm, cfg["k"]), fresh.solve(rm, cfg["k"])
                chk["random_coalitions"] = int(N_RANDOM)
                chk["random_bitwise"] = bool(np.array_equal(a_v, f_v))
                chk["random_max_abs_diff"] = float(np.max(np.abs(a_v - f_v)))
                b_v = None
                if cfg["family"] != "dt" or rep == 1:
                    rf, wf = C.run_sampler(fresh, cfg, att["eps"])
                    chk["sampler_same_as_fresh"] = C.same_result(r, rf)
                    chk["fresh_p5_s"] = wf
                else:
                    chk["sampler_same_as_fresh"] = None
                chk["revision_changes_values"] = None
                fresh.close()
                rec["correctness"] = chk

                t0 = time.perf_counter()
                if R == "R1":
                    eng.update_policy(comp)
                elif R == "R2":
                    eng.update_values(comp)
                else:
                    eng.rebuild(comp)
                back = eng.solve(C.endpoint_masks(cfg), cfg["k"])
                rec["revert_endpoints_bitwise"] = bool(np.array_equal(back, base_vals))
                b_v = eng.solve(rm, cfg["k"])
                chk["revision_changes_values"] = bool(not np.array_equal(b_v, a_v))
                if cfg["family"] != "dt":
                    attribute(eng, cfg)
                rec["revert_s"] = time.perf_counter() - t0
                rec["load_after"] = rb.load_snapshot()
                rec["gpu_apps"] = C.gpu_apps()
                rec["hardware"] = hw
                rec["e3_provenance"] = prov
                rec["base"] = {k_: v for k_, v in base.items() if k_ != "p5_calls"}
                rec["timestamp"] = rb.now_iso()
                rb.write_json(rpath(cfg, R, rep), rec)
                ok = chk["endpoints_bitwise"] and chk["random_bitwise"] and chk["sampler_same_as_fresh"] in (True, None)
                print(f"[{rb.now_iso()}] {cfg['name']} {R} rep{rep}: update {upd_s * 1e3:.1f} ms, U {att['ups_solve_s'] * 1e3:.1f} ms, "
                      f"P5 {att['p5_s']:.3f} s, reattr {rec['reattr_s']:.3f} s, fresh create {rec['fresh_create_submitted_s'] * 1e3:.1f} ms, "
                      f"bitwise={ok} changes={chk['revision_changes_values']} revert_ok={rec['revert_endpoints_bitwise']}", flush=True)
        eng.close()
        print(f"[{rb.now_iso()}] {cfg['name']}: GPU work took {time.perf_counter() - t_lock:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = [c for c in C.all_configs() if a.only is None or c["name"] in a.only]

    order = sorted(cfgs, key=lambda c: (len(types_for(c)) != 3, c["family"] == "dt", c["n"], c["states_target"] or 0, c["k"]))
    print(f"[{rb.now_iso()}] E3 revisions: {[c['name'] for c in order]}", flush=True)


    groups = [[c for c in order if c["family"] == "oc"],
              [c for c in order if c["family"] == "grid" and len(types_for(c)) == 3],
              [c for c in order if c["family"] == "grid" and len(types_for(c)) == 1],
              [c for c in order if c["family"] == "dt"]]
    groups = [g for g in groups if g]
    for g in groups:
        prepared = [P for P in (prepare(c, a.reps, a.force) for c in g) if P is not None]
        if not prepared:
            continue
        with rb.gpu_lock(f"E3 revisions {'+'.join(P['cfg']['name'] for P in prepared)}") as waited:
            t0 = time.perf_counter()
            for P in prepared:
                run_prepared(P, waited)
            print(f"[{rb.now_iso()}] held the GPU lock {time.perf_counter() - t0:.0f}s for "
                  f"{[P['cfg']['name'] for P in prepared]}", flush=True)
    print(f"[{rb.now_iso()}] E3 revisions: done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
