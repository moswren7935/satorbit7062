#!/usr/bin/env python3


from __future__ import annotations

import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import argparse
import json
import math
import os
import sys
import time
import traceback
import zlib
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))
import rb
import e1_models
import e1_ext

OUT = rb.results_dir("E1")
RAW = OUT / "raw" / "precision"
U64 = 2.0 ** -53
REPO = rb.REPO
MIN_COAL = 256


def audit_specs():
    out = []
    for s in e1_models.specs_tables13():
        out.append(dict(s, audit="tables13", kinds=["agent"]))
    tasks = {(t["n"], t["states_target"], t["k"], t["gen_seed"]): t for t in rb.paper_tasks(include_records=False)}
    t16 = tasks[(16, 1000, 50, 20260829)]
    out.append(dict(family="primary", model_id=t16["iid"], n=16, states_target=1000, k=50, gen_seed=20260829,
                    pattern="none", hazard=0.0, audit="primary_n16", kinds=["agent"], source="paper task",
                    cached_table=True))
    t32 = tasks[(32, 10000, 100, 20260829)]
    out.append(dict(family="primary", model_id=t32["iid"], n=32, states_target=10000, k=100, gen_seed=20260829,
                    pattern="none", hazard=0.0, audit="primary_largest", kinds=["agent"], source="paper task",
                    min_coal=128))
    for s in e1_models.specs_prism9():
        out.append(dict(s, audit="prism9", kinds=["agent", "step"]))
    for s in e1_models.specs_decision6():
        out.append(dict(s, audit="decision6_supplement", kinds=["step"]))
    out.append(dict(e1_models.specs_overcooked()[0], audit="overcooked_k10_supplement", kinds=["step"]))
    return out


def _name(spec, kind):
    mid = spec.get("model_id") or f"{spec['family']}_n{spec.get('n')}_S{spec.get('states_target')}_k{spec['k']}"
    return f"{mid}__{kind}"


def build(spec):
    comp, csg, info = e1_models.build(spec)
    return comp, info["iid"]


def design_masks(spec, comp, kind):


    n, k = int(comp.n_agents), int(spec["k"])
    d = n if kind == "agent" else n * k
    seed = zlib.crc32(_name(spec, kind).encode()) & 0x7FFFFFFF
    rng = np.random.default_rng(seed)
    if kind == "agent":
        onehot = (np.uint64(1) << np.arange(n, dtype=np.uint64)).reshape(-1, 1)
    else:
        from cubcr.masks import step_mask_matrix
        onehot = np.stack([step_mask_matrix([p], n, k) for p in range(d)], axis=0)
    kc = onehot.shape[1]
    keys = {}
    rows = []

    def add(row):
        key = row.tobytes()
        if key not in keys:
            keys[key] = len(rows)
            rows.append(row.copy())
        return keys[key]

    add(np.zeros(kc, dtype=np.uint64))
    add(np.bitwise_or.reduce(onehot, axis=0))
    if kind == "agent" and d <= 3:
        for m in range(1 << n):
            add(np.array([m], dtype=np.uint64))
    pairs = []
    target = int(spec.get("min_coal", MIN_COAL))
    if d <= 62:
        target = min(target, 1 << d)
    while len(rows) < target or not pairs:
        perm = rng.permutation(d)
        idx = []
        for pm in (perm, perm[::-1]):
            acc = np.zeros(kc, dtype=np.uint64)
            seq = [add(acc)]
            for p in pm:
                acc = acc | onehot[p]
                seq.append(add(acc))
            idx.append(np.array(seq))
        pairs.append((idx[0], idx[1], perm))
        if len(pairs) > 4096:
            break
    R = np.stack(rows, axis=0)
    masks = R[:, 0].copy() if kind == "agent" else np.ascontiguousarray(R.T)
    return masks, pairs


def mp_subset(spec, comp, B):
    k = int(spec["k"])
    heavy = comp.n_rows * k > 5e6
    n_mp = 3 if heavy else 10
    rng = np.random.default_rng(7)
    rest = rng.choice(np.arange(2, B), size=min(B - 2, n_mp - 2), replace=False) if B > 2 else np.array([], int)
    return np.concatenate([[0, 1], np.sort(rest)]).astype(int)[:n_mp]


def gpu_phase(spec):

    from cubcr.engine_gpu import GPUEngine
    comp, iid = build(spec)
    res = {}
    for kind in spec["kinds"]:
        p = RAW / f"{_name(spec, kind)}__gpu.npz"
        if p.exists():
            continue
        masks, pairs = design_masks(spec, comp, kind)
        k = int(spec["k"])
        with rb.gpu_lock(f"E1 precision {iid} {kind}"):
            eng = GPUEngine(comp, dtype="fp64", use_graph=True)
            v = np.asarray(eng.solve(masks, k), dtype=np.float64)
            v2 = np.asarray(eng.solve(masks, k), dtype=np.float64)
            eng.close()
        stored = np.full(v.shape, np.nan)
        src = "none"
        if spec.get("table_file"):
            tab = np.load(REPO / spec["table_file"], mmap_mode="r")
            stored = np.asarray(tab[masks.astype(np.int64)], dtype=np.float64)
            src = spec["table_file"]
        elif spec.get("cached_table"):
            t = [x for x in rb.paper_tasks(include_records=False) if x["iid"] == iid][0]
            tab = rb.exact_table(t)
            stored = np.asarray(tab[masks.astype(np.int64)], dtype=np.float64)
            src = "rebuttal/cache/tables (rb.exact_table)"
        elif spec["audit"] == "prism9" and kind == "agent":
            d = json.load(open(REPO / "results" / "e3b" / f"e3__{iid}.json"))
            gv = {int(r["mask"]): float(r["gpu_value"]) for r in d["prism"]["rows"]}
            stored = np.array([gv.get(int(m), np.nan) for m in masks], dtype=np.float64)
            src = f"results/e3b/e3__{iid}.json prism.rows[*].gpu_value"
        RAW.mkdir(parents=True, exist_ok=True)
        np.savez(p, masks=masks, v_gpu=v, v_gpu_repeat=v2, v_stored=stored, stored_source=src,
                 hardware=json.dumps(rb.hardware_tags()), load=json.dumps(rb.load_snapshot()), time=rb.now_iso())
        res[kind] = dict(bitwise_repeat=bool(np.array_equal(v, v2)),
                         stored_bitwise=None if src == "none" else bool(np.array_equal(v, stored)))
        print(f"[gpu] {iid} {kind}: B={v.size} repeat_bitwise={res[kind]['bitwise_repeat']} stored={src} "
              f"stored_bitwise={res[kind]['stored_bitwise']}", flush=True)
    return res


def gpu_phase_batched(specs):


    from cubcr.engine_gpu import GPUEngine
    todo = []
    for spec in specs:
        comp, iid = build(spec)
        for kind in spec["kinds"]:
            p = RAW / f"{_name(spec, kind)}__gpu.npz"
            if p.exists():
                continue
            masks, pairs = design_masks(spec, comp, kind)
            todo.append((spec, comp, iid, kind, masks, p))
    if not todo:
        return
    results = []
    with rb.gpu_lock(f"E1 precision audit ({len(todo)} model/mask sets)"):
        load = rb.load_snapshot()
        t0 = time.perf_counter()
        for spec, comp, iid, kind, masks, p in todo:
            eng = GPUEngine(comp, dtype="fp64", use_graph=True)
            v = np.asarray(eng.solve(masks, int(spec["k"])), dtype=np.float64)
            v2 = np.asarray(eng.solve(masks, int(spec["k"])), dtype=np.float64)
            eng.close()
            results.append((spec, iid, kind, masks, p, v, v2))
        gpu_s = time.perf_counter() - t0
    print(f"[gpu] {len(todo)} sets solved in {gpu_s:.1f}s under one lock", flush=True)
    hw = json.dumps(rb.hardware_tags())
    for spec, iid, kind, masks, p, v, v2 in results:
        stored = np.full(v.shape, np.nan)
        src = "none"
        if spec.get("table_file"):
            tab = np.load(REPO / spec["table_file"], mmap_mode="r")
            stored = np.asarray(tab[masks.astype(np.int64)], dtype=np.float64)
            src = spec["table_file"]
        elif spec.get("cached_table"):
            t = [x for x in rb.paper_tasks(include_records=False) if x["iid"] == iid][0]
            tab = rb.exact_table(t)
            stored = np.asarray(tab[masks.astype(np.int64)], dtype=np.float64)
            src = "rebuttal/cache/tables (rb.exact_table)"
        elif spec["audit"] == "prism9" and kind == "agent":
            d = json.load(open(REPO / "results" / "e3b" / f"e3__{iid}.json"))
            gv = {int(r["mask"]): float(r["gpu_value"]) for r in d["prism"]["rows"]}
            stored = np.array([gv.get(int(m), np.nan) for m in masks], dtype=np.float64)
            src = f"results/e3b/e3__{iid}.json prism.rows[*].gpu_value"
        np.savez(p, masks=masks, v_gpu=v, v_gpu_repeat=v2, v_stored=stored, stored_source=src, hardware=hw,
                 load=json.dumps(load), time=rb.now_iso(), gpu_s_all_sets=gpu_s)
        print(f"[gpu] {iid} {kind}: B={v.size} repeat_bitwise={np.array_equal(v, v2)} stored={src} "
              f"stored_bitwise={None if src == 'none' else bool(np.array_equal(v, stored))}", flush=True)


def cpu_phase(spec):
    try:
        os.nice(19)
    except Exception:
        pass
    out = []
    try:
        comp, iid = build(spec)
        struct = e1_ext.TemplateStructure(comp)
        k = int(spec["k"])
        for kind in spec["kinds"]:
            p = RAW / f"{_name(spec, kind)}__ext.npz"
            if p.exists():
                out.append(str(p))
                continue
            masks, pairs = design_masks(spec, comp, kind)
            B = masks.shape[-1]
            t0 = time.perf_counter()
            v_ld = e1_ext.values_numpy(struct, masks, k, np.longdouble)
            t_ld = time.perf_counter() - t0
            t0 = time.perf_counter()
            v_64 = e1_ext.values_numpy(struct, masks, k, np.float64)
            t_64 = time.perf_counter() - t0
            sub = mp_subset(spec, comp, B)
            mpr = e1_ext.MpReference(struct, 50)
            import mpmath
            v_mp_str = []
            ld_mp = []
            t0 = time.perf_counter()
            with mpmath.workdps(50):
                for i in sub:
                    m = int(masks[i]) if kind == "agent" else [int(x) for x in masks[:, i]]
                    vm = mpr.value(m, k)
                    num, den = v_ld[i].as_integer_ratio()
                    ld_mp.append(float(abs(mpmath.mpf(num) / den - vm)))
                    v_mp_str.append(mpmath.nstr(vm, 45))
            t_mp = time.perf_counter() - t0

            hi = v_ld.astype(np.float64)
            lo = (v_ld - hi.astype(np.longdouble)).astype(np.float64)
            np.savez(p, masks=masks, v_ld_hi=hi, v_ld_lo=lo, v_fp64_mine=v_64, mp_idx=sub,
                     v_mp=np.array(v_mp_str), ld_minus_mp_abs=np.array(ld_mp),
                     pair_fwd=np.stack([q[0] for q in pairs]), pair_rev=np.stack([q[1] for q in pairs]),
                     pair_perm=np.stack([q[2] for q in pairs]), t_ld=t_ld, t_64=t_64, t_mp=t_mp,
                     iid=iid, k=k, kind=kind)
            print(f"[cpu] {iid} {kind}: B={B} ld {t_ld:.1f}s fp64 {t_64:.1f}s mp({len(sub)}) {t_mp:.1f}s "
                  f"max|ld-mp|={max(ld_mp):.2e}", flush=True)
            out.append(str(p))
    except Exception:
        print(f"[cpu] ERROR {spec.get('model_id')}: {traceback.format_exc()}", flush=True)
    return out


def summarize(specs):
    import pandas as pd
    from cubcr.fp_error import forward_error_bound
    rows, coal_rows = [], []
    for spec in specs:
        comp, iid = build(spec)
        k = int(spec["k"])
        fp = forward_error_bound(comp, k, dtype="fp64")
        for kind in spec["kinds"]:
            pg = RAW / f"{_name(spec, kind)}__gpu.npz"
            pe = RAW / f"{_name(spec, kind)}__ext.npz"
            if not (pg.exists() and pe.exists()):
                rows.append(dict(audit=spec["audit"], model_id=iid, mask_kind=kind, status="MISSING"))
                continue
            g = np.load(pg, allow_pickle=False)
            e = np.load(pe, allow_pickle=False)
            assert np.array_equal(g["masks"], e["masks"])
            v_gpu = g["v_gpu"]
            v_ld = e["v_ld_hi"].astype(np.longdouble) + e["v_ld_lo"].astype(np.longdouble)
            diff = np.abs(v_gpu.astype(np.longdouble) - v_ld)
            e_max = float(diff.max())
            arg = int(np.argmax(diff))
            mine = float(np.max(np.abs(e["v_fp64_mine"] - v_gpu)))
            stored = g["v_stored"]
            has_stored = bool(np.all(np.isfinite(stored)))

            zd = 0.0
            n_pairs = int(e["pair_fwd"].shape[0])
            d = int(e["pair_perm"].shape[1])
            for r in range(n_pairs):
                perm = e["pair_perm"][r]
                vals64 = np.stack([v_gpu[e["pair_fwd"][r]], v_gpu[e["pair_rev"][r]]])
                valsld = np.stack([v_ld[e["pair_fwd"][r]], v_ld[e["pair_rev"][r]]])
                perms = np.stack([perm, perm[::-1]])
                m64 = np.empty((2, d)); mld = np.empty((2, d), dtype=np.longdouble)
                m64[np.arange(2)[:, None], perms] = np.diff(vals64, axis=1)
                mld[np.arange(2)[:, None], perms] = np.diff(valsld, axis=1)
                Z = m64.reshape(-1, 1, d).mean(axis=0)[0]
                D = (mld[0] + mld[1]) / 2
                zd = max(zd, float(np.max(np.abs(Z.astype(np.longdouble) - D))))
            ups = float(v_gpu[1] - v_gpu[0])
            implied = 2 * e_max + 2 * U64 * (abs(ups) + 2 * e_max)
            rows.append(dict(
                audit=spec["audit"], family=spec["family"], model_id=iid, mask_kind=kind, n_agents=int(comp.n_agents),
                n_players=d, n_states=int(comp.n_states), n_rows=int(comp.n_rows), nnz=int(comp.nnz), k=k,
                n_coalitions=int(v_gpu.size), n_pairs=n_pairs, v_empty=float(v_gpu[0]), v_full=float(v_gpu[1]),
                upsilon_fp=ups, e_max=e_max, e_max_coalition_value=float(v_gpu[arg]),
                e_max_over_ulp1=e_max / U64 / 2, max_abs_fp64_mine_minus_gpu=mine,
                gpu_repeat_bitwise=bool(np.array_equal(g["v_gpu"], g["v_gpu_repeat"])),
                stored_source=str(g["stored_source"]), stored_bitwise=(bool(np.array_equal(v_gpu, stored)) if has_stored else None),
                n_mp=int(e["mp_idx"].size), max_abs_ld_minus_mp=float(np.max(e["ld_minus_mp_abs"])),
                observed_max_abs_Z_minus_D=zd, implied_bound_Z_minus_D=implied,
                eta_v=fp.eta_v, eps_upsilon=fp.eps_upsilon, fp_term_eps_X=fp.eps_unit(2), M=fp.M, rho=fp.rho,
                eta_v_over_e_max=(fp.eta_v / e_max if e_max > 0 else float("inf")),
                t_longdouble_s=float(e["t_ld"]), t_mp_s=float(e["t_mp"]), status="OK"))
            for i in range(v_gpu.size):
                coal_rows.append(dict(model_id=iid, mask_kind=kind, idx=i,
                                      mask=(hex(int(g["masks"][i])) if kind == "agent" else
                                            ";".join(hex(int(x)) for x in g["masks"][:, i])),
                                      v_gpu=repr(float(v_gpu[i])), v_ext=repr(float(v_ld[i])),
                                      abs_err=float(diff[i])))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "precision_audit.csv", index=False)
    pd.DataFrame(coal_rows).to_csv(OUT / "raw" / "precision_coalitions.csv.gz", index=False, compression="gzip")
    ok = df[df["status"] == "OK"]
    req = ok[ok["audit"].isin(["tables13", "primary_n16", "primary_largest", "prism9"])]
    e_req = float(req["e_max"].max())
    e_all = float(ok["e_max"].max())

    def beta_from(e):
        x = 10 * (2 * e)
        return 10.0 ** math.ceil(math.log10(x)) if x > 0 else 0.0

    def beta_from_full(e):
        x = 10 * (2 * e + 2 * U64 * (1 + 2 * e))
        return 10.0 ** math.ceil(math.log10(x))

    summ = {
        "n_models_audited": int(ok["model_id"].nunique()), "n_rows": int(len(ok)),
        "n_coalitions_total": int(ok["n_coalitions"].sum()),
        "n_coalitions_required_set": int(req["n_coalitions"].sum()),
        "e_max_required_set": e_req, "e_max_all": e_all,
        "e_max_agent_level": float(ok[ok["mask_kind"] == "agent"]["e_max"].max()),
        "e_max_step_level": float(ok[ok["mask_kind"] == "step"]["e_max"].max()) if (ok["mask_kind"] == "step").any() else None,
        "max_abs_ld_minus_mp": float(ok["max_abs_ld_minus_mp"].max()), "n_mp_total": int(ok["n_mp"].sum()),
        "max_abs_fp64_mine_minus_gpu": float(ok["max_abs_fp64_mine_minus_gpu"].max()),
        "observed_max_abs_Z_minus_D": float(ok["observed_max_abs_Z_minus_D"].max()),
        "implied_bound_Z_minus_D_all": 2 * e_all + 2 * U64 * (1 + 2 * e_all),
        "beta_rebuttal_brief_rule": beta_from(e_all),
        "beta_rebuttal_incl_ulp_term": beta_from_full(e_all),
        "min_eta_v": float(ok["eta_v"].min()), "max_eta_v": float(ok["eta_v"].max()),
        "min_fp_term_eps_X": float(ok["fp_term_eps_X"].min()), "max_fp_term_eps_X": float(ok["fp_term_eps_X"].max()),
        "min_eta_v_over_e_max": float(ok["eta_v_over_e_max"].replace(np.inf, np.nan).min()),
        "all_gpu_repeat_bitwise": bool(ok["gpu_repeat_bitwise"].all()),
        "stored_bitwise_all": bool(ok["stored_bitwise"].dropna().astype(bool).all()),
        "n_stored_checked": int(ok["stored_bitwise"].notna().sum()),
        "git_commit": rb.git_commit(), "hardware": rb.hardware_tags(), "time": rb.now_iso(),
    }
    rb.write_json(OUT / "precision_audit_summary.json", summ)
    print(df[["audit", "model_id", "mask_kind", "n_coalitions", "e_max", "observed_max_abs_Z_minus_D",
              "max_abs_ld_minus_mp", "eta_v", "fp_term_eps_X", "stored_bitwise"]].to_string())
    print(json.dumps({kk: v for kk, v in summ.items() if kk != "hardware"}, indent=1))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["gpu", "cpu", "summary", "all"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only", default="")
    a = ap.parse_args(argv)
    specs = audit_specs()
    if a.only:
        keep = a.only.split(",")
        specs = [s for s in specs if any(t in (s.get("model_id") or "") or t == s["audit"] for t in keep)]
    RAW.mkdir(parents=True, exist_ok=True)
    if a.phase in ("gpu", "all"):
        gpu_phase_batched(specs)
    if a.phase in ("cpu", "all"):
        order = sorted(specs, key=lambda s: -(s.get("states_target") or 30000) * s["k"])
        with Pool(min(a.workers, 6)) as pool:
            pool.map(cpu_phase, order, chunksize=1)
    if a.phase in ("summary", "all"):
        summarize(specs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
