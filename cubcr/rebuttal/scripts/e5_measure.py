#!/usr/bin/env python3


from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import rb
import e5_gen as G
from e5_lock import gpu_hold, lock_mode

RES = rb.RB / "results" / "E5"


def unit_name(engine, a, m, size, B, tag=""):
    return f"e5__{engine}__a{a}_m{m}_S{size}_B{B}{tag}"


def byte_model(gm, B, k, nrow_bytes, d_slots):


    nt = gm.state_kind == 0
    rows = np.diff(gm.state_row_start)
    nnz_state = gm.row_ptr[gm.state_row_start[1:]] - gm.row_ptr[gm.state_row_start[:-1]]
    per_c = float(np.sum(8 * nnz_state[nt] + 16 * rows[nt] + 8) + 8 * np.sum(~nt))
    per_w = float(np.sum((12 + nrow_bytes) * nnz_state[nt]) + (54 + 2 * d_slots) * gm.n_states)
    total = B * k * per_c + k * (B / 32.0) * per_w + 8.0 * gm.n_states * B

    floor = k * (16.0 * gm.n_states * B)
    return {"bytes_logical": total, "bytes_per_coalition_sweep": per_c, "bytes_per_warp_sweep_uniform": per_w,
            "bytes_floor_V_only": floor}


def analytic_memory(gm, B, k, table_bytes, nrow_bytes, d_slots, engine):
    S, R, nnz = gm.n_states, gm.n_rows, gm.nnz
    model = (S * 1 + (S + 1) * 8 + S * d_slots * 2 + S * 1 + (R + 1) * 8 + nnz * 4 + nnz * nrow_bytes
             + nnz * 8 + nnz * 4 + (S + 1) * 8 + S * 4)
    if engine != "paper":
        model += S * d_slots
    vbuf = 2 * S * B * 8
    masks = B * 8
    return {"model_bytes": int(model), "table_bytes": int(table_bytes), "v_buffers_bytes": int(vbuf),
            "mask_bytes": int(masks), "total_bytes": int(model + table_bytes + vbuf + masks)}


def hardware_tags_subprocess() -> dict:
    import subprocess
    code = ("import sys, json; sys.path.insert(0, %r); import rb; print(json.dumps(rb.hardware_tags()))"
            % str(Path(__file__).resolve().parents[1] / "lib"))
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {"error": r.stderr[-500:]}


def measure_one(eng_kind, make, gm, masks, k, reps, mem_info):

    rec = {}
    rec["load_before"] = rb.load_snapshot()
    f0, tot = mem_info()
    t0 = time.perf_counter()
    eng = make()
    rec["engine_setup_s"] = time.perf_counter() - t0
    f1, _ = mem_info()
    t0 = time.perf_counter()
    vw = eng.solve(masks, k)
    rec["first_solve_s"] = time.perf_counter() - t0
    rec["first_solve_stats"] = dict(eng.stats)
    f2, _ = mem_info()
    walls, kms, tms, stats = [], [], [], []
    vals = vw
    for _ in range(reps):
        t0 = time.perf_counter()
        vals = eng.solve(masks, k)
        walls.append(time.perf_counter() - t0)
        st = dict(eng.stats)
        stats.append(st)
        kms.append(float(st["kernel_ms"]))
        tms.append(float(st["total_ms"]))
    rec["load_after"] = rb.load_snapshot()
    if eng_kind != "paper":
        import e5_engine as X
        rec["class_info"] = eng.class_info()
        rec["device_attr"] = X.device_attr()
    eng.close()
    f3, _ = mem_info()
    rec["mem"] = {"total_bytes": tot, "free_before_create": f0, "free_after_create": f1,
                  "free_after_first_solve": f2, "free_after_close": f3,
                  "delta_create_bytes": f0 - f1, "delta_first_solve_bytes": f1 - f2,
                  "delta_total_bytes": f0 - f2}
    rec["all_stats"] = stats
    rec["wall_s_all"] = walls
    rec["kernel_ms_all"] = kms
    rec["total_ms_all"] = tms
    return rec, vals, vw


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", nargs="+", choices=("paper", "smem", "acc"), required=True,
                    help="'paper' alone, or variant modes (smem/acc share the variant library and one lock hold)")
    ap.add_argument("--configs", nargs="+", required=True, help="a,m pairs (run in ONE lock hold, in order)")
    ap.add_argument("--configs-smem", nargs="+", default=None, help="override the config list for smem")
    ap.add_argument("--size", choices=tuple(G.E5_SIZES), required=True)
    ap.add_argument("--B", type=int, default=65536)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--subdir", default="batch")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save-values", action="store_true")
    ap.add_argument("--order", choices=("random", "sorted"), default="random")
    a = ap.parse_args(argv)
    out_dir = RES / "raw" / a.subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    k = G.E5_K
    masks_all = G.e5_batch_masks()
    masks = np.ascontiguousarray(np.sort(masks_all)[:a.B] if a.order == "sorted" else masks_all[:a.B])
    engines = list(dict.fromkeys(a.engine))
    if "paper" in engines and len(engines) > 1:
        raise SystemExit("the paper engine runs in its own process")
    is_paper = engines == ["paper"]

    tags = rb.hardware_tags() if is_paper else hardware_tags_subprocess()
    if is_paper:
        from cubcr.engine_gpu import mem_info
        nrow_bytes, d_slots = 1, 3
        lib_info = {"engine_lib": str(rb.LIB_GPU.relative_to(rb.REPO)), "engine_lib_sha256": rb.sha256(rb.LIB_GPU)}
    else:
        import e5_engine as X
        from e5_engine import mem_info
        nrow_bytes, d_slots = 2, 4
        lib_info = {"engine_lib": str(X.E5_LIB.relative_to(rb.REPO)), "engine_lib_sha256": X.lib_sha256()}
    todo = []
    for eng_kind, c in [(e, c) for e in engines for c in ((a.configs_smem or a.configs) if e == "smem" else a.configs)]:
        aa, mm = map(int, c.split(","))
        name = unit_name(eng_kind, aa, mm, a.size, a.B, a.tag + ("" if a.order == "random" else "_sorted"))
        out = out_dir / f"{name}.json"
        if out.exists():
            try:
                if json.load(open(out)).get("status") == "OK":
                    print(f"skip {name}", flush=True)
                    continue
            except Exception:
                pass
        t0 = time.perf_counter()
        gm = G.e5_model(aa, mm, a.size)
        build_s = time.perf_counter() - t0
        todo.append((eng_kind, aa, mm, name, out, gm, build_s, G.validate(gm)))
    if not todo:
        return 0
    rc = 0
    with gpu_hold(f"E5 {'+'.join(engines)} S{a.size} B{a.B} {' '.join(a.configs)}") as waited:
        for i, (eng_kind, aa, mm, name, out, gm, build_s, vstat) in enumerate(todo):
            t_all = time.perf_counter()
            if eng_kind == "paper":
                comp = G.to_compiled3(gm)
                make = lambda: rb.make_gpu_engine(comp)
            else:
                make = lambda: X.E5Engine(gm, mode=eng_kind, dtype="fp64", use_graph=True)
            rec = {"exp": "E5", "unit": name, "engine": eng_kind, "a": aa, "m": mm, "size": a.size,
                   "n": gm.n_agents, "k": k, "B": int(a.B), "reps": a.reps, "mask_order": a.order,
                   "mask_sha256": hashlib.sha256(masks.tobytes()).hexdigest(), "gen_seed": G.E5_GEN_SEED,
                   "generator": gm.meta, "model_fingerprint": gm.fingerprint(), "validation": vstat,
                   "states": gm.n_states, "n_rows": gm.n_rows, "nnz": gm.nnz, "z_model": gm.z_model,
                   "rows_per_state": int(mm ** aa), "model_build_s": build_s, "hardware": tags, **lib_info,
                   "timing_convention": "wall = perf_counter around engine.solve(B masks, k) after one discarded "
                                        "full-batch warm-up (experiments/grid.py protocol); kernel_ms = CUDA events "
                                        "around init + k sweeps",
                   "lock_wait_s": waited if i == 0 else 0.0, "position_in_lock_hold": i, "gpu_lock_mode": lock_mode(),
                   "started": rb.now_iso()}
            m_rec, vals, vw = measure_one(eng_kind, make, gm, masks, k, a.reps, mem_info)
            rec.update(m_rec)
            stats = rec["all_stats"]
            st = stats[-1] if stats else rec["first_solve_stats"]
            rec["engine_stats_last"] = st
            rec["waves"] = sorted({int(s["waves"]) for s in stats + [rec["first_solve_stats"]]})
            rec["batch_used"] = sorted({int(s["batch"]) for s in stats + [rec["first_solve_stats"]]})
            rec["graph_captures"] = [int(s["graph_captures"]) for s in stats]
            rec["wall_s"] = float(np.median(rec["wall_s_all"]))
            rec["kernel_s"] = float(np.median(rec["kernel_ms_all"])) / 1e3
            rec["deterministic_across_reps"] = bool(np.array_equal(vals, vw))
            rec["values_sha256"] = hashlib.sha256(np.ascontiguousarray(vals).tobytes()).hexdigest()
            rec["v_empty_full"] = None
            if a.B == 65536:
                i0 = int(np.nonzero(masks == 0)[0][0]); i1 = int(np.nonzero(masks == (1 << 16) - 1)[0][0])
                rec["v_empty_full"] = [float(vals[i0]), float(vals[i1])]
            tb = int(st["table_bytes"])
            rec["byte_model"] = byte_model(gm, a.B, k, nrow_bytes, d_slots)
            rec["analytic_memory"] = analytic_memory(gm, a.B, k, tb, nrow_bytes, d_slots, eng_kind)
            work_rows = float(a.B) * k * gm.n_rows
            work_z = float(a.B) * k * gm.z_model
            rec["throughput"] = {"row_updates_per_s_wall": work_rows / rec["wall_s"],
                                 "z_updates_per_s_wall": work_z / rec["wall_s"],
                                 "nnz_updates_per_s_wall": float(a.B) * k * gm.nnz / rec["wall_s"],
                                 "row_updates_per_s_kernel": work_rows / rec["kernel_s"],
                                 "z_updates_per_s_kernel": work_z / rec["kernel_s"],
                                 "gb_per_s_logical_kernel": rec["byte_model"]["bytes_logical"] / rec["kernel_s"] / 1e9,
                                 "gb_per_s_floor_kernel": rec["byte_model"]["bytes_floor_V_only"] / rec["kernel_s"] / 1e9,
                                 "coalitions_per_s_wall": a.B / rec["wall_s"]}
            ok = rec["waves"] == [1] and rec["batch_used"] == [a.B] and rec["deterministic_across_reps"]
            rec["status"] = "OK" if ok else "CHECK_FAILED"
            rec["process_elapsed_s"] = time.perf_counter() - t_all
            rec["finished"] = rb.now_iso()
            if a.save_values:
                vpath = out_dir / "values" / f"{name}.npy"
                vpath.parent.mkdir(parents=True, exist_ok=True)
                np.save(vpath, vals)
                rec["values_path"] = str(vpath.relative_to(rb.REPO))
            rb.write_json(out, rec)
            mem = rec["mem"]
            print(f"[E5] {name}: {rec['status']} wall={rec['wall_s']:.4f}s kernel={rec['kernel_s']:.4f}s "
                  f"setup={rec['engine_setup_s']:.3f}s Z={gm.z_model} "
                  f"rows/s={rec['throughput']['row_updates_per_s_wall']:.3e} "
                  f"mem+={mem['delta_total_bytes'] / 2**30:.3f}GiB waves={rec['waves']} lockwait={rec['lock_wait_s']:.1f}s",
                  flush=True)
            rc |= 0 if ok else 3
    return rc


if __name__ == "__main__":
    sys.exit(main())
