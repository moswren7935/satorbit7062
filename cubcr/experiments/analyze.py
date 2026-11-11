#!/usr/bin/env python3


from __future__ import annotations
import glob, json, math, os, sys
from collections import defaultdict
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(REPO, "results")


def load(exp):
    out = []
    for f in sorted(glob.glob(os.path.join(RES, exp, "*.json"))):
        try:
            d = json.load(open(f))
            if isinstance(d, dict):
                out.append(d)
        except Exception:
            pass
    return out


def fmt(x, p=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    if isinstance(x, (float, np.floating)):
        if x != 0 and (abs(x) >= 1e5 or abs(x) < 0.01):
            return f"{x:.2e}"
        return f"{x:.{p}f}"
    return str(x)


def table(header, rows):
    esc = lambda s: str(s).replace("|", "\\|")
    out = ["| " + " | ".join(esc(h) for h in header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(esc(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def b90_table():

    recs = [r for r in load("e1cal") if r.get("status") == "OK" and r["engine"] == "gpu_batched"]
    by_s = defaultdict(list)
    for r in recs:
        by_s[r["states"]].append((r["n_coalitions"], r["coalition_solves_per_s"]))
    out = {}
    for S, pts in by_s.items():
        pts.sort()
        plateau = pts[-1][1]
        b90 = next((B for B, t in pts if t >= 0.9 * plateau), pts[-1][0])
        out[S] = (b90, plateau)
    return out


def S_req(states):

    return int(round(10 ** round(math.log10(max(1, states)))))


def nearest_S(S, keys):
    return min(keys, key=lambda x: abs(math.log10(x) - math.log10(S))) if keys else None


def pair_cells(recs):

    cells = defaultdict(dict)
    for r in recs:
        cid = r.get("cell_id") or r["instance_id"]
        if r["engine"] == "gpu_batched":
            cells[cid]["gpu"] = r
        elif r["engine"] == "cpu_multicore":
            cells[cid][f"cpu{r['threads']}"] = r
    return cells


GIT_WARNINGS = set()


def check_pair(g, c):


    problems = []
    pg, pc = g.get("provenance") or {}, c.get("provenance") or {}
    for lib in ("libcubcr_gpu.so", "libcubcr_cpu.so"):
        if pg.get(lib) and pc.get(lib) and not (pg[lib].startswith(pc[lib]) or pc[lib].startswith(pg[lib])):
            problems.append(f"{lib} hash mismatch")
    if g.get("git") != c.get("git"):
        if pg and pc and all((pg.get(l) or "").startswith(pc.get(l) or "?") or (pc.get(l) or "").startswith(pg.get(l) or "?") for l in ("libcubcr_gpu.so", "libcubcr_cpu.so")):
            GIT_WARNINGS.add(f"{g.get('cell_id')}: git {str(g.get('git'))[:8]} vs {str(c.get('git'))[:8]} (identical engine binaries)")
        else:
            problems.append("git mismatch")
    if g.get("mask_sha256") and c.get("mask_sha256") and not (g["mask_sha256"].startswith(c["mask_sha256"]) or c["mask_sha256"].startswith(g["mask_sha256"])):
        problems.append("mask mismatch")
    if g.get("n_coalitions") != c.get("n_coalitions"):
        problems.append("B mismatch")
    if g.get("dtype") != c.get("dtype"):
        problems.append("dtype mismatch")
    return problems


def e1(exp="e1", title="E1 — GPU batched vs CPU-32 on the IDENTICAL coalition batch (FP64, TicketBench; median over seeds of per-seed throughput ratios)"):
    recs = load(exp)
    if not recs:
        return f"{exp.upper()}: NOT RUN"
    GIT_WARNINGS.clear()
    b90 = b90_table()
    cells = pair_cells(recs)
    agg = defaultdict(list)
    problems, incomplete = [], []
    for cid, arms in cells.items():
        g, c, c64 = arms.get("gpu"), arms.get("cpu32"), arms.get("cpu64")
        if not g or not c:
            incomplete.append(cid + " (missing arm)")
            continue
        if g["status"] != "OK" or c["status"] != "OK":
            incomplete.append(f"{cid} (gpu {g['status']}, cpu32 {c['status']})")
            continue
        pr = check_pair(g, c)
        if pr:
            problems.append(cid + ": " + ", ".join(pr))
            continue
        import re as _re
        m_s = _re.search(r"_S(\d+)_", g.get("cell_id") or "")
        S_key = int(m_s.group(1)) if (exp == "e1prune" and m_s) else S_req(g["states"])
        key = (g["n"], S_key, g["k"])
        S_cal = nearest_S(g["states"], list(b90))
        batched = (S_cal is not None) and g["n_coalitions"] >= b90[S_cal][0]
        r64 = (g["coalition_solves_per_s"] / c64["coalition_solves_per_s"]) if (c64 and c64["status"] == "OK" and not check_pair(g, c64)) else None
        agg[key].append(dict(sp=g["coalition_solves_per_s"] / c["coalition_solves_per_s"], sp64=r64, B=g["n_coalitions"], S_eff=g["states"],
                             gth=g["coalition_solves_per_s"], cth=c["coalition_solves_per_s"], gg=g["gfma_per_s"], cg=c["gfma_per_s"], batched=batched,
                             gsd=np.std(g["all_s"]) / np.median(g["all_s"]), csd=np.std(c["all_s"]) / np.median(c["all_s"])))
    rows = []
    sps_all, sps_batched = [], []
    for key in sorted(agg):
        v = agg[key]
        sp = [x["sp"] for x in v]
        sp64 = [x["sp64"] for x in v if x["sp64"] is not None]
        med = float(np.median(sp))
        sps_all.append(med)
        if all(x["batched"] for x in v):
            sps_batched.append(med)
        rows.append([key[0], (f"{key[1]} → {v[0]['S_eff']} pruned" if exp == "e1prune" else key[1]), key[2], len(v), v[0]["B"], "yes" if all(x["batched"] for x in v) else "no",
                     fmt(float(np.median([x["gth"] for x in v])), 0), fmt(float(np.median([x["gg"] for x in v])), 1),
                     fmt(float(np.median([x["cth"] for x in v])), 0), fmt(float(np.median([x["cg"] for x in v])), 1),
                     f"{med:.2f} [{min(sp):.2f}, {max(sp):.2f}]", f"{np.median(sp64):.2f}" if sp64 else "—",
                     f"{100*np.median([x['gsd'] for x in v]):.1f}% / {100*np.median([x['csd'] for x in v]):.1f}%"])
    hdr = ["n", "|S|", "k", "seeds", "B (both arms)", "batched", "GPU coal/s", "GPU GFMA/s", "CPU-32 coal/s", "CPU-32 GFMA/s", "speedup vs CPU-32 median [min,max]", "vs CPU-64", "timing CV GPU/CPU"]
    s = f"\n{len(rows)} cells with complete valid pairs. Median speedup over ALL cells: {np.median(sps_all):.2f}x (cells ≥5x: {sum(x >= 5 for x in sps_all)}/{len(sps_all)}); over BATCHED cells (B ≥ B90(|S|)): {np.median(sps_batched):.2f}x ({len(sps_batched)} cells, ≥5x: {sum(x >= 5 for x in sps_batched)}).\n"
    s += "B90(|S|) from e1cal: " + ", ".join(f"|S|={S}: B90={b}" for S, (b, _) in sorted(b90.items())) + "\n"
    if incomplete:
        s += f"Incomplete/censored/crashed cells ({len(incomplete)}): " + "; ".join(incomplete) + "\n"
    if problems:
        s += f"REJECTED pairs (provenance/batch mismatch) ({len(problems)}): " + "; ".join(problems) + "\n"
    if GIT_WARNINGS:
        s += f"Pairs with different repo hashes but identical engine binaries (accepted, {len(GIT_WARNINGS)}): " + "; ".join(sorted(GIT_WARNINGS)[:6]) + (" …" if len(GIT_WARNINGS) > 6 else "") + "\n"
    return f"### {title}\n" + table(hdr, rows) + s


def e0():
    recs = load("e0")
    if not recs:
        return "E0: NOT RUN"
    keys = sorted({(r["n"], r["states"], r["k"]) for r in recs})
    rows = []
    for n, S, k in keys:
        sel = lambda eng, dt, tag="": next((r for r in recs if (r["n"], r["states"], r["k"]) == (n, S, k) and r["engine"] == eng and r["dtype"] == dt and (r.get("tag") or "") == tag), None)
        g = next((r for r in recs if (r["n"], r["states"], r["k"]) == (n, S, k) and r["engine"] == "gpu_batched" and r["dtype"] == "fp64" and "fp32tau" not in json.dumps(r.get("instance_meta", "")) and r.get("kendall_tau") is None), None)
        c = sel("cpu_multicore", "fp64")
        f = next((r for r in recs if (r["n"], r["states"], r["k"]) == (n, S, k) and r["dtype"] == "fp32"), None)
        def cell(r, field):
            if r is None:
                return "NOT RUN"
            if r["status"] != "OK":
                return f">= {r['wall_s']:.0f} (cap)" if r["status"] == "TO" else r["status"]
            return fmt(r.get(field), 3)
        sp_solve = (c["wall_s"] / g["wall_s"]) if (g and c and g["status"] == "OK" and c["status"] == "OK") else None
        sp_e2e = (c["end_to_end_s"] / g["end_to_end_s"]) if (g and c and g.get("end_to_end_s") and c.get("end_to_end_s")) else None
        rows.append([n, S, k, 1 << n, cell(g, "wall_s"), cell(c, "wall_s"), fmt(sp_solve, 1), cell(g, "end_to_end_s"), cell(c, "end_to_end_s"), fmt(sp_e2e, 1),
                     f"{g['engine_setup_s']:.2f}/{g.get('shapley_s', 0):.2f}" if g and g["status"] == "OK" else "—",
                     f"{g['n_repeat']}" if g else "—", fmt(g.get("efficiency_gap"), 1) if g else "—", fmt(g.get("attributable_value"), 4) if g else "—",
                     f"{f['kendall_tau']:.3f} / {f['max_abs_err_vs_fp64']:.1e}" if f and f.get("kendall_tau") is not None else "—"])
    return "### E0 — exact agent-level Shapley over all 2^n coalitions (FP64; CPU = 32 threads). 'solve' = median of n_repeat solves of the full batch; 'end-to-end' = model build + engine set-up + one solve + Shapley aggregation\n" + \
        table(["n", "|S|", "k", "coalitions", "GPU solve [s]", "CPU solve [s]", "speedup (solve)", "GPU end-to-end [s]", "CPU end-to-end [s]", "speedup (e2e)", "GPU setup/Shapley [s]", "repeats", "efficiency gap", "Υ", "FP32: τ / max|Δv|"], rows)


def e0hz():
    recs = load("e0hz")
    if not recs:
        return "E0-hz: NOT RUN"
    rows = []
    for key in sorted({(r["n"], r["states"], r["k"]) for r in recs}):
        g = next((r for r in recs if (r["n"], r["states"], r["k"]) == key and r["engine"] == "gpu_batched"), None)
        c = next((r for r in recs if (r["n"], r["states"], r["k"]) == key and r["engine"] == "cpu_multicore"), None)
        if g and g["status"] == "OK":
            vs = g["v_summary"]
            rows.append([key[0], key[1], key[2], g.get("hazard"), fmt(vs.get("v_empty"), 4), fmt(vs.get("v_full"), 4), fmt(g.get("attributable_value"), 4), fmt(vs.get("frac_zero"), 3), vs.get("n_distinct"),
                         fmt(g["wall_s"], 3), fmt(c["wall_s"], 3) if c and c["status"] == "OK" else (c["status"] if c else "NOT RUN"),
                         fmt(c["wall_s"] / g["wall_s"], 1) if c and c["status"] == "OK" else "—", fmt(g.get("efficiency_gap"), 1)])
    return "### E0-hz — hazard variant (unavoidable failure hazard 0.02 at the commit stage): 0 < v(∅) < v(N); exact Shapley, identical full enumeration on both engines\n" + \
        table(["n", "|S|", "k", "hazard", "v(∅)", "v(N)", "Υ", "frac zero v", "distinct v", "GPU solve [s]", "CPU-32 solve [s]", "speedup", "efficiency gap"], rows)


def e1cal():
    recs = [r for r in load("e1cal") if r["status"] == "OK"]
    if not recs:
        return "E1-cal: NOT RUN"
    rows = []
    for r in sorted(recs, key=lambda r: (r["states"], r["engine"], r["threads"], r["n_coalitions"])):
        rows.append([r["states"], r["engine"], r["threads"], r["n_coalitions"], fmt(r["wall_s"], 4), fmt(r["coalition_solves_per_s"], 0), fmt(r["gfma_per_s"], 1)])
    b90 = b90_table()
    return "### E1-cal — throughput vs batch size B per |S| (n=16, k=25); B90 = smallest B with GPU ≥ 90% of its largest-B throughput\n" + \
        table(["|S|", "engine", "threads", "B", "wall [s]", "coal/s", "GFMA/s"], rows) + "\n" + ", ".join(f"|S|={S}: B90={b} (plateau {p:.0f} coal/s)" for S, (b, p) in sorted(b90.items()))


def e2():
    recs = load("e2")
    if not recs:
        return "E2: NOT RUN"
    keys = sorted({(r["n"], r["states"], r["k"]) for r in recs})
    rows = []
    for key in keys:
        g = next((r for r in recs if (r["n"], r["states"], r["k"]) == key and r["engine"] == "gpu_batched"), None)
        u = next((r for r in recs if (r["n"], r["states"], r["k"]) == key and r["engine"] == "gpu_unbatched"), None)
        if g and u and g["status"] == "OK" and u["status"] == "OK":
            rows.append([key[0], key[1], key[2], 1 << key[0], fmt(g["wall_s"], 4), fmt(u["wall_s"], 3), fmt(u["wall_s"] / g["wall_s"], 1), f"{g['n_repeat']}/{u['n_repeat']}"])
        elif g and u:
            rows.append([key[0], key[1], key[2], 1 << key[0], fmt(g["wall_s"], 4), u["status"], "—", "—"])
    return "### E2 — batching ablation on identical coalition sets (GPU FP64; one-coalition-per-wave uses the same kernel)\n" + table(["n", "|S|", "k", "coalitions", "batched wall [s]", "one-at-a-time wall [s]", "batching gain", "repeats"], rows)


def e6():
    recs = load("e6")
    if not recs:
        return "E6: NOT RUN"
    keys = sorted({(r["n"], r["states"], r["k"]) for r in recs})
    rows = []
    for key in keys:
        pick = lambda eng, dt: next((r for r in recs if (r["n"], r["states"], r["k"]) == key and r["engine"] == eng and r["dtype"] == dt and r["status"] == "OK"), None)
        g64, g32, c32, c64 = pick("gpu_batched", "fp64"), pick("gpu_batched", "fp32"), pick("cpu_multicore", "fp32"), pick("cpu_multicore", "fp64")
        arms = [r for r in (g64, g32, c32, c64) if r]
        gpu_bins = {(r.get("provenance") or {}).get("libcubcr_gpu.so") for r in arms if r["engine"] == "gpu_batched"}
        cpu_bins = {(r.get("provenance") or {}).get("libcubcr_cpu.so") for r in arms if r["engine"] == "cpu_multicore"}
        if len({(r.get("mask_sha256"), r.get("n_coalitions")) for r in arms}) > 1 or len(gpu_bins) > 1 or len(cpu_bins) > 1:
            rows.append([key[0], key[1], key[2], "REJECTED: arms differ in batch/binary", "", "", "", "", "", "", "", ""])
            continue
        if g64 and g32:
            rows.append([key[0], key[1], key[2], g64["n_coalitions"], fmt(g64["coalition_solves_per_s"], 0), fmt(g32["coalition_solves_per_s"], 0), fmt(g32["coalition_solves_per_s"] / g64["coalition_solves_per_s"], 2),
                         fmt(g32.get("max_abs_err_vs_fp64")), fmt(g32.get("mean_abs_err_vs_fp64")),
                         fmt(c64["coalition_solves_per_s"], 0) if c64 else "—", fmt(c32["coalition_solves_per_s"], 0) if c32 else "—",
                         fmt(g32["coalition_solves_per_s"] / c32["coalition_solves_per_s"], 2) if c32 else "—"])
    return "### E6 — FP32 vs FP64 (n=16, identical batches)\n" + table(["n", "|S|", "k", "B", "GPU fp64 coal/s", "GPU fp32 coal/s", "GPU fp32/fp64", "max|v32−v64|", "mean|v32−v64|", "CPU-32 fp64 coal/s", "CPU-32 fp32 coal/s", "GPU fp32 / CPU fp32"], rows)


def e3(exp="e3", title=None):
    recs = [r for r in load(exp) if "prism" in r]
    if not recs:
        return "E3: NOT RUN"
    rows = []
    for r in sorted(recs, key=lambda r: (r["source"], r["n"], r["states"], r["k"])):
        p = r["prism"]
        tot = fmt(p["total_wall_s"], 1) + ("" if p.get("complete") else f" (partial: {p['n_ok']}/{r['coalitions']} OK, {p.get('n_to', 0)} TO, {p.get('n_mo', 0)} MO, {p.get('n_err', 0)} ERR, {p.get('n_not_run', 0)} not run)")
        gate = "PASS" if r.get("values_match", p.get("complete") and (p.get("max_abs_err_vs_gpu") or 0) <= 1e-6) else "FAIL"
        sp = r.get("speedup_gpu_vs_prism")
        spl = r.get("speedup_gpu_vs_prism_lower_bound")
        rows.append([r["source"], r["n"], r["states"], r["k"], r["coalitions"], tot, fmt(p.get("solver_total_s") or p.get("prism_solver_s"), 2), fmt(p.get("median_wall_s"), 2),
                     fmt(p.get("max_abs_err_vs_gpu")), gate, fmt(r.get("gpu_wall_s"), 4), fmt(r.get("cpu32_wall_s"), 4),
                     fmt(sp, 0) if sp else (f">= {spl:.0f}" if spl else "—"), fmt(r.get("speedup_cpu_vs_prism"), 0) if r.get("speedup_cpu_vs_prism") else "—"])
    return (title or "### E3 — PRISM-games on identical instances (one collapsed model per coalition; PRISM CLI wall includes JVM start-up; 'solver-only' = PRISM-reported construction + checking)") + "\n" + \
        table(["source", "n", "|S|", "k", "coalitions", "PRISM CLI total [s]", "PRISM solver-only [s]", "median/coalition [s]", "max|v_PRISM−v_GPU|", "gate", "GPU wall [s]", "CPU-32 wall [s]", "GPU/PRISM", "CPU/PRISM"], rows)


def e9():
    recs = load("e9")
    if not recs:
        return "E9: NOT RUN"
    by_cell = defaultdict(dict)
    for r in recs:
        key = (r.get("pattern"), r.get("hazard") or 0.0, r["n"], S_req(r["states"]), r["k"])
        arm = r.get("arm")
        if arm and arm.startswith("cert"):
            by_cell[key].setdefault(arm, []).append(r)
        elif arm:
            prev = by_cell[key].get(arm)
            rank = {"OK": 3, "TO": 2}.get(r.get("status"), 1)
            prank = {"OK": 3, "TO": 2}.get(prev.get("status"), 1) if prev else 0
            if rank >= prank:
                by_cell[key][arm] = r
    rows = []
    heads = []
    for key in sorted(by_cell, key=lambda x: (x[1], x[2], x[3], x[4])):
        arms = by_cell[key]
        eg, ec = arms.get("exact_gpu"), arms.get("exact_cpu")
        def wall(r):
            if r is None:
                return None, "NOT RUN"
            if r["status"] == "OK":
                return r["wall_s"], fmt(r["wall_s"], 2)
            if r["status"] == "TO":
                return None, f">= {r['wall_s']:.0f} (genuine cap)"
            return None, f"NOT RUN (proj. {fmt(r.get('projected_full_s'), 0)} s)"
        weg, seg = wall(eg); wec, sec = wall(ec)
        def cert(arm):
            rs = arms.get(arm, [])
            ok = [r for r in rs if r.get("certified")]
            if not rs:
                return None, "—", None, None
            if len(ok) < len(rs):
                return None, f"{len(ok)}/{len(rs)} certified", None, None
            med = float(np.median([r["wall_s"] for r in ok]))
            vals = [r.get("covers_exact") for r in ok]
            cov = True if all(v is True for v in vals) else (None if all(v is None for v in vals) else all(v in (True, None) for v in vals) and None)
            solves = int(np.median([r["n_solves"] for r in ok]))
            return med, fmt(med, 2), cov, solves
        wsg, ssg, covg, slv = cert("cert_gpu")
        wsc, ssc, covc, _ = cert("cert_cpu")
        ups = next((r.get("attributable_value") for r in [eg, ec] + (arms.get("cert_gpu") or []) if r and r.get("attributable_value") is not None), None)
        head = (wec / wsg) if (wec and wsg) else None
        head_lb = None
        if head is None and ec is not None and ec.get("status") == "TO" and wsg:
            head_lb = ec["wall_s"] / wsg
        if head:
            heads.append(head)
        dec = ""
        if head and weg:
            dec = f"= {wec/weg:.1f} × {weg/wsg:.1f}"
        rows.append([key[0] + (f"+hz{key[1]}" if key[1] else ""), key[2], key[3], key[4], fmt(ups, 4),
                     f"{(1 << key[2]):.1e}", slv or "—", sec, seg, ssc, ssg,
                     (f"{head:.0f}× {dec}" if head else (f"> {int(head_lb)}× (exact CPU hit its cap; conservatively rounded down)" if head_lb else "—")),
                     f"{weg/wsg:.1f}" if (weg and wsg) else "—", f"{wsc/wsg:.1f}" if (wsc and wsg) else "—",
                     "yes" if covg else ("—" if covg is None else "NO")])
    hdr = ["instance", "n", "|S|", "k", "Υ", "2^n", "cert. solves (median)", "exact CPU-32 [s]", "exact GPU [s]",
           "cert CPU-32 [s]", "cert GPU [s]", "PIPELINE: exact-CPU / cert-GPU (= (E_C/E_G) × (E_G/S_G))", "E_G/S_G (alg on GPU)", "S_C/S_G (eng on certified)", "covers exact"]
    s = ""
    if heads:
        s = (f"\nPipeline ratios (exact-exhaustive CPU-32 wall / certified-±Υ/20 GPU wall) over the {len(heads)} cells where both are measured: "
             f"median {np.median(heads):.0f}×, range {min(heads):.0f}–{max(heads):.0f}×. This is a cross-algorithm, cross-accuracy comparison "
             f"(exact enumeration vs certified ±Υ/20 sampling), NOT an engine speedup; the decomposition column shows (E_C/E_G) × (E_G/S_G).\n")
    return ("### E9 — certified agent-level attribution vs exhaustive enumeration (certified = Castro sampling, range-rescaled two-sided EB, "
            "Bonferroni over players + looks, eps = Υ/20, α=0.05, 3 sampler seeds; exact = full 2^n + shapley_fast)\n" + table(hdr, rows) + s)


def e4(exp="e4"):
    files = sorted(glob.glob(os.path.join(RES, exp, "e4__*.json")))
    recs = [json.load(open(f)) for f in files]
    recs = [r for r in recs if "sampling" in r]
    if not recs:
        return "E4: NOT RUN"
    rows = []
    for r in sorted(recs, key=lambda r: (r["n"], r["states"], r.get("rule", ""), r.get("seed", 0))):
        s = r["sampling"]; al = r.get("alpha_allocation") or {}; cert = r.get("certificate") or {}
        rows.append([r["instance_id"], r["n"], r["states"], r["k"], r["n_players"], r.get("rule"), r.get("seed"), s.get("status"),
                     "yes" if r.get("certified") else "no", fmt(r.get("time_to_certify_s"), 1) if r.get("certified") else f"— ({fmt(s.get('wall_s'), 1)} s run)",
                     s.get("n_perms"), s.get("n_solves"), fmt(s.get("solves_per_s"), 0), fmt(s.get("hw_max_final"), 4),
                     fmt(al.get("alpha_player")), fmt(al.get("alpha_look")), fmt(s.get("efficiency_gap")),
                     (f"{r['exact']['max_abs_err']:.1e} / {r['exact']['all_inside_ci']}" if r.get("exact") else "—")])

    agg = defaultdict(list)
    for r in recs:
        if r.get("rule") and r.get("rule") != "trajectory":
            agg[(r["instance_id"], r["rule"])].append(r.get("time_to_certify_s") if r.get("certified") else None)
    lines = []
    for key in sorted(agg):
        v = agg[key]; ok = [x for x in v if x is not None]
        lines.append(f"{key[0]} / {key[1]}: certified {len(ok)}/{len(v)} seeds" + (f", median time-to-certify {np.median(ok):.1f} s" if ok and len(ok) == len(v) else (f" (partial: certified seeds took {min(ok):.1f}–{max(ok):.1f} s; median not reported)" if ok else "")))
    return ("" if exp == "e4" else f"### {exp.upper()} — CS sampler (monotone range + confidence sequence + geometric batches)\n") + "### E4 — sampled step-level heat-map with real stopping rules (GPU FP64; two-sided EB intervals, Bonferroni over players, union bound over looks; 3 seeds per rule)\n" + \
        table(["instance", "n", "|S|", "k", "players", "rule", "seed", "status", "certified", "time-to-certify [s]", "perms", "solves", "solves/s", "final max hw", "α_player", "α_look", "efficiency gap", "vs exact"], rows) + \
        "\n\nPer instance × rule: " + "; ".join(lines) + "\n"


def e3storm():
    files = sorted(glob.glob(os.path.join(RES, "e3storm", "e3storm__*.json")))
    recs = [json.load(open(f)) for f in files if "__" not in os.path.basename(f)[9:].replace("__", "", 0) or f.endswith(".json")]
    recs = [r for r in recs if isinstance(r, dict) and "storm" in r]
    if not recs:
        return "E3-Storm: NOT RUN"
    rows = []
    for r in sorted(recs, key=lambda r: (r["n"], r["states"])):
        st = r["storm"]
        rows.append([r["instance_id"], r["n"], r["states"], r["k"], r.get("coalitions"),
                     fmt(st.get("total_wall_s"), 1), fmt(st.get("solver_total_s"), 2), fmt(st.get("median_wall_s"), 3),
                     fmt(st.get("max_abs_err_vs_gpu")), "PASS" if r.get("values_match") else "FAIL",
                     fmt(r.get("gpu_wall_s"), 4), fmt(r.get("cpu32_wall_s"), 4),
                     fmt(r.get("speedup_gpu_vs_storm"), 0) if r.get("speedup_gpu_vs_storm") else "—"])
    return ("### E3-Storm — Coalition MDP validation\n" +
            table(["instance", "n", "|S|", "k", "coalitions", "Storm total [s]", "solver-only [s]", "median/coal [s]",
                   "max|Δv| vs GPU", "gate", "GPU wall [s]", "CPU-32 wall [s]", "GPU/Storm"], rows))


def e10():
    recs = load("e10")
    if not recs:
        return "E10: NOT RUN"
    by = defaultdict(list)
    for r in recs:
        by[(r["instance_id"], r.get("tool"), r.get("budget_factor") or 1)].append(r)
    rows = []
    for key in sorted(by, key=lambda x: (x[0], str(x[1]), x[2])):
        rs = by[key]
        rows.append([key[0][:40], key[1], key[2], len(rs), fmt(float(np.median([r["wall_s"] for r in rs])), 3),
                     fmt(int(np.median([r.get("n_oracle_rows") or 0 for r in rs])), 0),
                     fmt(float(np.median([r.get("max_abs_err_vs_exact") or np.nan for r in rs]))),
                     "EB-certified" if key[1] == "cubcr_sampler" else "none"])
    return ("### E10 — shap PermutationExplainer vs the cuBCR certified sampler on the SAME coalition oracle "
            "(matched engine-row budgets; 3 seeds; error vs the exact 2^n table)\n" +
            table(["instance", "tool", "budget ×", "seeds", "median wall [s]", "oracle rows", "median max|φ̂−φ|", "accuracy statement"], rows))


def _p(xs, q):
    return float(np.percentile(xs, q)) if len(xs) else None


def e9v3():
    recs = [r for r in load("e9v3") if r.get("exp") == "e9v3" and not r.get("concurrent_load")]
    if not recs:
        return "E9-v3: NOT RUN"
    inst = defaultdict(dict)
    for r in recs:
        arm = r.get("arm")
        if arm is None:
            continue
        if arm.startswith("cert"):
            inst[r["instance_id"]].setdefault(arm, []).append(r)
        else:
            prev = inst[r["instance_id"]].get(arm)
            rank = {"OK": 3, "TO": 2}.get(r.get("status"), 1)
            if prev is None or rank >= {"OK": 3, "TO": 2}.get(prev.get("status"), 1):
                inst[r["instance_id"]][arm] = r
    cells = defaultdict(list)
    for iid, arms in inst.items():
        r0 = next(iter(v[0] if isinstance(v, list) else v for v in arms.values()))
        cells[(r0["n"], S_req(r0["states"]), r0["k"])].append((iid, arms))
    rows = []
    all_ratios, all_ratios_setup, below = [], [], []
    for key in sorted(cells):
        ratios, ratios_setup, ratios_eg, ratios_sc, ratios_eb, ups_l, cov = [], [], [], [], [], [], []
        lower_bounds, not_run, n_inst = [], 0, 0
        for iid, arms in cells[key]:
            if "cell" in arms and arms["cell"].get("status") == "NOT RUN":
                below.append(iid); continue
            n_inst += 1
            eg, ec = arms.get("exact_gpu"), arms.get("exact_cpu")
            cg = [r for r in arms.get("cert_gpu", []) if r.get("certified")]
            cc = [r for r in arms.get("cert_cpu", []) if r.get("certified")]
            ce = [r for r in arms.get("cert_gpu_eblooks", []) if r.get("certified")]
            for r in cg:
                if r.get("attributable_value") is not None:
                    ups_l.append(r["attributable_value"])
                if r.get("covers_exact") is not None:
                    cov.append(bool(r["covers_exact"]))
            if not cg:
                continue
            for r in cg:
                if ec is not None and ec.get("status") == "OK":
                    ratios.append(ec["wall_s"] / r["wall_s"])
                    ratios_setup.append((ec.get("pipeline_setup_inclusive_s") or ec["wall_s"]) / (r.get("pipeline_setup_inclusive_s") or r["wall_s"]))
                elif ec is not None and ec.get("status") == "TO":
                    lower_bounds.append(ec["wall_s"] / r["wall_s"])
                else:
                    not_run += 1
                if eg is not None and eg.get("status") == "OK":
                    ratios_eg.append(eg["wall_s"] / r["wall_s"])
            if cc:
                ratios_sc.append(float(np.median([x["wall_s"] for x in cc])) / float(np.median([x["wall_s"] for x in cg])))
            if ce:
                ratios_eb.append(float(np.median([x["wall_s"] for x in ce])) / float(np.median([x["wall_s"] for x in cg])))
        all_ratios += ratios; all_ratios_setup += ratios_setup
        if ratios:
            head = f"{np.median(ratios):.0f}× (p10 {_p(ratios, 10):.0f}×, IQR {_p(ratios, 25):.0f}–{_p(ratios, 75):.0f}×; {len(ratios)} runs)"
        elif lower_bounds:
            head = f"> {int(min(lower_bounds))}× (exact CPU-32 hit its cap in every instance; {len(lower_bounds)} runs)"
        else:
            head = "— (exact CPU NOT RUN)"
        rows.append([key[0], key[1], key[2], n_inst, f"{min(ups_l):.3f}–{max(ups_l):.3f}" if ups_l else "—", head,
                     f"{np.median(ratios_setup):.0f}×" if ratios_setup else "—",
                     f"{np.median(ratios_eg):.1f}×" if ratios_eg else "—",
                     f"{np.median(ratios_sc):.1f}×" if ratios_sc else "—",
                     f"{np.median(ratios_eb):.1f}×" if ratios_eb else "—",
                     ("yes" if all(cov) else "NO") if cov else "—",
                     (f"{len(lower_bounds)} capped" if lower_bounds else "") + (f" {not_run} NOT RUN" if not_run else "")])
    hdr = ["n", "|S|", "k", "instances", "Υ range", "PIPELINE exact-CPU-32 / certified-GPU (median over instances×seeds)",
           "setup-inclusive median", "E_G/S_G", "S_C/S_G", "EB-looks / CS", "covers exact", "censoring"]
    s = ""
    if all_ratios:
        s = (f"\nPreregistered aggregate over all {len(all_ratios)} measured (instance, sampler-seed) runs with a measured exact CPU-32 wall: "
             f"median {np.median(all_ratios):.0f}×, 10th percentile {_p(all_ratios, 10):.0f}×, IQR {_p(all_ratios, 25):.0f}–{_p(all_ratios, 75):.0f}×; "
             f"setup-inclusive median {np.median(all_ratios_setup):.0f}×.  Censored runs enter only as '>' bounds; NOT RUN cells are listed.\n")
    if below:
        s += f"\nCells below the preregistered Υ-floor (0.05), NOT RUN: {', '.join(sorted(below))}\n"
    return ("### E9-v3 — certified agent-level attribution vs exhaustive enumeration, sampling protocol "
            "(Υ ≥ 0.05, k ∈ {50,100}, 3 generator seeds × 3 sampler seeds, confidence-sequence sampler with geometric batches, "
            "eps = Υ/20, α = 0.05; exact = full 2^n + shapley_fast; records taken under concurrent load excluded)\n" + table(hdr, rows) + s)


def e11():
    recs = [r for r in load("e11") if r.get("status") == "OK"]
    if not recs:
        return "E11: NOT RUN"
    by = defaultdict(list)
    for r in recs:
        by[(r["arm"], r["budget"])].append(r)
    arms = sorted({r["arm"] for r in recs}, key=lambda a: (0 if a.startswith("ours") else 1, a))
    budgets = sorted({r["budget"] for r in recs})
    rows = []
    for arm in arms:
        row = [arm]
        for b in budgets:
            rs = by.get((arm, b), [])
            if not rs:
                row.append("—"); continue
            rel = [r["max_abs_err"] / r["attributable_value"] for r in rs if r.get("attributable_value")]
            dist = [r.get("n_distinct") or 0 for r in rs]
            row.append(f"{np.median(rel):.3f} ({int(np.median(dist))})")
        rows.append(row)
    n_inst = len({r["instance_id"] for r in recs})
    cert = [r for r in recs if r["arm"] == "ours_cs" and r.get("covers_exact") is not None]
    covs = f"; ours_cs interval coverage of the exact φ: {sum(1 for r in cert if r['covers_exact'])}/{len(cert)} runs" if cert else ""
    return ("### E11 — Shapley estimator bake-off on a shared cached oracle (max|φ̂−φ|/Υ, median over instances × seeds; "
            "in parentheses the median number of DISTINCT coalition evaluations actually used at that requested budget)\n"
            + table(["arm \\ requested budget"] + [str(b) for b in budgets], rows)
            + f"\n{n_inst} instances (saved exact tables), 3 seeds; shapiq 1.3.0 arms, shap PermutationExplainer, "
              f"ours = cubcr sampler (CS = confidence sequence; strat = Latin-square stratified unit; EB = legacy looks){covs}.\n")


def e12():
    recs = load("e12")
    if not recs:
        return "E12: NOT RUN"
    by = defaultdict(list)
    for r in recs:
        by[(r.get("source"), r.get("pattern") or "—")].append(r)
    rows = []
    for key in sorted(by, key=lambda x: (str(x[0]), str(x[1]))):
        rs = by[key]; m = [r["remove_vs_shapley"] for r in rs]; ma = [r["add_vs_shapley"] for r in rs]
        rows.append([key[0], key[1], len(rs), f"{np.median([x['tau'] for x in m]):.2f} / {min(x['tau'] for x in m):.2f}",
                     f"{100 * np.mean([x['top1_agree'] for x in m]):.0f} %", f"{np.mean([x['top3_overlap'] for x in m]):.2f}",
                     max(x["max_rank_disp"] for x in m), f"{np.median([x['misordered_pair_frac'] for x in m]):.3f}",
                     f"{np.median([x['tau'] for x in ma]):.2f}", f"{100 * np.mean([x['top1_agree'] for x in ma]):.0f} %"])
    return ("### E12 — leave-one-out / difference-reward screening vs exact Shapley responsibility "
            "(remove[p] = v(N) − v(N∖{p}); add[p] = v({p}) − v(∅))\n" +
            table(["source", "pattern", "instances", "τ(remove, φ) median / min", "top-1 agree", "top-3 overlap",
                   "max rank displacement", "mis-ordered pairs (median frac.)", "τ(add, φ) median", "top-1 agree (add)"], rows))


def e13():
    recs = [r for r in load("e13") if r.get("status") == "OK"]
    if not recs:
        return "E13: NOT RUN"
    rows = []
    for n in sorted({r["n"] for r in recs}):
        rs = [r for r in recs if r["n"] == n]
        g = np.array([r["gap_max"] for r in rs]); z = np.array([r["gap_zero_frac"] for r in rs])
        tau = np.array([r["tau_step_vs_memrand"] for r in rs]); t1 = np.array([r["top1_agree"] for r in rs])
        sd = np.array([r["shapley_max_abs_diff"] for r in rs])
        rows.append([n, len(rs), f"{100 * np.mean(g > 1e-9):.0f} %", f"{np.median(g):.4f} / {g.max():.4f}",
                     f"{100 * z.mean():.0f} %", f"{tau.mean():.2f}", f"{100 * t1.mean():.0f} %", f"{np.median(sd):.4f} / {sd.max():.4f}"])
    return ("### E13 — semantic bracket: step-dependent (engine, PRISM-games F<=k) vs memoryless free strategies on tiny random CSGs "
            "(v_step ≤ v_mem ≤ v_memdet; bracket [v_step, v_memrand] per coalition; Shapley values compared at both ends)\n" +
            table(["n", "instances", "instances with a non-zero gap", "max gap median / max", "coalitions with zero gap",
                   "Kendall τ(φ_step, φ_mem) mean", "top-1 agreement", "max|Δφ| median / max"], rows))


def e14():
    recs = load("e14")
    if not recs:
        return "E14: NOT RUN"
    out = []
    for iid in sorted({r["instance_id"] for r in recs}):
        rs = [r for r in recs if r["instance_id"] == iid]
        ag = next((r for r in rs if r.get("arm") == "agent_exact"), None)
        ex = next((r for r in rs if r.get("arm") == "exact_step"), None)
        line = f"**{iid}**"
        if ag:
            v = ag["values"]
            line += (f": |S| = {ag['states']}, v(∅) = {v['empty']:.4f}, v({{P0}}) = {v['P0']:.4f}, v({{P1}}) = {v['P1']:.4f}, "
                     f"v(N) = {v['N']:.4f}, Υ = {ag['attributable_value']:.4f}, agent-level φ = {np.round(ag['shapley_agent'], 4).tolist()}")
        if ex:
            line += (f": exact step-level (N = {2 * ex['k']}, {ex['n_coalitions']} coalitions, {ex['wall_s']:.1f} s): Υ = {ex['attributable_value']:.4f}, "
                     f"top-5 (player, tick) by φ = {ex['top5_shapley']}, by LOO = {ex['top5_remove']}, τ(LOO, φ) = {ex['tau_remove_vs_shapley']:.2f}")
        out.append(line)
        rows = []
        for r in sorted([r for r in rs if (r.get("arm") or "").startswith("cert")], key=lambda r: (r["arm"], r["seed"])):
            rows.append([r["arm"], r["seed"], r.get("status"), r.get("certified"), fmt(r.get("wall_s")), r.get("n_perms"), r.get("n_solves"),
                         fmt(r.get("hw_max"), 4), fmt(r.get("eps_abs"), 4), str(r.get("top5_players") or r.get("top_k_certified") or "—")[:80],
                         fmt(r.get("tau_remove_vs_est")), r.get("covers_exact", "—")])
        if rows:
            out.append(table(["arm", "seed", "status", "certified", "wall [s]", "perms", "solves", "max hw", "eps", "top-5 (player, tick)", "τ(LOO, φ̂)", "covers exact"], rows))
    return "### E14 — Overcooked-AI cramped_room: who and when (step-level responsibility for the late pot)\n" + "\n\n".join(out)


if __name__ == "__main__":
    sections = [e1cal(), e0(), e0hz(), e9(), e1(), e1("e1perm", "E1-perm — locality-hostile sensitivity (random state permutation), identical batches"), e1("e1prune", "E1-prune — horizon-pruned track (exact k-reachable sub-model on BOTH engines; |S| column = unpruned size, see pruned_states in records)"), e2(), e6(), e4(), e3(),
                e3("e3b", "### E3b — PRISM-games benchmark suite [KNPS18/KNPS19] instances imported with uniform observed policies (same protocol as E3)"), e3storm(), e10(),
                e9v3(), e4("e4v2"), e11(), e12(), e13(), e14()]
    text = "\n\n".join(sections)
    print(text)
    if "--write" in sys.argv:
        from datetime import datetime
        with open(os.path.join(REPO, "RESULTS.md"), "a") as fh:
            fh.write(f"\n\n## Auto-generated summary tables ({datetime.now().isoformat(timespec='minutes')}, from results/*.json)\n\n" + text + "\n")
