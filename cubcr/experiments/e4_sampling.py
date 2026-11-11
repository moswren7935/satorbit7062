#!/usr/bin/env python3


from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments import protocol
from generators.ticketbench import build_compiled
from cubcr.engine_gpu import GPUEngine
from cubcr.shapley_sampled import sample_shapley, topk_certified, exact_player_table, kendall_tau
from cubcr.screening import singleton_screening
from cubcr.shapley import shapley_from_table

INSTANCE_SEED = 20260829
SEEDS = [20260829, 20260830, 20260831]
TRAJECTORY_SEED = 20260829

CONFIGS = {
    "default": [(3, 60, 4, True), (8, 1000, 25, False), (8, 10000, 25, False), (16, 1000, 25, False),
                (16, 10000, 25, False), (12, 100000, 25, False)],
    "smoke": [(3, 60, 4, True)],
}
EPS_LIST_OBS = [0.05, 0.02, 0.01, 0.005, 0.002]


def gpu_lib_provenance() -> dict:
    path = os.environ.get("CUBCR_GPU_LIB") or os.path.join(REPO, "csrc", "build", "libcubcr_gpu.so")
    out = {"libcubcr_gpu_path": path, "libcubcr_gpu_sha256": None, "libcubcr_gpu_bytes": None}
    if os.path.exists(path):
        data = open(path, "rb").read()
        out["libcubcr_gpu_sha256"] = hashlib.sha256(data).hexdigest()
        out["libcubcr_gpu_bytes"] = len(data)
        out["libcubcr_gpu_mtime"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(path)))
    return out


def _cert_json(c):
    if c is None:
        return None
    d = {}
    for key, val in c.items():
        d[key] = val.tolist() if isinstance(val, np.ndarray) else val
    return d


def rule_list(top_k, eps_list):
    rules = [(f"top{top_k}", {"top_k": int(top_k)})]
    rules += [(f"eps{e:g}", {"eps": float(e)}) for e in eps_list]
    return rules


def result_name(iid, rule, seed):
    return f"e4__{iid}__{rule}__s{seed}.json"


def run_one(eng, comp, meta, n, k, rule_name, rule_kw, seed, a, tags, prov, screening_rec, exact, label,
            warmup_s):
    iid = meta["instance_id"]
    N = n * k
    curve = []
    obs = {f"eps_{e:g}": None for e in EPS_LIST_OBS}
    obs["top_k_separated"] = None
    t0 = time.perf_counter()

    def cb(m, est, hw):
        t = time.perf_counter() - t0
        row = {"m": int(m), "t_s": t, "hw_max": float(hw.max()), "hw_median": float(np.median(hw)),
               "hw_mean": float(hw.mean())}
        if label == "trajectory":
            ok, _ = topk_certified(est, hw, a.top_k)
            row["top_k_separated"] = bool(ok)
            for e in EPS_LIST_OBS:
                if obs[f"eps_{e:g}"] is None and hw.max() <= e:
                    obs[f"eps_{e:g}"] = {"m": int(m), "t_s": t}
            if ok and obs["top_k_separated"] is None:
                obs["top_k_separated"] = {"m": int(m), "t_s": t}
        curve.append(row)

    v2_kw = {}
    if a.value_range != "none":
        v2_kw["value_range"] = a.value_range
    if a.ci != "eb_looks":
        v2_kw.update(ci=a.ci, batch_schedule=a.batch_schedule)
    res = sample_shapley(eng, kind="step", n=n, k=k, seed=seed, batch_perms=a.batch_perms, alpha=a.alpha,
                         max_perms=a.max_perms, time_budget=a.time_budget, antithetic=True, callback=cb,
                         two_sided=True, bonferroni=True, sequential=True, **rule_kw, **v2_kw)
    wall = res.wall_time
    est = res.estimates.reshape(n, k)
    hw = res.half_widths.reshape(n, k)
    cert = res.certificate
    fired = res.certified


    time_censored = res.status == "time"
    perm_censored = (label == "certification") and (res.status == "max_perms")
    status = "TO" if time_censored else "OK"
    top5 = np.argsort(-res.estimates, kind="stable")[:a.top_k].tolist()
    time_to_certify = {}
    if fired:
        time_to_certify[cert["kind"] if cert["kind"] == "eps" else "top_k"] = {"m": cert["m"], "t_s": cert["wall_s"]}
    sampling = {"wall_s": wall, "engine_s": res.engine_time, "n_perms": res.n_perms, "n_solves": res.n_solves,
                "n_requested": res.n_coalitions_requested, "cache_hits": res.cache_hits, "n_samples": res.n_samples,
                "n_engine_calls": res.n_engine_calls, "n_batches": res.n_batches,
                "solves_per_s": res.n_solves / wall if wall > 0 else None,
                "status": res.status, "certified": fired, "anytime_valid": res.anytime_valid,
                "batch_perms": a.batch_perms, "perms_per_unit": res.perms_per_unit, "max_perms": a.max_perms,
                "time_budget_s": a.time_budget, "budget_overshoot_s": res.budget_overshoot_s,
                "v_empty": res.v_empty, "v_full": res.v_full,
                "efficiency_gap": float(abs(res.estimates.sum() - (res.v_full - res.v_empty))),
                "estimates": est.tolist(), "half_widths": hw.tolist(), "hw_max_final": float(hw.max()),
                "hw_median_final": float(np.median(hw)), "top5_players": top5,
                "time_to_certify": time_to_certify, "curve": curve}
    if label == "trajectory":
        sampling["observed_crossings_not_certificates"] = obs
    rec = protocol.make_record(
        instance_id=iid, exp=a.exp, label=label, rule=rule_name, rule_params=rule_kw,
        sampler_v2={"value_range": a.value_range, "ci": a.ci, "batch_schedule": a.batch_schedule, "prune": a.prune,
                    "ci_method": res.ci_method, "fp_term": res.fp_term, "eps_upsilon": res.eps_upsilon,
                    "batch_sizes": res.batch_sizes, "cs_anomaly": res.cs_anomaly, "n_padded_rows": res.n_padded_rows,
                    "graph_captures": res.graph_captures, "value_range_used": list(res.value_range) if res.value_range else None},
        engine="gpu_batched", dtype="fp64", n=n, states=int(comp.n_states), k=k, seed=seed,
        instance_seed=INSTANCE_SEED, n_players=N, threads=0, cap_s=float(a.time_budget), cap_scope="sample_shapley time_budget",
        status=status, censored=bool(time_censored),
        wall_s=max(wall, float(a.time_budget)) if time_censored else wall,
        coalition_solves_per_s=None if time_censored else (res.n_solves / wall if wall > 0 else None),
        certified=bool(fired), certificate_kind=None if cert is None else cert["kind"],
        time_to_certify_s=None if not fired else float(cert["wall_s"]),
        certify_censored=bool(not fired and label == "certification"),
        censor_reason=("time" if time_censored else ("max_perms" if perm_censored else None)),
        certificate=_cert_json(cert), alpha_allocation=res.alpha_allocation(), alpha=a.alpha,
        sampling=sampling, screening=screening_rec, warmup_s=warmup_s,
        gpu_stats_last=getattr(eng, "stats", None), use_graph=True,
        driver=tags.get("driver"), nvcc=tags.get("nvcc"), provenance=prov,
        instance_meta={kk: vv for kk, vv in meta.items() if kk != "params"},
        note=("certification run: terminated by its own stopping rule or censored" if label == "certification"
              else "trajectory run (no stopping rule): CI-convergence curve only; crossings are observations, not certificates"))
    if exact is not None:
        phi = exact["phi"]
        err = np.abs(phi - res.estimates)
        rec["exact"] = {"phi": phi.reshape(n, k).tolist(), "max_abs_err": float(err.max()),
                        "all_inside_ci": bool(np.all(err <= res.half_widths)),
                        "exact_top5": exact["top5"], "kendall_tau": kendall_tau(res.estimates, phi),
                        "n_coalitions": int(exact["table"].shape[0])}
        rec["kendall_tau"] = rec["exact"]["kendall_tau"]
        if fired and cert["kind"] == "top_k":
            rec["exact"]["certified_top_set_equals_exact"] = bool(set(cert["top_set"]) == set(exact["top_k_set"]))
    path = protocol.write_result(a.out, rec, filename=result_name(iid, rule_name, seed), tags=tags)
    print(f"[e4] {iid} {label} {rule_name} s{seed}: status={res.status} certified={fired} "
          f"t={wall:.1f}s perms={res.n_perms} solves={res.n_solves} looks={res.n_looks_used}/{res.n_looks_max} "
          f"hw_max={hw.max():.4f} overshoot={res.budget_overshoot_s:.2f}s", flush=True)
    return rec


def run_instance(n, S, k, exact_check, a, tags, prov):
    t_build = time.perf_counter()
    comp, meta = build_compiled(n, S, k, 0.05, "none", INSTANCE_SEED)
    if a.prune:
        from cubcr.model import restrict_to_horizon
        full_states = int(comp.n_states)
        comp = restrict_to_horizon(comp, k)
        meta = dict(meta, pruned_from_states=full_states, pruned_states=int(comp.n_states))
    build_s = time.perf_counter() - t_build
    iid = meta["instance_id"]
    N = n * k
    rules = rule_list(a.top_k, a.eps)
    todo = [(r, kw, s) for s in a.seeds for (r, kw) in rules]
    if not a.no_trajectory:
        todo.append(("trajectory", {}, TRAJECTORY_SEED))
    if not a.force:
        todo = [t for t in todo if not os.path.exists(os.path.join(a.out, result_name(iid, t[0], t[2])))]
    if not todo:
        print(f"[e4] {iid}: all records present, skipping", flush=True)
        return
    eng = GPUEngine(comp, dtype="fp64", use_graph=True)
    try:

        rng = np.random.default_rng(0)
        B = a.batch_perms * 2 * (N + 1)
        warm = rng.integers(0, 1 << n, size=(k, B), dtype=np.uint64)
        t0 = time.perf_counter()
        eng.solve(warm, k)
        warmup_s = time.perf_counter() - t0

        scr = singleton_screening(eng, n, k, kind="step")
        screening_rec = {"wall_s": scr.wall_time, "n_solves": scr.n_solves, "n_engine_calls": scr.n_engine_calls,
                         "add": scr.add.tolist(), "remove": scr.remove.tolist(), "v_empty": scr.v_empty,
                         "v_full": scr.v_full, "attributable": scr.attributable,
                         "top5_by_remove": scr.top_players(a.top_k, by="remove").tolist(),
                         "top5_by_add": scr.top_players(a.top_k, by="add").tolist()}
        exact = None
        if exact_check and N <= 16:
            t0 = time.perf_counter()
            table = exact_player_table(eng, N, kind="step", n=n, k=k)
            phi = shapley_from_table(np.asarray(table), N)
            order = np.argsort(-phi, kind="stable")
            exact = {"table": table, "phi": phi, "top5": order[:5].tolist(), "top_k_set": order[:a.top_k].tolist(),
                     "wall_s": time.perf_counter() - t0}
            screening_rec["exact_tau_remove_vs_shapley"] = kendall_tau(scr.remove.reshape(-1), phi)
            screening_rec["exact_tau_add_vs_shapley"] = kendall_tau(scr.add.reshape(-1), phi)
        exact_msg = "" if exact is None else f" exact table {exact['wall_s']:.2f}s"
        print(f"[e4] {iid}: build {build_s:.2f}s warm-up {warmup_s:.2f}s screening {scr.wall_time:.3f}s "
              f"({scr.n_solves} solves){exact_msg}", flush=True)
        recs = []
        for rule_name, kw, seed in todo:
            label = "trajectory" if rule_name == "trajectory" else "certification"
            recs.append(run_one(eng, comp, meta, n, k, rule_name, kw, seed, a, tags, prov, screening_rec, exact, label,
                                warmup_s))
    finally:
        eng.close()
    write_summary(iid, n, S, k, a, tags, prov, build_s, warmup_s, screening_rec, exact)


def write_summary(iid, n, S, k, a, tags, prov, build_s, warmup_s, screening_rec, exact):

    rules = [r for r, _ in rule_list(a.top_k, a.eps)]
    per_rule = {}
    for rule in rules:
        runs = []
        for seed in a.seeds:
            p = os.path.join(a.out, result_name(iid, rule, seed))
            if not os.path.exists(p):
                continue
            r = protocol.read_result(p)
            runs.append({"seed": seed, "status": r["status"], "sampler_status": r["sampling"]["status"],
                         "certified": r["certified"], "censored": r["censored"], "censor_reason": r.get("censor_reason"),
                         "time_to_certify_s": r.get("time_to_certify_s"), "wall_s": r["sampling"]["wall_s"],
                         "n_perms": r["sampling"]["n_perms"], "n_solves": r["sampling"]["n_solves"],
                         "n_looks_used": r["alpha_allocation"]["n_looks_used"],
                         "hw_max_final": r["sampling"]["hw_max_final"],
                         "top_set": None if r.get("certificate") is None else r["certificate"].get("top_set"),
                         "exact_top_ok": (r.get("exact") or {}).get("certified_top_set_equals_exact")})
        cert_times = [x["time_to_certify_s"] for x in runs if x["certified"]]
        per_rule[rule] = {"runs": runs, "n_runs": len(runs), "n_certified": len(cert_times),
                          "n_censored_time": sum(1 for x in runs if x["censor_reason"] == "time"),
                          "n_censored_max_perms": sum(1 for x in runs if x["censor_reason"] == "max_perms"),
                          "time_to_certify_median_s": float(np.median(cert_times)) if len(cert_times) == len(runs) and runs else None,
                          "time_to_certify_note": ("median over seeds (all certified)" if runs and len(cert_times) == len(runs)
                                                   else "not reported: at least one seed did not certify (time cap -> >= cap_s; "
                                                        "or max_perms exhausted) -> only a lower bound is given"),
                          "time_to_certify_lower_bound_s": (None if not runs or len(cert_times) == len(runs)
                                                            else float(min([x["wall_s"] for x in runs if not x["certified"]] + cert_times)))}
    traj_path = os.path.join(a.out, result_name(iid, "trajectory", TRAJECTORY_SEED))
    traj = None
    if os.path.exists(traj_path):
        t = protocol.read_result(traj_path)
        traj = {"seed": TRAJECTORY_SEED, "n_perms": t["sampling"]["n_perms"], "wall_s": t["sampling"]["wall_s"],
                "status": t["sampling"]["status"], "hw_max_final": t["sampling"]["hw_max_final"],
                "observed_crossings_not_certificates": t["sampling"].get("observed_crossings_not_certificates"),
                "alpha_allocation": t["alpha_allocation"]}
    summ = {"instance_id": iid, "exp": "e4", "n": n, "states_requested": S, "k": k, "n_players": n * k,
            "instance_seed": INSTANCE_SEED, "seeds": a.seeds, "alpha": a.alpha, "top_k": a.top_k, "eps_list": a.eps,
            "max_perms": a.max_perms, "time_budget_s": a.time_budget, "batch_perms": a.batch_perms,
            "engine": "gpu_batched", "dtype": "fp64", "build_s": build_s, "warmup_s": warmup_s,
            "screening": {kk: vv for kk, vv in screening_rec.items() if kk not in ("add", "remove")},
            "exact": None if exact is None else {"wall_s": exact["wall_s"], "top5": exact["top5"],
                                                  "n_coalitions": int(exact["table"].shape[0])},
            "per_rule": per_rule, "trajectory": traj, "gpu": tags.get("gpu"), "driver": tags.get("driver"),
            "cuda": tags.get("cuda"), "git": tags.get("git"), "provenance": prov, "timestamp": protocol.now_iso()}
    path = os.path.join(a.out, f"e4_summary__{iid}.json")
    with open(path, "w") as fh:
        json.dump(summ, fh, indent=1, default=protocol._json_default)
    print(f"[e4] summary -> {path}", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="default", choices=sorted(CONFIGS))
    ap.add_argument("--instances", default=None, help="comma-separated indices into the config list")
    ap.add_argument("--max-perms", type=int, default=None, help="default 20000 (smoke: 500)")
    ap.add_argument("--time-budget", type=float, default=None, help="seconds per run; default 300 (smoke: 60)")
    ap.add_argument("--batch-perms", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--eps", default="0.02,0.01,0.005")
    ap.add_argument("--seeds", default=None, help="comma-separated sampler seeds; default 20260829,20260830,20260831 (smoke: first only)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "e4"))
    ap.add_argument("--force", action="store_true", help="re-run even if the record exists")
    ap.add_argument("--no-trajectory", action="store_true")

    ap.add_argument("--value-range", default="none", choices=["none", "monotone"])
    ap.add_argument("--ci", default="eb_looks", choices=["eb_looks", "prpl_eb_cs"])
    ap.add_argument("--batch-schedule", default="fixed", choices=["fixed", "geometric"])
    ap.add_argument("--prune", action="store_true", help="restrict_to_horizon(comp, k) before sampling")
    ap.add_argument("--exp", default="e4", help="exp label written into the records (e4 | e4v2)")
    a = ap.parse_args(argv)
    smoke = a.configs == "smoke"
    a.max_perms = a.max_perms if a.max_perms is not None else (500 if smoke else 20000)
    a.time_budget = a.time_budget if a.time_budget is not None else (60.0 if smoke else 300.0)
    a.seeds = [int(s) for s in a.seeds.split(",")] if a.seeds else (SEEDS[:1] if smoke else list(SEEDS))
    a.eps = [float(e) for e in a.eps.split(",") if e]
    tags = protocol.machine_tags()
    prov = gpu_lib_provenance()
    os.makedirs(a.out, exist_ok=True)
    cfgs = CONFIGS[a.configs]
    if a.instances:
        cfgs = [cfgs[int(i)] for i in a.instances.split(",")]
    print(f"[e4] gpu={tags.get('gpu')} driver={tags.get('driver')} cuda={tags.get('cuda')} git={tags.get('git')} "
          f"lib_sha256={prov['libcubcr_gpu_sha256']} max_perms={a.max_perms} budget={a.time_budget}s seeds={a.seeds} "
          f"rules={[r for r, _ in rule_list(a.top_k, a.eps)]} out={a.out}", flush=True)
    for n, S, k, ex in cfgs:
        run_instance(n, S, k, ex, a, tags, prov)
    return 0


if __name__ == "__main__":
    sys.exit(main())
