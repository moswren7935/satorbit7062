from __future__ import annotations

import itertools
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubcr import vi_ref
from cubcr.model import CSG
from cubcr.shapley_sampled import kendall_tau
from experiments import protocol
from generators.vehicle import vehicle_csg
from tests.helpers import random_csg


def exact_shapley(table: np.ndarray, N: int) -> np.ndarray:
    phi = np.zeros(N)
    for S in range(1 << N):
        s = bin(S).count("1")
        for p in range(N):
            if not (S >> p) & 1:
                w = math.factorial(s) * math.factorial(N - s - 1) / math.factorial(N)
                phi[p] += w * (table[S | (1 << p)] - table[S])
    return phi


def reach_within_k(T: np.ndarray, kind: np.ndarray, init: int, k: int) -> np.ndarray:

    B, S, _ = T.shape
    x = np.tile((kind == 1).astype(np.float64), (B, 1))
    tgt = kind == 1
    avd = kind == 2
    for _ in range(k):
        x = np.einsum("bij,bj->bi", T, x)
        x[:, tgt] = 1.0
        x[:, avd] = 0.0
    return x[:, init]


def build_dtmcs(csg: CSG, mask: int, free_choice):


    S = csg.n_states
    kind = np.zeros(S, dtype=np.int8)
    kind[csg.fail_states] = 1
    kind[csg.avoid_states] = 2
    B = next(iter(free_choice.values())).shape[0] if free_choice else 1
    T = np.zeros((B, S, S))
    for s, d in enumerate(csg.state_defs):
        if kind[s] != 0:
            T[:, s, s] = 1.0
            continue
        for joint, succ in d.rows:
            w = np.ones(B)
            for slot, a in enumerate(d.active):
                ai = joint[slot]
                if (mask >> a) & 1:
                    w = w * csg.observed_policy[(s, a)][ai]
                else:
                    w = w * free_choice[(s, a)][:, ai]
            for sp, p in succ:
                T[:, s, sp] += w * p
    return T, kind


def free_pairs(csg: CSG, mask: int):
    return [(s, a, csg.state_defs[s].n_actions[slot])
            for s, d in enumerate(csg.state_defs) if s not in csg.fail_states and s not in csg.avoid_states
            for slot, a in enumerate(d.active) if not (mask >> a) & 1]


def v_memdet(csg: CSG, mask: int, k: int, max_profiles: int = 1 << 18):
    pairs = free_pairs(csg, mask)
    if not pairs:
        T, kind = build_dtmcs(csg, mask, {})
        return float(reach_within_k(T, kind, csg.init_state, k)[0]), 0, None
    radices = [m for (_, _, m) in pairs]
    n_prof = int(np.prod(radices))
    if n_prof > max_profiles:
        return None, n_prof, None
    grid = np.array(list(itertools.product(*[range(m) for m in radices])), dtype=np.int64)
    choice = {}
    for j, (s, a, m) in enumerate(pairs):
        oh = np.zeros((n_prof, m))
        oh[np.arange(n_prof), grid[:, j]] = 1.0
        choice[(s, a)] = oh
    T, kind = build_dtmcs(csg, mask, choice)
    vals = reach_within_k(T, kind, csg.init_state, k)
    j = int(np.argmin(vals))
    return float(vals[j]), n_prof, grid[j]


def v_memrand(csg: CSG, mask: int, k: int, best_det, rng, restarts: int = 8, iters: int = 60):

    pairs = free_pairs(csg, mask)
    if not pairs:
        return None
    radices = [m for (_, _, m) in pairs]

    def evaluate(params_list):
        B = len(params_list)
        choice = {}
        for j, (s, a, m) in enumerate(pairs):
            choice[(s, a)] = np.stack([p[j] for p in params_list], axis=0)
        T, kind = build_dtmcs(csg, mask, choice)
        return reach_within_k(T, kind, csg.init_state, k)

    def project(p):

        u = np.sort(p)[::-1]
        css = np.cumsum(u)
        rho = np.nonzero(u * np.arange(1, len(p) + 1) > (css - 1))[0][-1]
        theta = (css[rho] - 1) / (rho + 1)
        return np.maximum(p - theta, 0.0)

    best = math.inf
    starts = []
    if best_det is not None:
        starts.append([np.eye(m)[best_det[j]] for j, m in enumerate(radices)])
    for _ in range(restarts):
        starts.append([rng.dirichlet(np.ones(m)) for m in radices])
    h = 1e-6
    for params in starts:
        params = [p.copy() for p in params]
        f0 = float(evaluate([params])[0])
        step = 0.5
        for it in range(iters):

            pert = []
            for j, m in enumerate(radices):
                for ai in range(m):
                    q = [p.copy() for p in params]
                    q[j][ai] += h
                    pert.append(q)
            fs = evaluate(pert)
            g = [(fs[idx:idx + m] - f0) / h for idx, m in zip(np.cumsum([0] + radices[:-1]), radices)]
            new = [project(p - step * gj) for p, gj in zip(params, g)]
            f1 = float(evaluate([new])[0])
            if f1 < f0 - 1e-12:
                params, f0 = new, f1
            else:
                step *= 0.5
                if step < 1e-4:
                    break
        best = min(best, f0)
    return float(best)


def run_instance(args):
    name, csg_json, k, seed = args
    csg = CSG.from_json(csg_json) if isinstance(csg_json, dict) else csg_json
    rng = np.random.default_rng(seed + 977)
    n = csg.n_agents
    t0 = time.perf_counter()
    rows = []
    v_s = np.zeros(1 << n); v_d = np.zeros(1 << n); v_r = np.zeros(1 << n)
    for mask in range(1 << n):
        vs = float(vi_ref.solve(csg, mask, k=k))
        vd, n_prof, best = v_memdet(csg, mask, k)
        if vd is None:
            return {"instance_id": name, "status": "NOT RUN", "note": f"{n_prof} deterministic profiles > cap"}
        vr = v_memrand(csg, mask, k, best, rng) if best is not None else vd
        vr = min(vd, vr) if vr is not None else vd
        if vs > vr + 1e-9:
            raise RuntimeError(f"bracket violated: v_step {vs} > v_memrand {vr} (mask {mask})")
        v_s[mask], v_d[mask], v_r[mask] = vs, vd, vr
        rows.append({"mask": mask, "v_step": vs, "v_memdet": vd, "v_memrand": vr, "n_profiles": n_prof})
    phi_s = exact_shapley(v_s, n); phi_r = exact_shapley(v_r, n); phi_d = exact_shapley(v_d, n)
    gaps = v_r - v_s
    rec = protocol.make_record(
        instance_id=name, exp="e13", engine="naive_python", dtype="fp64", n=n, states=csg.n_states, k=k, seed=seed,
        threads=1, cap_s=3600, wall_s=time.perf_counter() - t0, n_coalitions=1 << n, coalitions=rows,
        upsilon_step=float(v_s[-1] - v_s[0]), upsilon_memrand=float(v_r[-1] - v_r[0]),
        upsilon_memdet=float(v_d[-1] - v_d[0]),
        gap_max=float(gaps.max()), gap_mean=float(gaps.mean()), gap_zero_frac=float((gaps <= 1e-9).mean()),
        gap_memdet_max=float((v_d - v_s).max()),
        shapley_step=phi_s.tolist(), shapley_memrand=phi_r.tolist(), shapley_memdet=phi_d.tolist(),
        tau_step_vs_memrand=kendall_tau(phi_s, phi_r), tau_step_vs_memdet=kendall_tau(phi_s, phi_d),
        top1_agree=bool(int(np.argmax(phi_s)) == int(np.argmax(phi_r))),
        shapley_max_abs_diff=float(np.abs(phi_s - phi_r).max()),
        n_free_pairs_max=max(len(free_pairs(csg, m)) for m in range(1 << n)))
    return rec


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/e13")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    jobs = []
    for n in (2, 3):
        for S in (5, 6, 8):
            for k in (2, 3, 4):
                for seed in range(a.seeds):
                    csg = random_csg(n_agents=n, n_states=S, k=k, seed=seed, max_active=2, max_actions=2,
                                     min_actions=2, succ_per_row=2)
                    jobs.append((f"e13_rand_n{n}_S{S}_k{k}_s{seed}", csg.to_json(), k, seed))
    csg = vehicle_csg()
    jobs.append(("e13_munajib_vehicle_k1", csg.to_json(), csg.horizon_k, 0))
    print(f"[e13] {len(jobs)} instances, {a.procs} processes", flush=True)
    t0 = time.perf_counter()
    with Pool(a.procs) as pool:
        for i, rec in enumerate(pool.imap_unordered(run_instance, jobs)):
            protocol.write_result(a.out, rec, filename=f"{rec['instance_id']}.json")
            if i % 20 == 0 or rec.get("status") != "OK":
                print(f"  [{i + 1}/{len(jobs)}] {rec['instance_id']} gap_max={rec.get('gap_max')} "
                      f"tau={rec.get('tau_step_vs_memrand')} ({time.perf_counter() - t0:.0f}s)", flush=True)
    print(f"[e13] done in {time.perf_counter() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
