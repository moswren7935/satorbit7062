from __future__ import annotations

import itertools

import numpy as np
import pytest

from cubcr import shapley, vi_ref
from cubcr.engine_cpu import CPUEngine
from cubcr.model import CSG, DMAX, CompiledCSG, StateDef, permute_states, reachable_within_k
from generators import ticketbench as tb
from tests.helpers import random_csg

SEED = 20260829


def python_bfs(csg: CSG, k: int) -> set:

    stop = set(csg.fail_states) | set(csg.avoid_states)
    depth = {csg.init_state: 0}
    frontier = [csg.init_state]
    for d in range(k):
        nxt = []
        for s in frontier:
            if s in stop:
                continue
            for _joint, succ in csg.state_defs[s].rows:
                for sp, _p in succ:
                    if sp not in depth:
                        depth[sp] = d + 1
                        nxt.append(sp)
        frontier = nxt
    return set(depth)


def compiled_vi(c: CompiledCSG, cols) -> np.ndarray:


    S = c.n_states
    kind = np.asarray(c.state_kind)
    V = (kind == 1).astype(np.float64)
    for t in reversed(range(len(cols))):
        m = int(cols[t])
        Vn = np.empty(S)
        for s in range(S):
            if kind[s] == 1:
                Vn[s] = 1.0
                continue
            if kind[s] == 2:
                Vn[s] = 0.0
                continue
            r0, r1 = int(c.state_row_start[s]), int(c.state_row_start[s + 1])
            q = [float(np.dot(c.val[c.row_ptr[r]:c.row_ptr[r + 1]], V[c.col[c.row_ptr[r]:c.row_ptr[r + 1]]]))
                 for r in range(r0, r1)]
            rad = [int(c.state_radix[s, d]) for d in range(DMAX)]
            ag = [int(c.state_agent[s, d]) for d in range(DMAX)]
            pinned = [ag[d] >= 0 and ((m >> ag[d]) & 1) == 1 for d in range(DMAX)]
            acc = {}
            for digits in itertools.product(*[range(r) for r in rad]):
                r_local = 0
                w = 1.0
                for d in range(DMAX):
                    r_local = r_local * rad[d] + digits[d]
                    if pinned[d]:
                        w *= float(c.weights[s, d, digits[d]])
                key = tuple(digits[d] for d in range(DMAX) if not pinned[d])
                acc[key] = acc.get(key, 0.0) + w * q[r_local]
            Vn[s] = min(acc.values())
        V = Vn
    return V


def chain_csg(L: int, branch: bool = False) -> CSG:

    sds = []
    for s in range(L):
        succ_a = [(s + 1, 1.0)]
        succ_b = [(L - 1, 1.0)] if (branch and s == 0) else [(s + 1, 1.0)]
        sds.append(StateDef(active=[0], n_actions=[2], rows=[((0,), succ_a), ((1,), succ_b)]))
    sds.append(StateDef(active=[], n_actions=[], rows=[((), [(L, 1.0)])]))
    obs = {(s, 0): np.array([1.0, 0.0]) for s in range(L)}
    return CSG(n_agents=1, horizon_k=L, init_state=0, fail_states=[L], state_defs=sds, observed_policy=obs)


@pytest.mark.parametrize("L", [1, 2, 5, 9])
def test_reachable_within_k_chain_exact_depths(L):
    comp = chain_csg(L).compile()
    for k in range(0, L + 3):
        n_reach, frac, mask = reachable_within_k(comp, k)
        expect = min(k, L) + 1
        assert n_reach == expect and frac == pytest.approx(expect / (L + 1))
        assert np.array_equal(np.flatnonzero(mask), np.arange(expect))
        assert bool(mask[L]) == (k >= L)

    comp = chain_csg(6, branch=True).compile()
    assert bool(reachable_within_k(comp, 1)[2][5]) and not bool(reachable_within_k(comp, 1)[2][6])
    assert bool(reachable_within_k(comp, 2)[2][6])


def test_reachable_within_k_target_init_and_avoid_not_expanded():
    csg = chain_csg(4)
    csg.init_state = 4
    comp = csg.compile()
    assert reachable_within_k(comp, 10)[0] == 1
    csg = chain_csg(4)
    csg.avoid_states = [2]
    comp = csg.compile()
    assert reachable_within_k(comp, 10)[0] == 3
    with pytest.raises(ValueError):
        reachable_within_k(comp, -1)


@pytest.mark.parametrize("seed", [31, 32, 33, 34])
def test_reachable_within_k_random_csg_vs_python_bfs(seed):
    csg = random_csg(n_agents=4, n_states=40, k=6, seed=seed, max_active=3, max_actions=3, n_fail=2, n_avoid=2,
                     succ_per_row=2)
    comp = csg.compile()
    for k in range(0, 9):
        n_reach, frac, mask = reachable_within_k(comp, k)
        ref = python_bfs(csg, k)
        assert set(np.flatnonzero(mask).tolist()) == ref, (seed, k)
        assert n_reach == len(ref) and frac == pytest.approx(len(ref) / comp.n_states)
    assert reachable_within_k(comp, 0)[0] == 1


@pytest.mark.parametrize("seed", [41, 42])
def test_permute_states_values_invariant_random_csg(seed):
    csg = random_csg(n_agents=4, n_states=24, k=5, seed=seed, max_active=3, max_actions=3, n_fail=2, n_avoid=1)
    c0 = csg.compile()
    perm = np.random.default_rng(seed).permutation(c0.n_states)
    c1 = permute_states(c0, perm)
    assert c1.init_state == perm[c0.init_state]
    assert set(np.flatnonzero(c1.state_kind == 1).tolist()) == set(perm[csg.fail_states].tolist())
    masks = np.arange(1 << 4, dtype=np.uint64)
    v_ref = vi_ref.solve_many(csg, [int(m) for m in masks])
    v0 = CPUEngine(c0, n_threads=2, block=4).solve(masks)
    v1 = CPUEngine(c1, n_threads=2, block=4).solve(masks)
    assert np.max(np.abs(v0 - v_ref)) <= 1e-12
    assert np.max(np.abs(v1 - v0)) <= 1e-15

    for m in (0, 5, 10, 15):
        vp = compiled_vi(c1, [m] * csg.horizon_k)[c1.init_state]
        assert abs(vp - v_ref[m]) <= 1e-12

    sm = np.random.default_rng(seed + 100).integers(0, 16, size=(csg.horizon_k, 6)).astype(np.uint64)
    s0 = CPUEngine(c0, n_threads=2, block=4).solve(sm)
    s1 = CPUEngine(c1, n_threads=2, block=4).solve(sm)
    assert np.max(np.abs(s0 - s1)) <= 1e-15
    cols = [int(x) for x in sm[:, 0]]
    assert np.max(np.abs(compiled_vi(c1, cols)[perm] - compiled_vi(c0, cols))) <= 1e-15
    assert abs(compiled_vi(c0, cols)[c0.init_state] - vi_ref.solve(csg, cols)) <= 1e-12
    assert abs(s0[0] - vi_ref.solve(csg, cols)) <= 1e-12

    inv = np.empty_like(perm)
    inv[perm] = np.arange(c0.n_states)
    assert permute_states(c1, inv).fingerprint() == c0.fingerprint()


def test_permute_states_rejects_non_permutations():
    c0 = chain_csg(5).compile()
    S = c0.n_states
    for bad in (np.arange(S) + 1, np.arange(S) - 1, np.r_[0, 0, np.arange(2, S)], np.arange(S - 1), np.arange(S + 1)):
        with pytest.raises(ValueError):
            permute_states(c0, bad)
    assert permute_states(c0, np.arange(S)).fingerprint() == c0.fingerprint()


def test_hazard_row_algebra_on_csr_and_stochasticity():
    h = 0.02
    for pattern in ("none", "symmetric"):
        c0, m0 = tb.build_compiled(4, 100, 25, 0.1, pattern, SEED)
        c1, m1 = tb.build_compiled(4, 100, 25, 0.1, pattern, SEED, hazard=h)
        assert m1["instance_id"] == m0["instance_id"] + f"_hz{h:g}" and m1["hazard"] == h
        assert c1.n_states == c0.n_states and c1.n_rows == c0.n_rows
        assert np.array_equal(c1.state_row_start, c0.state_row_start) and np.array_equal(c1.weights, c0.weights)
        fail = int(np.flatnonzero(c1.state_kind == 1)[0])
        ns, C = m0["n_stages"], m0["retry_budget"]
        commit = {s for s in range(c0.n_states - 1) if ((s // 2) // C) % ns == ns - 1}
        assert commit
        row_sums = np.add.reduceat(c1.val, c1.row_ptr[:-1])
        assert np.max(np.abs(row_sums - 1.0)) <= 1e-12 and c1.val.min() > 0.0
        for s in range(c0.n_states):
            for r in range(int(c0.state_row_start[s]), int(c0.state_row_start[s + 1])):
                d0 = dict(zip(c0.col[c0.row_ptr[r]:c0.row_ptr[r + 1]].tolist(), c0.val[c0.row_ptr[r]:c0.row_ptr[r + 1]]))
                d1 = dict(zip(c1.col[c1.row_ptr[r]:c1.row_ptr[r + 1]].tolist(), c1.val[c1.row_ptr[r]:c1.row_ptr[r + 1]]))
                if s in commit:
                    expect = {sp: (1 - h) * p for sp, p in d0.items()}
                    expect[fail] = expect.get(fail, 0.0) + h
                else:
                    expect = d0
                assert set(d1) == set(expect), (pattern, s, r)
                for sp in expect:
                    assert abs(d1[sp] - expect[sp]) <= 1e-15, (pattern, s, r, sp)


@pytest.mark.parametrize("pattern,C", [("none", 3), ("none", 2), ("dummy", 2), ("joint", 3)])
def test_hazard_v_empty_threshold_is_stall_budget(pattern, C):

    lay = tb.make_layout(tb.Params(n=4, states=60, k=1, pfail=0.1, pattern=pattern, seed=SEED, retries=C, hazard=0.02))
    thr = (lay.n_stage - 1) * C + 1
    below = tb.build_instance(4, 60, thr - 1, 0.1, pattern, SEED, retries=C, hazard=0.02)
    above = tb.build_instance(4, 60, thr, 0.1, pattern, SEED, retries=C, hazard=0.02)
    assert below.meta["fail_reachable_within_k"] and thr - 1 >= lay.n_stage
    assert vi_ref.solve(below, 0) == 0.0 and vi_ref.solve(below, 15) > 0.0
    v0, vN = vi_ref.solve(above, 0), vi_ref.solve(above, 15)
    assert 0.0 < v0 < vN
    phi = shapley.shapley_from_table(vi_ref.solve_many(above, range(16)), 4)
    assert phi.sum() == pytest.approx(vN - v0, abs=1e-12)


def test_degenerate_parameters_reported_not_rejected():
    comp, meta = tb.build_compiled(4, 100, 25, 0.0, "none", SEED, pleak=0.0)
    assert meta["fail_reachable_within_k"] is False and meta["n_reachable_k"] == reachable_within_k(comp, 25)[0]
    comp, meta = tb.build_compiled(4, 100, 25, 0.0, "none", SEED, pleak=0.0, hazard=0.01)
    assert meta["fail_reachable_within_k"] is True
    csg = tb.build_instance(4, 100, 25, 0.0, "none", SEED, pleak=0.0)
    assert csg.meta["fail_reachable_within_k"] is False and vi_ref.solve(csg, 15) == 0.0


def test_gpu_permutation_invariance_light():
    gpu = pytest.importorskip("cubcr.engine_gpu")
    if gpu.device_count() == 0:
        pytest.skip("no CUDA device")
    c0, _ = tb.build_compiled(4, 100, 25, 0.1, "none", SEED, hazard=0.02)
    c1, _ = tb.build_compiled(4, 100, 25, 0.1, "none", SEED, hazard=0.02, permute="random")
    masks = np.arange(16, dtype=np.uint64)
    g0 = gpu.GPUEngine(c0).solve(masks)
    g1 = gpu.GPUEngine(c1).solve(masks)
    cpu = CPUEngine(c0, n_threads=2, block=8).solve(masks)
    assert np.max(np.abs(g0 - g1)) <= 1e-15
    assert np.max(np.abs(g1 - cpu)) <= 1e-12
    assert g1[0] > 0.0 and g1[-1] > g1[0]
