import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubcr import vi_ref, shapley
from generators import ticketbench as tb

SEED = 20260829


def _table(csg):
    return np.array([vi_ref.solve(csg, m) for m in range(1 << csg.n_agents)])


def _sid(csg, r, j, c, h):
    m = csg.meta
    return ((r * m["n_stages"] + j) * m["retry_budget"] + c) * 2 + h


def _decode(csg, s):
    m = csg.meta
    C, ns = m["retry_budget"], m["n_stages"]
    h = s % 2
    s //= 2
    c = s % C
    s //= C
    return s // ns, s % ns, c, h


@pytest.mark.parametrize("kw", [dict(pleak=1.5), dict(prec=-0.1), dict(pcomp=1.2), dict(q=2.0), dict(q=-0.5),
                                dict(rho=-1.0), dict(retries=0), dict(concentration=0.0)])
@pytest.mark.parametrize("pattern", ["none", "joint"])
def test_bad_parameters_raise_on_both_build_paths(kw, pattern):
    with pytest.raises(ValueError):
        tb.build_instance(4, 50, 6, 0.1, pattern, SEED, **kw)
    with pytest.raises(ValueError):
        tb.build_compiled(4, 50, 6, 0.1, pattern, SEED, **kw)


@pytest.mark.parametrize("args", [(1, 50, 6, 0.1, "none"), (2, 50, 6, 0.1, "dummy"), (2, 50, 6, 0.1, "joint"),
                                  (4, 50, -1, 0.1, "none"), (4, 50, 6, 1.5, "none"), (4, 50, 6, 0.1, "bogus")])
def test_bad_shape_arguments_raise(args):
    with pytest.raises(ValueError):
        tb.build_instance(*args, SEED)


def _simulate(csg, n_samples, rng):
    k, fail = csg.horizon_k, set(csg.fail_states)
    hits = 0
    for _ in range(n_samples):
        s = csg.init_state
        for _t in range(k):
            if s in fail:
                break
            d = csg.state_defs[s]
            idx = 0
            for ag, m in zip(d.active, d.n_actions):
                idx = idx * m + rng.choice(m, p=csg.observed_policy[(s, ag)])
            succ = d.rows[idx][1]
            pr = np.array([p for _, p in succ])
            s = succ[rng.choice(len(succ), p=pr / pr.sum())][0]
        hits += s in fail
    return hits / n_samples


@pytest.mark.parametrize("pattern,kw", [("none", dict(retries=2)), ("redundant", dict(retries=2)),
                                        ("symmetric", dict(retries=3))])
def test_all_pinned_value_matches_monte_carlo(pattern, kw):
    csg = tb.build_instance(4, 50, 8, 0.2, pattern, SEED, **kw)
    csg.validate()
    vN = vi_ref.solve(csg, (1 << csg.n_agents) - 1)
    N = 20000
    est = _simulate(csg, N, np.random.default_rng(1))
    se = np.sqrt(max(vN * (1 - vN), 1e-9) / N)
    assert abs(est - vN) < 4.5 * se, (pattern, vN, est)


@pytest.mark.parametrize("pattern", tb.PATTERNS)
def test_structural_invariants(pattern):
    csg = tb.build_instance(5, 400, 12, 0.2, pattern, 3, retries=3)
    csg.validate()
    m = csg.meta
    C, ns, R, F = m["retry_budget"], m["n_stages"], m["n_requests"], m["fail_state"]
    assert csg.n_states == 2 * C * ns * R + 1
    last = ns - 1
    for s in range(csg.n_states - 1):
        r, j, c, h = _decode(csg, s)
        d = csg.state_defs[s]
        assert d.active == m["stage_agents"][j]
        allowed = {_sid(csg, (r + 1) % R, 0, 0, 0), F}
        if j < last:
            allowed |= {_sid(csg, r, j + 1, 0, 0), _sid(csg, r, j + 1, 0, 1)}
        if c < C - 1:
            allowed |= {_sid(csg, r, j, c + 1, 0), _sid(csg, r, j, c + 1, 1)}
        for joint, succ in d.rows:
            cols = [sp for sp, _ in succ]
            assert cols == sorted(cols) and len(set(cols)) == len(cols)
            assert abs(sum(p for _, p in succ) - 1.0) < 1e-12
            for sp, p in succ:
                assert sp in allowed and 0.0 < p <= 1.0 + 1e-12
                if sp == F:
                    assert j == last, "FAIL must only be reachable from the last stage"
        kind = m["stage_kinds"][j]
        if kind == "single_dummy":
            slot = d.active.index(m["pattern_info"]["dummy_agents"][0])
            groups = {}
            for joint, succ in d.rows:
                groups.setdefault(tuple(a for i, a in enumerate(joint) if i != slot), []).append(tuple(succ))
            assert all(all(x == lst[0] for x in lst) for lst in groups.values())
        if kind == "pair":
            rule, q = m["pattern_info"]["rule"], m["params"]["q"]
            qh = q if h == 0 else 1.0
            for (a, b), succ in d.rows:
                pf = dict(succ).get(F, 0.0)
                ba, bb = a == tb.PROCEED, b == tb.PROCEED
                exp = {"and": qh if (ba and bb) else 0.0, "or": qh if (ba or bb) else 0.0,
                       "add": qh * (ba + bb) / 2.0}[rule]
                assert abs(pf - exp) < 1e-15


@pytest.mark.parametrize("seed", [1, 7])
@pytest.mark.parametrize("pattern", tb.PATTERNS)
def test_signatures_and_monotonicity_other_seeds(seed, pattern):
    csg = tb.build_instance(4, 50, 8, 0.3, pattern, seed, retries=3, q=0.8)
    csg.validate()
    n = csg.n_agents
    v = _table(csg)
    for m in range(1 << n):
        for i in range(n):
            if not (m >> i) & 1:
                assert v[m | (1 << i)] >= v[m] - 1e-12, "v must be monotone in the pinned set"
    assert v[-1] > v[0]
    phi = shapley.shapley_from_table(v, n)
    assert abs(phi.sum() - (v[-1] - v[0])) < 1e-12
    info = csg.meta["pattern_info"]
    if pattern == "dummy":
        d = info["dummy_agents"][0]
        assert max(abs(v[m | (1 << d)] - v[m]) for m in range(1 << n) if not (m >> d) & 1) < 1e-14
        assert abs(phi[d]) < 1e-12
    elif pattern != "none":
        a, b = info["pair"]
        A, B = 1 << a, 1 << b
        others = [m for m in range(1 << n) if not (m & A) and not (m & B)]
        assert abs(phi[a] - phi[b]) < 1e-13
        if pattern == "joint":
            assert all(abs(v[m | A] - v[m]) < 1e-14 and abs(v[m | B] - v[m]) < 1e-14 for m in others)
            full = (1 << n) - 1
            assert v[full] - v[full & ~A] > 1e-3
        elif pattern == "redundant":
            assert all(abs(v[m | A] - v[m | A | B]) < 1e-14 and abs(v[m | B] - v[m | A | B]) < 1e-14 for m in others)
            rest = (1 << n) - 1 - A - B
            assert v[rest | A] - v[rest] > 1e-3


def test_horizon_edge_cases():
    csg = tb.build_instance(4, 50, 0, 0.1, "none", SEED)
    csg.validate()
    assert csg.meta["fail_reachable_within_k"] is False
    assert all(vi_ref.solve(csg, m) == 0.0 for m in range(16))
    csg = tb.build_instance(4, 50, 3, 0.1, "none", SEED)
    assert csg.meta["fail_reachable_within_k"] is False and vi_ref.solve(csg, 15) == 0.0
    csg = tb.build_instance(4, 50, 4, 0.1, "none", SEED)
    assert csg.meta["fail_reachable_within_k"] is True and vi_ref.solve(csg, 15) > 0.0


def test_small_n_and_single_request_layouts():
    csg = tb.build_instance(2, 40, 6, 0.1, "none", SEED)
    csg.validate()
    assert csg.meta["stage_agents"] == [[0], [0, 1]]
    assert vi_ref.solve(csg, 3) > vi_ref.solve(csg, 0)
    csg = tb.build_instance(20, 100, 30, 0.1, "none", SEED)
    csg.validate()
    assert csg.meta["n_requests"] == 1 and csg.n_states == 2 * 3 * 20 + 1
    assert len(csg.agent_names) == 20 == csg.n_agents


def test_step_level_masks_cpu_engine_vs_reference():
    from cubcr.engine_cpu import CPUEngine
    csg = tb.build_instance(4, 50, 6, 0.1, "none", SEED, retries=2)
    eng = CPUEngine(csg.compile(), n_threads=2, block=8)
    k = csg.horizon_k
    cols = np.random.default_rng(0).integers(0, 16, size=(k, 12)).astype(np.uint64)
    got = eng.solve(cols)
    ref = np.array([vi_ref.solve(csg, [int(cols[t, b]) for t in range(k)]) for b in range(12)])
    assert np.max(np.abs(got - ref)) <= 1e-12


def test_build_compiled_rejects_nothing_valid_and_row_sums():
    comp, meta = tb.build_compiled(6, 3000, 25, 0.05, "symmetric", SEED)
    rs = np.add.reduceat(comp.val, comp.row_ptr[:-1])
    assert np.max(np.abs(rs - 1.0)) < 1e-12 and comp.val.min() > 0.0 and comp.val.max() <= 1.0
    assert meta["states"] == comp.n_states and abs(comp.n_states - 3000) <= 600
    assert np.allclose(comp.weights.sum(axis=2), 1.0)


def test_prism_crosscheck_dummy_and_symmetric(tmp_path):
    try:
        from cubcr import io_prism
    except ImportError:
        pytest.skip("cubcr.io_prism not available")
    if not io_prism.prism_available():
        pytest.skip("PRISM-games not built")
    for pattern, masks in [("dummy", [7, 6]), ("symmetric", [4, 15])]:
        csg = tb.build_instance(4, 50, 6, 0.1, pattern, SEED, retries=2)
        res = io_prism.crosscheck(csg, masks=masks, workdir=str(tmp_path), timeout_s=120)
        assert res["all_match"], res
        assert res["max_abs_err"] <= 1e-9
