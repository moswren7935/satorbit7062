import json
import os
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cubcr import vi_ref, shapley
from cubcr.model import load_csg, DMAX
from generators import ticketbench as tb

SEED = 20260829
TINY = dict(retries=2)


def full_table(csg):
    return np.array([vi_ref.solve(csg, m) for m in range(1 << csg.n_agents)])


@pytest.mark.parametrize("states", [200, 2000, 20000])
@pytest.mark.parametrize("n", [4, 8])
def test_validate_and_state_count(states, n):
    csg = tb.build_instance(n, states, 25, 0.05, "none", SEED)
    csg.validate()
    assert abs(csg.n_states - states) <= 0.2 * states
    assert csg.meta["states"] == csg.n_states
    assert csg.meta["states_requested"] == states
    assert csg.meta["horizon_k"] == csg.horizon_k == 25
    assert csg.meta["n_agents"] == csg.n_agents == n
    assert csg.meta["seed"] == SEED
    assert csg.meta["instance_id"] and csg.meta["generator_version"]
    assert csg.meta["params"]["pattern"] == "none"
    assert max(len(d.active) for d in csg.state_defs) <= 2 <= DMAX
    assert max((max(d.n_actions) if d.n_actions else 1) for d in csg.state_defs) <= 4


def test_linear_in_n_for_fixed_requests():

    sizes = {}
    for n in (4, 8, 16):
        csg = tb.build_instance(n, 2 * 3 * n * 10 + 1, 10, 0.05, "none", SEED)
        assert csg.meta["n_requests"] == 10
        sizes[n] = csg.n_states
    assert sizes[8] - 1 == 2 * (sizes[4] - 1) and sizes[16] - 1 == 2 * (sizes[8] - 1)


def test_locality_stage_major():
    csg = tb.build_instance(8, 5000, 25, 0.05, "none", SEED)
    loc = csg.meta["locality"]
    c = csg.compile()
    src = np.repeat(np.arange(c.n_states), np.diff(c.state_row_start))
    src = np.repeat(src, np.diff(c.row_ptr))
    jump = np.abs(c.col.astype(np.int64) - src)
    nonfail = c.col != csg.fail_states[0]
    assert np.mean(jump[nonfail] <= loc["window_states"]) > 0.99
    assert loc["frac_nonfail_edges_within_window"] > 0.99


@pytest.mark.parametrize("pattern", tb.PATTERNS)
def test_all_patterns_validate_and_are_labelled(pattern):
    csg = tb.build_instance(4, 50, 6, 0.1, pattern, SEED, **TINY)
    csg.validate()
    assert csg.n_states <= 60
    info = csg.meta["pattern_info"]
    assert info["pattern"] == pattern
    assert csg.meta["params"]["pattern"] == pattern
    assert len(csg.agent_names) == csg.n_agents
    if pattern == "dummy":
        assert info["dummy_agents"] == [3]
    elif pattern != "none":
        assert info["pair"] == [2, 3]


@pytest.mark.parametrize("pattern", tb.PATTERNS)
def test_failure_is_avoidable(pattern):
    csg = tb.build_instance(4, 50, 6, 0.1, pattern, SEED, **TINY)
    n = csg.n_agents
    vN = vi_ref.solve(csg, (1 << n) - 1)
    v0 = vi_ref.solve(csg, 0)
    assert vN > v0 + 1e-3
    assert vN > 0.0


def test_dummy_agent_has_zero_shapley():
    csg = tb.build_instance(4, 50, 6, 0.1, "dummy", SEED, **TINY)
    n = csg.n_agents
    d = csg.meta["pattern_info"]["dummy_agents"][0]
    v = full_table(csg)
    for m in range(1 << n):
        if not (m >> d) & 1:
            assert abs(v[m | (1 << d)] - v[m]) <= 1e-15
    phi = shapley.shapley_from_table(v, n)
    assert abs(phi[d]) < 1e-12
    assert phi.sum() == pytest.approx(v[-1] - v[0], abs=1e-12)


def test_symmetric_pair_has_equal_shapley():
    csg = tb.build_instance(4, 50, 6, 0.1, "symmetric", SEED, **TINY)
    a, b = csg.meta["pattern_info"]["pair"]
    v = full_table(csg)
    phi = shapley.shapley_from_table(v, csg.n_agents)
    assert phi[a] == pytest.approx(phi[b], abs=1e-13)
    assert phi[a] > 1e-3
    A, B = 1 << a, 1 << b
    for m in range(1 << csg.n_agents):
        if not (m & A) and not (m & B):
            assert v[m | A] == pytest.approx(v[m | B], abs=1e-15)


def test_joint_pair_signature():


    csg = tb.build_instance(4, 50, 6, 0.1, "joint", SEED, **TINY)
    n = csg.n_agents
    a, b = csg.meta["pattern_info"]["pair"]
    A, B = 1 << a, 1 << b
    v = full_table(csg)
    others = [m for m in range(1 << n) if not (m & A) and not (m & B)]
    for m in others:
        assert abs(v[m | A] - v[m]) <= 1e-15
        assert abs(v[m | B] - v[m]) <= 1e-15
        assert v[m | A | B] >= v[m | A] - 1e-15
    full = (1 << n) - 1
    assert v[full] - v[full & ~A] > 1e-3
    assert v[full] - v[full & ~B] > 1e-3
    phi = shapley.shapley_from_table(v, n)
    assert phi[a] == pytest.approx(phi[b], abs=1e-13)


def test_redundant_pair_signature():

    csg = tb.build_instance(4, 50, 6, 0.1, "redundant", SEED, **TINY)
    n = csg.n_agents
    a, b = csg.meta["pattern_info"]["pair"]
    A, B = 1 << a, 1 << b
    v = full_table(csg)
    others = [m for m in range(1 << n) if not (m & A) and not (m & B)]
    for m in others:
        assert abs(v[m | A] - v[m | A | B]) <= 1e-15
        assert abs(v[m | B] - v[m | A | B]) <= 1e-15
        assert v[m | A] >= v[m] - 1e-15
    rest = (1 << n) - 1 - A - B
    assert v[rest | A] - v[rest] > 1e-3
    assert v[rest | B] - v[rest] > 1e-3
    phi = shapley.shapley_from_table(v, n)
    assert phi[a] == pytest.approx(phi[b], abs=1e-13)


def test_determinism_same_seed_identical_json(tmp_path):
    a = tb.build_instance(5, 300, 12, 0.07, "none", SEED)
    b = tb.build_instance(5, 300, 12, 0.07, "none", SEED)
    ja = json.dumps(a.to_json(), sort_keys=True)
    jb = json.dumps(b.to_json(), sort_keys=True)
    assert ja == jb
    c = tb.build_instance(5, 300, 12, 0.07, "none", SEED + 1)
    assert json.dumps(c.to_json(), sort_keys=True) != ja

    out = tmp_path / "tb"
    rc = tb.main(["--n", "5", "--states", "300", "--k", "12", "--pfail", "0.07", "--pattern", "none",
                  "--seed", str(SEED), "--out", str(out), "--emit", "json,npz"])
    assert rc == 0
    path = out / (a.meta["instance_id"] + ".json")
    assert path.exists()
    loaded = load_csg(str(path))
    assert json.dumps(loaded.to_json(), sort_keys=True) == ja
    comp, meta = tb.load_compiled_npz(str(out / (a.meta["instance_id"] + ".npz")))
    assert comp.fingerprint() == a.compile().fingerprint()
    assert meta["instance_id"] == a.meta["instance_id"]


def test_prism_emission_optional(tmp_path):
    csg = tb.build_instance(4, 50, 6, 0.1, "none", SEED, **TINY)
    files = tb.emit_prism(csg, str(tmp_path), csg.meta["instance_id"])
    try:
        import cubcr.io_prism
    except ImportError:
        assert files == []
        return
    assert len(files) == 4
    txt = open(files[0]).read()
    assert txt.startswith("//") or txt.startswith("csg")
    assert 'label "fail"' in txt
    assert "Pmin=?" in open(files[1]).read()


@pytest.mark.parametrize("pattern", tb.PATTERNS)
def test_build_compiled_matches_compile(pattern):
    csg = tb.build_instance(4, 60, 6, 0.1, pattern, SEED)
    comp, meta = tb.build_compiled(4, 60, 6, 0.1, pattern, SEED)
    assert comp.fingerprint() == csg.compile().fingerprint()
    assert meta["instance_id"] == csg.meta["instance_id"]


def test_cpu_engine_agrees_with_reference_all_coalitions():
    from cubcr.engine_cpu import CPUEngine
    csg = tb.build_instance(4, 50, 8, 0.1, "none", SEED, **TINY)
    masks = np.arange(1 << 4, dtype=np.uint64)
    ref = np.array([vi_ref.solve(csg, int(m)) for m in masks])
    got = CPUEngine(csg.compile(), n_threads=2, block=8).solve(masks)
    assert np.max(np.abs(got - ref)) <= 1e-12
    assert ref[-1] > ref[0]


def test_build_1e5_states_fast():
    t0 = time.perf_counter()
    csg = tb.build_instance(8, 100_000, 25, 0.05, "none", SEED)
    dt = time.perf_counter() - t0
    assert abs(csg.n_states - 100_000) <= 20_000
    assert dt < 15.0, f"build_instance(1e5) took {dt:.1f}s"


from cubcr.model import reachable_within_k, permute_states

REVIEW = dict(n=4, states=100, k=25, pfail=0.1)


def _python_bfs(csg, k):

    fail = set(csg.fail_states) | set(csg.avoid_states)
    seen = {csg.init_state}
    frontier = {csg.init_state}
    for _ in range(k):
        nxt = set()
        for s in frontier:
            if s in fail:
                continue
            for _joint, succ in csg.state_defs[s].rows:
                for sp, p in succ:
                    if p > 0.0 and sp not in seen:
                        seen.add(sp)
                        nxt.add(sp)
        if not nxt:
            break
        frontier = nxt
    return seen


@pytest.mark.parametrize("k", [0, 1, 3, 4, 7, 25, 400])
def test_reachable_within_k_matches_python_bfs(k):
    csg = tb.build_instance(**REVIEW, pattern="none", seed=SEED)
    comp = csg.compile()
    n_reach, frac, mask = reachable_within_k(comp, k)
    ref = _python_bfs(csg, k)
    assert set(np.flatnonzero(mask).tolist()) == ref
    assert n_reach == len(ref) and frac == pytest.approx(len(ref) / comp.n_states)
    assert mask.dtype == bool and mask.shape == (comp.n_states,) and mask[comp.init_state]

    assert bool(mask[csg.fail_states[0]]) == (k >= csg.meta["n_stages"])


def test_reachability_closed_form_and_meta_fields():


    for n, states, k in ((8, 10_000, 25), (8, 10_000, 100), (12, 1_000, 400), (16, 2_000, 60)):
        comp, meta = tb.build_compiled(n, states, k, 0.05, "none", SEED)
        ns, C, R = meta["n_stages"], meta["retry_budget"], meta["n_requests"]
        closed = int(k >= ns) + sum((2 if j >= 1 else 1) for r in range(R) for j in range(ns) for c in range(C)
                                    if 2 * r + j + c <= k)
        assert meta["n_reachable_k"] == closed == reachable_within_k(comp, k)[0]
        assert meta["frac_reachable_k"] == pytest.approx(closed / comp.n_states)
        for key in ("n_states", "n_rows", "nnz", "max_rows_per_state", "n_reachable_k", "frac_reachable_k",
                    "hazard", "permute"):
            assert key in meta
        assert meta["n_states"] == comp.n_states and meta["n_rows"] == comp.n_rows and meta["nnz"] == comp.nnz
        assert meta["max_rows_per_state"] == comp.max_rows_per_state == 16
        assert meta["hazard"] == 0.0 and meta["permute"] == "none" and meta["v_stats"] is None

    csg = tb.build_instance(**REVIEW, pattern="none", seed=SEED)
    assert csg.meta["n_reachable_k"] == reachable_within_k(csg.compile(), 25)[0]
    assert 0.0 < csg.meta["frac_reachable_k"] < 1.0


def test_hazard_makes_failure_unavoidable_but_attributable():

    base = tb.build_instance(**REVIEW, pattern="none", seed=SEED)
    assert base.meta["hazard"] == 0.0 and "_hz" not in base.meta["instance_id"]
    assert vi_ref.solve(base, 0) == 0.0
    for pattern in ("none", "dummy"):
        csg = tb.build_instance(**REVIEW, pattern=pattern, seed=SEED, hazard=0.02)
        csg.validate()
        n = csg.n_agents
        assert 90 <= csg.n_states <= 110
        assert csg.meta["hazard"] == 0.02 and csg.meta["params"]["hazard"] == 0.02
        assert csg.meta["instance_id"].endswith("_hz0.02")
        v = full_table(csg)
        assert v[0] > 1e-3
        assert v[-1] > v[0] + 1e-3
        assert np.all(np.diff(np.sort(v)) >= -1e-15)
        phi = shapley.shapley_from_table(v, n)
        assert phi.sum() == pytest.approx(v[-1] - v[0], abs=1e-12)
        assert np.all(phi >= -1e-12)
        if pattern == "dummy":
            d = csg.meta["pattern_info"]["dummy_agents"][0]
            for m in range(1 << n):
                if not (m >> d) & 1:
                    assert abs(v[m | (1 << d)] - v[m]) <= 1e-15
            assert abs(phi[d]) < 1e-12

        from cubcr.engine_cpu import CPUEngine
        got = CPUEngine(csg.compile(), n_threads=2, block=8).solve(np.arange(1 << n, dtype=np.uint64))
        assert np.max(np.abs(got - v)) <= 1e-12

    csg = tb.build_instance(**REVIEW, pattern="none", seed=SEED, hazard=0.02)
    fail = csg.fail_states[0]
    ns, C = csg.meta["n_stages"], csg.meta["retry_budget"]
    for s, d in enumerate(csg.state_defs[:-1]):
        j = (s // 2 // C) % ns
        for _joint, succ in d.rows:
            pf = sum(p for sp, p in succ if sp == fail)
            if j == ns - 1:
                assert pf >= 0.02 - 1e-12
            else:
                assert pf == 0.0

    comp, meta = tb.build_compiled(**REVIEW, pattern="none", seed=SEED, hazard=0.02)
    assert comp.fingerprint() == csg.compile().fingerprint() and meta["instance_id"] == csg.meta["instance_id"]
    with pytest.raises(ValueError):
        tb.build_compiled(**REVIEW, pattern="none", seed=SEED, hazard=1.5)


def test_permute_states_preserves_all_coalition_values():

    from cubcr.engine_cpu import CPUEngine
    masks = np.arange(1 << 4, dtype=np.uint64)
    for hazard in (0.0, 0.02):
        c0, m0 = tb.build_compiled(**REVIEW, pattern="none", seed=SEED, hazard=hazard)
        c1, m1 = tb.build_compiled(**REVIEW, pattern="none", seed=SEED, hazard=hazard, permute="random")
        assert m1["permute"] == "random" and m1["instance_id"] == m0["instance_id"] + "_permrandom"
        assert c1.n_states == c0.n_states and c1.n_rows == c0.n_rows and c1.nnz == c0.nnz
        assert c1.fingerprint() != c0.fingerprint() and c1.init_state != 0
        assert (m1["n_reachable_k"], m1["frac_reachable_k"]) == (m0["n_reachable_k"], m0["frac_reachable_k"])
        v0 = CPUEngine(c0, n_threads=2, block=8).solve(masks)
        v1 = CPUEngine(c1, n_threads=2, block=8).solve(masks)
        assert np.max(np.abs(v0 - v1)) <= 1e-15
        assert v1[-1] > v1[0]

        sm = np.random.default_rng(3).integers(0, 16, size=(25, 8)).astype(np.uint64)
        assert np.max(np.abs(CPUEngine(c0, n_threads=2, block=8).solve(sm) -
                             CPUEngine(c1, n_threads=2, block=8).solve(sm))) <= 1e-15


    c0, _ = tb.build_compiled(**REVIEW, pattern="none", seed=SEED)
    perm = np.random.default_rng(11).permutation(c0.n_states)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(c0.n_states)
    c1 = permute_states(c0, perm)
    assert permute_states(c1, inv).fingerprint() == c0.fingerprint()
    assert c1.init_state == perm[c0.init_state]
    assert np.array_equal(c1.state_kind[perm], c0.state_kind)
    assert np.array_equal(c1.weights[perm], c0.weights)
    for s in range(c0.n_states):
        r0, r1 = c0.state_row_start[s], c0.state_row_start[s + 1]
        q0, q1 = c1.state_row_start[perm[s]], c1.state_row_start[perm[s] + 1]
        assert r1 - r0 == q1 - q0
        for i in range(r1 - r0):
            a, b = c0.row_ptr[r0 + i], c0.row_ptr[r0 + i + 1]
            e, f = c1.row_ptr[q0 + i], c1.row_ptr[q0 + i + 1]
            assert np.array_equal(perm[c0.col[a:b]], c1.col[e:f]) and np.array_equal(c0.val[a:b], c1.val[e:f])

    assert tb.build_compiled(**REVIEW, pattern="none", seed=SEED, permute="random")[0].fingerprint() == \
        tb.build_compiled(**REVIEW, pattern="none", seed=SEED, permute="random")[0].fingerprint()
    with pytest.raises(ValueError):
        permute_states(c0, np.zeros(c0.n_states, dtype=np.int64))
    with pytest.raises(ValueError):
        tb.build_compiled(**REVIEW, pattern="none", seed=SEED, permute="shuffle")
    with pytest.raises(ValueError):
        tb.build_instance(**REVIEW, pattern="none", seed=SEED, permute="random")


def test_permute_random_destroys_locality():
    c0, m0 = tb.build_compiled(8, 5000, 25, 0.05, "none", SEED)
    c1, m1 = tb.build_compiled(8, 5000, 25, 0.05, "none", SEED, permute="random")
    win = m0["locality"]["window_states"]

    def frac_within_window(c):
        src = np.repeat(np.arange(c.n_states), np.diff(c.state_row_start))
        src = np.repeat(src, np.diff(c.row_ptr))
        fail = int(np.flatnonzero(c.state_kind == 1)[0])
        nonfail = c.col != fail
        return float(np.mean(np.abs(c.col.astype(np.int64) - src)[nonfail] <= win))
    assert frac_within_window(c0) > 0.99
    assert frac_within_window(c1) < 0.05
    assert "note" in m1["locality"]


def test_build_compiled_1e6_with_reachability_and_permutation_fast():
    t0 = time.perf_counter()
    comp, meta = tb.build_compiled(16, 1_000_000, 400, 0.05, "none", SEED, permute="random")
    dt = time.perf_counter() - t0
    assert abs(comp.n_states - 1_000_000) <= 1000
    assert meta["n_reachable_k"] == 18264 and meta["frac_reachable_k"] < 0.02
    assert dt < 60.0, f"build_compiled(1e6, permute=random) took {dt:.1f}s"


def test_cli_hazard_and_permute(tmp_path):
    rc = tb.main(["--n", "4", "--states", "100", "--k", "25", "--pfail", "0.1", "--hazard", "0.02",
                  "--permute", "random", "--seed", str(SEED), "--out", str(tmp_path), "--emit", "npz"])
    assert rc == 0
    comp0, meta0 = tb.build_compiled(**REVIEW, pattern="none", seed=SEED, hazard=0.02, permute="random")
    comp, meta = tb.load_compiled_npz(str(tmp_path / (meta0["instance_id"] + ".npz")))
    assert comp.fingerprint() == comp0.fingerprint()
    assert meta["hazard"] == 0.02 and meta["permute"] == "random" and meta["n_reachable_k"] == meta0["n_reachable_k"]
    with pytest.raises(SystemExit):
        tb.main(["--n", "4", "--states", "100", "--k", "25", "--permute", "random", "--out", str(tmp_path),
                 "--emit", "json"])
