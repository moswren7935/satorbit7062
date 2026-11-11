import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import e5_gen as G
import e5_ref as E


@pytest.mark.parametrize("a,m", G.E5_WITHIN + G.E5_BEYOND)
def test_exact_am_valid_probabilities_absorbing_fail(a, m):
    gm = G.e5_model(a, m, "1e3")
    v = G.validate(gm)
    assert v["active_agents_per_nonterminal_state"] == [a]
    assert v["actions_per_active_agent"] == [m]
    assert v["rows_per_nonterminal_state"] == [m ** a]
    assert v["max_row_sum_dev"] <= 1e-12 and v["max_policy_sum_dev"] <= 1e-12
    assert v["fail_absorbing"]
    assert gm.n_states == 993
    assert gm.nnz == 2 * (gm.n_rows - 1) + 1
    assert gm.z_model == gm.nnz + gm.n_rows

    assert G.agents_active_in_reach(gm) == list(range(16))


def test_sizes():
    assert G.e5_model(2, 4, "1e4").n_states == 10001


def test_deterministic():
    a = G.e5_model(3, 4, "1e3"); b = G.e5_model(3, 4, "1e3")
    assert a.fingerprint() == b.fingerprint()


def test_paper_layout_conversion():
    gm = G.e5_model(3, 4, "1e3")
    c3 = G.to_compiled3(gm)
    assert c3.state_agent.shape == (gm.n_states, 3) and c3.weights.shape == (gm.n_states, 3, 8)
    assert np.array_equal(c3.state_agent, gm.state_agent[:, :3]) and np.all(gm.state_agent[:, 3] == -1)
    with pytest.raises(ValueError):
        G.to_compiled3(G.e5_model(4, 4, "1e3"))


def test_batch_masks_full_enumeration_random_order():
    mk = G.e5_batch_masks()
    assert mk.shape == (65536,) and mk.dtype == np.uint64
    assert np.array_equal(np.sort(mk), np.arange(65536, dtype=np.uint64))
    assert not np.array_equal(mk[:64], np.arange(64, dtype=np.uint64))
    assert np.array_equal(mk, G.e5_batch_masks())


@pytest.mark.parametrize("cfg", [dict(n=5, a=2, m=3, R=2, k=4), dict(n=5, a=4, m=3, R=1, k=3),
                                 dict(n=4, a=3, m=4, R=1, k=3),
                                 dict(n=6, a=1, m=2, R=2, k=4,
                                      stage_config=[(1, 2), (2, 3), (3, 2), (4, 2), (2, 8), (1, 5)])])
def test_numpy_reference_equals_vi_ref(cfg):
    from cubcr import vi_ref
    gm = G.build(**cfg, seed=7)
    csg = G.to_csg(gm)
    rng = np.random.default_rng(1)
    masks = [0, (1 << gm.n_agents) - 1] + rng.integers(0, 1 << gm.n_agents, 6).tolist()
    a = np.array([vi_ref.solve(csg, int(mm), gm.horizon_k) for mm in masks])
    b = E.NumpyRef(gm).solve_many(masks, gm.horizon_k)
    assert np.max(np.abs(a - b)) <= 1e-15


def test_monotone_and_nonzero_gap():
    gm = G.e5_model(2, 4, "1e3")
    ref = E.NumpyRef(gm)
    v0, v1 = ref.solve_many([0, (1 << 16) - 1], gm.horizon_k)
    assert v1 - v0 > 0.05
    rng = np.random.default_rng(3)
    for _ in range(5):
        perm = rng.permutation(16)
        mask, prev = 0, v0
        for i in perm[:6]:
            mask |= 1 << int(i)
            cur = ref.solve(mask, gm.horizon_k)
            assert cur >= prev - 1e-15
            prev = cur
