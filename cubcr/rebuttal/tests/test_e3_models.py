import json

import numpy as np
import pytest

import rb
import e3_models as M

GRID = [(n, S, k) for n in rb.GRID_N for S in rb.GRID_STATES for k in rb.GRID_K]


@pytest.mark.parametrize("gen_seed", rb.PAPER_GEN_SEEDS)
def test_tb_phased_equals_build_compiled(gen_seed):
    from generators.ticketbench import build_compiled
    for n, S, k in GRID:
        c_ref, m_ref = build_compiled(n, S, k, M.PAPER_PFAIL, "none", gen_seed)
        c, m, t = M.tb_build_phased(n, S, k, gen_seed)
        assert c.fingerprint() == c_ref.fingerprint()
        assert json.dumps(m, sort_keys=True, default=str) == json.dumps(m_ref, sort_keys=True, default=str)
        assert all(v >= 0 for v in t.values())


def test_decision_time_model_id():
    from generators.ticketbench import build_compiled
    c, m, _ = M.tb_build_phased(**M.DT_SPEC)
    assert m["instance_id"] == "tb_none_n12_S100009_k25_pf0.05_C3_s20260829"
    c_ref, _ = build_compiled(12, 100000, 25, 0.05, "none", 20260829)
    assert c.fingerprint() == c_ref.fingerprint()


@pytest.fixture(scope="module")
def comp16():
    c, _, _ = M.tb_build_phased(16, 1000, 50, 20260829)
    return c


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_revise_policy(comp16, seed):
    c2, info = M.revise_policy(comp16, seed)
    M.validate_compiled(c2)
    for f in ("state_kind", "state_row_start", "state_agent", "state_radix", "row_ptr", "col", "val"):
        assert np.array_equal(getattr(c2, f), getattr(comp16, f))
    assert not np.array_equal(c2.weights, comp16.weights)
    changed = np.any(c2.weights != comp16.weights, axis=2)
    assert set(np.unique(comp16.state_agent[changed]).tolist()) == {info["agent"]}
    c3, info3 = M.revise_policy(comp16, seed)
    assert np.array_equal(c3.weights, c2.weights) and info3 == info


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_revise_values(comp16, seed):
    c2, info = M.revise_values(comp16, seed)
    M.validate_compiled(c2)
    assert np.array_equal(c2.row_ptr, comp16.row_ptr) and np.array_equal(c2.col, comp16.col)
    assert np.array_equal(c2.weights, comp16.weights)
    assert info["n_nnz_changed"] > 0 and not info["degenerate"]
    nz_changed = int(np.sum(c2.val != comp16.val))
    assert 0 < nz_changed <= info["n_nnz_changed"]

    rows = np.unique(np.repeat(np.arange(comp16.n_rows), np.diff(comp16.row_ptr))[c2.val != comp16.val])
    st = np.searchsorted(comp16.state_row_start, rows, side="right") - 1
    assert np.unique(st).size <= info["n_states_selected"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_revise_structure(comp16, seed):
    c2, info = M.revise_structure(comp16, seed)
    M.validate_compiled(c2)
    for f in ("state_kind", "state_row_start", "state_agent", "state_radix", "weights"):
        assert np.array_equal(getattr(c2, f), getattr(comp16, f))
    assert c2.n_rows == comp16.n_rows
    assert info["n_entries_added"] + info["n_entries_removed"] > 0
    assert c2.nnz == comp16.nnz + info["n_entries_added"] - info["n_entries_removed"]

    for r in range(c2.n_rows):
        cc = c2.col[c2.row_ptr[r]:c2.row_ptr[r + 1]]
        assert np.all(np.diff(cc) > 0)
