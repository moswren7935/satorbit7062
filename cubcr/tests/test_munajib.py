import numpy as np
import pytest

from cubcr import vi_ref, shapley
from cubcr.masks import agent_mask

from generators.vehicle import vehicle_csg


@pytest.fixture(scope="module")
def csg():
    return vehicle_csg()


@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_coalition_values(csg, k):
    v12 = vi_ref.solve(csg, agent_mask([0, 1]), k)
    v1 = vi_ref.solve(csg, agent_mask([0]), k)
    v2 = vi_ref.solve(csg, agent_mask([1]), k)
    v0 = vi_ref.solve(csg, agent_mask([]), k)
    assert v12 == pytest.approx(1.0, abs=1e-15)
    assert v1 == pytest.approx(0.6, abs=1e-15)
    assert v2 == pytest.approx(0.2, abs=1e-15)
    assert v0 == pytest.approx(0.12, abs=1e-15)


def test_shapley_values(csg):
    v = np.array([vi_ref.solve(csg, m, 1) for m in range(4)])
    phi = shapley.shapley_from_table(v, 2)
    assert phi[0] == pytest.approx(0.64, abs=1e-15)
    assert phi[1] == pytest.approx(0.24, abs=1e-15)
    assert phi.sum() == pytest.approx(v[3] - v[0], abs=1e-15)
    assert shapley.attributable_value(v, 2) == pytest.approx(0.88, abs=1e-15)
    phi_perm = shapley.shapley_by_permutations(lambda m: v[m], 2)
    assert np.allclose(phi, phi_perm, atol=1e-15)


def test_step_level_masks_reduce_to_agent_level(csg):

    cols = [agent_mask([0])] * 3
    assert vi_ref.solve(csg, cols, 3) == pytest.approx(0.6, abs=1e-15)

    cols = [agent_mask([0]), 0, 0]
    assert vi_ref.solve(csg, cols, 3) == pytest.approx(0.6, abs=1e-15)

    cols = [0, agent_mask([0]), 0]
    assert vi_ref.solve(csg, cols, 3) == pytest.approx(0.12, abs=1e-15)


def test_compiled_arrays(csg):
    c = csg.compile()
    assert c.n_rows == 6 and c.nnz == 9
    assert c.state_kind.tolist() == [0, 1, 0]
    assert c.max_rows_per_state == 4
