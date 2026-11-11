import numpy as np
import pytest

from cubcr import vi_ref, shapley
from tests.helpers import random_csg


def full_table(csg, k=None):
    return np.array([vi_ref.solve(csg, m, k) for m in range(1 << csg.n_agents)])


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_efficiency(seed):
    csg = random_csg(n_agents=4, n_states=14, k=4, seed=seed, max_active=3)
    v = full_table(csg)
    phi = shapley.shapley_from_table(v, 4)
    assert phi.sum() == pytest.approx(v[-1] - v[0], abs=1e-12)
    assert np.all(phi >= -1e-12)


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_dummy_agent(seed):
    csg = random_csg(n_agents=4, n_states=14, k=4, seed=seed, max_active=3, dummy_agents=(2,))
    v = full_table(csg)
    phi = shapley.shapley_from_table(v, 4)
    for m in range(16):
        if not (m >> 2) & 1:
            assert v[m | 4] == pytest.approx(v[m], abs=1e-15)
    assert abs(phi[2]) < 1e-14


@pytest.mark.parametrize("seed", [7, 8, 9])
def test_symmetry(seed):
    csg = random_csg(n_agents=4, n_states=14, k=4, seed=seed, max_active=3, symmetric_pair=(0, 1))
    v = full_table(csg)
    phi = shapley.shapley_from_table(v, 4)
    assert phi[0] == pytest.approx(phi[1], abs=1e-13)


@pytest.mark.parametrize("seed", [10, 11])
def test_shapley_formula_matches_permutation_average(seed):
    csg = random_csg(n_agents=4, n_states=10, k=3, seed=seed)
    v = full_table(csg)
    phi = shapley.shapley_from_table(v, 4)
    phi_p = shapley.shapley_by_permutations(lambda m: v[m], 4)
    assert np.allclose(phi, phi_p, atol=1e-13)


def test_monotone_in_horizon():
    csg = random_csg(n_agents=3, n_states=12, k=6, seed=12)
    prev = -1.0
    for k in range(0, 7):
        cur = vi_ref.solve(csg, 0b111, k)
        assert cur >= prev - 1e-15
        prev = cur
