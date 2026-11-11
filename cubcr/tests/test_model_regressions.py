import numpy as np
import pytest

from cubcr import vi_ref
from cubcr.model import CSG, StateDef, RMAX
from cubcr.masks import agent_mask
from cubcr.engine_cpu import CPUEngine
from cubcr.shapley import shapley_from_table
from tests.helpers import random_csg


def matching_pennies():

    rows = [((0, 0), [(1, 1.0)]), ((0, 1), [(2, 1.0)]), ((1, 0), [(2, 1.0)]), ((1, 1), [(1, 1.0)])]
    sds = [StateDef(active=[0, 1], n_actions=[2, 2], rows=rows),
           StateDef(active=[], n_actions=[], rows=[((), [(1, 1.0)])]),
           StateDef(active=[], n_actions=[], rows=[((), [(2, 1.0)])])]
    obs = {(0, 0): np.array([0.5, 0.5]), (0, 1): np.array([0.5, 0.5])}
    return CSG(n_agents=2, horizon_k=1, init_state=0, fail_states=[1], state_defs=sds, observed_policy=obs)


def test_min_over_free_is_outside_expectation():

    csg = matching_pennies()
    assert vi_ref.solve(csg, agent_mask([0])) == pytest.approx(0.5)
    assert vi_ref.solve(csg, agent_mask([1])) == pytest.approx(0.5)
    assert vi_ref.solve(csg, 0) == pytest.approx(0.0)
    assert vi_ref.solve(csg, 3) == pytest.approx(0.5)
    eng = CPUEngine(csg.compile(), n_threads=1)
    got = eng.solve(np.arange(4, dtype=np.uint64))
    assert np.allclose(got, [0.0, 0.5, 0.5, 0.5], atol=1e-15)


def test_radix_one_active_slots_cpu():
    csg = random_csg(n_agents=4, n_states=20, k=5, seed=101, max_active=3, max_actions=3, min_actions=1)
    masks = np.arange(16, dtype=np.uint64)
    ref = np.array([vi_ref.solve(csg, int(m)) for m in masks])
    for block in (1, 8):
        assert np.max(np.abs(CPUEngine(csg.compile(), n_threads=2, block=block).solve(masks) - ref)) <= 1e-12


def test_validate_rejects_bad_probabilities():
    csg = matching_pennies()
    csg.state_defs[0].rows[0] = ((0, 0), [(1, float("nan"))])
    with pytest.raises(ValueError):
        csg.validate()
    csg = matching_pennies()
    csg.state_defs[0].rows[0] = ((0, 0), [(1, 0.7), (2, 0.7)])
    with pytest.raises(ValueError):
        csg.validate()
    csg = matching_pennies()
    csg.observed_policy[(0, 0)] = np.array([0.5, float("inf")])
    with pytest.raises(ValueError):
        csg.validate()


def test_validate_rejects_more_than_rmax_rows():
    rows = []
    for a in range(5):
        for b in range(5):
            for c in range(5):
                rows.append(((a, b, c), [(1, 1.0)]))
    sds = [StateDef(active=[0, 1, 2], n_actions=[5, 5, 5], rows=rows),
           StateDef(active=[], n_actions=[], rows=[((), [(1, 1.0)])])]
    obs = {(0, i): np.full(5, 0.2) for i in range(3)}
    csg = CSG(n_agents=3, horizon_k=1, init_state=0, fail_states=[1], state_defs=sds, observed_policy=obs)
    assert 125 > RMAX
    with pytest.raises(ValueError):
        csg.validate()


def test_shapley_table_shape_check():
    with pytest.raises(ValueError):
        shapley_from_table(np.arange(4.0)[:, None], 2)


@pytest.mark.skipif(__import__("cubcr.engine_gpu", fromlist=["device_count"]).device_count() == 0, reason="no GPU")
def test_gpu_multiwave_step_masks_odd_k_and_tiling():
    from cubcr.engine_gpu import GPUEngine
    csg = random_csg(n_agents=5, n_states=30, k=5, seed=202, max_active=3, max_actions=3, min_actions=1, n_avoid=1)
    rng = np.random.default_rng(3)
    B = 100
    cols = rng.integers(0, 32, size=(5, B)).astype(np.uint64)
    ref = CPUEngine(csg.compile(), n_threads=2).solve(cols)
    for kw in (dict(max_batch=7, use_graph=True), dict(max_batch=33, use_graph=False), dict(l2_tile=64, max_batch=0, use_graph=False), dict(l2_tile=8, tpb=32, use_graph=False)):
        eng = GPUEngine(csg.compile(), **kw)
        got = eng.solve(cols)
        assert np.max(np.abs(got - ref)) <= 1e-12, kw
        eng.close()


@pytest.mark.skipif(__import__("cubcr.engine_gpu", fromlist=["device_count"]).device_count() == 0, reason="no GPU")
def test_gpu_tiling_with_graph_is_refused_and_default_paths_are_deterministic():


    from cubcr.engine_gpu import GPUEngine
    csg = random_csg(n_agents=5, n_states=30, k=5, seed=202, max_active=3, max_actions=3, min_actions=1, n_avoid=1)
    comp = csg.compile()
    with pytest.raises(ValueError):
        GPUEngine(comp, l2_tile=64, use_graph=True)
    cols = np.random.default_rng(3).integers(0, 32, size=(5, 100)).astype(np.uint64)
    ref = CPUEngine(comp, n_threads=2).solve(cols)
    for kw in (dict(use_graph=True), dict(max_batch=7, use_graph=True), dict(use_graph=False)):
        for _ in range(6):
            eng = GPUEngine(comp, **kw)
            got = eng.solve(cols)
            eng.close()
            assert np.max(np.abs(got - ref)) <= 1e-12, kw
