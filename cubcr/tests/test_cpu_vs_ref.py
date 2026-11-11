import numpy as np
import pytest

from cubcr import vi_ref
from cubcr.engine_cpu import CPUEngine
from tests.helpers import random_csg


def cases():
    yield random_csg(n_agents=4, n_states=16, k=5, seed=21, max_active=3, max_actions=4, n_avoid=1)
    yield random_csg(n_agents=5, n_states=24, k=4, seed=22, max_active=3, max_actions=3, n_fail=2)
    yield random_csg(n_agents=3, n_states=9, k=6, seed=23, max_active=2, max_actions=2)
    yield random_csg(n_agents=6, n_states=30, k=3, seed=24, max_active=3, max_actions=4, deterministic_policy=True)


@pytest.mark.parametrize("block", [1, 4, 8, 16])
@pytest.mark.parametrize("threads", [1, 4])
def test_agent_level_all_coalitions(block, threads):
    for csg in cases():
        n = csg.n_agents
        masks = np.arange(1 << n, dtype=np.uint64)
        ref = np.array([vi_ref.solve(csg, int(m)) for m in masks])
        eng = CPUEngine(csg.compile(), n_threads=threads, block=block)
        got = eng.solve(masks)
        assert np.max(np.abs(got - ref)) <= 1e-12


def test_step_level_random_masks():
    rng = np.random.default_rng(0)
    for csg in cases():
        n, k = csg.n_agents, csg.horizon_k
        B = 37
        cols = rng.integers(0, 1 << n, size=(k, B)).astype(np.uint64)
        ref = np.array([vi_ref.solve(csg, cols[:, c].tolist()) for c in range(B)])
        eng = CPUEngine(csg.compile(), n_threads=3, block=8)
        got = eng.solve(cols)
        assert np.max(np.abs(got - ref)) <= 1e-12


def test_fp32_close():
    csg = random_csg(n_agents=4, n_states=16, k=5, seed=31, max_active=3, max_actions=4)
    masks = np.arange(16, dtype=np.uint64)
    ref = CPUEngine(csg.compile(), n_threads=2, block=8, dtype="fp64").solve(masks)
    got = CPUEngine(csg.compile(), n_threads=2, block=8, dtype="fp32").solve(masks)
    assert np.max(np.abs(got - ref)) <= 1e-5


def test_k_zero_and_horizon_override():
    csg = random_csg(n_agents=3, n_states=10, k=4, seed=41)
    eng = CPUEngine(csg.compile(), n_threads=1, block=1)
    assert eng.solve(np.zeros(1, dtype=np.uint64), k=0)[0] == 0.0
    for k in (1, 2, 7):
        assert abs(eng.solve(np.array([5], dtype=np.uint64), k=k)[0] - vi_ref.solve(csg, 5, k)) <= 1e-12
