import numpy as np
import pytest

from cubcr import vi_ref
from cubcr.engine_cpu import CPUEngine
from tests.helpers import random_csg

gpu = pytest.importorskip("cubcr.engine_gpu")
if gpu.device_count() == 0:
    pytest.skip("no CUDA device", allow_module_level=True)
from cubcr.engine_gpu import GPUEngine


def cases():
    yield random_csg(n_agents=4, n_states=16, k=5, seed=21, max_active=3, max_actions=4, n_avoid=1)
    yield random_csg(n_agents=5, n_states=24, k=4, seed=22, max_active=3, max_actions=3, n_fail=2)
    yield random_csg(n_agents=3, n_states=9, k=6, seed=23, max_active=2, max_actions=2)
    yield random_csg(n_agents=6, n_states=30, k=3, seed=24, max_active=3, max_actions=4, deterministic_policy=True)
    yield random_csg(n_agents=4, n_states=40, k=7, seed=25, max_active=1, max_actions=2)


@pytest.mark.parametrize("use_graph", [False, True])
def test_agent_level_all_coalitions_vs_reference(use_graph):
    for csg in cases():
        n = csg.n_agents
        masks = np.arange(1 << n, dtype=np.uint64)
        ref = np.array([vi_ref.solve(csg, int(m)) for m in masks])
        eng = GPUEngine(csg.compile(), use_graph=use_graph)
        got = eng.solve(masks)
        assert np.max(np.abs(got - ref)) <= 1e-12, (csg.meta, np.max(np.abs(got - ref)))
        eng.close()


def test_step_level_random_masks_vs_reference():
    rng = np.random.default_rng(0)
    for csg in cases():
        n, k = csg.n_agents, csg.horizon_k
        B = 300
        cols = rng.integers(0, 1 << n, size=(k, B)).astype(np.uint64)
        ref = np.array([vi_ref.solve(csg, cols[:, c].tolist()) for c in range(B)])
        eng = GPUEngine(csg.compile())
        got = eng.solve(cols)
        assert np.max(np.abs(got - ref)) <= 1e-12
        eng.close()


@pytest.mark.parametrize("max_batch,tpb", [(7, 32), (64, 64), (1, 256)])
def test_waves_and_tpb(max_batch, tpb):
    csg = random_csg(n_agents=5, n_states=24, k=4, seed=22, max_active=3, max_actions=3, n_fail=2)
    masks = np.arange(32, dtype=np.uint64)
    ref = CPUEngine(csg.compile(), n_threads=2).solve(masks)
    eng = GPUEngine(csg.compile(), max_batch=max_batch, tpb=tpb, use_graph=True)
    got = eng.solve(masks)
    assert eng.stats["waves"] == (32 + max_batch - 1) // max_batch
    assert np.max(np.abs(got - ref)) <= 1e-12
    eng.close()


def test_fp32_path():
    csg = random_csg(n_agents=4, n_states=16, k=5, seed=31, max_active=3, max_actions=4)
    masks = np.arange(16, dtype=np.uint64)
    ref = CPUEngine(csg.compile(), n_threads=2).solve(masks)
    got = GPUEngine(csg.compile(), dtype="fp32").solve(masks)
    assert np.max(np.abs(got - ref)) <= 1e-5


def test_large_batch_matches_cpu():

    csg = random_csg(n_agents=12, n_states=200, k=10, seed=77, max_active=2, max_actions=3, n_avoid=2)
    masks = np.arange(1 << 12, dtype=np.uint64)
    ref = CPUEngine(csg.compile(), n_threads=8, block=8).solve(masks)
    got = GPUEngine(csg.compile(), use_graph=True).solve(masks)
    assert np.max(np.abs(got - ref)) <= 1e-12
