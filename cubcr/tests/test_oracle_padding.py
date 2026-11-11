import numpy as np
import pytest

from cubcr.engine_cpu import CPUEngine
from cubcr.shapley_sampled import CoalitionOracle, exact_player_table, sample_shapley
from tests.helpers import random_csg

SEED = 20260829


def _has_gpu():
    try:
        from cubcr.engine_gpu import device_count
        return device_count() > 0
    except Exception:
        return False


def test_bucket_rows():
    b = CoalitionOracle.bucket_rows
    assert b(1) == 1024 and b(1024) == 1024 and b(1025) == 2048 and b(16384) == 16384
    assert b(16385) == 32768 and b(70000) == 131072


def test_padding_is_bitwise_identical_cpu():
    csg = random_csg(n_agents=4, n_states=20, k=6, seed=SEED, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    a = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=64, batch_perms=16, pad_rows=False)
    b = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=64, batch_perms=16, pad_rows=True)
    assert a.n_padded_rows == 0 and b.n_padded_rows > 0 and b.pad_rows
    assert np.array_equal(a.estimates, b.estimates) and np.array_equal(a.half_widths, b.half_widths)
    assert a.n_solves == b.n_solves and a.n_coalitions_requested == b.n_coalitions_requested


def test_large_step_level_is_pass_through():

    csg = random_csg(n_agents=3, n_states=12, k=10, seed=SEED, max_active=2, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    res = sample_shapley(eng, 30, kind="step", n=3, k=10, seed=SEED, max_perms=40, batch_perms=10)
    orc = CoalitionOracle(eng, 30, "step", 3, 10)
    assert len(orc.cache) == 0
    assert res.cache_hits == 0 and res.n_solves == res.n_coalitions_requested

    rows = np.concatenate([orc.prefix_masks(np.random.default_rng(3).permutation(30)) for _ in range(4)])
    ref = eng.solve(np.ascontiguousarray(rows.T), k=10)
    assert np.array_equal(orc.values(rows), ref)


@pytest.mark.skipif(not _has_gpu(), reason="GPU library not available")
def test_padding_is_bitwise_identical_gpu_and_reuses_graph():
    from cubcr.engine_gpu import GPUEngine
    csg = random_csg(n_agents=4, n_states=30, k=8, seed=SEED, max_active=3, max_actions=3)
    comp = csg.compile()
    eng = GPUEngine(comp, dtype="fp64", use_graph=True)
    try:
        a = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=96, batch_perms=8, pad_rows=False,
                           use_cache=False)
        b = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=96, batch_perms=8, pad_rows=True,
                           use_cache=False)
        assert np.array_equal(a.estimates, b.estimates)
        assert b.graph_captures is not None and b.graph_captures <= 2
        assert a.graph_captures is not None
        ref = exact_player_table(eng, 4)
        orc = CoalitionOracle(eng, 4, pad_rows=True, use_cache=False)
        rows = np.arange(16, dtype=np.uint64).reshape(-1, 1)
        assert np.array_equal(orc.values(rows), ref)
    finally:
        eng.close()
