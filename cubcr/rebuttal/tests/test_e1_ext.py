import numpy as np
import pytest

import e1_ext


def _random(seed, max_active, max_actions, n_avoid=0):
    from tests.helpers import random_csg
    return random_csg(n_agents=4, n_states=14, k=5, seed=seed, max_active=max_active, max_actions=max_actions,
                      n_avoid=n_avoid, n_fail=2)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize("shape", [(2, 3, 0), (3, 4, 1), (3, 2, 0)])
def test_agent_level_matches_vi_ref(seed, shape):
    from cubcr import vi_ref
    csg = _random(seed, *shape)
    st = e1_ext.TemplateStructure(csg.compile())
    masks = np.arange(16, dtype=np.uint64)
    ref = vi_ref.solve_many(csg, [int(m) for m in masks], 5)
    for dt in (np.float64, np.longdouble):
        v = e1_ext.values_numpy(st, masks, 5, dt).astype(np.float64)
        assert np.max(np.abs(v - ref)) <= 4e-16


@pytest.mark.parametrize("seed", [0, 5])
def test_step_level_matches_vi_ref(seed):
    from cubcr import vi_ref
    csg = _random(seed, 3, 3, 1)
    st = e1_ext.TemplateStructure(csg.compile())
    rng = np.random.default_rng(seed)
    sm = rng.integers(0, 16, size=(5, 12), dtype=np.uint64)
    v = e1_ext.values_numpy(st, sm, 5, np.longdouble).astype(np.float64)
    ref = np.array([vi_ref.solve(csg, sm[:, j].tolist(), 5) for j in range(sm.shape[1])])
    assert np.max(np.abs(v - ref)) <= 4e-16


def test_longdouble_agrees_with_mpmath():
    import mpmath
    csg = _random(11, 3, 3, 1)
    st = e1_ext.TemplateStructure(csg.compile())
    masks = np.array([0, 5, 15], dtype=np.uint64)
    v = e1_ext.values_numpy(st, masks, 5, np.longdouble)
    mp = e1_ext.MpReference(st, 50)
    for i, m in enumerate(masks):
        vm = mp.value(int(m), 5)
        num, den = v[i].as_integer_ratio()
        with mpmath.workdps(50):
            assert abs(mpmath.mpf(num) / den - vm) < 1e-18


def test_longdouble_is_extended():
    assert np.finfo(np.longdouble).nmant >= 63, "x86 80-bit extended precision expected"


def test_failure_states_fixed_at_one():
    csg = _random(3, 2, 3)
    comp = csg.compile()
    fail = int(np.flatnonzero(comp.state_kind == 1)[0])
    comp.init_state = fail
    st = e1_ext.TemplateStructure(comp)
    v = e1_ext.values_numpy(st, np.arange(16, dtype=np.uint64), 5, np.longdouble)
    assert np.all(v == 1)


def _gpu():
    try:
        from cubcr.engine_gpu import device_count
        return device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu(), reason="needs the CUDA engine")
def test_matches_paper_gpu_engine():
    import rb
    from experiments.grid import build_model
    comp, iid, _, _ = build_model("ticketbench", 8, 1000, 25, 20260829, "none", 0.05)
    st = e1_ext.TemplateStructure(comp)
    masks = np.random.default_rng(0).integers(0, 256, size=32, dtype=np.uint64)
    with rb.gpu_lock("test_e1_ext"):
        eng = rb.make_gpu_engine(comp)
        vg = eng.solve(masks, 25)
        eng.close()
    v64 = e1_ext.values_numpy(st, masks, 25, np.float64)
    vld = e1_ext.values_numpy(st, masks, 25, np.longdouble)
    assert np.max(np.abs(v64 - vg)) <= 1e-15
    assert float(np.max(np.abs(vg.astype(np.longdouble) - vld))) <= 1e-15
