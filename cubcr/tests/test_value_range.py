import numpy as np
import pytest

from cubcr.shapley_sampled import sample_shapley, empirical_bernstein_half_width
from cubcr.shapley import shapley_from_table
from cubcr.engine_cpu import CPUEngine
from tests.helpers import random_csg
from generators.ticketbench import build_compiled


def test_eb_range_scaling_math():
    var = np.array([0.0])
    hw_wide = empirical_bernstein_half_width(var, 100, 0.05, lo=-1, hi=1, two_sided=True)
    hw_tight = empirical_bernstein_half_width(var, 100, 0.05, lo=0.0, hi=0.01, two_sided=True)
    assert hw_tight[0] == pytest.approx(hw_wide[0] * 0.01 / 2)


def engine(n=6, S=60, k=5, seed=3):
    return CPUEngine(random_csg(n_agents=n, n_states=S, k=k, seed=seed, max_active=2, max_actions=3).compile(), n_threads=4)


def test_monotone_mode_certifies_faster_and_covers_truth():
    eng = engine()
    n = 6
    v = eng.solve(np.arange(1 << n, dtype=np.uint64))
    phi = shapley_from_table(v, n)
    ups = float(v[-1] - v[0])
    assert ups > 0
    r = sample_shapley(eng, kind="agent", n=n, seed=1, batch_perms=64, eps=ups / 20, alpha=0.05,
                       max_perms=200000, value_range="monotone", two_sided=True, bonferroni=True, sequential=True)
    r_legacy = sample_shapley(eng, kind="agent", n=n, seed=1, batch_perms=64, eps=ups / 20, alpha=0.05,
                              max_perms=200000, two_sided=True, bonferroni=True, sequential=True)
    assert r.certified and r.status == "eps"
    assert np.all(np.abs(r.estimates - phi) <= r.half_widths + 1e-12)
    assert r.n_perms < r_legacy.n_perms

    assert r.value_range[0] == pytest.approx(0.0, abs=1e-8) and r.value_range[1] == pytest.approx(ups, abs=1e-8)


def test_null_game_short_circuit():

    comp, meta = build_compiled(4, 60, 2, 0.05, "none", 20260829)
    eng = CPUEngine(comp, n_threads=2)
    r = sample_shapley(eng, kind="agent", n=4, k=2, seed=1, eps=0.01, value_range="monotone",
                       two_sided=True, bonferroni=True, sequential=True)
    assert r.certified and r.n_perms == 0 and np.all(r.estimates == 0) and np.all(r.half_widths == 0)


def test_range_violation_fails_loud():
    eng = engine()
    with pytest.raises(RuntimeError, match="outside the declared value range"):
        sample_shapley(eng, kind="agent", n=6, seed=1, batch_perms=8, max_perms=64,
                       value_range=(0.0, 1e-9))


def test_coverage_with_monotone_range_mock():

    rng = np.random.default_rng(0)
    n = 5
    inside = 0
    total = 0

    class MockEngine:
        def __init__(self, table):
            self.table = table

        def solve(self, masks, k=None):
            return self.table[np.asarray(masks, dtype=np.int64)]

    for trial in range(200):
        w = rng.random(n)
        masks = np.arange(1 << n)
        v = np.zeros(1 << n)
        for i in range(n):
            v += w[i] * ((masks >> i) & 1)
        v = v ** 1.5 / (v.max() ** 1.5) * rng.random() * 0.5
        phi = shapley_from_table(v, n)
        r = sample_shapley(MockEngine(v), kind="agent", n=n, n_players=n, k=1, seed=trial,
                           batch_perms=32, max_perms=512, value_range="monotone",
                           two_sided=True, bonferroni=True, sequential=True, eps=None, use_cache=False)
        inside += int(np.all(np.abs(r.estimates - phi) <= r.half_widths + 1e-12))
        total += 1
    assert inside / total >= 0.95, (inside, total)
