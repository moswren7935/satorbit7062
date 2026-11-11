import numpy as np, time
from cubcr.shapley import shapley_from_table
from cubcr.shapley_fast import shapley_from_table_fast


def test_fast_matches_reference():
    rng = np.random.default_rng(0)
    for n in (2, 5, 12, 16):
        v = np.sort(rng.random(1 << n))
        a = shapley_from_table(v, n); b = shapley_from_table_fast(v, n)
        assert np.max(np.abs(a - b)) <= 1e-12 * max(1.0, np.max(np.abs(a)))


def test_fast_is_faster_for_n20():
    v = np.random.default_rng(1).random(1 << 20)
    shapley_from_table_fast(v, 20)
    t = time.perf_counter(); shapley_from_table(v, 20); t_ref = time.perf_counter() - t
    t = time.perf_counter(); shapley_from_table_fast(v, 20); t_fast = time.perf_counter() - t
    assert t_fast < t_ref
