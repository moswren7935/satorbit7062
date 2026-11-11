import itertools
import numpy as np
import rb


def brute(a, b):
    C = D = 0
    for i, j in itertools.combinations(range(len(a)), 2):
        s = (a[i] - a[j]) * (b[i] - b[j])
        if s > 0:
            C += 1
        elif s < 0:
            D += 1
    return (C - D) / (len(a) * (len(a) - 1) / 2)


def test_identical_and_reversed():
    a = np.arange(10.0)
    assert rb.kendall_tau_a(a, a) == 1.0
    assert rb.kendall_tau_a(a, a[::-1]) == -1.0


def test_ties_count_for_neither():
    a = np.array([1.0, 1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0, 4.0])

    assert abs(rb.kendall_tau_a(a, b) - 5 / 6) < 1e-15


def test_random_vs_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(200):
        d = int(rng.integers(2, 12))
        a = rng.integers(0, 4, d).astype(float)
        b = rng.integers(0, 4, d).astype(float)
        assert abs(rb.kendall_tau_a(a, b) - brute(a, b)) < 1e-15


def test_cs_separation_and_topk():
    phi = np.array([0.5, 0.3, 0.1, 0.05])
    L = np.array([0.45, 0.25, 0.05, 0.0]); U = np.array([0.55, 0.35, 0.15, 0.1])
    m = rb.topk_metrics(phi, phi, L, U, ks=(1, 3))
    assert m["top1_correct"]
    assert m["top3_exact"]
    assert m["top3_recall"] == 1.0 and m["top3_jaccard"] == 1.0
    assert m["sep1"] is True
    assert m["sep3"] is False
    assert rb.cs_separated(L, U, 2) is True


def test_topk_imperfect_ranking():
    phi = np.array([0.5, 0.3, 0.1, 0.05, 0.0])
    phi_hat = np.array([0.31, 0.32, 0.02, 0.2, 0.0])
    m = rb.topk_metrics(phi_hat, phi, ks=(1, 3))
    assert m["top1_correct"] is False
    assert m["top3_exact"] is False
    assert abs(m["top3_recall"] - 2 / 3) < 1e-15
    assert abs(m["top3_jaccard"] - 2 / 4) < 1e-15


def test_cs_separated_rejects_invalid():
    import pytest
    with pytest.raises(ValueError):
        rb.cs_separated([2.0, 0.0], [0.0, 1.0], 1)
    with pytest.raises(ValueError):
        rb.cs_separated([0.0, np.nan], [1.0, 1.0], 1)
    assert rb.cs_separated([0.5, -np.inf], [0.6, np.inf], 1) is False
