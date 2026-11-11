import math
import time

import numpy as np
import pytest

from cubcr.engine_cpu import CPUEngine
from cubcr.shapley import shapley_from_table
from cubcr.shapley_sampled import (empirical_bernstein_half_width, look_schedule_size, sample_shapley,
                                   topk_certified)
from tests.helpers import random_csg

SEED = 20260829


class MockCompiled:
    def __init__(self, n, k):
        self.n_agents, self.horizon_k = n, k


class TableEngine:


    def __init__(self, table, n, delay_per_coalition=0.0):
        self.table = np.asarray(table, dtype=np.float64)
        self.compiled = MockCompiled(n, 1)
        self.delay = delay_per_coalition
        self.call_starts = []
        self.call_sizes = []

    def solve(self, masks, k=None):
        self.call_starts.append(time.perf_counter())
        masks = np.asarray(masks, dtype=np.uint64)
        assert masks.ndim == 1
        self.call_sizes.append(int(masks.shape[0]))
        if self.delay > 0:
            time.sleep(self.delay * masks.shape[0])
        return self.table[masks.astype(np.int64)].copy()


def random_game(rng, N, kind="uniform"):

    tab = rng.uniform(0.0, 1.0, size=1 << N)
    if kind == "monotone":
        w = rng.uniform(0.0, 0.6, size=N)
        for S in range(1 << N):
            tab[S] = 1.0 - np.prod([1 - w[i] for i in range(N) if (S >> i) & 1]) if S else 0.0
        tab = np.clip(tab + 0.05 * rng.uniform(0, 1, size=tab.shape) * (np.arange(1 << N) > 0), 0, 1)
    tab[0] = 0.0
    return tab


def agent6():

    csg = random_csg(n_agents=6, n_states=24, k=4, seed=166, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=4)
    v = eng.solve(np.arange(64, dtype=np.uint64))
    return csg, eng, v, shapley_from_table(v, 6)


def test_two_sided_constant_is_ln4_over_alpha():
    alpha, m = 0.05, 100
    var = np.array([0.0, 0.04, 0.25])
    one = empirical_bernstein_half_width(var, m, alpha, two_sided=False)
    two = empirical_bernstein_half_width(var, m, alpha, two_sided=True)
    L1, L2 = math.log(2 / alpha), math.log(4 / alpha)
    for j in range(3):
        vy = var[j] / 4.0
        assert one[j] == pytest.approx(2 * (math.sqrt(2 * vy * L1 / m) + 7 * L1 / (3 * (m - 1))), rel=1e-12)
        assert two[j] == pytest.approx(2 * (math.sqrt(2 * vy * L2 / m) + 7 * L2 / (3 * (m - 1))), rel=1e-12)
    assert np.all(two > one)

    assert np.allclose(two, empirical_bernstein_half_width(var, m, alpha / 2, two_sided=False))


def test_two_sided_coverage_mock_oracle_200_games():

    rng = np.random.default_rng(SEED)
    alpha, N = 0.05, 5
    per_player, family = [], []
    for g in range(200):
        tab = random_game(rng, N, kind="uniform" if g % 2 else "monotone")
        phi = shapley_from_table(tab, N)
        eng = TableEngine(tab, N)
        r = sample_shapley(eng, N, kind="agent", seed=g, batch_perms=50, max_perms=400, alpha=alpha,
                           two_sided=True, bonferroni=False, sequential=False)
        assert r.two_sided and not r.bonferroni and not r.sequential and not r.certified
        assert r.n_samples == 200 and r.n_perms == 400
        assert np.all(np.isfinite(r.half_widths))
        inside = np.abs(r.estimates - phi) <= r.half_widths
        per_player.append(inside)

        hw = empirical_bernstein_half_width(r.sample_var, r.n_samples, alpha, two_sided=True)
        assert np.allclose(r.half_widths, hw, atol=1e-12) and r.alpha_eff == alpha
        rb = sample_shapley(eng, N, kind="agent", seed=g, batch_perms=50, max_perms=400, alpha=alpha,
                            two_sided=True, bonferroni=True, sequential=False)
        assert rb.alpha_player == alpha / N and np.all(rb.half_widths > r.half_widths)
        assert np.array_equal(rb.estimates, r.estimates)
        family.append(bool(np.all(np.abs(rb.estimates - phi) <= rb.half_widths)))

        assert abs(r.estimates.sum() - (tab[-1] - tab[0])) <= 1e-12
    cov_player = float(np.mean(per_player))
    cov_family = float(np.mean(family))
    assert cov_player >= 1 - alpha, cov_player
    assert cov_family >= 1 - alpha, cov_family


def test_certificate_is_snapshot_at_stopping_time():
    csg, eng, v, phi = agent6()
    seen = []

    def cb(m, est, hw):
        seen.append((int(m), est.copy(), hw.copy()))

    res = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=2, max_perms=200000, batch_perms=64, callback=cb)
    assert res.status == "top_k" and res.certified and res.anytime_valid
    c = res.certificate
    assert c is not None and c["kind"] == "top_k"
    m_last, est_last, hw_last = seen[-1]
    assert c["m"] == m_last == res.n_samples and c["n_perms"] == res.n_perms
    assert np.array_equal(c["estimates"], est_last) and np.array_equal(c["half_widths"], hw_last)
    assert np.array_equal(c["estimates"], res.estimates) and np.array_equal(c["half_widths"], res.half_widths)
    assert set(c["top_set"].tolist()) == set(res.top_k_players.tolist()) == set(np.argsort(-phi)[:2].tolist())
    assert c["look_index"] == res.n_looks_used and c["wall_s"] <= res.wall_time
    assert c["n_solves"] == res.n_solves and c["alpha_eff"] == res.alpha_eff and c["anytime_valid"]

    res.estimates[:] = -1.0
    res.half_widths[:] = 0.0
    assert np.array_equal(c["estimates"], est_last) and np.array_equal(c["half_widths"], hw_last)

    ok, top = topk_certified(c["estimates"], c["half_widths"], 2)
    assert ok and set(top.tolist()) == set(c["top_set"].tolist())

    r2 = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=2, max_perms=4, batch_perms=2)
    assert r2.status == "max_perms" and not r2.certified and r2.certificate is None
    r3 = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=64, batch_perms=32)
    assert r3.status == "max_perms" and not r3.certified and r3.certificate is None


def test_alpha_allocation_fields_and_look_schedule():
    csg, eng, v, phi = agent6()
    alpha, N, batch, max_perms = 0.05, 6, 64, 10000
    res = sample_shapley(eng, N, kind="agent", seed=SEED, eps=1e-9, alpha=alpha, max_perms=max_perms,
                         batch_perms=batch)
    assert res.status == "max_perms" and not res.certified and res.certificate is None
    assert res.two_sided and res.bonferroni and res.sequential and res.anytime_valid
    L_max = math.ceil(math.log2(max_perms / (2 * batch))) + 1
    assert res.n_looks_max == L_max == look_schedule_size(max_perms, 2 * batch)
    assert res.alpha == alpha and res.alpha_player == alpha / N
    assert res.alpha_look == pytest.approx(alpha / N / L_max) and res.alpha_eff == res.alpha_look
    assert 1 <= res.n_looks_used <= res.n_looks_max and len(res.look_m) == res.n_looks_used

    m0 = res.look_m[0]
    assert m0 == batch
    for j, mm in enumerate(res.look_m[:-1]):
        assert mm >= m0 * 2 ** j
    assert res.look_m[-1] == res.n_samples and res.n_perms == max_perms

    hw = empirical_bernstein_half_width(res.sample_var, res.n_samples, res.alpha_eff, two_sided=True)
    assert np.allclose(res.half_widths, hw, atol=1e-12)
    d = res.alpha_allocation()
    for key in ("alpha", "alpha_player", "alpha_look", "alpha_eff", "n_looks_max", "n_looks_used", "look_m",
                "two_sided", "bonferroni", "sequential", "anytime_valid"):
        assert key in d
    s = res.summary()
    assert s["alpha_look"] == res.alpha_look and s["n_looks_used"] == res.n_looks_used

    ex = sample_shapley(eng, N, kind="agent", seed=SEED, alpha=alpha, max_perms=max_perms, batch_perms=batch)
    assert not (ex.two_sided or ex.bonferroni or ex.sequential) and not ex.anytime_valid
    assert ex.alpha_player == ex.alpha_look == ex.alpha_eff == alpha and ex.n_looks_max == 1
    assert np.array_equal(ex.estimates, res.estimates) and np.all(ex.half_widths < res.half_widths)

    ex2 = sample_shapley(eng, N, kind="agent", seed=SEED, alpha=alpha, max_perms=max_perms, batch_perms=batch,
                         two_sided=True, bonferroni=True, sequential=True)
    assert ex2.anytime_valid and np.array_equal(ex2.half_widths, res.half_widths)
    ex3 = sample_shapley(eng, N, kind="agent", seed=SEED, alpha=alpha, max_perms=256, batch_perms=batch,
                         eps=1e-9, sequential=False)
    assert ex3.two_sided and ex3.bonferroni and not ex3.sequential and not ex3.anytime_valid
    assert ex3.n_looks_used == ex3.n_batches == 2 and ex3.n_looks_max == 1


@pytest.mark.parametrize("max_perms,batch,anti", [(100, 4, True), (1000, 7, True), (64, 8, True), (3, 8, False),
                                                  (513, 16, False), (2048, 1, True)])
def test_look_count_bounded_and_max_perms_hit_exactly(max_perms, batch, anti):
    rng = np.random.default_rng(max_perms * 31 + batch)
    N = 4
    eng = TableEngine(random_game(rng, N), N)
    r = sample_shapley(eng, N, kind="agent", seed=1, eps=0.0, max_perms=max_perms, batch_perms=batch, antithetic=anti)
    ppu = 2 if anti else 1
    assert r.status == "max_perms" and r.n_perms == max_perms
    assert r.n_looks_max == look_schedule_size(max_perms, batch * ppu)
    assert r.n_looks_max == max(1, math.ceil(math.log2(max_perms / (batch * ppu)) - 1e-12) + 1) or \
        max_perms <= batch * ppu
    assert 1 <= r.n_looks_used <= r.n_looks_max, (r.n_looks_used, r.n_looks_max, r.look_m)
    assert r.look_m[-1] == r.n_samples
    assert r.n_samples == math.ceil(max_perms / ppu)


def test_topk_stopping_certifies_exact_top_set_agent6():
    csg, eng, v, phi = agent6()
    assert np.min(np.diff(np.sort(phi))) > 0.01
    for K in (1, 2, 3):
        exact_top = set(np.argsort(-phi)[:K].tolist())
        res = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=K, alpha=0.05, max_perms=400000, batch_perms=128)
        assert res.status == "top_k" and res.certified and res.anytime_valid, (K, res.status)
        assert set(res.top_k_players.tolist()) == exact_top
        assert set(res.certificate["top_set"].tolist()) == exact_top
        assert res.n_perms < 400000 and res.n_looks_used <= res.n_looks_max

        order = res.ranking()
        assert np.min(res.lower[order[:K]]) > np.max(res.upper[order[K:]])

        assert np.all(np.abs(res.estimates - phi) <= res.half_widths)

    res = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=2, max_perms=64, batch_perms=32)
    assert res.status == "max_perms" and not res.certified and res.certificate is None


def test_time_budget_overshoot_bounded_by_one_batch():
    rng = np.random.default_rng(7)
    N, batch = 4, 8
    delay = 0.004
    eng = TableEngine(random_game(rng, N), N, delay_per_coalition=delay)
    budget = 1.15
    t0 = time.perf_counter()
    res = sample_shapley(eng, N, kind="agent", seed=1, eps=0.0, max_perms=10 ** 9, batch_perms=batch,
                         time_budget=budget, use_cache=False)
    wall_outer = time.perf_counter() - t0
    full_batch_s = delay * 2 * batch * (N + 1)
    assert res.status == "time" and not res.certified
    assert res.time_budget == budget
    assert res.budget_overshoot_s == pytest.approx(max(0.0, res.wall_time - budget))
    assert res.budget_overshoot_s <= full_batch_s, (res.budget_overshoot_s, full_batch_s)
    assert wall_outer <= budget + 1.5 * full_batch_s

    assert all(t - t0 <= budget + 0.05 for t in eng.call_starts)

    assert eng.call_sizes[-1] < eng.call_sizes[0] and res.n_perms % (2 * batch) != 0
    assert res.n_batches == len(eng.call_sizes) >= 2

    assert res.look_m[-1] == res.n_samples and res.n_looks_used <= res.n_looks_max

    r0 = sample_shapley(eng, N, kind="agent", seed=1, max_perms=10 ** 9, batch_perms=2, time_budget=0.0,
                        use_cache=False)
    assert r0.status == "time" and r0.n_perms == 4 and r0.n_batches == 1
