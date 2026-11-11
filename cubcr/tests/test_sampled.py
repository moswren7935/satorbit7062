import math
import warnings

import numpy as np
import pytest
from scipy.stats import kendalltau

from cubcr import vi_ref
from cubcr.engine_cpu import CPUEngine
from cubcr.masks import step_mask_matrix
from cubcr.shapley import shapley_from_table
from cubcr.shapley_sampled import (CoalitionOracle, empirical_bernstein_half_width, exact_player_table,
                                   heatmap, kendall_tau, sample_shapley, topk_certified)
from cubcr.screening import exact_marginal_crosscheck, singleton_screening
from tests.helpers import random_csg

SEED = 20260829


def agent6():

    csg = random_csg(n_agents=6, n_states=24, k=4, seed=166, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=4)
    v = eng.solve(np.arange(64, dtype=np.uint64))
    return csg, eng, v, shapley_from_table(v, 6)


def step_exact(csg, eng):

    n, k = csg.n_agents, csg.horizon_k
    N = n * k
    cols = np.stack([step_mask_matrix([p for p in range(N) if (S >> p) & 1], n, k)
                     for S in range(1 << N)], axis=1)
    v = eng.solve(cols)
    return v, shapley_from_table(v, N)


def test_eb_half_width_formula():
    alpha, m = 0.05, 100
    var = np.array([0.0, 0.04, 0.25])
    hw = empirical_bernstein_half_width(var, m, alpha)
    L = math.log(2 / alpha)
    for j in range(3):
        vy = var[j] / 4.0
        expect = 2.0 * (math.sqrt(2 * vy * L / m) + 7 * L / (3 * (m - 1)))
        assert hw[j] == pytest.approx(expect, rel=1e-12)
    assert np.all(np.isinf(empirical_bernstein_half_width(var, 1, alpha)))
    assert np.all(empirical_bernstein_half_width(var, 1000, alpha) < hw)
    assert np.all(empirical_bernstein_half_width(var, m, 0.5) < hw)


def test_topk_certified_helper():
    est = np.array([0.5, 0.4, 0.1, 0.05])
    ok, top = topk_certified(est, np.full(4, 0.01), 2)
    assert ok and set(top.tolist()) == {0, 1}
    ok, top = topk_certified(est, np.full(4, 0.2), 2)
    assert not ok and top is None


def test_agent_level_within_ci_and_tau_one():
    csg, eng, v, phi = agent6()
    assert np.min(np.diff(np.sort(phi))) > 0.01
    res = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=20000, batch_perms=256, alpha=0.05)
    assert res.n_players == 6 and res.status == "max_perms"
    assert np.all(np.abs(res.estimates - phi) <= res.half_widths)
    assert np.max(res.half_widths) < 0.01
    assert kendalltau(res.estimates, phi).statistic == pytest.approx(1.0, abs=1e-12)
    assert kendall_tau(res.estimates, phi) == 1.0

    assert res.estimates.sum() == pytest.approx(v[-1] - v[0], abs=1e-10)
    assert res.v_full == v[-1] and res.v_empty == v[0]

    res2 = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=20000, batch_perms=256, antithetic=False)
    assert res2.n_samples == res2.n_perms
    assert np.all(np.abs(res2.estimates - phi) <= res2.half_widths)
    assert res2.estimates.sum() == pytest.approx(v[-1] - v[0], abs=1e-10)


def test_agent_level_topk_and_eps_stopping():
    csg, eng, v, phi = agent6()
    top2 = set(np.argsort(-phi)[:2].tolist())
    res = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=2, max_perms=200000, batch_perms=256)
    assert res.status == "top_k" and res.certified
    assert set(res.top_k_players.tolist()) == top2
    assert res.n_perms < 200000
    order = res.ranking()
    assert np.min(res.lower[order[:2]]) > np.max(res.upper[order[2:]])
    res = sample_shapley(eng, 6, kind="agent", seed=SEED, eps=0.005, max_perms=200000, batch_perms=256)
    assert res.status == "eps" and np.all(res.half_widths <= 0.005)
    res = sample_shapley(eng, 6, kind="agent", seed=SEED, time_budget=0.0, max_perms=10 ** 9, batch_perms=8)
    assert res.status == "time" and res.n_perms == 16

    a = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=512, batch_perms=256)
    b = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=512, batch_perms=256, bonferroni=True)
    assert np.allclose(a.estimates, b.estimates) and np.all(b.half_widths > a.half_widths)


@pytest.mark.parametrize("seed", [SEED, 3])
def test_step_level_within_ci_and_efficiency(seed):
    csg = random_csg(n_agents=3, n_states=16, k=4, seed=seed, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=4)
    n, k, N = 3, 4, 12
    v, phi = step_exact(csg, eng)
    assert len(v) == 4096

    assert np.array_equal(exact_player_table(eng, N, "step", n, k), v)
    res = sample_shapley(eng, N, kind="step", n=n, k=k, seed=SEED, max_perms=6000, batch_perms=64)
    assert res.n_players == 12 and res.kind == "step"
    assert np.all(np.abs(res.estimates - phi) <= res.half_widths)
    assert res.estimates.sum() == pytest.approx(v[-1] - v[0], abs=1e-10)
    assert res.v_full == pytest.approx(v[-1], abs=1e-15) and res.v_empty == pytest.approx(v[0], abs=1e-15)
    assert kendall_tau(res.estimates, phi) >= 0.9
    hm = res.heatmap()
    assert hm.shape == (3, 4) and np.array_equal(hm, res.estimates.reshape(3, 4))
    assert np.array_equal(heatmap(phi, n, k), phi.reshape(3, 4))
    assert res.n_solves <= 4096 < res.n_coalitions_requested


def test_step_level_stratified_within_ci():
    csg = random_csg(n_agents=3, n_states=16, k=4, seed=3, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=4)
    v, phi = step_exact(csg, eng)
    res = sample_shapley(eng, 12, kind="step", seed=SEED, max_perms=6000, batch_perms=32, stratify=True)
    assert res.stratify and res.n_samples * 12 * 2 == res.n_perms
    assert np.all(np.abs(res.estimates - phi) <= res.half_widths)
    assert np.max(np.abs(res.estimates - phi)) < 0.005
    assert res.estimates.sum() == pytest.approx(v[-1] - v[0], abs=1e-10)


def test_eb_coverage_over_50_seeds():
    hits = []
    for s in range(50):
        csg = random_csg(n_agents=5, n_states=20, k=4, seed=1000 + s, max_active=3, max_actions=3)
        eng = CPUEngine(csg.compile(), n_threads=2)
        phi = shapley_from_table(eng.solve(np.arange(32, dtype=np.uint64)), 5)
        res = sample_shapley(eng, 5, kind="agent", seed=s, max_perms=400, batch_perms=25, alpha=0.05)
        assert np.all(np.isfinite(res.half_widths)) and np.max(res.half_widths) < 0.2
        hits.append(np.abs(res.estimates - phi) <= res.half_widths)
    cov = float(np.mean(hits))
    assert cov >= 0.95, cov


def _shap_values(oracle, N, seeds, perms):
    shap = pytest.importorskip("shap")

    def f(X):
        X = np.atleast_2d(np.asarray(X))
        z = X > 0.5
        bits = [int(sum(1 << j for j in range(N) if row[j])) for row in z]
        return oracle.values(oracle.masks_from_bitsets(bits))

    vals = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for sd in seeds:
            ex = shap.PermutationExplainer(f, np.zeros((1, N)), seed=sd)
            e = ex(np.ones((1, N)), max_evals=(2 * N + 1) * perms)
            vals.append(np.asarray(e.values[0], dtype=np.float64))
            assert float(e.base_values[0]) == pytest.approx(oracle.value(oracle.empty), abs=1e-12)
    return np.array(vals)


@pytest.mark.parametrize("kind", ["agent", "step"])
def test_shap_permutation_explainer_crossval(kind):
    if kind == "agent":
        csg = random_csg(n_agents=4, n_states=14, k=3, seed=SEED, max_active=3, max_actions=3)
        n, k, N = 4, 3, 4
    else:
        csg = random_csg(n_agents=2, n_states=12, k=3, seed=5, max_active=2, max_actions=3)
        n, k, N = 2, 3, 6
    eng = CPUEngine(csg.compile(), n_threads=2)
    table = exact_player_table(eng, N, kind, n, k)
    phi = shapley_from_table(table, N)
    orc = CoalitionOracle(eng, N, kind, n, k)
    sv = _shap_values(orc, N, seeds=(0, 1, 2), perms=200)
    shap_mean, shap_std = sv.mean(axis=0), sv.std(axis=0, ddof=1)

    assert np.all(np.abs(sv.sum(axis=1) - (table[-1] - table[0])) < 1e-10)
    res = sample_shapley(eng, N, kind=kind, n=n, k=k, seed=SEED, max_perms=600, batch_perms=50)
    assert np.all(np.abs(res.estimates - phi) <= res.half_widths)

    assert np.all(np.abs(res.estimates - shap_mean) <= res.half_widths + 3 * shap_std + 1e-12)
    assert np.all(np.abs(shap_mean - phi) <= 3 * shap_std + 0.01)


def test_cache_reduces_solves():
    csg = random_csg(n_agents=4, n_states=14, k=3, seed=SEED, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=2)
    a = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=100, batch_perms=10, use_cache=True)
    b = sample_shapley(eng, 4, kind="agent", seed=SEED, max_perms=100, batch_perms=10, use_cache=False)
    assert a.n_coalitions_requested == b.n_coalitions_requested == 100 * 5
    assert b.n_solves == 500
    assert a.n_solves <= 16 < a.n_coalitions_requested
    assert a.cache_hits == a.n_coalitions_requested - a.n_solves
    assert np.array_equal(a.estimates, b.estimates)

    c = sample_shapley(eng, 12, kind="step", seed=SEED, max_perms=2000, batch_perms=50)
    assert c.n_solves <= 4096 < c.n_coalitions_requested

    orc = CoalitionOracle(eng, 4, "agent")
    rows = orc.masks_from_bitsets([0, 1, 3, 1, 0, 15])
    v1 = orc.values(rows)
    assert orc.n_solves == 4 and orc.cache_hits == 2
    v2 = orc.values(rows)
    assert orc.n_solves == 4 and np.array_equal(v1, v2)
    assert np.allclose(v1, [vi_ref.solve(csg, m) for m in (0, 1, 3, 1, 0, 15)], atol=1e-12)


@pytest.mark.parametrize("seed", [SEED, 2])
def test_singleton_screening_and_crosscheck(seed):
    csg = random_csg(n_agents=3, n_states=16, k=4, seed=seed, max_active=3, max_actions=3)
    eng = CPUEngine(csg.compile(), n_threads=4)
    sc = singleton_screening(eng, 3, 4, kind="step")
    assert sc.add.shape == (3, 4) and sc.remove.shape == (3, 4)
    assert sc.n_engine_calls == 2 and sc.n_solves == 2 * 12 + 2
    assert np.all(sc.add >= -1e-15) and np.all(sc.remove >= -1e-15)

    cc = exact_marginal_crosscheck(sc, csg=csg)
    assert cc.method == "vi_ref" and cc.max_abs_diff_add <= 1e-12 and cc.max_abs_diff_remove <= 1e-12

    cc2 = exact_marginal_crosscheck(sc, engine=eng)
    assert cc2.max_abs_diff_add <= 1e-12 and cc2.max_abs_diff_remove <= 1e-12
    v, phi = step_exact(csg, eng)
    assert np.array_equal(cc2.table, v)
    assert np.allclose(cc2.shapley_exact, phi.reshape(3, 4), atol=1e-13)

    assert phi.sum() == pytest.approx(sc.attributable, abs=1e-12)

    sa = singleton_screening(eng, kind="agent")
    va = eng.solve(np.arange(8, dtype=np.uint64))
    assert sa.add.shape == (3, 1) and sa.n_solves == 2 * 3 + 2
    for i in range(3):
        assert sa.add[i, 0] == pytest.approx(va[1 << i] - va[0], abs=1e-15)
        assert sa.remove[i, 0] == pytest.approx(va[7] - va[7 ^ (1 << i)], abs=1e-15)


def test_gpu_engine_sampled_matches_cpu():
    gpu = pytest.importorskip("cubcr.engine_gpu")
    if gpu.device_count() == 0:
        pytest.skip("no CUDA device")
    csg = random_csg(n_agents=5, n_states=20, k=4, seed=1000, max_active=3, max_actions=3)
    comp = csg.compile()
    ce = CPUEngine(comp, n_threads=2)
    ge = gpu.GPUEngine(comp)
    try:
        phi = shapley_from_table(ce.solve(np.arange(32, dtype=np.uint64)), 5)
        rc = sample_shapley(ce, 5, kind="agent", seed=3, max_perms=400, batch_perms=25)
        rg = sample_shapley(ge, 5, kind="agent", seed=3, max_perms=400, batch_perms=25)
        assert np.max(np.abs(rc.estimates - rg.estimates)) <= 1e-12
        assert np.all(np.abs(rg.estimates - phi) <= rg.half_widths)
        rc = sample_shapley(ce, 20, kind="step", seed=3, max_perms=200, batch_perms=25)
        rg = sample_shapley(ge, 20, kind="step", seed=3, max_perms=200, batch_perms=25)
        assert np.max(np.abs(rc.estimates - rg.estimates)) <= 1e-12
        assert rg.heatmap().shape == (5, 4)
        sg = singleton_screening(ge, 5, 4)
        sc = singleton_screening(ce, 5, 4)
        assert np.max(np.abs(sg.add - sc.add)) <= 1e-12 and np.max(np.abs(sg.remove - sc.remove)) <= 1e-12
    finally:
        ge.close()


def test_review2_max_perms_hit_exactly_and_legacy_defaults_without_rule():


    csg, eng, v, phi = agent6()
    res = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=1000, batch_perms=64)
    assert res.status == "max_perms" and res.n_perms == 1000 and res.n_samples == 500
    assert not res.certified and res.certificate is None and not res.anytime_valid
    assert res.two_sided is False and res.bonferroni is False and res.sequential is False
    assert res.alpha_player == res.alpha_look == res.alpha_eff == 0.05 and res.n_looks_max == 1
    assert np.all(np.abs(res.estimates - phi) <= res.half_widths)
    assert res.estimates.sum() == pytest.approx(v[-1] - v[0], abs=1e-10)


def test_review2_certifying_run_uses_full_allocation_and_snapshot():


    csg, eng, v, phi = agent6()
    base = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=1024, batch_perms=64)
    cert = sample_shapley(eng, 6, kind="agent", seed=SEED, max_perms=1024, batch_perms=64, eps=1e-9)
    assert np.array_equal(base.estimates, cert.estimates) and np.all(cert.half_widths > base.half_widths)
    assert cert.two_sided and cert.bonferroni and cert.sequential and cert.anytime_valid
    L = math.log(4 / cert.alpha_eff)
    vy = cert.sample_var / 4.0
    m = cert.n_samples
    expect = 2.0 * (np.sqrt(2 * vy * L / m) + 7 * L / (3 * (m - 1)))
    assert np.allclose(cert.half_widths, expect, rtol=1e-12)
    assert cert.alpha_eff == pytest.approx(0.05 / 6 / cert.n_looks_max)
    assert cert.n_looks_max == 1 + math.ceil(math.log2(1024 / 128))
    assert cert.status == "max_perms" and not cert.certified and cert.certificate is None

    top = sample_shapley(eng, 6, kind="agent", seed=SEED, top_k=2, max_perms=200000, batch_perms=128)
    assert top.certified and top.certificate["kind"] == "top_k" and top.certificate["m"] == top.n_samples
    assert np.array_equal(top.certificate["estimates"], top.estimates)
    assert set(top.certificate["top_set"].tolist()) == set(np.argsort(-phi)[:2].tolist())
