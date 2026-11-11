import json
import os

import numpy as np
import pytest

import rb
import e2a_lib as L


@pytest.mark.parametrize("n,k", [(2, 10), (3, 4), (2, 3)])
def test_step_table_engine_maps_masks_to_player_bitsets(n, k):
    from cubcr.shapley_sampled import CoalitionOracle
    from cubcr.masks import step_mask_matrix
    d = n * k
    rng = np.random.default_rng(0)
    table = rng.random(1 << d)
    eng = L.StepTableEngine(table, compiled=None, n=n, k=k)
    orc = CoalitionOracle(eng, d, "step", n=n, k=k, use_cache=False)
    bitsets = rng.integers(0, 1 << d, size=500)
    bitsets[:2] = [0, (1 << d) - 1]
    rows = orc.masks_from_bitsets(bitsets)
    vals = orc.values(rows)
    assert np.array_equal(vals, table[bitsets])

    for b in bitsets[:50]:
        players = [p for p in range(d) if (int(b) >> p) & 1]
        cols = step_mask_matrix(players, n, k)
        assert eng.bitsets(cols.reshape(k, 1))[0] == int(b)


def test_step_table_engine_rejects_bad_masks():
    eng = L.StepTableEngine(np.zeros(1 << 20), compiled=None, n=2, k=10)
    with pytest.raises(ValueError):
        eng.solve(np.zeros((9, 3), dtype=np.uint64))
    bad = np.zeros((10, 1), dtype=np.uint64); bad[0, 0] = 4
    with pytest.raises(ValueError):
        eng.solve(bad)


def test_snap_ties():
    x = np.array([0.3, 0.1 + 1e-17, 0.1, 0.2, 0.1 - 2e-17, 0.5])
    s = L.snap_ties(x)
    assert s[1] == s[2] == s[4]
    assert s[0] == 0.3 and s[3] == 0.2 and s[5] == 0.5
    assert np.array_equal(L.snap_ties(np.array([1.0, 2.0])), np.array([1.0, 2.0]))


def test_boundary_well_defined():
    phi = np.array([0.3, 0.3 + 1e-17, 0.3 - 1e-17, 0.2, 0.1])
    assert not L.boundary_well_defined(phi, 1)
    assert not L.boundary_well_defined(phi, 2)
    assert L.boundary_well_defined(phi, 3)


def test_concentration_metrics():
    phi = np.array([0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
    c = L.concentration_metrics(phi, 1.0, 0.05)
    assert c["share_top1"] == 0.4 and abs(c["share_top3"] - 0.8) < 1e-15
    assert c["n_cover80"] == 3
    assert c["n_gt_eps20"] == 4
    assert c["n_gt_ups100"] == 6
    assert abs(c["gap12_eps20"] - 2.0) < 1e-12 and abs(c["gap34_eps20"]) < 1e-12
    assert abs(c["gini"] - rb.gini(phi)) == 0


def test_run_metrics_separation_and_coverage():
    phi = np.array([0.30, 0.20, 0.15, 0.05, 0.02])
    est = np.array([0.31, 0.19, 0.16, 0.05, 0.01])

    Lb = np.array([0.26, 0.17, 0.12, 0.03, 0.00])
    Rb = np.array([0.33, 0.24, 0.18, 0.07, 0.03])
    H = np.maximum(est - Lb, Rb - est)
    m = L.run_metrics(est, H, Lb, Rb, phi, ups=0.72, eps_run=0.05, eps20=0.036)
    assert m["top1_correct"] and m["top3_exact"] == 1.0 and m["top3_jaccard"] == 1.0
    assert m["sep1"] == 1.0 and m["sep1_and_correct"] == 1.0
    assert m["sep2"] == 0.0
    assert m["sep3"] == 1.0
    assert m["covers_cs"] and m["covers_reported_strict"] and m["n_cover_cs"] == 5
    assert abs(m["max_abs_err"] - 0.01) < 1e-15
    assert m["tau_a"] == 1.0

    phi2 = phi.copy(); phi2[0] = 0.40
    m2 = L.run_metrics(est, H, Lb, Rb, phi2, ups=0.82, eps_run=0.05, eps20=0.041)
    assert not m2["covers_cs"] and m2["n_cover_cs"] == 4


def test_run_metrics_tied_top_is_na():
    phi = np.array([0.3, 0.3 + 1e-17, 0.1, 0.05])
    est = np.array([0.29, 0.31, 0.11, 0.04])
    H = np.full(4, 0.02)
    m = L.run_metrics(est, H, est - H, est + H, phi, ups=0.75, eps_run=0.05, eps20=0.0375)
    assert m["top1_correct"]
    assert np.isnan(m["sep1"]) and np.isnan(m["top1_exact"])
    assert m["top2_exact"] == 1.0 and m["sep2"] in (0.0, 1.0)


def test_stored_payload_matches_record_fields():
    estimate = np.array([0.3, 0.2])
    lower = np.array([0.27, 0.19])
    upper = np.array([0.32, 0.24])
    fp = 0.001
    half_widths = np.maximum(estimate - lower, upper - estimate) + fp
    rec = {
        "certificate": {"cs_lower": lower.tolist(), "cs_upper": upper.tolist()},
        "fp_term": fp, "n": 2, "sampling_status": "eps", "certified": True,
        "wall_s": 0.1, "n_units": 16, "n_perms": 32, "n_solves": 4,
        "n_batches": 1, "batch_sizes": [16], "eps_abs": 0.05,
        "shapley_sampled": estimate.tolist(), "half_widths": half_widths.tolist(),
        "hw_max": float(half_widths.max()),
    }
    p = L.stored_payload(rec)
    est = np.asarray(p["phi_hat"]); H = np.asarray(p["half_widths"])
    cl = np.asarray(p["cs_lower"]); cu = np.asarray(p["cs_upper"])

    assert np.array_equal(H, np.maximum(est - cl, cu - est) + p["fp_term"])

    assert np.all(np.asarray(p["L"]) >= est - H - 1e-15) and np.all(np.asarray(p["R"]) <= est + H + 1e-15)
    assert np.all(np.asarray(p["L"]) < cl) and np.all(np.asarray(p["R"]) > cu)
    assert p["n_evals_unique"] == 4 and p["n_evals_total"] == 98
    assert p["status"] == "eps" and p["certified"] is True


needs_oc_table = pytest.mark.skipif(not L.OC_TABLE.exists(), reason="Overcooked k=10 table not generated yet")


@needs_oc_table
def test_overcooked_table_matches_stored_phi():
    from cubcr.shapley_fast import shapley_from_table_fast
    tab = L.load_overcooked_table()
    oc = L.overcooked_task()
    assert tab.shape == (1 << 20,)
    assert float(tab[-1] - tab[0]) == oc["upsilon"]
    phi = shapley_from_table_fast(tab, 20)
    assert np.max(np.abs(phi - oc["phi_exact"])) <= 1e-12


@needs_oc_table
@pytest.mark.parametrize("seed", list(L.PAPER_SEEDS))
def test_overcooked_step_oracle_reproduces_stored_runs_bitwise(seed):
    from cubcr.shapley_sampled import sample_shapley
    oc = L.overcooked_task()
    rec = oc["cert_gpu"][seed]
    comp, _ = _oc_comp()
    eng = L.StepTableEngine(L.load_overcooked_table(), compiled=comp, n=2, k=10)
    tab = eng.table
    eps = float(tab[-1] - tab[0]) / 20
    assert eps == rec["eps_abs"]
    r = sample_shapley(eng, seed=seed, eps=eps, **L.OC_SAMPLER_KW)
    assert r.certified and r.status == "eps"
    assert np.array_equal(r.estimates, np.asarray(rec["estimates"]).reshape(-1))
    assert np.array_equal(r.half_widths, np.asarray(rec["half_widths"]).reshape(-1))
    assert r.n_perms == rec["n_perms"] and r.n_solves == rec["n_solves"]


_OC_COMP = {}


def _oc_comp():
    if "c" not in _OC_COMP:
        _OC_COMP["c"] = L.build_overcooked_k10()
    return _OC_COMP["c"]
