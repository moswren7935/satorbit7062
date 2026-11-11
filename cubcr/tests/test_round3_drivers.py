import numpy as np

from experiments import analyze, e9v3_certified
from experiments.e12_loo import rank_metrics, from_table
from experiments.e13_semantics import run_instance, exact_shapley
from tests.helpers import random_csg


def test_e12_rank_metrics_and_table_helpers():
    phi = np.array([0.5, 0.3, 0.2, 0.0])
    m = rank_metrics(phi, phi.copy())
    assert m["tau"] == 1.0 and m["top1_agree"] and m["top3_overlap"] == 1.0 and m["max_rank_disp"] == 0
    m2 = rank_metrics(phi, phi[::-1])
    assert m2["tau"] == -1.0 and not m2["top1_agree"] and m2["misordered_pair_frac"] == 1.0
    table = np.array([0.0, 0.2, 0.3, 0.9])
    ph, add, rem = from_table(table, 2)
    assert np.allclose(add, [0.2, 0.3]) and np.allclose(rem, [0.6, 0.7]) and abs(ph.sum() - 0.9) < 1e-12


def test_e13_bracket_on_tiny_instance():
    csg = random_csg(n_agents=2, n_states=5, k=3, seed=1, max_active=2, max_actions=2, min_actions=2, succ_per_row=2)
    rec = run_instance(("t", csg.to_json(), 3, 1))
    assert rec["status"] == "OK" and rec["n_coalitions"] == 4
    for c in rec["coalitions"]:
        assert c["v_step"] <= c["v_memrand"] + 1e-9 <= c["v_memdet"] + 2e-9
    assert rec["gap_max"] >= 0 and 0 <= rec["gap_zero_frac"] <= 1
    v = np.array([c["v_step"] for c in rec["coalitions"]])
    assert np.allclose(exact_shapley(v, 2).sum(), v[-1] - v[0])


def test_e9v3_grid_and_floor():
    cells = e9v3_certified.cells("full")
    assert len(cells) == 20 and all(c["k"] in (50, 100) for c in cells) and {c["n"] for c in cells} == {16, 20, 24, 28, 32}
    assert e9v3_certified.UPSILON_FLOOR == 0.05 and len(e9v3_certified.GEN_SEEDS) == 3


def test_analyzer_sections_render():
    for fn in (analyze.e9v3, analyze.e11, analyze.e12, analyze.e13, analyze.e14, lambda: analyze.e4("e4v2")):
        out = fn()
        assert isinstance(out, str) and len(out) > 5
