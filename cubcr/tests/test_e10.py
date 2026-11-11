import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr.shapley import shapley_from_table
from experiments import protocol
from experiments.e10_shap import (
    CountingEngine, make_value_fn, masks_from_rows, npermutations_for_budget,
    run_sampler_arm, shap_explain, tool_record,
)

N, S_REQ, K = 5, 60, 8
SEED = 20260829


@pytest.fixture(scope="module")
def setup():
    from generators.ticketbench import build_compiled
    from cubcr.engine_cpu import CPUEngine
    comp, meta = build_compiled(N, S_REQ, K, 0.05, "none", SEED)
    eng = CPUEngine(comp, n_threads=1, block=8, dtype="fp64")
    v = eng.solve(np.arange(1 << N, dtype=np.uint64), K)
    phi = shapley_from_table(v, N)
    ups = float(v[-1] - v[0])
    assert ups > 0, "test instance must not be a null game"
    yield dict(comp=comp, eng=eng, v=v, phi=phi, ups=ups,
               iid=meta["instance_id"], cell=dict(pattern="none", hazard=0.0, n=N,
                                                  states=S_REQ, k=K))
    eng.close()


def all_binary_rows(n):
    return ((np.arange(1 << n)[:, None] >> np.arange(n)[None, :]) & 1).astype(np.float64)


def test_masks_from_rows_enumerates_all_masks():
    X = all_binary_rows(N)
    m = masks_from_rows(X, N)
    assert m.dtype == np.uint64
    assert np.array_equal(m, np.arange(1 << N, dtype=np.uint64))


def test_masks_from_rows_bit_direction():

    for i in range(N):
        row = np.zeros((1, N))
        row[0, i] = 1.0
        assert masks_from_rows(row, N)[0] == np.uint64(1) << np.uint64(i)

    assert masks_from_rows(np.ones(N), N)[0] == np.uint64((1 << N) - 1)
    with pytest.raises(ValueError):
        masks_from_rows(np.ones((1, N + 1)), N)


def test_value_fn_matches_engine_table_one_call(setup):
    ce = CountingEngine(setup["eng"])
    f = make_value_fn(ce, N, K)
    fv = f(all_binary_rows(N))
    assert np.array_equal(fv, setup["v"]), "f(X) must equal v(mask) bit-for-bit"
    assert ce.n_calls == 1, "the whole batch must go through ONE engine.solve call"
    assert ce.n_rows == 1 << N


def test_shap_matches_shapley_from_table(setup):
    P = 300
    sh = shap_explain(setup["eng"], N, K, npermutations=P, seed=0, warmup=False)

    assert abs(sh["base"] - setup["v"][0]) <= 1e-12, "base value must be v(empty)"
    assert abs(sh["base"] + sh["phi"].sum() - setup["v"][-1]) <= 1e-10, \
        "local accuracy: base + sum(shap values) = v(N)"


    tol = 0.05 * setup["ups"]
    err = np.max(np.abs(sh["phi"] - setup["phi"]))
    assert err <= tol, f"|shap - exact| = {err} > combined tolerance {tol}"


def test_counting_engine_counts_rows_and_calls(setup):
    ce = CountingEngine(setup["eng"])
    ce.solve(np.arange(4, dtype=np.uint64), K)
    ce.solve(np.arange(7, dtype=np.uint64), K)
    assert (ce.n_rows, ce.n_calls) == (11, 2)


def test_shap_oracle_row_count(setup):
    P = 20
    sh = shap_explain(setup["eng"], N, K, npermutations=P, seed=0, warmup=False)

    assert sh["n_engine_calls"] == P
    assert sh["n_oracle_rows"] == P * (2 * N + 1)
    assert sh["max_evals"] == P * (2 * N + 1)
    assert npermutations_for_budget(sh["n_oracle_rows"], N) == P
    assert npermutations_for_budget(1, N) == 1


def test_sampler_oracle_row_count_and_coverage(setup):
    r, wall, ce, ups, fields = run_sampler_arm(setup["eng"], setup["cell"], SEED,
                                               eps_div=20, cap_s=60)

    assert ce.n_rows == r.n_solves + 2
    assert ce.n_calls == r.n_engine_calls + 1
    assert r.n_coalitions_requested == fields["n_oracle_rows_requested"]
    assert r.n_solves <= (1 << N) + 2, "cache must collapse requests to distinct masks"
    assert fields["certified"] and r.status == "eps"
    assert np.all(np.abs(r.estimates - setup["phi"]) <= r.half_widths + 1e-12), \
        "certificate must cover the exact Shapley value"
    assert np.max(r.half_widths) <= ups / 20 + 1e-12


def test_record_schema(setup, tmp_path):
    import shap
    phi = setup["phi"]
    exact_ref = {"n_coalitions": 1 << N, "wall_s": 0.001, "source": "test"}
    recs = {}
    recs["cubcr_sampler"] = tool_record(
        setup["iid"], setup["cell"], setup["comp"], "cubcr_sampler", SEED, 1, 60.0,
        0.02, 34, 3, phi, phi, setup["ups"], exact_ref, shap.__version__,
        certified=True, covers_exact=True, eps_abs=setup["ups"] / 20,
        accuracy_statement="CERTIFIED")
    recs["shap_permutation"] = tool_record(
        setup["iid"], setup["cell"], setup["comp"], "shap_permutation", SEED, 1, 60.0,
        0.01, 33, 3, phi, phi, setup["ups"], exact_ref, shap.__version__,
        certified=False, budget_factor=1, npermutations=3,
        accuracy_statement="NOT certified")
    for tool, rec in recs.items():
        path = protocol.write_result(tmp_path, rec, filename=f"{tool}.json")
        got = json.loads(path.read_text())
        protocol.validate_record(got)
        assert got["engine"] == "cpu_multicore"
        assert got["exp"] == "e10"
        assert got["tool"] == tool
        assert got["shap_version"] == shap.__version__
        assert got["n"] == N and got["k"] == K and got["seed"] == SEED
        assert got["n_oracle_rows"] > 0 and got["threads"] == 1
        assert got["max_abs_err_vs_exact"] == 0.0
        assert got["kendall_tau"] == pytest.approx(1.0)
        assert isinstance(got["certified"], bool)
    assert recs["cubcr_sampler"]["certified"] is True
    assert recs["shap_permutation"]["certified"] is False
