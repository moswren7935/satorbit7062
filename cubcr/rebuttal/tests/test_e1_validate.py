import copy
import importlib.util
import os

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("e1_validate_inputs", os.path.join(HERE, "..", "scripts", "e1_validate_inputs.py"))
V = importlib.util.module_from_spec(spec)
spec.loader.exec_module(V)


def _model():
    from generators.ticketbench import build_compiled
    comp, meta = build_compiled(8, 1000, 25, 0.05, "none", 20260829)
    return comp


def _clone(comp):
    return copy.deepcopy(comp)


def test_clean_model_passes():
    out = V.check_arrays(_model())
    assert out["finite_P"] and out["finite_pi"] and out["n_negative_P"] == 0
    assert out["max_row_sum_dev"] < 1e-15 and out["max_policy_sum_dev"] < 1e-15
    assert out["fail_structurally_absorbing"] and out["unused_slots_exact_1_0"] and out["rows_eq_prod_radix"]


def test_detects_negative_probability_and_row_sum():
    c = _clone(_model())
    c.val = c.val.copy(); c.val[5] = -1e-3
    out = V.check_arrays(c)
    assert out["n_negative_P"] == 1 and out["max_row_sum_dev"] > 1e-4


def test_detects_nan():
    c = _clone(_model())
    c.val = c.val.copy(); c.val[3] = np.nan
    assert not V.check_arrays(c)["finite_P"]


def test_detects_policy_defects():
    c = _clone(_model())
    W = c.weights.copy()
    s = int(np.flatnonzero((c.state_kind == 0) & (c.state_agent[:, 0] >= 0))[0])
    m = int(c.state_radix[s, 0])
    W[s, 0, 0] += 1e-6
    W[s, 0, m] = 0.5 if m < W.shape[2] else W[s, 0, m - 1]
    c.weights = W
    out = V.check_arrays(c)
    assert out["max_policy_sum_dev"] > 1e-7
    if m < W.shape[2]:
        assert out["max_pi_mass_beyond_radix"] == 0.5


def test_detects_non_absorbing_failure_state():
    c = _clone(_model())
    f = int(np.flatnonzero(c.state_kind == 1)[0])
    r0 = int(c.state_row_start[f]); p0 = int(c.row_ptr[r0])
    c.col = c.col.copy(); c.col[p0] = 0
    out = V.check_arrays(c)
    assert not out["fail_structurally_absorbing"]
