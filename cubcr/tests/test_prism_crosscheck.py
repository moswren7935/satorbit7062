import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import random_csg

from cubcr import vi_ref
from cubcr import io_prism

pytestmark = pytest.mark.skipif(not io_prism.prism_available(),
                                reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")

from generators.vehicle import vehicle_csg
EXAMPLES = os.path.join(io_prism.REPO_ROOT, "external", "prism-games", "prism-examples", "csgs")
N_JOBS = 8
TOL = 1e-6


@pytest.fixture(scope="module")
def WORK(tmp_path_factory):

    keep = os.environ.get("CUBCR_PRISM_TEST_WORK")
    if keep:
        os.makedirs(keep, exist_ok=True)
        return keep
    return str(tmp_path_factory.mktemp("prism_work"))


def _check(csg, workdir, k=None, masks=None):
    r = io_prism.crosscheck(csg, masks=masks, k=k, tol=TOL, timeout_s=120, workdir=workdir, n_jobs=N_JOBS)
    bad = [x for x in r["rows"] if x["status"] != "OK" or x["abs_err"] > TOL]
    assert not bad, f"PRISM mismatch/failure: {bad}"
    return r


def test_munajib_vehicle_all_coalitions(WORK):
    csg = vehicle_csg()
    r = _check(csg, os.path.join(WORK, "vehicle"), k=1)
    got = {x["mask"]: x["prism"] for x in r["rows"]}

    assert got[3] == pytest.approx(1.0, abs=1e-6)
    assert got[1] == pytest.approx(0.6, abs=1e-6)
    assert got[2] == pytest.approx(0.2, abs=1e-6)
    assert got[0] == pytest.approx(0.12, abs=1e-6)


def test_step_counting_k0_k1_k2(WORK):

    csg = vehicle_csg()
    pt, pr = io_prism.emit_coalition_model(csg, 0, 0)
    r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=os.path.join(WORK, "semantics"), tag="k0")
    assert r["status"] == "OK" and r["value"] == pytest.approx(0.0, abs=1e-12)
    assert vi_ref.solve(csg, 0, 0) == 0.0
    import copy
    c2 = copy.deepcopy(csg)
    c2.init_state = 1
    pt, pr = io_prism.emit_coalition_model(c2, 0, 0)
    r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=os.path.join(WORK, "semantics"), tag="k0_initfail")
    assert r["status"] == "OK" and r["value"] == pytest.approx(1.0, abs=1e-12)
    assert vi_ref.solve(c2, 0, 0) == 1.0

    for m in (0, 1, 2):
        pt, pr = io_prism.emit_coalition_model(csg, m, 2)
        r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=os.path.join(WORK, "semantics"), tag=f"k2_{m}")
        assert r["status"] == "OK"
        assert r["value"] == pytest.approx(vi_ref.solve(csg, m, 2), abs=TOL)


def test_all_pinned_property_form(WORK):


    csg = vehicle_csg()
    pt, pr = io_prism.emit_coalition_model(csg, 3, 1)
    assert pr.strip() == '<<p0>> Pmin=? [ F<=1 "fail" ]'
    r_alt = io_prism.run_prism(pt, '<<p1>> Pmax=? [ F<=1 "fail" ]\n', timeout_s=60,
                               workdir=os.path.join(WORK, "semantics"), tag="allpin_alt")
    assert r_alt["status"] == "OK" and r_alt["value"] == pytest.approx(1.0, abs=1e-12)
    r_empty = io_prism.run_prism(pt, '<<>> Pmin=? [ F<=1 "fail" ]\n', timeout_s=60,
                                 workdir=os.path.join(WORK, "semantics"), tag="allpin_empty")
    assert r_empty["status"] == "ERR"


RANDOM_CASES = [

    ("n3_S12_k4_avoid", dict(n_agents=3, n_states=12, k=4, seed=20260829, max_active=2, max_actions=3,
                             n_fail=1, n_avoid=1)),
    ("n4_S30_k5_det_avoid", dict(n_agents=4, n_states=30, k=5, seed=20260830, max_active=3, max_actions=3,
                                 n_fail=2, n_avoid=2, deterministic_policy=True)),
    ("n4_S20_k3_dummy_avoid", dict(n_agents=4, n_states=20, k=3, seed=20260831, max_active=3, max_actions=4,
                                   n_fail=1, n_avoid=1, dummy_agents=(2,))),
]


@pytest.mark.parametrize("name,kw", RANDOM_CASES, ids=[c[0] for c in RANDOM_CASES])
def test_random_instances_all_coalitions(name, kw, WORK):
    csg = random_csg(**kw)
    r = _check(csg, os.path.join(WORK, name))
    vals = np.array([x["prism"] for x in r["rows"]])
    assert len(vals) == 1 << kw["n_agents"]

    assert vals.max() - vals.min() > 1e-3, vals
    assert r["max_abs_err"] <= TOL


IMPORT_CASES = [
    ("robot_coordination2", "robot_coordination/robot_coordination2.prism", {"l": 3, "q": 0.1}, 3, None, "crash"),
    ("medium_access2", "simple/medium_access2.prism", {"emax": 2, "q1": 0.9, "q2": 0.75}, 3, "s1=1", "fail"),
    ("rps2", "simple/rps2.prism", None, 2, "win=1", "fail"),
]


@pytest.mark.parametrize("name,rel,consts,k,fail_expr,fail_label", IMPORT_CASES, ids=[c[0] for c in IMPORT_CASES])
def test_import_prism_csg(name, rel, consts, k, fail_expr, fail_label, WORK):
    path = os.path.join(EXAMPLES, rel)
    if not os.path.exists(path):
        pytest.skip(f"example {rel} not found")
    r = io_prism.crosscheck_import(path, consts, k=k, fail_expr=fail_expr, fail_label=fail_label,
                                   workdir=os.path.join(WORK, "import_" + name), timeout_s=120)
    bad = [c for c in r["checks"] if c["status"] != "OK" or c["abs_err"] is None or c["abs_err"] > TOL]
    assert not bad, bad
    csg = r["csg"]
    assert csg.n_agents == 2
    c0 = r["checks"][0]
    assert c0["abs_err"] <= TOL
    if name == "robot_coordination2":
        assert c0["prism"] > 0.0
        assert csg.n_states == 65

    assert any(c["name"].startswith("orig_vs_emit_Pmax") and 0.0 < c["prism_orig"] < 1.0 for c in r["checks"]), r["checks"]
