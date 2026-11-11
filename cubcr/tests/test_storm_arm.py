import json
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from helpers import random_csg
from cubcr import vi_ref, io_prism, io_storm
from experiments import protocol

from generators.vehicle import vehicle_csg
TOL = 1e-6
MISSING_BIN = "/nonexistent/storm-not-built-yet/storm"

needs_prism = pytest.mark.skipif(not io_prism.prism_available(),
                                 reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")


@pytest.fixture(scope="module")
def WORK(tmp_path_factory):
    return str(tmp_path_factory.mktemp("storm_arm_work"))


def _rnd():
    return random_csg(n_agents=3, n_states=14, k=3, seed=7)


def test_mdp_emission_structure_and_reuse():


    csg = _rnd()
    special = set(csg.fail_states) | set(csg.avoid_states)
    for mask in (0, 5, 7):
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, 3)
        assert mdp.splitlines()[3] == "mdp" and "player" not in mdp and "csg" not in mdp.split()
        assert props.strip() == 'Pmin=? [ F<=3 "fail" ]'
        n_cmds = sum(1 for ln in mdp.splitlines() if ln.strip().startswith("[]"))
        expected = sum(len(io_prism._mixed_rows(csg, s, mask)) for s in range(csg.n_states) if s not in special)
        assert n_cmds == expected + len(special)

    full = (1 << csg.n_agents) - 1
    mdp, _ = io_prism.emit_coalition_mdp(csg, full, 3)
    assert sum(1 for ln in mdp.splitlines() if ln.strip().startswith("[]")) == csg.n_states


@needs_prism
def test_mdp_emission_matches_vi_ref_munajib(WORK):

    csg = vehicle_csg()
    for mask in (0, 1, 3):
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, 1)
        r = io_prism.run_prism(mdp, props, timeout_s=120, workdir=WORK, tag=f"veh_mdp_{mask}",
                               include_identity=False)
        assert r["status"] == "OK", (r["status"], r["error"], r["stdout_path"])
        assert abs(r["value"] - vi_ref.solve(csg, mask, 1)) <= TOL


@needs_prism
def test_mdp_emission_matches_vi_ref_random(WORK):

    csg = _rnd()
    for mask in (0, 5, 7):
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, 3)
        r = io_prism.run_prism(mdp, props, timeout_s=120, workdir=WORK, tag=f"rnd_mdp_{mask}",
                               include_identity=False)
        assert r["status"] == "OK", (r["status"], r["error"], r["stdout_path"])
        assert abs(r["value"] - vi_ref.solve(csg, mask, 3)) <= TOL


def test_run_storm_skip_without_binary(tmp_path):
    csg = vehicle_csg()
    mdp, props = io_prism.emit_coalition_mdp(csg, 0, 1)
    r = io_storm.run_storm(mdp, props, timeout_s=10, workdir=str(tmp_path), tag="skiptest",
                           storm_bin=MISSING_BIN)
    assert r["status"] == "SKIP"
    assert r["value"] is None and r["wall_s"] is None and r["storm_reported_time_s"] is None
    assert "storm binary missing" in r["error"] and MISSING_BIN in r["error"]
    assert r["cmd"] is None and r["rc"] is None and r["log_path"] is None
    ident = r["tool_identity"]
    assert ident["available"] is False and ident["storm_version"] is None and ident["storm_bin_sha256"] is None

    r2 = io_storm.storm_value(csg, 1, 1, workdir=str(tmp_path), storm_bin=MISSING_BIN)
    assert r2["status"] == "SKIP"
    assert not io_storm.storm_available(MISSING_BIN)


def test_parse_storm_stdout_and_status():
    out = (
        "Storm 1.9.0\n\n"
        "Time for model input parsing: 0.005s.\n\n"
        "Time for model construction: 0.012s.\n\n"
        "-------------------------------------------------------------- \n"
        "Model type: \tMDP (sparse)\n"
        "States: \t26\n"
        "Transitions: \t58\n"
        "Choices: \t42\n"
        "Reward Models:  none\n"
        "-------------------------------------------------------------- \n\n"
        'Model checking property "P1": Pmin=? [true U<=5 "fail"] ...\n'
        "Time for model checking: 0.001s.\n"
        "Result (for initial states): 0.0525\n")
    err = ("\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:00.31\n"
           "\tMaximum resident set size (kbytes): 24196\n")
    d = io_storm.parse_storm_stdout(out, err)
    assert d["value"] == pytest.approx(0.0525)
    assert d["storm_construction_s"] == pytest.approx(0.012)
    assert d["storm_checking_s"] == pytest.approx(0.001)
    assert d["storm_reported_time_s"] == pytest.approx(0.013)
    assert (d["storm_states"], d["storm_transitions"], d["storm_choices"]) == (26, 58, 42)
    assert d["peak_rss_kb"] == 24196 and d["time_elapsed_s"] == pytest.approx(0.31)
    assert io_storm.classify_storm_status(False, 0, out, err, d["value"]) == ("OK", None)

    st, _ = io_storm.classify_storm_status(True, -1, "", "", None)
    assert st == "TO"
    st, e = io_storm.classify_storm_status(False, 1, "terminate called after throwing an instance of 'std::bad_alloc'", "", None)
    assert st == "MO" and "bad_alloc" in e
    st, e = io_storm.classify_storm_status(False, 1, "ERROR (ExpressionParser.cpp): something went wrong", "", None)
    assert st == "ERR" and e.startswith("ERROR")
    st, e = io_storm.classify_storm_status(False, 0, "no result line here", "", None)
    assert st == "ERR" and "Result (for initial states)" in e


def test_protocol_engines_has_storm():
    assert "storm" in protocol.ENGINES


@needs_prism
def test_driver_smoke_storm_absent(tmp_path):


    from cubcr import engine_gpu
    if engine_gpu.device_count() == 0:
        pytest.skip("no CUDA device")
    from experiments import e3storm
    out, work = str(tmp_path / "out"), str(tmp_path / "work")
    rc = e3storm.main(["--configs", "smoke", "--out", out, "--work", work,
                       "--storm-bin", MISSING_BIN, "--n-repeat", "2", "--timeout", "60",
                       "--cap-total", "300"])
    assert rc == 0
    recs = [f for f in os.listdir(out) if f.startswith("e3storm__") and "__" not in f[9:]]
    inst = [f for f in os.listdir(out) if f.startswith("e3storm__") and f.count("__") == 1]
    assert len(inst) == 1, sorted(os.listdir(out))
    with open(os.path.join(out, inst[0])) as fh:
        rec = json.load(fh)
    assert rec["n"] == 2 and rec["coalitions"] == 4

    assert [r["status"] for r in rec["storm"]["rows"]] == ["SKIP"] * 4
    assert rec["storm"]["n_skip"] == 4 and rec["storm"]["skipped"] is True
    assert rec["storm_outcome"] == "SKIP" and rec["gate_failed"] is None and rec["values_match"] is True
    for f in e3storm.SPEEDUP_FIELDS:
        assert rec[f] is None

    pm = rec["prism_mdp"]
    assert pm is not None and pm["n_ok"] == 4 and pm["n_mismatch"] == 0 and pm["n_fail"] == 0
    assert pm["max_abs_err_vs_gpu"] <= TOL

    with open(os.path.join(out, f"e3storm__{rec['instance_id']}__storm.json")) as fh:
        srec = json.load(fh)
    assert srec["engine"] == "storm" and srec["status"] == "NOT RUN" and srec["censored"] is False
    assert srec["wall_s"] is None and srec["coalition_solves_per_s"] is None
    assert srec["storm_n_skip"] == 4 and srec["storm_outcome"] == "SKIP"
    for eng in ("gpu_batched", "cpu_multicore"):
        with open(os.path.join(out, f"e3storm__{rec['instance_id']}__{eng}.json")) as fh:
            erec = json.load(fh)
        assert erec["status"] == "OK" and erec["speedup_vs_storm"] is None
    assert recs == inst
