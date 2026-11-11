import copy
import math
import os
import re
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr import io_prism, vi_ref
from experiments import protocol
from experiments import e3_prism as e3

from generators.vehicle import vehicle_csg
K = 1
IID = "vehicle_review2"
needs_prism = pytest.mark.skipif(not io_prism.prism_available(), reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")


@pytest.fixture(scope="module")
def csg():
    return vehicle_csg()


@pytest.fixture(scope="module")
def tags():
    return protocol.machine_tags()


@pytest.fixture(scope="module")
def vref(csg):
    return np.array([vi_ref.solve(csg, m, K) for m in range(1 << csg.n_agents)], dtype=np.float64)


@needs_prism
def test_real_oom_is_MO_and_ok_run_parses_like_the_log(csg, vref, tmp_path):
    pt, pr = io_prism.emit_coalition_model(csg, 0, K)

    r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=str(tmp_path), tag="oom", javamaxmem="2m", include_identity=False)
    assert r["status"] == "MO" and r["timed_out"] is False and r["value"] is None
    assert "OutOfMemoryError" in r["error"] and r["returncode"] != 0
    assert "-javamaxmem 2m" in r["cmd"] and r["prism_construction_s"] is None

    r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=str(tmp_path), tag="ok", include_identity=False)
    assert r["status"] == "OK" and r["error"] is None and r["returncode"] == 0
    log = open(r["stdout_path"]).read()
    assert log.splitlines()[0] == r["cmd"] and "=== STDOUT ===" in log and "=== STDERR ===" in log
    constr = float(re.search(r"Time for model construction: ([0-9.]+) seconds", log).group(1))
    check = float(re.search(r"Time for model checking: ([0-9.]+) seconds", log).group(1))
    states = int(re.search(r"^States:\s+(\d+)", log, re.M).group(1))
    trans = int(re.search(r"^Transitions:\s+(\d+)", log, re.M).group(1))
    result = float(re.search(r"^Result: (\S+)", log, re.M).group(1))
    rss = int(re.search(r"Maximum resident set size \(kbytes\): (\d+)", log).group(1))
    assert r["prism_construction_s"] == constr and r["prism_checking_s"] == check
    assert r["prism_solver_s"] == pytest.approx(constr + check) and 0 <= r["prism_solver_s"] < r["wall_s"]
    assert r["prism_states"] == states == 3 and r["prism_transitions"] == trans and r["peak_rss_kb"] == rss > 0
    assert r["value"] == result == pytest.approx(vref[0], abs=1e-9) == pytest.approx(0.12, abs=1e-9)

    exported = [ln.strip() for ln in open(r["results_path"]) if ln.strip()]
    assert exported[0].lower().startswith("result") and float(exported[1]) == r["value"] == r["value_exported"]
    argv = r["cmd_argv"]
    i = argv.index("-javamaxmem")
    assert argv[i + 1] == io_prism.DEFAULT_JAVAMAXMEM == "16g" and argv[argv.index("-maxiters") + 1] == "100000"
    assert argv[argv.index("-exportresults") + 1] == r["results_path"] and argv[argv.index("-prop") + 1] == "1"
    assert r["time_elapsed_s"] is not None and abs(r["time_elapsed_s"] - r["wall_s"]) < 1.0


def _eng(vref, gate_ok=True):
    t = {"median_s": 0.01, "min_s": 0.009, "max_s": 0.011, "mean_s": 0.01, "all_s": [0.01] * 5, "warmup_s": [0.05],
         "n_repeat": 5, "warmup": 1}
    return {"vg": vref.copy(), "vc": vref.copy(), "gpu": dict(t), "cpu": dict(t, median_s=0.02), "build_s": 0.001,
            "threads": 32, "n_repeat": 5, "warmup": 1, "max_abs_err_gpu_vs_cpu": 0.0 if gate_ok else 3e-12,
            "gpu_cpu_gate_ok": gate_ok, "gpu_values_finite_in_unit": True}


def _row(mask, value, status="OK", wall=1.5, solver=(0.03, 0.02)):
    c, k = solver if solver is not None else (None, None)
    return {"mask": mask, "status": status, "value": value if status == "OK" else None, "wall_s": wall,
            "prism_construction_s": c, "prism_checking_s": k, "prism_solver_s": (c + k) if c is not None else None,
            "prism_states": 3, "prism_transitions": 6, "peak_rss_kb": 100000, "time_elapsed_s": wall, "cmd": "prism ...",
            "rc": 0 if status == "OK" else (-1 if status == "TO" else 1), "log": "/x.log", "model_path": "/x.prism",
            "props_path": "/x.props", "results_path": "/x.results.txt", "emit_s": 0.001,
            "error": None if status == "OK" else f"{status} line", "timed_out": status == "TO", "reason": None}


def _rows(vref, overrides=None, statuses=None, timeout=60.0):
    rows = []
    for m in range(len(vref)):
        st = (statuses or {}).get(m, "OK")
        v = (overrides or {}).get(m, float(vref[m]))
        rows.append(_row(m, v, st, wall=timeout if st == "TO" else 1.5))
    return rows


def _all_speedups(rec):
    return {k: v for k, v in rec.items() if k.startswith("speedup") and v is not None}


CASES = {
    "correct": dict(),
    "nan": dict(overrides={1: float("nan")}),
    "inf": dict(overrides={2: float("inf")}),
    "above_one_within_tol": dict(overrides={3: 1.0 + 5e-7}),
    "below_zero_within_tol": dict(overrides={0: -1e-7}),
    "boundary_pass": dict(overrides={2: 0.2 + 9e-7}),
    "boundary_fail": dict(overrides={2: 0.2 + 1.5e-6}),
    "to_only": dict(statuses={3: "TO"}),
    "all_to": dict(statuses={0: "TO", 1: "TO", 2: "TO", 3: "TO"}),
    "to_plus_mo": dict(statuses={3: "TO", 1: "MO"}),
    "to_plus_mismatch": dict(statuses={3: "TO"}, overrides={1: 0.6 + 1e-3}),
    "err": dict(statuses={2: "ERR"}),
    "gpu_cpu_fail": dict(gate_ok=False),
}
EXPECT = {
    "correct": (True, "OK", True, False), "nan": (False, "GATE_FAILED", False, False), "inf": (False, "GATE_FAILED", False, False),
    "above_one_within_tol": (False, "GATE_FAILED", False, False), "below_zero_within_tol": (False, "GATE_FAILED", False, False),
    "boundary_pass": (True, "OK", True, False), "boundary_fail": (False, "GATE_FAILED", False, False),
    "to_only": (True, "TO", False, True), "all_to": (True, "TO", False, True), "to_plus_mo": (True, "MO", False, False),
    "to_plus_mismatch": (False, "GATE_FAILED", False, False), "err": (False, "ERR", False, False),
    "gpu_cpu_fail": (False, "GATE_FAILED", False, False),
}


def _summarize_case(name, csg, vref, tags, timeout=60.0, cap_total=600.0):
    spec = CASES[name]
    rows = _rows(vref, spec.get("overrides"), spec.get("statuses"), timeout)
    total = sum(r["wall_s"] for r in rows)
    eng = _eng(vref, spec.get("gate_ok", True))
    rec = e3.summarize(csg, K, IID + "_" + name, rows, total, eng, tags, "fixture", timeout, cap_total, emit_total=0.004)
    return rec, total


@pytest.mark.parametrize("name", sorted(CASES))
def test_gate_cannot_be_bypassed(name, csg, vref, tags):
    rec, total = _summarize_case(name, csg, vref, tags)
    vm, outcome, primary, lower = EXPECT[name]
    assert rec["values_match"] is vm and rec["prism_outcome"] == outcome, (name, rec["gate_failed"])
    assert (rec["gate_failed"] is None) is vm
    sp = _all_speedups(rec)
    if primary:
        assert set(sp) == {"speedup_gpu_vs_prism", "speedup_gpu_vs_prism_solver_only", "speedup_cpu_vs_prism"}
        assert sp["speedup_gpu_vs_prism"] == pytest.approx(total / 0.01) and sp["speedup_cpu_vs_prism"] == pytest.approx(total / 0.02)
        assert sp["speedup_gpu_vs_prism_solver_only"] == pytest.approx(rec["prism"]["solver_total_s"] / 0.01)
        assert rec["prism"]["solver_total_s"] == pytest.approx(0.05 * 4) and rec["prism"]["jvm_overhead_s"] == pytest.approx(6.0 - 0.2)
        assert "shapley_prism" in rec and rec["shapley_max_abs_err"] <= 1e-6
    elif lower:
        assert set(sp) == {"speedup_gpu_vs_prism_lower_bound", "speedup_cpu_vs_prism_lower_bound"}
        assert sp["speedup_gpu_vs_prism_lower_bound"] == pytest.approx(total / 0.01) and total >= 60.0
        assert "shapley_prism" not in rec
    else:
        assert sp == {}, (name, sp)
        assert "shapley_prism" not in rec
    p = rec["prism"]
    assert p["n_ok"] + p["n_to"] + p["n_mo"] + p["n_err"] + p["n_not_run"] == 4
    if name == "boundary_fail":
        row = next(r for r in p["rows"] if r["mask"] == 2)
        assert row["mismatch"] is True and row["abs_err"] == pytest.approx(1.5e-6, rel=1e-6) and p["n_mismatch"] == 1
    if name == "boundary_pass":
        assert p["n_mismatch"] == 0 and p["max_abs_err_vs_gpu"] == pytest.approx(9e-7, rel=1e-6)
    if name in ("nan", "inf"):
        bad = next(r for r in p["rows"] if r["mismatch"])
        assert bad["abs_err"] is None and (bad["value"] is None or not math.isfinite(bad["value"]))
    if name == "gpu_cpu_fail":
        assert rec["gate_failed"].startswith("gpu_cpu_mismatch") and "3.000e-12" in rec["gate_failed"]
    if name == "err":
        assert "prism_err" in rec["gate_failed"]


@pytest.mark.parametrize("name", sorted(CASES))
def test_protocol_records_status_matrix(name, csg, vref, tags, tmp_path):
    rec, total = _summarize_case(name, csg, vref, tags)
    paths = e3.write(rec, tags, str(tmp_path / name), 600.0)
    recs = {}
    for pth in paths:
        r = protocol.read_result(pth)
        protocol.validate_record(r)
        recs[r["engine"]] = r
    assert set(recs) == {"prism_games", "gpu_batched", "cpu_multicore"}
    pr = recs["prism_games"]
    outcome, p = rec["prism_outcome"], rec["prism"]

    if p["n_err"]:
        assert pr["status"] == "CRASH" and pr["censored"] is False and pr["wall_s"] is None and outcome == "ERR"
    elif p["n_mo"]:
        assert pr["status"] == "MO" and pr["censored"] is False and pr["wall_s"] is None and outcome == "MO"
    elif p["complete"]:
        assert pr["status"] == "OK" and pr["censored"] is False and pr["wall_s"] == pytest.approx(total)
        assert outcome == ("OK" if rec["values_match"] else "GATE_FAILED")
    else:
        assert pr["status"] == "TO" and pr["censored"] is True and pr["cap_s"] == 60.0 and pr["wall_s"] >= pr["cap_s"]
        assert (rec["speedup_gpu_vs_prism_lower_bound"] is not None) is rec["values_match"]
        assert outcome == ("TO" if rec["values_match"] else "GATE_FAILED")
    assert pr["censored"] is (pr["status"] == "TO")
    assert pr["values_match"] is rec["values_match"] and pr["gate_failed"] == rec["gate_failed"]

    assert (pr["coalition_solves_per_s"] is not None) is (rec["values_match"] and rec["prism"]["complete"])

    g, c = recs["gpu_batched"], recs["cpu_multicore"]
    assert g["speedup_vs_prism"] == rec["speedup_gpu_vs_prism"] and g["speedup_vs_prism_lower_bound"] == rec["speedup_gpu_vs_prism_lower_bound"]
    assert g["speedup_vs_prism_solver_only"] == rec["speedup_gpu_vs_prism_solver_only"]
    assert c["speedup_vs_prism"] == rec["speedup_cpu_vs_prism"] and c["speedup_vs_prism_lower_bound"] == rec["speedup_cpu_vs_prism_lower_bound"]
    assert g["speedup_is_lower_bound"] is (rec["speedup_gpu_vs_prism_lower_bound"] is not None)
    assert pr["cap_per_call_s"] == 60.0 and pr["cap_total_s"] == 600.0
    for r in recs.values():
        assert r["prism_outcome"] == outcome and r["cap_s"] > 0
        assert r["tool_identity"]["prism_launcher_sha256"] == io_prism.tool_identity()["prism_launcher_sha256"]
        assert r["engine_binaries"]["libcubcr_cpu.so"]["path"]


def test_missing_prism_times_null_the_solver_only_metric_only(csg, vref, tags):
    rows = _rows(vref)
    rows[1].update(prism_construction_s=None, prism_checking_s=None, prism_solver_s=None)
    total = sum(r["wall_s"] for r in rows)
    rec = e3.summarize(csg, K, IID, rows, total, _eng(vref), tags, "fixture", 60.0, 600.0)
    p = rec["prism"]
    assert rec["values_match"] is True and rec["prism_outcome"] == "OK"
    assert p["solver_times_complete"] is False and p["solver_total_s"] is None and p["jvm_overhead_s"] is None
    assert rec["speedup_gpu_vs_prism_solver_only"] is None
    assert rec["speedup_gpu_vs_prism"] == pytest.approx(total / 0.01)


def test_aggregate_cap_not_run_uses_cap_total_and_stays_a_lower_bound(csg, vref, tags, tmp_path):
    rows = _rows(vref, statuses={1: "TO"}, timeout=60.0)
    rows[2] = e3._not_run_row(2, "aggregate cap: cumulative PRISM wall 61.5s > cap_total 50.0s")
    rows[3] = e3._not_run_row(3, "aggregate cap: cumulative PRISM wall 61.5s > cap_total 50.0s")
    total = 1.5 + 60.0
    rec = e3.summarize(csg, K, IID, rows, total, _eng(vref), tags, "fixture", 60.0, 50.0)
    assert rec["prism"]["n_ok"] == 1 and rec["prism"]["n_to"] == 1 and rec["prism"]["n_not_run"] == 2
    assert rec["prism_outcome"] == "TO" and rec["speedup_gpu_vs_prism_lower_bound"] == pytest.approx(61.5 / 0.01)
    assert rec["speedup_gpu_vs_prism"] is None
    pr = protocol.read_result(e3.write(rec, tags, str(tmp_path), 50.0)[0])
    protocol.validate_record(pr)
    assert pr["status"] == "TO" and pr["censored"] is True and pr["cap_s"] == 50.0 and pr["wall_s"] == pytest.approx(61.5)

    rows2 = _rows(vref, statuses={1: "ERR"})
    rows2[2] = e3._not_run_row(2, "aborted after ERR on mask 1")
    rows2[3] = e3._not_run_row(3, "aborted after ERR on mask 1")
    rec2 = e3.summarize(csg, K, IID, rows2, 3.0, _eng(vref), tags, "fixture", 60.0, 50.0)
    assert rec2["prism_outcome"] == "ERR" and _all_speedups(rec2) == {} and rec2["values_match"] is False
    assert e3._prism_record_status(rec2)[0] == "CRASH"


def test_err_stops_further_prism_calls(csg, monkeypatch, tmp_path):
    calls = []

    def fake(prism_text, props_text, timeout_s=600.0, workdir=None, tag="model", **kw):
        mask = int(tag.rsplit("mask", 1)[1])
        calls.append(mask)
        if mask == 1:
            return dict(_row_to_result(_row(mask, None, "ERR")), error="Error: Unknown switch -nosuchflag")
        return _row_to_result(_row(mask, 0.5))

    monkeypatch.setattr(e3.io_prism, "run_prism", fake)
    rows, total, emit_total = e3.prism_all_coalitions(csg, K, IID, 60.0, 600.0, str(tmp_path), verbose=False)
    assert calls == [0, 1] and [r["status"] for r in rows] == ["OK", "ERR", "NOT RUN", "NOT RUN"]
    assert total == pytest.approx(3.0) and emit_total > 0

    calls.clear()
    rows, _, _ = e3.prism_all_coalitions(csg, K, IID, 60.0, 600.0, str(tmp_path), stop_on_err=False, verbose=False)
    assert calls == [0, 1, 2, 3] and rows[1]["status"] == "ERR"


def _row_to_result(row):
    r = dict(row)
    r["returncode"] = r.pop("rc")
    r["stdout_path"] = r.pop("log")
    return r
