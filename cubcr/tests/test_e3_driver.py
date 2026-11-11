import copy
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr import io_prism
from cubcr import engine_gpu
from experiments import protocol
from experiments import e3_prism as e3

pytestmark = [pytest.mark.skipif(not io_prism.prism_available(), reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}"),
              pytest.mark.skipif(engine_gpu.device_count() == 0, reason="no CUDA device")]

from generators.vehicle import vehicle_csg
K = 1
IID = "vehicle_e3test"


class _Args:
    timeout = 60.0
    cap_total = 600.0
    javamaxmem = None
    maxiters = io_prism.DEFAULT_MAXITERS
    n_repeat = 5
    threads = 32


@pytest.fixture(scope="module")
def work(tmp_path_factory):
    return str(tmp_path_factory.mktemp("e3_work"))


@pytest.fixture(scope="module")
def tags():
    return protocol.machine_tags()


@pytest.fixture(scope="module")
def real_run(work, tags):

    csg = vehicle_csg()
    eng = e3.time_engines(csg, K, n_repeat=5, threads=32)
    rows, total, emit_total = e3.prism_all_coalitions(csg, K, IID, timeout_s=60.0, cap_total=600.0, workdir=work, verbose=False)
    rec = e3.summarize(csg, K, IID, copy.deepcopy(rows), total, eng, tags, "fixture", 60.0, 600.0, emit_total=emit_total)
    return {"csg": csg, "eng": eng, "rows": rows, "total": total, "emit_total": emit_total, "rec": rec}


def _fake_run_prism_from_rows(rows, perturb=None, status_override=None):

    by_mask = {r["mask"]: r for r in rows}

    def fake(prism_text, props_text, timeout_s=600.0, workdir=None, tag="model", javamaxmem=None, maxiters=None,
             include_identity=False, **kw):
        mask = int(tag.rsplit("mask", 1)[1])
        r = by_mask[mask]
        out = {"value": r["value"], "wall_s": r["wall_s"], "status": r["status"], "error": r["error"], "timed_out": r["timed_out"],
               "prism_construction_s": r["prism_construction_s"], "prism_checking_s": r["prism_checking_s"],
               "prism_solver_s": r["prism_solver_s"], "prism_states": r["prism_states"], "prism_transitions": r["prism_transitions"],
               "peak_rss_kb": r["peak_rss_kb"], "time_elapsed_s": r["time_elapsed_s"], "cmd": r["cmd"], "returncode": r["rc"],
               "stdout_path": r["log"], "model_path": r["model_path"], "props_path": r["props_path"], "results_path": r["results_path"]}
        if perturb and mask in perturb:
            out["value"] = out["value"] + perturb[mask]
        if status_override and mask in status_override:
            st, err = status_override[mask]
            out.update({"status": st, "error": err, "value": None, "timed_out": st == "TO",
                        "prism_construction_s": None, "prism_checking_s": None, "prism_solver_s": None,
                        "returncode": -1 if st == "TO" else 1})
            if st == "TO":
                out["wall_s"] = timeout_s
        return out
    return fake


def test_gate_passes_on_correct_values(real_run):
    rec = real_run["rec"]
    p = rec["prism"]
    assert p["n_ok"] == 4 and p["n_to"] == p["n_mo"] == p["n_err"] == p["n_not_run"] == 0 and p["complete"]
    assert rec["values_match"] is True and rec["gate_failed"] is None and rec["prism_outcome"] == "OK"
    assert p["max_abs_err_vs_gpu"] is not None and p["max_abs_err_vs_gpu"] <= 1e-6
    assert p["n_mismatch"] == 0 and all(r["mismatch"] is False and r["abs_err"] <= 1e-6 for r in p["rows"])

    got = {r["mask"]: r["value"] for r in p["rows"]}
    assert got[0] == pytest.approx(0.12, abs=1e-9) and got[1] == pytest.approx(0.6, abs=1e-9)
    assert got[2] == pytest.approx(0.2, abs=1e-9) and got[3] == pytest.approx(1.0, abs=1e-9)

    assert rec["gpu_cpu_gate_ok"] and rec["max_abs_err_gpu_vs_cpu"] <= 1e-12

    assert rec["speedup_gpu_vs_prism"] == pytest.approx(p["total_wall_s"] / rec["gpu_wall_s"])
    assert rec["speedup_cpu_vs_prism"] == pytest.approx(p["total_wall_s"] / rec["cpu32_wall_s"])
    assert rec["speedup_gpu_vs_prism_solver_only"] == pytest.approx(p["solver_total_s"] / rec["gpu_wall_s"])
    assert rec["speedup_gpu_vs_prism_lower_bound"] is None
    assert rec["shapley_max_abs_err"] <= 1e-6


def test_engine_timing_protocol(real_run):
    rec = real_run["rec"]
    for key in ("gpu_timing", "cpu_timing"):
        t = rec[key]
        assert t["n_repeat"] == 5 and t["warmup"] == 1 and len(t["all_s"]) == 5 and len(t["warmup_s"]) == 1
        assert t["median_s"] == pytest.approx(float(np.median(t["all_s"])))
        assert t["min_s"] <= t["median_s"] <= t["max_s"]
    assert rec["gpu_wall_s"] == rec["gpu_timing"]["median_s"] and rec["cpu32_wall_s"] == rec["cpu_timing"]["median_s"]
    assert rec["cpu_threads"] == 32 and rec["cpu_threads_label"].startswith("32 OpenMP threads")


def test_prism_reported_times_and_provenance(real_run):
    rec = real_run["rec"]
    p = rec["prism"]
    for r in p["rows"]:
        assert r["prism_construction_s"] is not None and r["prism_construction_s"] >= 0
        assert r["prism_checking_s"] is not None and r["prism_checking_s"] >= 0
        assert r["prism_solver_s"] == pytest.approx(r["prism_construction_s"] + r["prism_checking_s"])
        assert r["prism_states"] is not None and 1 <= r["prism_states"] <= 3
        assert r["prism_transitions"] is not None and r["prism_transitions"] > 0
        assert r["wall_s"] > r["prism_solver_s"] and r["peak_rss_kb"] is not None and r["peak_rss_kb"] > 0
        assert r["rc"] == 0 and os.path.isfile(r["log"]) and os.path.isfile(r["model_path"]) and os.path.isfile(r["props_path"])
        assert "-javamaxmem 16g" in r["cmd"] and "-maxiters 100000" in r["cmd"] and "-exportresults" in r["cmd"]
        assert os.path.isfile(r["results_path"])
        assert r["emit_s"] is not None and r["emit_s"] >= 0
    assert p["solver_total_s"] == pytest.approx(sum(r["prism_solver_s"] for r in p["rows"]))
    assert p["jvm_overhead_s"] == pytest.approx(p["total_wall_ok_s"] - p["solver_total_s"]) and p["jvm_overhead_s"] > 0
    assert p["min_wall_s"] <= p["median_wall_s"] <= p["max_wall_s"]
    ident = rec["tool_identity"]
    assert ident["prism_version"] and ident["prism_version"].startswith("PRISM-games")
    assert ident["java_version"] and "version" in ident["java_version"]
    assert ident["prism_launcher_sha256"] and len(ident["prism_launcher_sha256"]) == 64
    assert (ident["prism_jar_sha256"] and len(ident["prism_jar_sha256"]) == 64) or (ident["prism_classes_sha256"] and ident["prism_classes_n_files"] > 0)
    assert rec["prism_version"] == ident["prism_version"] and rec["javamaxmem"] == "16g" and rec["maxiters"] == 100000
    eb = rec["engine_binaries"]
    assert eb["libcubcr_gpu.so"]["sha256"] and eb["libcubcr_cpu.so"]["sha256"]


def test_gate_fails_on_perturbed_value(real_run, monkeypatch, tags, work):
    rr = real_run
    monkeypatch.setattr(e3.io_prism, "run_prism", _fake_run_prism_from_rows(rr["rows"], perturb={2: 1e-3}))
    rows, total, emit_total = e3.prism_all_coalitions(rr["csg"], K, IID, 60.0, 600.0, work, verbose=False)
    rec = e3.summarize(rr["csg"], K, IID, rows, total, rr["eng"], tags, "fixture", 60.0, 600.0, emit_total=emit_total)
    p = rec["prism"]
    assert p["complete"] and p["n_ok"] == 4 and p["n_mismatch"] == 1
    row = next(r for r in p["rows"] if r["mask"] == 2)
    assert row["mismatch"] is True and row["abs_err"] == pytest.approx(1e-3, rel=1e-3)
    assert rec["values_match"] is False and rec["gate_failed"].startswith("prism_value_mismatch")
    assert rec["prism_outcome"] == "GATE_FAILED"
    for f in e3.SPEEDUP_FIELDS:
        assert rec[f] is None, f
    assert "shapley_prism" not in rec

    paths = e3.write(rec, tags, os.path.join(work, "out_perturbed"), 600.0)
    recs = {protocol.read_result(pth)["engine"]: protocol.read_result(pth) for pth in paths}
    assert recs["prism_games"]["status"] == "OK" and recs["prism_games"]["coalition_solves_per_s"] is None
    assert recs["prism_games"]["values_match"] is False and recs["prism_games"]["gate_failed"].startswith("prism_value_mismatch")
    assert recs["gpu_batched"]["speedup_vs_prism"] is None and recs["gpu_batched"]["speedup_vs_prism_lower_bound"] is None


def test_gate_fails_on_gpu_cpu_mismatch(real_run, tags):
    rr = real_run
    eng = dict(rr["eng"])
    eng["gpu_cpu_gate_ok"] = False
    eng["max_abs_err_gpu_vs_cpu"] = 5e-9
    rec = e3.summarize(rr["csg"], K, IID, copy.deepcopy(rr["rows"]), rr["total"], eng, tags, "fixture", 60.0, 600.0)
    assert rec["values_match"] is False and rec["gate_failed"].startswith("gpu_cpu_mismatch")
    for f in e3.SPEEDUP_FIELDS:
        assert rec[f] is None, f


def test_status_TO_is_our_timeout(work):
    csg = vehicle_csg()
    pt, pr = io_prism.emit_coalition_model(csg, 0, K)
    r = io_prism.run_prism(pt, pr, timeout_s=0.2, workdir=work, tag="forced_to")
    assert r["status"] == "TO" and r["timed_out"] is True and r["value"] is None and r["returncode"] == -1
    assert 0.2 <= r["wall_s"] < 5.0 and "timeout" in r["error"]
    assert "-javamaxmem 16g" in r["cmd"] and "-maxiters 100000" in r["cmd"]


def test_status_ERR_bad_switch(work):
    csg = vehicle_csg()
    pt, pr = io_prism.emit_coalition_model(csg, 0, K)
    r = io_prism.run_prism(pt, pr, timeout_s=60, workdir=work, tag="forced_err", extra_args=["-nosuchflag"])
    assert r["status"] == "ERR" and r["timed_out"] is False and r["value"] is None and r["returncode"] != 0
    assert r["error"] and "nosuchflag" in r["error"]
    assert r["prism_construction_s"] is None and r["prism_checking_s"] is None


def test_classify_status_MO_vs_ERR():
    st, err = io_prism.classify_prism_status(False, 1, "Building model...\nException in thread \"main\" java.lang.OutOfMemoryError: Java heap space\n", "", None)
    assert st == "MO" and "OutOfMemoryError" in err
    st, err = io_prism.classify_prism_status(False, 0, "Result: 0.5\n", "Error: out of memory\n", 0.5)
    assert st == "MO"
    st, err = io_prism.classify_prism_status(False, 0, "Result: 0.5\n", "", 0.5)
    assert st == "OK" and err is None
    st, err = io_prism.classify_prism_status(False, 0, "\nError: Coalitions must not be empty.\n", "", None)
    assert st == "ERR" and err.startswith("Error: Coalitions")
    st, err = io_prism.classify_prism_status(True, -1, "", "", None)
    assert st == "TO"


def test_parse_prism_stdout():
    out = ("Time for model construction: 0.075 seconds.\n\nType:        CSG\nPlayers:     4\nStates:      43 (1 initial)\n"
           "Transitions: 484\nChoices:     313\n\nTime for model checking: 0.027 seconds.\n\nResult: 0.0\n")
    err = "\tElapsed (wall clock) time (h:mm:ss or m:ss): 0:01.43\n\tMaximum resident set size (kbytes): 168564\n"
    d = io_prism.parse_prism_stdout(out, err)
    assert d["prism_construction_s"] == 0.075 and d["prism_checking_s"] == 0.027 and d["prism_solver_s"] == pytest.approx(0.102)
    assert d["prism_states"] == 43 and d["prism_transitions"] == 484 and d["prism_choices"] == 313
    assert d["value"] == 0.0 and d["peak_rss_kb"] == 168564 and d["time_elapsed_s"] == pytest.approx(1.43)


def test_lower_bound_only_for_TO_or_NOT_RUN(real_run, monkeypatch, tags, work):
    rr = real_run

    monkeypatch.setattr(e3.io_prism, "run_prism", _fake_run_prism_from_rows(rr["rows"], status_override={3: ("TO", "killed by cuBCR timeout")}))
    rows, total, emit_total = e3.prism_all_coalitions(rr["csg"], K, IID, 60.0, 600.0, work, verbose=False)
    rec = e3.summarize(rr["csg"], K, IID, rows, total, rr["eng"], tags, "fixture", 60.0, 600.0, emit_total=emit_total)
    p = rec["prism"]
    assert p["n_ok"] == 3 and p["n_to"] == 1 and not p["complete"] and rec["values_match"] is True and rec["gate_failed"] is None
    assert rec["prism_outcome"] == "TO" and rec["speedup_gpu_vs_prism"] is None
    assert rec["speedup_gpu_vs_prism_lower_bound"] == pytest.approx(total / rec["gpu_wall_s"]) and total >= 60.0
    paths = e3.write(rec, tags, os.path.join(work, "out_to"), 600.0)
    pr = protocol.read_result([x for x in paths if "prism_games" in str(x)][0])
    assert pr["status"] == "TO" and pr["censored"] is True and pr["cap_s"] == 60.0 and pr["wall_s"] >= pr["cap_s"]
    assert pr["coalition_solves_per_s"] is None
    gr = protocol.read_result([x for x in paths if "gpu_batched" in str(x)][0])
    assert gr["speedup_vs_prism"] is None and gr["speedup_vs_prism_lower_bound"] is not None and gr["speedup_is_lower_bound"] is True

    monkeypatch.setattr(e3.io_prism, "run_prism", _fake_run_prism_from_rows(rr["rows"]))
    rows, total, _ = e3.prism_all_coalitions(rr["csg"], K, IID, 60.0, 1.0, work, verbose=False)
    rec = e3.summarize(rr["csg"], K, IID, rows, total, rr["eng"], tags, "fixture", 60.0, 1.0)
    assert rec["prism"]["n_not_run"] >= 1 and rec["speedup_gpu_vs_prism_lower_bound"] is not None and rec["speedup_gpu_vs_prism"] is None
    nr = next(r for r in rec["prism"]["rows"] if r["status"] == "NOT RUN")
    assert nr["reason"].startswith("aggregate cap") and nr["value"] is None
    pr = protocol.read_result(e3.write(rec, tags, os.path.join(work, "out_cap"), 1.0)[0])
    assert pr["status"] == "TO" and pr["censored"] is True and pr["cap_s"] == 1.0 and pr["wall_s"] >= 1.0


def test_MO_reports_MO_and_no_bound(real_run, monkeypatch, tags, work):
    rr = real_run
    monkeypatch.setattr(e3.io_prism, "run_prism", _fake_run_prism_from_rows(rr["rows"], status_override={1: ("MO", "java.lang.OutOfMemoryError: Java heap space")}))
    rows, total, _ = e3.prism_all_coalitions(rr["csg"], K, IID, 60.0, 600.0, work, verbose=False)
    rec = e3.summarize(rr["csg"], K, IID, rows, total, rr["eng"], tags, "fixture", 60.0, 600.0)
    assert rec["prism"]["n_mo"] == 1 and rec["prism"]["n_ok"] == 3 and rec["prism_outcome"] == "MO" and rec["gate_failed"] is None
    for f in e3.SPEEDUP_FIELDS:
        assert rec[f] is None, f
    pr = protocol.read_result(e3.write(rec, tags, os.path.join(work, "out_mo"), 600.0)[0])
    assert pr["status"] == "MO" and pr["censored"] is False and pr["wall_s"] is None and pr["coalition_solves_per_s"] is None


def test_ERR_fails_loud_and_stops(real_run, monkeypatch, tags, work, capsys):
    rr = real_run
    monkeypatch.setattr(e3.io_prism, "run_prism", _fake_run_prism_from_rows(rr["rows"], status_override={1: ("ERR", "Error: Unknown switch -nosuchflag")}))
    rows, total, _ = e3.prism_all_coalitions(rr["csg"], K, IID, 60.0, 600.0, work, verbose=False)
    assert [r["status"] for r in rows] == ["OK", "ERR", "NOT RUN", "NOT RUN"]
    assert rows[2]["reason"].startswith("aborted after ERR on mask 1")
    assert "ERROR: PRISM-games returned ERR" in capsys.readouterr().err
    rec = e3.summarize(rr["csg"], K, IID, rows, total, rr["eng"], tags, "fixture", 60.0, 600.0)
    assert rec["prism"]["n_err"] == 1 and rec["gate_failed"] and "prism_err" in rec["gate_failed"] and rec["prism_outcome"] == "ERR"
    for f in e3.SPEEDUP_FIELDS:
        assert rec[f] is None, f
    pr = protocol.read_result(e3.write(rec, tags, os.path.join(work, "out_err"), 600.0)[0])
    assert pr["status"] == "CRASH" and pr["censored"] is False and pr["wall_s"] is None


def test_records_validate_and_carry_provenance(real_run, tags, work):
    rec = real_run["rec"]
    paths = e3.write(rec, tags, os.path.join(work, "out_ok"), 600.0)
    assert len(paths) == 3
    for pth in paths:
        r = protocol.read_result(pth)
        protocol.validate_record(r)
        assert r["tool_identity"]["prism_version"].startswith("PRISM-games") and r["java_version"]
        assert r["javamaxmem"] == "16g" and r["maxiters"] == 100000
        assert r["engine_binaries"]["libcubcr_gpu.so"]["sha256"] and r["engine_binaries"]["libcubcr_cpu.so"]["sha256"]
        assert r["git"] == tags["git"]
        assert r["values_match"] is True and r["gate_failed"] is None
    recs = {protocol.read_result(p)["engine"]: protocol.read_result(p) for p in paths}
    assert recs["prism_games"]["status"] == "OK" and recs["prism_games"]["wall_s"] == pytest.approx(rec["prism"]["total_wall_s"])
    assert recs["prism_games"]["emit_s"] == pytest.approx(rec["prism"]["emit_total_s"]) and recs["prism_games"]["prism_solver_total_s"] > 0
    assert recs["gpu_batched"]["n_repeat"] == 5 and recs["gpu_batched"]["warmup"] == 1 and len(recs["gpu_batched"]["all_s"]) == 5
    assert recs["cpu_multicore"]["threads"] == 32 and recs["cpu_multicore"]["threads_label"].startswith("32 OpenMP threads")
    assert recs["gpu_batched"]["speedup_vs_prism"] == pytest.approx(rec["speedup_gpu_vs_prism"])
