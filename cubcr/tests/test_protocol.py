from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import protocol, summarize


def test_timed_run_discards_warmup_and_reports_median():
    calls = []
    sleeps = iter([0.05, 0.004, 0.001, 0.002, 0.003, 0.010])

    def fn():
        d = next(sleeps)
        calls.append(d)
        time.sleep(d)
        return len(calls)

    r = protocol.timed_run(fn, n_repeat=5, warmup=1)
    assert len(calls) == 6
    assert r["n_repeat"] == 5 and r["warmup"] == 1
    assert len(r["all_s"]) == 5 and len(r["warmup_s"]) == 1
    assert r["warmup_s"][0] >= 0.05
    assert r["warmup_s"][0] not in r["all_s"]
    assert r["median_s"] == pytest.approx(float(np.median(r["all_s"])))
    assert r["min_s"] == min(r["all_s"]) and r["max_s"] == max(r["all_s"])
    assert r["max_s"] < 0.05
    assert r["result"] == 6


def test_timed_run_exact_median_with_fake_timer():
    ticks = iter([0, 100, 100, 101, 101, 105, 105, 108, 108, 110, 110, 117])
    r = protocol.timed_run(lambda: None, n_repeat=5, warmup=1, timer=lambda: next(ticks))
    assert r["warmup_s"] == [100.0]
    assert r["all_s"] == [1.0, 4.0, 3.0, 2.0, 7.0]
    assert r["median_s"] == 3.0 and r["min_s"] == 1.0 and r["max_s"] == 7.0


def test_timed_run_rejects_bad_args():
    with pytest.raises(ValueError):
        protocol.timed_run(lambda: None, n_repeat=0)
    with pytest.raises(ValueError):
        protocol.timed_run(lambda: None, warmup=-1)


def test_run_with_timeout_thread_ok_and_to():
    r = protocol.run_with_timeout(lambda x: x * 2, 5.0, 21)
    assert r["status"] == "OK" and r["result"] == 42 and r["censored"] is False
    r = protocol.run_with_timeout(lambda: time.sleep(0.5), 0.05)
    assert r["status"] == "TO" and r["censored"] is True and r["elapsed_s"] < 0.5
    r = protocol.run_with_timeout(lambda: 1 / 0, 5.0)
    assert r["status"] == "ERR" and "ZeroDivisionError" in r["error"]


def _sleep_then(x):
    time.sleep(x)
    return "done"


def test_run_with_timeout_process_ok_and_to():
    r = protocol.run_with_timeout(_sleep_then, 5.0, 0.01, mode="process")
    assert r["status"] == "OK" and r["result"] == "done"
    r = protocol.run_with_timeout(_sleep_then, 0.2, 5.0, mode="process")
    assert r["status"] == "TO" and r["censored"] and r["elapsed_s"] < 2.0


def _record(**over):
    rec = protocol.make_record(
        instance_id="rnd_n4_s12_k4", engine="gpu_batched", dtype="fp64", n=4, states=12, k=4,
        seed=20260829, wall_s=0.5, coalition_solves_per_s=32.0, speedup_vs_cpu=2.0,
        max_abs_err_vs_fp64=0.0, kendall_tau=1.0, cap_s=600.0, threads=0, n_coalitions=16)
    rec.update(over)
    return rec


FAKE_TAGS = {"gpu": "FakeGPU", "cuda": "12.2", "git": "deadbeef", "hostname": "test"}


def test_write_result_fills_schema_and_env(tmp_path):
    p = protocol.write_result(tmp_path, _record(), tags=FAKE_TAGS)
    assert p.exists() and p.suffix == ".json"
    rec = json.loads(p.read_text())
    for f in protocol.REQUIRED_FIELDS:
        assert f in rec, f
    assert rec["gpu"] == "FakeGPU" and rec["cuda"] == "12.2" and rec["git"] == "deadbeef"
    assert rec["machine"] == FAKE_TAGS
    assert set(rec["env"]) >= {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"}
    assert rec["timestamp"] and rec["schema_version"] == protocol.SCHEMA_VERSION
    assert rec["status"] == "OK" and rec["censored"] is False

    p2 = protocol.write_result(tmp_path, _record(), tags=FAKE_TAGS)
    assert p2 != p and p2.exists()


def test_write_result_env_capture_reflects_omp(tmp_path, monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "32")
    monkeypatch.setenv("OMP_PROC_BIND", "close")
    monkeypatch.setenv("OMP_PLACES", "cores")
    rec = json.loads(protocol.write_result(tmp_path, _record(), tags=FAKE_TAGS).read_text())
    assert rec["env"]["OMP_NUM_THREADS"] == "32"
    assert rec["env"]["OMP_PROC_BIND"] == "close"
    assert rec["env"]["OMP_PLACES"] == "cores"


@pytest.mark.parametrize("missing", ["instance_id", "cap_s", "threads", "censored", "seed", "k"])
def test_write_result_missing_required_field_raises(tmp_path, missing):
    rec = _record()
    del rec[missing]
    with pytest.raises(ValueError, match=missing):
        protocol.write_result(tmp_path, rec, tags=FAKE_TAGS)
    assert not list(tmp_path.glob("*.json"))


def test_write_result_rejects_bad_values(tmp_path):
    with pytest.raises(ValueError, match="engine"):
        protocol.write_result(tmp_path, _record(engine="fast_thing"), tags=FAKE_TAGS)
    with pytest.raises(ValueError, match="cap_s"):
        protocol.write_result(tmp_path, _record(cap_s=0), tags=FAKE_TAGS)
    with pytest.raises(ValueError, match="censored"):
        protocol.write_result(tmp_path, _record(censored="yes"), tags=FAKE_TAGS)

    with pytest.raises(ValueError):
        protocol.write_result(tmp_path, _record(censored=True, status="TO", wall_s=10.0, cap_s=600.0,
                                                coalition_solves_per_s=None, speedup_vs_cpu=None),
                              tags=FAKE_TAGS)
    with pytest.raises(ValueError, match="throughput"):
        protocol.write_result(tmp_path, _record(censored=True, status="TO", wall_s=600.0, cap_s=600.0,
                                                speedup_vs_cpu=None), tags=FAKE_TAGS)
    ok = protocol.write_result(tmp_path, _record(censored=True, status="TO", wall_s=600.0, cap_s=600.0,
                                                 coalition_solves_per_s=None, speedup_vs_cpu=None),
                               tags=FAKE_TAGS)
    rec = json.loads(ok.read_text())
    assert protocol.format_wall(rec) == ">= 600 s"
    assert protocol.format_wall(_record()) == "0.5 s"
    assert protocol.format_speedup(_record(speedup_is_lower_bound=True)) == ">= 2.00x"


def test_machine_tags_keys():
    tags = protocol.machine_tags()
    for key in ("hostname", "gpu", "driver", "cuda", "nvcc", "cpu_model", "physical_cores",
                "logical_cpus", "python", "numpy", "git", "os"):
        assert key in tags, key
    assert tags["hostname"]
    assert tags["python"] == ".".join(map(str, sys.version_info[:3]))
    assert tags["numpy"] == np.__version__
    assert tags["logical_cpus"] == os.cpu_count()
    if tags["physical_cores"] is not None:
        assert 0 < tags["physical_cores"] <= tags["logical_cpus"]
    if tags["git"] is not None:
        assert len(tags["git"]) == 40
    env = protocol.capture_env()
    assert set(env) >= {"OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES"}


def test_summarize_empty_dir(tmp_path):
    md = summarize.summarize(tmp_path / "does_not_exist", tmp_path / "summary.md")
    assert "No result records" in md
    assert (tmp_path / "summary.md").read_text() == md
    df = summarize.load_results(tmp_path)
    assert df.empty and "censored" in df.columns


def test_summarize_censored_excluded_from_median(tmp_path):
    res = tmp_path / "results"

    for wall, thr in ((2.0, 8.0), (4.0, 4.0)):
        protocol.write_result(res, _record(engine="cpu_multicore", threads=32, wall_s=wall,
                                           coalition_solves_per_s=thr, speedup_vs_cpu=1.0), tags=FAKE_TAGS)
    protocol.write_result(res, _record(engine="cpu_multicore", threads=32, censored=True, status="TO",
                                       wall_s=600.0, cap_s=600.0, coalition_solves_per_s=None,
                                       speedup_vs_cpu=None), tags=FAKE_TAGS)

    protocol.write_result(res, _record(engine="gpu_batched", censored=True, status="TO", wall_s=600.0,
                                       cap_s=600.0, coalition_solves_per_s=None, speedup_vs_cpu=None),
                          tags=FAKE_TAGS)

    protocol.write_result(res, _record(engine="gpu_batched", dtype="fp32", speedup_vs_cpu=50.0,
                                       speedup_is_lower_bound=True, max_abs_err_vs_fp64=3e-7,
                                       kendall_tau=0.98), tags=FAKE_TAGS)

    (res / "notes.json").write_text(json.dumps({"hello": "world"}))

    df = summarize.load_results(res)
    assert len(df) == 5
    agg = summarize.aggregate(df)
    cpu = agg[(agg.engine == "cpu_multicore")].iloc[0]
    assert cpu["runs"] == 3 and cpu["censored"] == 1
    assert cpu["throughput"] == pytest.approx(6.0)
    gpu64 = agg[(agg.engine == "gpu_batched") & (agg.dtype == "fp64")].iloc[0]
    assert gpu64["runs"] == 1 and gpu64["censored"] == 1 and pd.isna(gpu64["throughput"])
    gpu32 = agg[(agg.engine == "gpu_batched") & (agg.dtype == "fp32")].iloc[0]
    assert gpu32["speedup"] == 50.0 and bool(gpu32["speedup_lower_bound"]) is True
    assert gpu32["max_err"] == pytest.approx(3e-7) and gpu32["kendall_tau"] == pytest.approx(0.98)

    md = summarize.summarize(res, tmp_path / "summary.md")
    lines = [ln for ln in md.splitlines() if ln.startswith("| ")]
    assert lines[0].startswith("| engine")
    body = "\n".join(lines[1:])
    assert "1/3" in body
    assert "TO (>= 600 s)" in body
    assert ">= 50.00x" in body
    assert "| 6 " in body or "| 6" in body
    assert "600" not in [c.strip() for c in lines[1].split("|")]
    assert (tmp_path / "summary.md").exists()
