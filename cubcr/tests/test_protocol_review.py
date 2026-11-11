from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments import protocol, summarize

FAKE_TAGS = {"gpu": "FakeGPU", "cuda": "12.2", "git": "deadbeef"}


def _rec(**over):
    r = protocol.make_record(
        instance_id="rev", engine="cpu_multicore", dtype="fp64", n=4, states=30, k=5, seed=20260829,
        wall_s=1.0, coalition_solves_per_s=16.0, speedup_vs_cpu=1.0, max_abs_err_vs_fp64=0.0,
        kendall_tau=1.0, cap_s=600.0, threads=32)
    r.update(over)
    return r


def _big_result(n):
    return np.zeros(n, dtype=np.float64)


def _crash_hard():
    os._exit(3)


def _raise():
    raise RuntimeError("boom")


@pytest.mark.parametrize("n", [1000, 100_000, 1_000_000])
def test_process_mode_large_result_is_not_a_false_timeout(n):


    r = protocol.run_with_timeout(_big_result, 5.0, n, mode="process")
    assert r["status"] == "OK" and r["censored"] is False, r["error"]
    assert r["result"].shape == (n,) and r["elapsed_s"] < 5.0


def test_process_mode_true_timeout_err_and_crash():
    r = protocol.run_with_timeout(time.sleep, 0.3, 5.0, mode="process")
    assert r["status"] == "TO" and r["censored"] is True and 0.3 <= r["elapsed_s"] < 2.0
    r = protocol.run_with_timeout(_raise, 5.0, mode="process")
    assert r["status"] == "ERR" and "RuntimeError" in r["error"] and r["censored"] is False
    r = protocol.run_with_timeout(_crash_hard, 5.0, mode="process")
    assert r["status"] == "ERR" and "no result" in r["error"] and r["elapsed_s"] < 5.0
    with pytest.raises(ValueError):
        protocol.run_with_timeout(_raise, 5.0, mode="process", ctx="bogus")


def _solve_in_child(n_agents, n_states, k, seed, masks):

    from tests.helpers import random_csg
    from cubcr.engine_cpu import CPUEngine
    csg = random_csg(n_agents=n_agents, n_states=n_states, k=k, seed=seed)
    return CPUEngine(csg.compile(), n_threads=2).solve(np.asarray(masks, dtype=np.uint64))


def test_timed_run_and_thread_cap_match_vi_ref_and_spawn_after_openmp():
    from tests.helpers import random_csg
    from cubcr import vi_ref
    from cubcr.engine_cpu import CPUEngine

    n, S, k, seed = 4, 30, 5, 20260829
    csg = random_csg(n_agents=n, n_states=S, k=k, seed=seed)
    comp = csg.compile()
    eng = CPUEngine(comp, n_threads=4)
    masks = np.arange(1 << n, dtype=np.uint64)
    ref = np.array([vi_ref.solve(csg, int(m)) for m in masks])
    tr = protocol.timed_run(lambda: eng.solve(masks), n_repeat=3, warmup=1)
    assert len(tr["all_s"]) == 3 and len(tr["warmup_s"]) == 1
    assert np.max(np.abs(tr["result"] - ref)) <= 1e-12

    rng = np.random.default_rng(seed)
    step = rng.integers(0, 1 << n, size=(k, 8), dtype=np.uint64)
    ref_step = np.array([vi_ref.solve(csg, [int(step[t, c]) for t in range(k)]) for c in range(8)])
    r = protocol.run_with_timeout(eng.solve, 30.0, step, k=k, mode="thread")
    assert r["status"] == "OK" and np.max(np.abs(r["result"] - ref_step)) <= 1e-12


    r = protocol.run_with_timeout(_solve_in_child, 120.0, n, S, k, seed, masks.tolist(),
                                  mode="process", ctx="spawn")
    assert r["status"] == "OK", r["error"]
    assert np.max(np.abs(np.asarray(r["result"]) - ref)) <= 1e-12


def test_validate_record_rejects_inconsistent_timeout_and_bad_wall():
    with pytest.raises(ValueError, match="censored=True"):
        protocol.validate_record({**_rec(status="TO"), "gpu": "g", "cuda": "c", "git": "h", "timestamp": "t"})
    with pytest.raises(ValueError, match="wall_s"):
        protocol.validate_record({**_rec(wall_s=-1.0), "gpu": "g", "cuda": "c", "git": "h", "timestamp": "t"})
    with pytest.raises(ValueError, match="wall_s"):
        protocol.validate_record({**_rec(wall_s=float("nan")), "gpu": "g", "cuda": "c", "git": "h", "timestamp": "t"})
    with pytest.raises(ValueError, match="wall_s"):
        protocol.validate_record({**_rec(censored=True, status="TO", wall_s=None, coalition_solves_per_s=None,
                                         speedup_vs_cpu=None), "gpu": "g", "cuda": "c", "git": "h", "timestamp": "t"})

    protocol.validate_record({**_rec(censored=True, status="TO", wall_s=600.0, coalition_solves_per_s=None,
                                     speedup_vs_cpu=None), "gpu": "g", "cuda": "c", "git": "h", "timestamp": "t"})


def test_summarize_does_not_pool_thread_counts_and_escapes_pipes(tmp_path):
    res = tmp_path / "results"
    protocol.write_result(res, _rec(threads=1, coalition_solves_per_s=1.0), tags=FAKE_TAGS)
    protocol.write_result(res, _rec(threads=32, coalition_solves_per_s=30.0), tags=FAKE_TAGS)
    agg = summarize.aggregate(summarize.load_results(res))
    assert len(agg) == 2, "1-thread and 32-thread CPU runs must be separate cells"
    by_thr = {int(r.threads): r.throughput for r in agg.itertuples()}
    assert by_thr == {1: 1.0, 32: 30.0}
    md = summarize.to_markdown(agg)
    lines = [ln for ln in md.splitlines() if ln.startswith("|")]
    header, sep, body = lines[0], lines[1], lines[2:]

    n_cols = sep.count("|") - 1
    assert header.replace("\\|", "").count("|") - 1 == n_cols == len(summarize.TABLE_COLUMNS)
    assert "\\|S\\|" in header and "max \\|err\\| vs FP64" in header
    for row in body:
        assert row.count("|") - 1 == n_cols


def test_manifest_matches_files_on_disk():
    mpath = ROOT / "datasets" / "prism_csg" / "manifest.json"
    if not mpath.exists():
        pytest.skip("fetch.py has not been run")
    m = json.loads(mpath.read_text())
    files = [e for e in m["files"] if "error" not in e]
    assert files, "manifest has no fetched files"
    for e in files:
        p = ROOT / e["file"]
        assert p.exists(), e["file"]
        assert p.stat().st_size == e["size"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == e["sha256"], e["file"]
        assert p.suffix in (".prism", ".props", ".pm", ".nm")
        head = p.read_bytes()[:400].lower()
        assert b"<html" not in head and b"<pre" not in head, f"{e['file']} looks like HTML"
        if p.suffix == ".prism" and "prism_status" in e:
            assert e["prism_status"] in ("PARSE_OK", "NEEDS_CONSTS", "ERROR")

    direct = [e for e in files if e["origin"] == "direct" and e["file"].endswith(".prism")]
    assert len(direct) == 17
    assert all(e["prism_status"] in ("PARSE_OK", "NEEDS_CONSTS") for e in direct if "prism_status" in e)


def test_prism_status_classification_on_synthetic_models(tmp_path):
    from datasets import fetch
    if not fetch.prism_available():
        pytest.skip("PRISM-games binary not built")
    body = ("csg\n{const}\nplayer p1 m1 endplayer\nplayer p2 m2 endplayer\n"
            "module m1\n x : [0..2] init 0;\n [a1] x<K -> (x'=x+1);\n [b1] x<K -> (x'=x);\nendmodule\n"
            "module m2 = m1 [x=y, a1=a2, b1=b2] endmodule\nlabel \"fail\" = x=2;\n")
    undef = tmp_path / "undef.prism"; undef.write_text(body.format(const="const int K;"))
    defd = tmp_path / "def.prism"; defd.write_text(body.format(const="const int K = 2;"))
    bad = tmp_path / "bad.prism"; bad.write_text(body.format(const="const int K;").replace("(x'=x);", "(x'=x)"))
    r = fetch.verify_prism_file(undef, 60, try_build=False)
    assert r["prism_status"] == "NEEDS_CONSTS" and r["undefined_consts"] == ["K"]
    r = fetch.verify_prism_file(defd, 60, try_build=True)
    assert r["prism_status"] == "PARSE_OK" and r["build_status"] == "BUILD_OK" and "states=" in r["build_msg"]
    r = fetch.verify_prism_file(bad, 60, try_build=False)
    assert r["prism_status"] == "ERROR" and "Syntax error" in r["prism_msg"]
