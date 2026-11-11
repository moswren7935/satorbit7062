import hashlib
import importlib
import shutil
import subprocess

import pytest

import rb
import e2b_common as C
from experiments import protocol


def _git_repo(path):
    path.mkdir()
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    subprocess.run(["git", "-C", str(path), "-c", "user.name=Anonymous",
                    "-c", "user.email=anonymous", "commit", "-q", "--allow-empty", "-m", "test"], check=True)
    return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()


def _use_repo(monkeypatch, path):
    monkeypatch.setattr(rb, "REPO", path)
    monkeypatch.setattr(protocol, "REPO_ROOT", path)


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_git_metadata_does_not_inherit_parent_repository(tmp_path, monkeypatch):
    parent = tmp_path / "parent"
    _git_repo(parent)
    project = parent / "project"
    project.mkdir()
    _use_repo(monkeypatch, project)
    assert rb.git_commit() == "unknown"
    assert protocol.git_commit() is None
    assert protocol.git_dirty() is None


@pytest.mark.skipif(shutil.which("git") is None, reason="Git is unavailable")
def test_git_metadata_records_only_the_project_repository(tmp_path, monkeypatch):
    project = tmp_path / "project"
    revision = _git_repo(project)
    _use_repo(monkeypatch, project)
    assert rb.git_commit() == revision
    assert protocol.git_commit() == revision
    assert protocol.git_commit(short=True) == revision[:8]
    assert protocol.git_dirty() is False
    (project / "local.txt").write_text("changed")
    assert protocol.git_dirty() is True


def test_local_engine_builds_are_accepted_and_fingerprinted(tmp_path, monkeypatch):
    cpu = tmp_path / "cpu.so"
    gpu = tmp_path / "gpu.so"
    cpu.write_bytes(b"local cpu build")
    gpu.write_bytes(b"local gpu build")
    monkeypatch.setattr(rb, "LIB_CPU", cpu)
    monkeypatch.setattr(rb, "LIB_GPU", gpu)
    result = rb.check_engine_provenance(strict=True)
    assert result["lib_sha256"] == {
        "libcubcr_cpu.so": hashlib.sha256(b"local cpu build").hexdigest(),
        "libcubcr_gpu.so": hashlib.sha256(b"local gpu build").hexdigest(),
    }
    assert result["local_build_complete"] is True


def test_missing_engine_builds_are_reported(tmp_path, monkeypatch):
    cpu = tmp_path / "cpu.so"
    cpu.write_bytes(b"local cpu build")
    monkeypatch.setattr(rb, "LIB_CPU", cpu)
    monkeypatch.setattr(rb, "LIB_GPU", tmp_path / "missing.so")
    result = rb.check_engine_provenance()
    assert result["lib_sha256"]["libcubcr_gpu.so"] is None
    assert result["local_build_complete"] is False
    with pytest.raises(RuntimeError, match="missing"):
        rb.check_engine_provenance(strict=True)


@pytest.mark.parametrize("name", ["e2b_grid", "e2b_dt", "e2b_dt_big", "e2b_overcooked"])
def test_e2b_drivers_require_local_builds_without_historical_hashes(name, tmp_path, monkeypatch):
    module = importlib.import_module("rebuttal.scripts." + name)
    for key in ("RAW_GRID", "RAW_DT", "RAW_OC"):
        monkeypatch.setattr(C, key, tmp_path / key)
    cpu, gpu = tmp_path / "cpu.so", tmp_path / "gpu.so"
    monkeypatch.setattr(rb, "LIB_CPU", cpu)
    monkeypatch.setattr(rb, "LIB_GPU", gpu)
    monkeypatch.setattr(C, "hw_tags", rb.check_engine_provenance)
    with pytest.raises(SystemExit, match="missing"):
        module.main(["--seeds"])
    cpu.write_bytes(b"local cpu build")
    gpu.write_bytes(b"local gpu build")

    class InputsRequired(Exception):
        pass

    def need_inputs(*args, **kwargs):
        raise InputsRequired()

    if name == "e2b_grid":
        monkeypatch.setattr(rb, "paper_tasks", need_inputs)
    elif name == "e2b_dt_big":
        monkeypatch.setattr(C, "dt_build", need_inputs)
    if name in ("e2b_grid", "e2b_dt_big"):
        with pytest.raises(InputsRequired):
            module.main(["--seeds"])
    else:
        args = ["--seeds", "--models"] if name == "e2b_dt" else ["--seeds"]
        assert module.main(args) == 0
