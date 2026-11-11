import fcntl
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
import e5_lock as L


def _hold(path, pid):
    f = open(path, "a+")
    fcntl.flock(f, fcntl.LOCK_EX)
    f.seek(0); f.truncate(); f.write(f"{pid} test-hold 2026-11-07T00:00:00\n"); f.flush()
    return f


def test_verified_parent_hold_passes(tmp_path):
    p = tmp_path / "gpu.lock"
    f = _hold(p, 4242)
    try:
        L.verify_parent_holds(p, "4242", 4242)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close()


def test_free_lock_is_refused(tmp_path):
    p = tmp_path / "gpu.lock"
    p.write_text("4242 stale 2026-11-07T00:00:00\n")
    with pytest.raises(RuntimeError, match="FREE"):
        L.verify_parent_holds(p, "4242", 4242)


def test_wrong_parent_or_holder_is_refused(tmp_path):
    p = tmp_path / "gpu.lock"
    f = _hold(p, 4242)
    try:
        with pytest.raises(RuntimeError, match="parent pid"):
            L.verify_parent_holds(p, "4242", 1)
        with pytest.raises(RuntimeError, match="holder"):
            L.verify_parent_holds(p, "999", 999)
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close()


def test_gpu_hold_without_env_is_rb_gpu_lock(monkeypatch):
    monkeypatch.delenv(L.ENV_HOLDER, raising=False)
    cm = L.gpu_hold("t")
    assert cm.__class__.__name__ == "_GeneratorContextManager"
    assert L.lock_mode().startswith("own hold")


def test_gpu_hold_with_verified_parent_yields_parent_wait(tmp_path, monkeypatch):
    p = tmp_path / "gpu.lock"
    f = _hold(p, os.getppid())
    try:
        monkeypatch.setenv(L.ENV_HOLDER, str(os.getppid()))
        monkeypatch.setenv(L.ENV_WAIT, "12.5")
        with L.gpu_hold("t", lock_path=p) as waited:
            assert waited == 12.5
        assert L.lock_mode().startswith("parent hold")
    finally:
        fcntl.flock(f, fcntl.LOCK_UN); f.close()
