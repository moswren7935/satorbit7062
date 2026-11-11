import contextlib
import time

import numpy as np
import pytest

import rb
import e2b_common as C


class FakeEngine:
    def __init__(self, comp):
        self.closed = False
        self.stats = {"graph_captures": 1}

    def solve(self, masks, k=None):
        m = np.asarray(masks)
        return np.arange(m.shape[-1], dtype=np.float64) if m.ndim == 2 else np.arange(m.size, dtype=np.float64)

    def close(self):
        self.closed = True


@pytest.fixture
def fake(monkeypatch):
    calls = {"acquire": 0, "engines": []}

    @contextlib.contextmanager
    def fake_lock(tag=""):
        calls["acquire"] += 1
        yield 1.5

    def make(comp):
        e = FakeEngine(comp)
        calls["engines"].append(e)
        return e

    monkeypatch.setattr(rb, "gpu_lock", fake_lock)
    monkeypatch.setattr(rb, "make_gpu_engine", make)
    monkeypatch.setattr(rb, "load_snapshot", lambda top=6: {"loadavg": [0.0, 0.0, 0.0], "gpu_util_mem": "0, 0"})
    monkeypatch.setattr(C, "gpu_mem", lambda: (1, 2))
    return calls


def test_per_run_mode_acquires_each_time(fake):
    for _ in range(3):
        out, info = C.locked_timed_run("t", None, lambda e: None, lambda e: 42)
        assert out == 42 and info["lock_mode"] == "per_run" and info["lock_wait_s"] == 1.5
    assert fake["acquire"] == 3
    assert all(e.closed for e in fake["engines"]) and len(fake["engines"]) == 3


def test_chunk_holds_once_and_attributes_wait_once(fake):
    with C.ChunkedLock("t", max_hold_s=3600) as lock:
        held = lock.ensure()
        ups, vv, li = C.locked_upsilon("u", None, np.array([0, 3], dtype=np.uint64), 4, held=held)
        assert ups == 1.0 and li["lock_wait_s"] == 1.5 and li["lock_mode"] == "group"
        waits = []
        for _ in range(3):
            held = lock.ensure()
            out, info = C.locked_timed_run("t", None, lambda e: None, lambda e: 7, held=held)
            waits.append(info["lock_wait_s"])
            assert info["lock_mode"] == "group"
            lock.boundary()
    assert fake["acquire"] == 1
    assert waits == [0.0, 0.0, 0.0]
    assert len(lock.log) == 1 and lock.log[0]["hold_s"] is not None
    assert all(e.closed for e in fake["engines"]) and len(fake["engines"]) == 4


def test_chunk_releases_at_boundary_after_budget(fake):
    lock = C.ChunkedLock("t", max_hold_s=0.0)
    for _ in range(3):
        held = lock.ensure()
        C.locked_timed_run("t", None, lambda e: None, lambda e: 1, held=held)
        time.sleep(0.001)
        lock.boundary()
    lock.release(); lock.release()
    assert fake["acquire"] == 3 and lock.n_holds == 3 and len(lock.log) == 3


def test_engine_closed_on_error(fake):
    def boom(e):
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        C.locked_timed_run("t", None, lambda e: None, boom)
    assert fake["engines"][-1].closed
