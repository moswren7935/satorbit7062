import contextlib
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "scripts"))
import e7_gpu_runs as G


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def fake_env(monkeypatch):
    clock = FakeClock()
    ran, written, holds = [], [], []

    @contextlib.contextmanager
    def fake_lock(tag=""):
        holds.append(tag)
        yield 0.0

    def fake_body(t, comp, kw):
        ran.append((t["iid"], kw["seed"]))
        clock.t += t["cost"]
        return object(), t["cost"], {"lock_hold_s": t["cost"]}

    monkeypatch.setattr(G.rb, "gpu_lock", fake_lock)
    monkeypatch.setattr(G, "_gpu_body", fake_body)
    monkeypatch.setattr(G, "_kw", lambda t, rule, seed: ({"seed": seed}, 1))
    monkeypatch.setattr(G, "_write", lambda t, rule, seed, *a, **k: written.append((t["iid"], seed)))
    monkeypatch.setattr(G.time, "perf_counter", clock)
    return clock, ran, written, holds


def _jobs(costs):
    return [({"iid": f"task{i}", "cost": c}, "topk1", 20260829) for i, c in enumerate(costs)]


def test_no_budget_runs_everything_in_one_hold(fake_env):
    _, ran, written, holds = fake_env
    jobs = _jobs([100.0] * 5)
    rest = G.run_group(jobs, {}, {j[0]["iid"]: None for j in jobs})
    assert rest == [] and len(ran) == 5 and len(written) == 5 and len(holds) == 1


def test_budget_stops_starting_runs_and_returns_the_rest(fake_env):
    _, ran, written, holds = fake_env
    jobs = _jobs([100.0] * 5)
    rest = G.run_group(jobs, {}, {j[0]["iid"]: None for j in jobs}, hold_budget_s=250.0)

    assert [r[0] for r in ran] == ["task0", "task1", "task2"]
    assert [r[0] for r in written] == ["task0", "task1", "task2"]
    assert [j[0]["iid"] for j in rest] == ["task3", "task4"]
    assert len(holds) == 1


def test_first_run_always_starts_even_if_longer_than_budget(fake_env):
    _, ran, _, _ = fake_env
    jobs = _jobs([500.0, 10.0])
    rest = G.run_group(jobs, {}, {j[0]["iid"]: None for j in jobs}, hold_budget_s=60.0)
    assert [r[0] for r in ran] == ["task0"] and [j[0]["iid"] for j in rest] == ["task1"]


def test_requeue_covers_every_job_exactly_once(fake_env):
    _, ran, written, holds = fake_env
    queue = [_jobs([70.0, 70.0, 70.0, 70.0, 70.0, 70.0, 70.0])]
    while queue:
        unit = queue.pop(0)
        rest = G.run_group(unit, {}, {j[0]["iid"]: None for j in unit}, hold_budget_s=100.0)
        if rest:
            queue.insert(0, rest)
    assert sorted(r[0] for r in ran) == sorted(f"task{i}" for i in range(7))
    assert len(ran) == len(set(ran)) == len(written) == 7
    assert len(holds) == 4
