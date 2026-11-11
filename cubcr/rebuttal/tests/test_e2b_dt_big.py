import contextlib
import importlib.util
import json
import os

import pytest

import rb
import e2b_common as C

HERE = os.path.dirname(os.path.abspath(__file__))


def _load():
    spec = importlib.util.spec_from_file_location("e2b_dt_big", os.path.join(HERE, "..", "scripts", "e2b_dt_big.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class FakeDT:
    SCOPE_CUT_S = 900.0
    BIG = (12, 100000, 25)

    def __init__(self, walls):
        self.walls = list(walls)
        self.calls = []

    def run_one(self, comp, meta, n, k, eps, seed, hw, exact_phi=None, held=None):
        assert held is not None and held["group_tag"].startswith("E2B dt big")
        self.calls.append(seed)
        rec = {"time_to_certify_s": 1.0, "wall_s": self.walls.pop(0)}
        rb.write_json(C.dt_path(meta["instance_id"], eps, seed), rec)
        return rec


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "RAW_DT", tmp_path)
    acq = {"n": 0, "on_acquire": None}

    @contextlib.contextmanager
    def fake_group_lock(tag):
        acq["n"] += 1
        if acq["on_acquire"]:
            acq["on_acquire"]()
        g = {"group_tag": tag, "group_lock_wait_s": 2.0, "t_acquired": 0.0, "first": True}
        yield g
        g["group_hold_s"] = 1.0

    monkeypatch.setattr(C, "group_lock", fake_group_lock)
    return tmp_path, acq


META = {"instance_id": "tb_none_n12_S100009_k25_pf0.05_C3_s20260829"}
SEEDS = [20260829, 20260830, 20260831]


def test_single_acquisition_and_skip_seeds_on_disk(env):
    tmp, acq = env
    big = _load()

    acq["on_acquire"] = lambda: rb.write_json(C.dt_path(META["instance_id"], 0.001, SEEDS[0]), {"x": 1})
    dt = FakeDT([300.0, 310.0])
    out = big.run_remaining(None, META, 12, 25, 0.001, SEEDS, {}, dt)
    assert acq["n"] == 1 and out["n_acquisitions"] == 1
    assert dt.calls == SEEDS[1:] and out["ran"] == SEEDS[1:] and out["skipped_on_disk"] == SEEDS[:1]
    assert not out["scope_cut"]
    assert json.load(open(C.dt_path(META["instance_id"], 0.001, SEEDS[0]))) == {"x": 1}


def test_scope_cut_after_long_run(env):
    tmp, acq = env
    big = _load()
    dt = FakeDT([950.0, 1.0, 1.0])
    out = big.run_remaining(None, META, 12, 25, 0.001, SEEDS, {}, dt)
    assert dt.calls == SEEDS[:1] and out["scope_cut"]
    cut = json.load(open(tmp / f"partB_dt_scopecut__{META['instance_id']}__eps0.001.json"))
    assert cut["skipped_seeds"] == SEEDS[1:] and cut["ran_seed"] == SEEDS[0]

    dt2 = FakeDT([1.0])
    out2 = big.run_remaining(None, META, 12, 25, 0.001, SEEDS, {}, dt2)
    assert dt2.calls == [] and out2["ran"] == []


def test_wrapper_uses_e2b_dt_run_one_and_constants():
    big = _load()
    dt = big.load_e2b_dt()
    assert dt.BIG == (12, 100000, 25) and dt.SCOPE_CUT_S == 900.0
    assert callable(dt.run_one)
