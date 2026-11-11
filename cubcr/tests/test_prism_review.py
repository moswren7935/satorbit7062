import os
import subprocess
import sys
import time

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from helpers import random_csg

from cubcr import vi_ref, io_prism
from cubcr.model import CSG, StateDef

pytestmark = pytest.mark.skipif(not io_prism.prism_available(),
                                reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")

from generators.vehicle import vehicle_csg
ROBOT = os.path.join(io_prism.REPO_ROOT, "external", "prism-games", "prism-examples", "csgs",
                     "robot_coordination", "robot_coordination2.prism")


@pytest.fixture(scope="module")
def WORK(tmp_path_factory):
    return str(tmp_path_factory.mktemp("prism_review"))


def _roundtrip(csg: CSG, name: str, k: int, work: str) -> float:
    pt, _ = io_prism.emit_coalition_model(csg, 0, k)
    path = os.path.join(work, f"{name}.prism")
    with open(path, "w") as f:
        f.write(pt)
    special = set(csg.fail_states) | set(csg.avoid_states)

    def policy(sidx, stup, i, names):
        orig_s = int(stup[0])
        d = csg.state_defs[orig_s]
        if i not in d.active or orig_s in special:
            assert names in ([f"idle{i}"], ["-"]), (sidx, i, names)
            return [1.0]
        dist = csg.observed_policy[(orig_s, i)]
        for nm in names:
            assert nm.startswith(f"a{i}_"), (sidx, i, names)
        return [float(dist[int(nm.split("_")[1])]) for nm in names]

    imp = io_prism.import_prism_csg(path, None, policy=policy, k=k, fail_label="fail",
                                    out_base=os.path.join(work, name))
    assert imp.n_agents == csg.n_agents
    vals = np.array([vi_ref.solve(csg, m, k) for m in range(1 << csg.n_agents)])
    got = np.array([vi_ref.solve(imp, m, k) for m in range(1 << csg.n_agents)])
    assert vals.max() - vals.min() > 1e-2, "instance must be non-trivial"
    return float(np.max(np.abs(vals - got)))


def test_roundtrip_asymmetric_models(WORK):
    err = _roundtrip(vehicle_csg(), "vehicle", 1, WORK)
    assert err <= 1e-12
    err = _roundtrip(random_csg(n_agents=3, n_states=10, k=4, seed=20260829, max_active=2,
                                max_actions=3, n_fail=1, n_avoid=1), "rt_n3", 4, WORK)
    assert err <= 1e-12

    err = _roundtrip(random_csg(n_agents=4, n_states=12, k=3, seed=11, max_active=3,
                                max_actions=4, n_fail=2, n_avoid=0), "rt_n4", 3, WORK)
    assert err <= 1e-12


def _java_procs_with(tag: str):
    pids = subprocess.run(["pgrep", "-x", "java"], capture_output=True, text=True).stdout.split()
    out = []
    for p in pids:
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                args = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if tag in args:
            out.append(int(p))
    return out


@pytest.mark.skipif(not os.path.exists(ROBOT), reason="robot_coordination2 example missing")
def test_timeout_kills_jvm(WORK):
    with open(ROBOT) as f:
        txt = f.read()
    tag = "review_to_kill"
    t0 = time.perf_counter()
    r = io_prism.run_prism(txt, '<<robot1,robot2>> Pmin=? [ F<=200 "crash" ]\n', timeout_s=3,
                           javamaxmem="4g", workdir=WORK, tag=tag, consts={"l": 40, "q": 0.1})
    assert r["status"] == "TO" and r["value"] is None
    assert time.perf_counter() - t0 < 30
    time.sleep(0.5)
    left = _java_procs_with(tag)
    for p in left:
        os.kill(p, 9)
    assert left == [], f"JVM still running after timeout: {left}"


def test_edge_cases_vs_viref(WORK):

    sds = [StateDef(active=[0], n_actions=[3], rows=[((0,), [(1, 0.5), (2, 0.5)]),
                                                     ((1,), [(3, 0.7), (4, 0.3)]),
                                                     ((2,), [(1, 1.0)])]),
           StateDef(active=[], n_actions=[], rows=[((), [(3, 0.4), (4, 0.6)])]),
           StateDef(active=[0], n_actions=[2], rows=[((0,), [(3, 1.0)]), ((1,), [(4, 1.0)])]),
           StateDef(active=[], n_actions=[], rows=[((), [(3, 1.0)])]),
           StateDef(active=[], n_actions=[], rows=[((), [(4, 1.0)])])]
    c1 = CSG(n_agents=1, horizon_k=2, init_state=0, fail_states=[3], avoid_states=[4],
             state_defs=sds, observed_policy={(0, 0): np.array([0.2, 0.5, 0.3]),
                                              (2, 0): np.array([0.9, 0.1])})
    c1.validate()
    r = io_prism.crosscheck(c1, k=2, tol=1e-9, timeout_s=60, workdir=os.path.join(WORK, "n1"), n_jobs=2)
    assert r["all_match"], r["rows"]
    vals = {x["mask"]: x["prism"] for x in r["rows"]}
    assert vals[0] == pytest.approx(0.2, abs=1e-9)
    assert vals[1] == pytest.approx(0.6, abs=1e-9)


    v = vehicle_csg()
    v.n_agents = 3
    v.agent_names = None
    v.validate()
    pt, _ = io_prism.emit_coalition_model(v, 0, 1)
    assert "[idle2] (s=0|s=1|s=2) -> true;" in pt
    r = io_prism.crosscheck(v, k=1, tol=1e-9, timeout_s=60, workdir=os.path.join(WORK, "n3"), n_jobs=4)
    assert r["all_match"], r["rows"]
    vals = {x["mask"]: x["prism"] for x in r["rows"]}
    for m in range(4):
        assert vals[m] == pytest.approx(vals[m | 4], abs=1e-12)
    assert vals[0] == pytest.approx(0.12) and vals[3] == pytest.approx(1.0)
