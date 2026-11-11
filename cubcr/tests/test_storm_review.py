import os
import sys
import time

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr.model import CSG, StateDef
from cubcr import vi_ref, io_prism, io_storm
from experiments import protocol

TOL = 1e-6

needs_prism = pytest.mark.skipif(not io_prism.prism_available(),
                                 reason=f"PRISM-games binary not found at {io_prism.PRISM_BIN}")


def _adv_csg() -> CSG:


    sds = [
        StateDef(active=[0, 1], n_actions=[1, 2],
                 rows=[((0, 0), [(1, 0.5), (5, 0.5)]),
                       ((0, 1), [(2, 0.7), (4, 0.3)])]),
        StateDef(active=[1], n_actions=[2],
                 rows=[((0,), [(4, 0.6), (3, 0.4)]),
                       ((1,), [(3, 1.0)])]),
        StateDef(active=[0], n_actions=[2],
                 rows=[((0,), [(4, 0.2), (0, 0.8)]),
                       ((1,), [(5, 1.0)])]),
        StateDef(active=[], n_actions=[], rows=[((), [(3, 1.0)])]),
        StateDef(active=[], n_actions=[], rows=[((), [(4, 1.0)])]),
        StateDef(active=[], n_actions=[], rows=[((), [(5, 1.0)])]),
    ]
    obs = {(0, 0): np.array([1.0]), (0, 1): np.array([0.6, 0.4]),
           (1, 1): np.array([0.3, 0.7]), (2, 0): np.array([0.5, 0.5])}
    csg = CSG(n_agents=2, horizon_k=3, init_state=0, fail_states=[4], state_defs=sds,
              observed_policy=obs, avoid_states=[5], meta={})
    csg.validate()
    return csg


def test_mdp_emission_avoid_not_fail_labelled():


    csg = _adv_csg()
    mdp, props = io_prism.emit_coalition_mdp(csg, 0, 3)
    fail_line = next(ln for ln in mdp.splitlines() if ln.startswith('label "fail"'))
    assert "s=4" in fail_line and "s=5" not in fail_line
    assert '[] s=5 -> 1.0:(s\'=5);' in mdp
    assert sum(1 for ln in mdp.splitlines() if ln.strip().startswith("[] s=0 ")) == 2
    assert props.strip() == 'Pmin=? [ F<=3 "fail" ]'


@needs_prism
def test_mdp_emission_avoid_and_radix1_matches_vi_ref(tmp_path):


    csg = _adv_csg()
    for mask in (0, 2, 3):
        ref = vi_ref.solve(csg, mask, 3)
        mdp, props = io_prism.emit_coalition_mdp(csg, mask, 3)
        r = io_prism.run_prism(mdp, props, timeout_s=120, workdir=str(tmp_path),
                               tag=f"adv_{mask}", include_identity=False)
        assert r["status"] == "OK", (r["status"], r["error"], r["stdout_path"])
        assert abs(r["value"] - ref) <= TOL, (mask, r["value"], ref)

    assert vi_ref.solve(csg, 3, 3) == pytest.approx(0.21544)
    assert vi_ref.solve(csg, 2, 3) == pytest.approx(0.174)


def test_run_storm_timeout_kills_process_group(tmp_path):


    pidfile = tmp_path / "child.pid"
    fake = tmp_path / "fake_storm"
    fake.write_text('#!/bin/bash\nif [ "$1" = "--version" ]; then echo "Storm 0.0-fake"; exit 0; fi\n'
                    f'sleep 300 &\necho $! > {pidfile}\nwait\n')
    fake.chmod(0o755)
    t0 = time.perf_counter()
    r = io_storm.run_storm('mdp\nmodule m x:[0..1] init 0; [] x=0 -> 1.0:(x\'=1); '
                           '[] x=1 -> 1.0:(x\'=1); endmodule\nlabel "fail" = x=1;\n',
                           'Pmin=? [ F<=1 "fail" ]\n', timeout_s=1.5, workdir=str(tmp_path),
                           tag="kill", storm_bin=str(fake), include_identity=False)
    wall = time.perf_counter() - t0
    assert r["status"] == "TO" and r["timed_out"] is True and r["value"] is None
    assert wall < 10.0
    child = int(pidfile.read_text().strip())
    dead = False
    for _ in range(40):
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            dead = True
            break
        time.sleep(0.05)
    assert dead, f"child {child} of the timed-out storm process survived the group kill"


def test_protocol_storm_records_validate():


    base = dict(instance_id="t", n=2, states=6, k=3, seed=0, gpu="g", cuda="c", git="h",
                dtype="n/a", threads=1, max_abs_err_vs_fp64=None, kendall_tau=None,
                speedup_vs_cpu=None, timestamp="now")
    protocol.validate_record(protocol.make_record(**base, engine="storm", cap_s=600.0,
                                                  wall_s=None, coalition_solves_per_s=None,
                                                  status="NOT RUN", censored=False))
    protocol.validate_record(protocol.make_record(**base, engine="storm", cap_s=600.0,
                                                  wall_s=600.0, coalition_solves_per_s=None,
                                                  status="TO", censored=True))
    with pytest.raises(ValueError, match="engine"):
        protocol.validate_record(protocol.make_record(**base, engine="storm2", cap_s=600.0,
                                                      wall_s=1.0, status="OK", censored=False))
