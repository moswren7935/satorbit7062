from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import List, Optional

import numpy as np

import rb

REPO = rb.REPO
CACHE = rb.RB / "cache"
MODEL_CACHE = CACHE / "models"
PRISM_CACHE = CACHE / "prism_import"

PRIMARY_PFAIL = 0.05


def specs_primary() -> List[dict]:
    out = []
    for t in rb.paper_tasks(include_records=False):
        out.append(dict(family="primary", model_id=t["iid"], n=t["n"], states_target=t["states_target"], k=t["k"],
                        gen_seed=t["gen_seed"], pattern="none", hazard=0.0,
                        source="experiments/e9v3_certified.py (results/e9v3, paper seeds)"))
    return out


def specs_tables13() -> List[dict]:
    out = []
    files = sorted(glob.glob(str(REPO / "results" / "e0" / "values" / "*.npy"))) + \
        sorted(glob.glob(str(REPO / "results" / "e0hz" / "values" / "*.npy"))) + \
        sorted(glob.glob(str(REPO / "results" / "e11" / "values" / "*.npy")))
    for f in files:
        base = os.path.basename(f)
        iid = base.split("__")[1] if "__" in base else base[:-4]
        import re
        mm = re.search(r"_n(\d+)_S(\d+)_k(\d+)_", iid)
        n, S_act, k = int(mm.group(1)), int(mm.group(2)), int(mm.group(3))
        hazard = 0.02 if "_hz0.02" in iid else 0.0
        S_tgt = 1000 if S_act < 5000 else 10000
        out.append(dict(family="tables13", model_id=iid, n=n, states_target=S_tgt, k=k, gen_seed=20260829,
                        pattern="none", hazard=hazard, table_file=os.path.relpath(f, REPO),
                        source=os.path.relpath(os.path.dirname(os.path.dirname(f)), REPO)))
    return out


def specs_loo90() -> List[dict]:
    out = []
    for pattern in ("none", "symmetric", "redundant", "joint", "dummy"):
        for n in (8, 12, 16):
            for k in (25, 100):
                for seed in (20260829, 20260830, 20260831):
                    out.append(dict(family="loo90", model_id=None, n=n, states_target=1000, k=k, gen_seed=seed,
                                    pattern=pattern, hazard=0.0, source="experiments/e12_loo.py part (c)"))
    return out


DECISION6 = [(3, 60, 4), (8, 1000, 25), (8, 10000, 25), (16, 1000, 25), (16, 10000, 25), (12, 100000, 25)]


def specs_decision6() -> List[dict]:
    return [dict(family="decision6", model_id=None, n=n, states_target=S, k=k, gen_seed=20260829, pattern="none",
                 hazard=0.0, source="experiments/e4_sampling.py (results/e4v2, unpruned)") for n, S, k in DECISION6]


def specs_overcooked() -> List[dict]:
    out = [dict(family="overcooked", model_id="overcooked_cramped_room_k10", k=10, npz=None,
                source="experiments/e14_overcooked.py::run_exact_k10 (built: enumerate_layers + fit_human_policy(min_obs=3) + build_layered_compiled)")]
    for k in (20, 30, 40):
        out.append(dict(family="overcooked", model_id=f"overcooked_cramped_room_k{k}", k=k,
                        npz=f"datasets/overcooked/overcooked_cramped_room_k{k}.npz",
                        source=f"datasets/overcooked/overcooked_cramped_room_k{k}.npz"))
    return out


def specs_prism9() -> List[dict]:
    sys.path.insert(0, str(REPO))
    from experiments.e3b_bench import active_bench
    out = []
    for iid, path, consts, k, fkw, entry in active_bench():
        out.append(dict(family="prism9", model_id=iid, k=int(k), path=path, consts=dict(consts), fail=dict(fkw),
                        source=f"experiments/e3b_bench.py ({entry['source']}: {entry['path']})"))
    return out


STORM6 = [(4, 100), (6, 100), (8, 100), (4, 1000), (6, 1000), (8, 1000)]


def specs_storm6() -> List[dict]:
    return [dict(family="storm6", model_id=None, n=n, states_target=S, k=25, gen_seed=20260829, pattern="none",
                 hazard=0.0, source="experiments/e3storm.py (build_instance)") for n, S in STORM6]


def all_specs() -> List[dict]:
    return (specs_primary() + specs_tables13() + specs_loo90() + specs_decision6() + specs_overcooked() +
            specs_prism9() + specs_storm6())


def overcooked_k10(force: bool = False):

    from generators.ticketbench import load_compiled_npz, save_compiled_npz
    MODEL_CACHE.mkdir(parents=True, exist_ok=True)
    p = MODEL_CACHE / "overcooked_cramped_room_k10.npz"
    if p.exists() and not force:
        return load_compiled_npz(str(p))
    import warnings
    warnings.filterwarnings("ignore")
    from generators import overcooked_csg as oc
    k = 10
    mdp, keys, succ, layers = oc.enumerate_layers("cramped_room", k, verbose=False)
    pol, level, pmeta = oc.fit_human_policy(mdp, "cramped_room", keys, 3)
    comp, index, meta = oc.build_layered_compiled(keys, succ, layers, pol, k)
    meta = dict(meta, policy=pmeta)
    tmp = str(p) + f".tmp{os.getpid()}.npz"
    save_compiled_npz(comp, json.loads(json.dumps(meta, default=str)), tmp)
    os.replace(tmp, p)
    return load_compiled_npz(str(p))


def prism_csg(spec):


    from cubcr import io_prism
    PRISM_CACHE.mkdir(parents=True, exist_ok=True)
    out_base = str(PRISM_CACHE / spec["model_id"])
    return io_prism.import_prism_csg(spec["path"], spec["consts"], policy="uniform", k=spec["k"],
                                     out_base=out_base, **spec["fail"])


def build(spec):

    fam = spec["family"]
    if fam in ("primary", "tables13", "loo90"):
        from experiments.grid import build_model
        comp, iid, meta, build_s = build_model("ticketbench", spec["n"], spec["states_target"], spec["k"],
                                               spec["gen_seed"], spec["pattern"], PRIMARY_PFAIL,
                                               hazard=spec.get("hazard", 0.0))
        if spec.get("model_id") is not None and iid != spec["model_id"]:
            raise RuntimeError(f"instance id mismatch {iid} != {spec['model_id']}")
        return comp, None, dict(iid=iid, validated_at_build=False, build_path="ticketbench.build_compiled",
                                meta={kk: v for kk, v in meta.items() if kk in ("n_reachable_k", "frac_reachable_k")})
    if fam == "decision6":
        from generators.ticketbench import build_compiled
        comp, meta = build_compiled(spec["n"], spec["states_target"], spec["k"], 0.05, "none", spec["gen_seed"])
        return comp, None, dict(iid=meta["instance_id"], validated_at_build=False,
                                build_path="ticketbench.build_compiled", meta={})
    if fam == "storm6":
        from generators.ticketbench import build_instance
        csg = build_instance(spec["n"], spec["states_target"], spec["k"], 0.05, "none", spec["gen_seed"])
        comp = csg.compile()
        return comp, csg, dict(iid=csg.meta.get("instance_id"), validated_at_build=True,
                               build_path="ticketbench.build_instance -> CSG.compile (validate)", meta={})
    if fam == "overcooked":
        if spec["npz"] is None:
            comp, meta = overcooked_k10()
        else:
            from generators.ticketbench import load_compiled_npz
            comp, meta = load_compiled_npz(str(REPO / spec["npz"]))
        return comp, None, dict(iid=spec["model_id"], validated_at_build=False,
                                build_path="overcooked_csg.build_layered_compiled" + (" (npz)" if spec["npz"] else ""),
                                meta={})
    if fam == "prism9":
        csg = prism_csg(spec)
        comp = csg.compile()
        return comp, csg, dict(iid=spec["model_id"], validated_at_build=True,
                               build_path="io_prism.import_prism_csg (validate) -> compile", meta={})
    raise ValueError(fam)


def compiled_to_csg(comp):

    from cubcr.model import CSG, StateDef
    from e1_ext import deff_of
    S = int(comp.n_states)
    deff = deff_of(comp)
    sds = []
    obs = {}
    rp = comp.row_ptr
    for s in range(S):
        d = int(deff[s])
        active = [int(a) for a in comp.state_agent[s, :d]]
        radices = [int(m) for m in comp.state_radix[s, :d]]
        r0, r1 = int(comp.state_row_start[s]), int(comp.state_row_start[s + 1])
        rows = []
        for r in range(r0, r1):
            loc = r - r0
            joint = []
            for m in reversed(radices):
                joint.append(loc % m)
                loc //= m
            joint = tuple(reversed(joint))
            succ = list(zip(comp.col[rp[r]:rp[r + 1]].tolist(), comp.val[rp[r]:rp[r + 1]].tolist()))
            rows.append((joint, succ))
        sds.append(StateDef(active=active, n_actions=radices, rows=rows))
        if comp.state_kind[s] == 0:
            for slot, a in enumerate(active):
                obs[(s, a)] = np.asarray(comp.weights[s, slot, :radices[slot]], dtype=np.float64).copy()
    fail = [int(s) for s in np.flatnonzero(comp.state_kind == 1)]
    avoid = [int(s) for s in np.flatnonzero(comp.state_kind == 2)]
    return CSG(n_agents=int(comp.n_agents), horizon_k=int(comp.horizon_k), init_state=int(comp.init_state),
               fail_states=fail, state_defs=sds, observed_policy=obs, avoid_states=avoid)
