from __future__ import annotations

import ast
import json
import os
import sys
import time
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

np.Inf = np.inf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cubcr.model import AMAX, DMAX, CSG, CompiledCSG, StateDef

ACTIONS = [(0, -1), (0, 1), (1, 0), (-1, 0), (0, 0), "interact"]
ACTION_NAMES = ["N", "S", "E", "W", "STAY", "INTERACT"]
NA = len(ACTIONS)
HELD = {None: 0, "onion": 1, "dish": 2, "soup": 3, "tomato": 4}


ASSUMPTIONS = [
    "A1: no counter objects; counter-facing INTERACT recoded to STAY (56 of 1,493 human INTERACTs: 20 no-ops, 20 drops, 16 pickups); state key omits counter contents",
    "A2: old auto-cook semantics for the three-onion loading property (full pot == cooking; empty-handed INTERACT on the pot recoded to STAY)",
    "A3: failure means the pot is not loaded after k synchronous joint actions (ticks 0..k-1); goal absorbing SAFE, layer-k non-goal absorbing FAIL",
    "A4: train-only empirical marginal plug-in policy, all trajectory ticks, no smoothing, back-off L1-L4 (>=3 counts), no estimation uncertainty propagated",
    "A5: Markov product of role-indexed marginal policies (joint-action correlation not preserved); population policy pooled across workers and absolute time",
    "A6: fixed empty standard start (P0 at (1,2), P1 at (3,1), both facing north); P0/P1 are the data-array player indices, not human identities",
    "A7: both roles active with all six actions at every non-absorbing tick; package-1.1.0 collision/swap rules; simultaneous interactions resolved P0 then P1",
    "A8: pre-action tick alignment (action at layer t produces layer t+1; loading on the k-th action counts as success); policy ignores absolute time/parity; player id p = i*k + t",
    "A9: centralized jointly optimal controller for free roles (v(empty) is an optimal-team baseline, not independently optimizing humans)",
]


def _import_mdp():
    import warnings
    warnings.filterwarnings("ignore")
    from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, Action
    return OvercookedGridworld


def state_key(mdp, state) -> tuple:

    ps = []
    for p in state.players:
        held = p.held_object.name if p.held_object is not None else None
        ps.append((tuple(p.position), tuple(p.orientation), HELD[held]))
    pot = 0
    for pos in mdp.get_pot_locations():
        if state.has_object(tuple(pos)):
            soup = state.get_object(tuple(pos))
            pot = min(len(soup.ingredients), 3)
    return (ps[0], ps[1], pot)


def key_from_human_row(state_dict: dict, pot_positions) -> tuple:

    ps = []
    for p in state_dict["players"]:
        held = p.get("held_object")
        name = None if held is None else held.get("name")
        if name not in HELD:
            raise ValueError(f"unknown held object {name!r}")
        ps.append((tuple(p["position"]), tuple(p["orientation"]), HELD[name]))
    pot = 0
    raw_objects = state_dict.get("objects") or {}
    objects = raw_objects.values() if isinstance(raw_objects, dict) else raw_objects
    for obj in objects:
        pos = tuple(obj.get("position", ()))
        if obj.get("name") == "soup" and (tuple(pos) in pot_positions):
            st = obj.get("state")
            if isinstance(st, (list, tuple)) and len(st) >= 3:
                pot = min(int(st[1]), 3)
            else:
                pot = min(len(obj.get("_ingredients", obj.get("ingredients", []))), 3)
    return (ps[0], ps[1], pot)


def is_goal(key) -> bool:
    return key[2] == 3


def _effective_joint(mdp, state, ja):


    out = []
    for p, a in zip(state.players, ja):
        if a == "interact":
            x, y = p.position
            dx, dy = p.orientation
            terrain = mdp.get_terrain_type_at_pos((x + dx, y + dy))
            if terrain == "X" or (terrain == "P" and p.held_object is None):
                a = (0, 0)
        out.append(a)
    return tuple(out)


def enumerate_layers(layout: str, k: int, max_states: int = 2_000_000, verbose: bool = True):


    OG = _import_mdp()
    mdp = OG.from_layout_name(layout)
    s0 = mdp.get_standard_start_state()
    joint_actions = [(a, b) for a in ACTIONS for b in ACTIONS]
    key_id: Dict[tuple, int] = {}
    rep: List = []
    succ_rows: List[np.ndarray] = []

    def intern(state):
        kk = state_key(mdp, state)
        i = key_id.get(kk)
        if i is None:
            i = len(rep)
            key_id[kk] = i
            rep.append(state)
        return i

    root = intern(s0)
    layers = [np.array([root], dtype=np.int64)]
    expanded = 0
    t0 = time.perf_counter()
    for t in range(k):
        cur = layers[-1]
        nxt = set()
        for sid in cur:
            state = rep[sid]
            if is_goal(state_key(mdp, state)):
                continue
            if sid >= len(succ_rows):

                row = np.empty(NA * NA, dtype=np.int32)
                for j, ja in enumerate(joint_actions):
                    eff = _effective_joint(mdp, state, ja)
                    ns, _ = mdp.get_state_transition(state, eff)
                    row[j] = intern(ns)
                while len(succ_rows) < sid:
                    succ_rows.append(None)
                succ_rows.append(row)
                expanded += 1
                if len(rep) > max_states:
                    raise RuntimeError(f"state cap {max_states} exceeded at layer {t}")
            nxt.update(succ_rows[sid].tolist())
        layers.append(np.array(sorted(nxt), dtype=np.int64))
        if verbose:
            print(f"  layer {t + 1}: {len(nxt)} keys ({len(rep)} distinct so far, {expanded} expanded, "
                  f"{time.perf_counter() - t0:.1f}s)", flush=True)

    K = len(rep)
    succ = np.full((K, NA * NA), -1, dtype=np.int32)
    for sid, row in enumerate(succ_rows):
        if row is not None:
            succ[sid] = row
    keys = [state_key(mdp, s) for s in rep]
    return mdp, keys, succ, layers


def load_human_rows(layout: str, split: str = "train"):
    import pandas as pd
    import overcooked_ai_py
    d = os.path.join(os.path.dirname(overcooked_ai_py.__file__), "data", "human_data",
                     f"clean_{split}_trials.pickle")
    df = pd.read_pickle(d)
    df = df[df["layout_name"] == layout]
    return df


def _action_index(a) -> int:
    if isinstance(a, str):
        if a.lower() == "interact":
            return 5
        raise ValueError(f"unknown action string {a!r}")
    t = tuple(int(x) for x in a)
    if t not in ACTIONS[:5]:
        raise ValueError(f"unknown action vector {t!r}")
    return ACTIONS.index(t)


def fit_human_policy(mdp, layout: str, keys: List[tuple], min_obs: int = 3, splits=("train",)):

    pot_positions = {tuple(p) for p in mdp.get_pot_locations()}
    c1: Dict[Tuple[tuple, int], np.ndarray] = defaultdict(lambda: np.zeros(NA))
    c2: Dict[Tuple[tuple, int], np.ndarray] = defaultdict(lambda: np.zeros(NA))
    c3: Dict[Tuple[tuple, int], np.ndarray] = defaultdict(lambda: np.zeros(NA))
    c4 = np.zeros((2, NA))
    n_rows = 0
    counter_interacts = 0
    interacts = 0
    skipped_rows = 0
    n_workers = set()
    for split in splits:
        df = load_human_rows(layout, split)
        if "workerid_num" in df:
            n_workers |= set(df["workerid_num"].tolist())
        for st, ja in zip(df["state"], df["joint_action"]):
            if isinstance(st, str):
                st = ast.literal_eval(st)
            if isinstance(ja, str):
                ja = ast.literal_eval(ja)
            try:
                kk = key_from_human_row(st, pot_positions)
            except Exception as e:
                skipped_rows += 1
                if skipped_rows <= 3:
                    print(f"  [overcooked] skipped human row: {e!r}", flush=True)
                continue
            n_rows += 1
            for i in range(2):
                a = _action_index(ja[i])
                if a == 5:
                    interacts += 1
                    x, y = kk[i][0]
                    dx, dy = kk[i][1]
                    if mdp.get_terrain_type_at_pos((x + dx, y + dy)) == "X":
                        counter_interacts += 1
                        a = 4
                c1[(kk, i)][a] += 1
                c2[((kk[i], kk[2]), i)][a] += 1
                c3[((kk[i][2], kk[2]), i)][a] += 1
                c4[i, a] += 1
    K = len(keys)
    pol = np.zeros((K, 2, NA))
    level = np.zeros((K, 2), dtype=np.int8)
    for sid, kk in enumerate(keys):
        for i in range(2):
            for lv, cnt in ((1, c1.get((kk, i))), (2, c2.get(((kk[i], kk[2]), i))),
                            (3, c3.get(((kk[i][2], kk[2]), i)))):
                if cnt is not None and cnt.sum() >= min_obs:
                    pol[sid, i] = cnt / cnt.sum()
                    level[sid, i] = lv
                    break
            else:
                pol[sid, i] = c4[i] / c4[i].sum()
                level[sid, i] = 4
    meta = {"human_rows": int(n_rows), "skipped_rows": int(skipped_rows), "min_obs": int(min_obs), "splits": list(splits),
            "n_action_observations": int(2 * n_rows), "n_interacts": int(interacts), "n_counter_interacts": int(counter_interacts),
            "interact_frac": float(interacts / max(1, 2 * n_rows)),
            "counter_interact_frac_of_interacts": float(counter_interacts / max(1, interacts)),
            "level_counts": {f"L{lv}": int((level == lv).sum()) for lv in (1, 2, 3, 4)},
            "distinct_full_keys_in_data": int(len({kk for (kk, _) in c1.keys()})),
            "agent_marginal": c4.tolist(), "n_workerid_num": int(len(n_workers)),
            "source_split": "train" if list(splits) == ["train"] else list(splits), "uses_test_split": "test" in splits,
            "uses_all_trajectory_ticks": True, "conditions_on_absolute_timestep": False,
            "joint_policy_factorization": "product_of_agent_index_marginals",
            "counter_interacts_relabelled_as_stay": True, "policy_estimation_uncertainty_propagated": False,
            "smoothing": None}
    return pol, level, meta


def build_layered_compiled(keys, succ, layers, policy, k: int, layer_weights: Optional[np.ndarray] = None):


    goal = np.array([is_goal(kk) for kk in keys], dtype=bool)
    ids: Dict[Tuple[int, int], int] = {}
    order: List[Tuple[int, int]] = []
    for t in range(k):
        for sid in layers[t]:
            if not goal[sid]:
                ids[(t, int(sid))] = 2 + len(order)
                order.append((t, int(sid)))
    S = 2 + len(order)
    R = NA * NA
    kind = np.zeros(S, dtype=np.uint8)
    kind[1] = 1
    row_start = np.zeros(S + 1, dtype=np.int64)
    row_start[1] = 1
    row_start[2] = 2
    row_start[3:] = 2 + R * np.arange(1, S - 1)
    agent = np.full((S, DMAX), -1, dtype=np.int16)
    radix = np.ones((S, DMAX), dtype=np.uint8)
    weights = np.zeros((S, DMAX, AMAX), dtype=np.float64)
    weights[:, :, 0] = 1.0
    agent[2:, 0], agent[2:, 1] = 0, 1
    radix[2:, 0], radix[2:, 1] = NA, NA
    weights[2:, :2, :] = 0.0
    n_rows = int(row_start[-1])
    col = np.empty(n_rows, dtype=np.int32)
    col[0], col[1] = 0, 1
    val = np.ones(n_rows, dtype=np.float64)
    row_ptr = np.arange(n_rows + 1, dtype=np.int64)
    for s, (t, sid) in enumerate(order):
        weights[s + 2, 0, :NA] = policy[sid, 0]
        weights[s + 2, 1, :NA] = policy[sid, 1]
        nxt = succ[sid]
        if t + 1 == k:
            c = np.where(goal[nxt], 0, 1)
        else:
            c = np.array([0 if goal[j] else ids[(t + 1, int(j))] for j in nxt], dtype=np.int32)
        col[row_start[s + 2]:row_start[s + 3]] = c
    comp = CompiledCSG(n_agents=2, n_states=S, init_state=2, horizon_k=k, state_kind=kind,
                       state_row_start=row_start, state_agent=agent, state_radix=radix, weights=weights,
                       row_ptr=row_ptr, col=col, val=val)
    index = np.array([(-1, -1), (-1, -1)] + order, dtype=np.int32)
    meta = {"n_states": S, "n_rows": n_rows, "layers": [int(len(l)) for l in layers],
            "goal_keys": int(goal.sum()), "distinct_keys": int(len(keys))}
    return comp, index, meta


def to_csg(comp: CompiledCSG, keys=None, index=None) -> CSG:

    sds = []
    obs = {}
    for s in range(comp.n_states):
        r0, r1 = int(comp.state_row_start[s]), int(comp.state_row_start[s + 1])
        active = [int(a) for a in comp.state_agent[s] if a >= 0]
        radices = [int(m) for m, a in zip(comp.state_radix[s], comp.state_agent[s]) if a >= 0]
        rows = []
        for r in range(r0, r1):
            loc = r - r0
            joint = []
            for m in reversed(radices):
                joint.append(loc % m)
                loc //= m
            joint = tuple(reversed(joint))
            succ = [(int(comp.col[j]), float(comp.val[j])) for j in range(comp.row_ptr[r], comp.row_ptr[r + 1])]
            rows.append((joint, succ))
        sds.append(StateDef(active=active, n_actions=radices, rows=rows))
        if comp.state_kind[s] == 0:
            for slot, a in enumerate(active):
                obs[(s, a)] = comp.weights[s, slot, :radices[slot]].copy()
    fail = [int(s) for s in np.flatnonzero(comp.state_kind == 1)]
    return CSG(n_agents=comp.n_agents, horizon_k=comp.horizon_k, init_state=comp.init_state,
               fail_states=fail, state_defs=sds, observed_policy=obs, agent_names=["P0", "P1"])


def main(argv=None):
    import argparse
    from generators.ticketbench import save_compiled_npz
    ap = argparse.ArgumentParser()
    ap.add_argument("--layout", default="cramped_room")
    ap.add_argument("--k", type=int, nargs="+", default=[20, 30])
    ap.add_argument("--out", default="datasets/overcooked")
    ap.add_argument("--min-obs", type=int, default=3)
    ap.add_argument("--max-states", type=int, default=2_000_000)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    kmax = max(a.k)
    print(f"[overcooked] enumerating {a.layout} up to k={kmax}", flush=True)
    t0 = time.perf_counter()
    mdp, keys, succ, layers = enumerate_layers(a.layout, kmax, a.max_states)
    t_enum = time.perf_counter() - t0
    policy, level, pmeta = fit_human_policy(mdp, a.layout, keys, a.min_obs)
    print(f"[overcooked] policy: {pmeta}", flush=True)
    for k in a.k:
        comp, index, meta = build_layered_compiled(keys, succ, layers[:k + 1], policy, k)
        meta.update({"layout": a.layout, "k": k, "enumeration_s": t_enum, "policy": pmeta,
                     "assumptions": ASSUMPTIONS,
                     "fingerprint": comp.fingerprint()})
        path = os.path.join(a.out, f"overcooked_{a.layout}_k{k}.npz")
        save_compiled_npz(comp, meta, path)
        np.savez(os.path.join(a.out, f"overcooked_{a.layout}_k{k}_index.npz"), index=index,
                 keys=np.array([json.dumps(kk) for kk in keys]), level=level, policy=policy)
        print(f"[overcooked] k={k}: S={comp.n_states} rows={comp.n_rows} -> {path}", flush=True)
    json.dump({"layout": a.layout, "k": a.k, "policy": pmeta, "assumptions": ASSUMPTIONS,
               "layers": [int(len(l)) for l in layers], "distinct_keys": len(keys), "enumeration_s": t_enum},
              open(os.path.join(a.out, f"overcooked_{a.layout}_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
