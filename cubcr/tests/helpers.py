from __future__ import annotations

import numpy as np

from cubcr.model import CSG, StateDef


def random_csg(n_agents=3, n_states=12, k=4, seed=0, max_active=2, max_actions=3,
               n_fail=1, n_avoid=0, succ_per_row=3, dummy_agents=(), symmetric_pair=None,
               deterministic_policy=False, min_actions=2) -> CSG:


    rng = np.random.default_rng(seed)
    S = n_states
    special = set()
    fail = list(range(S - n_fail - n_avoid, S - n_avoid))
    avoid = list(range(S - n_avoid, S))
    special.update(fail)
    special.update(avoid)
    sds = []
    obs = {}
    for s in range(S):
        if s in special:
            sds.append(StateDef(active=[], n_actions=[], rows=[((), [(s, 1.0)])]))
            continue
        d = int(rng.integers(0, max_active + 1))
        active = sorted(rng.choice(n_agents, size=d, replace=False).tolist()) if d else []
        if symmetric_pair is not None and (symmetric_pair[0] in active) != (symmetric_pair[1] in active):

            i, j = symmetric_pair
            if i in active and j not in active and len(active) < max_active:
                active = sorted(active + [j])
            elif j in active and i not in active and len(active) < max_active:
                active = sorted(active + [i])
            else:
                active = [a for a in active if a not in (i, j)]
        nact = [int(rng.integers(min_actions, max_actions + 1)) for _ in active]
        if symmetric_pair is not None and symmetric_pair[0] in active:
            m = nact[active.index(symmetric_pair[0])]
            nact[active.index(symmetric_pair[1])] = m
        nrows = int(np.prod(nact)) if nact else 1
        rows = []
        digits = [0] * len(active)

        dists = {}
        for r in range(nrows):
            j = list(digits)
            key = tuple(j)

            key_eff = tuple(0 if active[i] in dummy_agents else j[i] for i in range(len(active)))
            if symmetric_pair is not None and symmetric_pair[0] in active:
                ia, ib = active.index(symmetric_pair[0]), active.index(symmetric_pair[1])
                ke = list(key_eff)
                if ke[ia] > ke[ib]:
                    ke[ia], ke[ib] = ke[ib], ke[ia]
                key_eff = tuple(ke)
            if key_eff not in dists:
                cnt = int(rng.integers(1, succ_per_row + 1))
                succ = rng.choice(S, size=cnt, replace=False)
                p = rng.dirichlet(np.ones(cnt))
                dists[key_eff] = [(int(a), float(b)) for a, b in zip(succ, p)]
            rows.append((key, list(dists[key_eff])))

            for i in reversed(range(len(active))):
                digits[i] += 1
                if digits[i] < nact[i]:
                    break
                digits[i] = 0
        sds.append(StateDef(active=active, n_actions=nact, rows=rows))
        for a, m in zip(active, nact):
            if deterministic_policy:
                dist = np.zeros(m)
                dist[int(rng.integers(0, m))] = 1.0
            else:
                dist = rng.dirichlet(np.ones(m))
            obs[(s, a)] = dist
        if symmetric_pair is not None and symmetric_pair[0] in active:
            obs[(s, symmetric_pair[1])] = obs[(s, symmetric_pair[0])].copy()
    init = 0
    csg = CSG(n_agents=n_agents, horizon_k=k, init_state=init, fail_states=fail, state_defs=sds,
              observed_policy=obs, avoid_states=avoid, meta={"seed": seed})
    csg.validate()
    return csg
