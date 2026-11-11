import numpy as np

from cubcr.model import CSG, StateDef


def vehicle_csg() -> CSG:
    csg = CSG(
        n_agents=2,
        horizon_k=1,
        init_state=0,
        fail_states=[1],
        state_defs=[
            StateDef(
                active=[0, 1],
                n_actions=[2, 2],
                action_labels=[["brake", "nobrake"], ["brake", "nobrake"]],
                rows=[
                    ((0, 0), [(1, 0.12), (2, 0.88)]),
                    ((0, 1), [(1, 0.2), (2, 0.8)]),
                    ((1, 0), [(1, 0.6), (2, 0.4)]),
                    ((1, 1), [(1, 1.0)]),
                ],
            ),
            StateDef(active=[], n_actions=[], rows=[((), [(1, 1.0)])]),
            StateDef(active=[], n_actions=[], rows=[((), [(2, 1.0)])]),
        ],
        observed_policy={(0, 0): np.array([0.0, 1.0]), (0, 1): np.array([0.0, 1.0])},
        agent_names=["A1", "A2"],
        meta={"source": "Mu & Najib arXiv:2605.13077 Fig.1 / Examples 2-4, outcome phi = X crash"},
    )
    csg.validate()
    return csg
