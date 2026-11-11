import numpy as np
from cubcr.model import restrict_to_horizon
from cubcr.engine_cpu import CPUEngine
from tests.helpers import random_csg
from generators.ticketbench import build_compiled


def test_pruned_values_identical_random():
    for seed in (1, 2):
        csg = random_csg(n_agents=4, n_states=60, k=4, seed=seed, max_active=2, max_actions=3, succ_per_row=2)
        comp = csg.compile()
        for k in (1, 2, 4):
            pr = restrict_to_horizon(comp, k)
            assert pr.n_states <= comp.n_states
            masks = np.arange(16, dtype=np.uint64)
            a = CPUEngine(comp, n_threads=2).solve(masks, k)
            b = CPUEngine(pr, n_threads=2).solve(masks, k)
            assert np.max(np.abs(a - b)) <= 1e-15


def test_pruned_values_identical_ticketbench():
    comp, meta = build_compiled(8, 2000, 25, 0.05, "none", 20260829)
    pr = restrict_to_horizon(comp, 25)
    assert pr.n_states < comp.n_states
    masks = np.random.default_rng(0).integers(0, 256, size=64, dtype=np.uint64)
    a = CPUEngine(comp, n_threads=4).solve(masks, 25)
    b = CPUEngine(pr, n_threads=4).solve(masks, 25)
    assert np.max(np.abs(a - b)) <= 1e-15
    assert abs(pr.n_states - meta["n_reachable_k"]) <= 1
