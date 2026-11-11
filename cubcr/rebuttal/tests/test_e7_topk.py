import json
import math

import numpy as np
import pytest

import rb
import e7_topk as e7


def _gpu_available():
    try:
        from cubcr.engine_gpu import device_count
        return device_count() > 0
    except Exception:
        return False


def test_topk_kwargs_are_paper_config():
    kw = e7.topk_kwargs(3, 20260829, 16, 50, time_budget=120.0)
    for key, val in rb.PAPER_SAMPLER_KW.items():
        if key in ("time_budget", "max_perms"):
            continue
        assert kw[key] == val, key
    assert kw["eps"] is None and kw["top_k"] == 3
    assert kw["max_perms"] == 2_000_000 and kw["time_budget"] == 120.0
    assert e7.topk_kwargs(1, 1, 16, 50, time_budget=None)["time_budget"] is None


def test_eps_kwargs_paper_caps():
    kw = e7.eps_kwargs(0.01, 20260829, 16, 50)
    assert kw["time_budget"] == 900.0 and kw["max_perms"] == 2_000_000 and kw["eps"] == 0.01
    assert e7.eps_kwargs(0.01, 20260829, 16, 50, time_budget=None)["time_budget"] is None


def test_dt_kwargs_match_stored_e4v2():
    p = rb.REPO / "results" / "e4v2" / "e4__tb_none_n3_S55_k4_pf0.05_C3_s20260829__top5__s20260829.json"
    if not p.exists():
        pytest.skip("e4v2 record not present")
    d = json.load(open(p))
    kw = e7.dt_topk_kwargs(1, 20260829, 3, 4)
    assert kw["batch_perms"] == d["sampling"]["batch_perms"] == 16
    assert kw["ci"] == d["sampler_v2"]["ci"] and kw["batch_schedule"] == d["sampler_v2"]["batch_schedule"]
    assert kw["value_range"] == d["sampler_v2"]["value_range"] and kw["alpha"] == d["alpha"]
    assert kw["kind"] == "step" and d["sampler_v2"]["prune"] is False


def test_exact_topk_and_ties():
    phi = np.array([0.1, 0.3, 0.3, 0.05, 0.2])
    ex1 = e7.exact_topk(phi, 1)
    assert ex1["tie"] and ex1["gap"] == 0.0 and ex1["n_tied_at_boundary"] == 2
    ex2 = e7.exact_topk(phi, 2)
    assert not ex2["tie"] and ex2["top_set"] == [1, 2] and math.isclose(ex2["gap"], 0.1)
    ex3 = e7.exact_topk(phi, 3)
    assert ex3["top_set"] == [1, 2, 4] and not ex3["tie"]
    with pytest.raises(ValueError):
        e7.exact_topk(phi, 5)


def test_separation_recheck():
    est = np.array([0.5, 0.4, 0.1, 0.0])
    lo = np.array([0.45, 0.35, 0.05, -0.01]); hi = np.array([0.55, 0.45, 0.15, 0.01])
    s1 = e7.separation_holds(lo, hi, 0.0, 1, est)
    assert s1 == {"brief_rule": False, "sampler_rule": False}
    s2 = e7.separation_holds(lo, hi, 0.0, 2, est)
    assert s2 == {"brief_rule": True, "sampler_rule": True}
    s2fp = e7.separation_holds(lo, hi, 0.11, 2, est)
    assert s2fp == {"brief_rule": False, "sampler_rule": False}


def _synthetic_table(w):
    n = len(w)
    masks = np.arange(1 << n)
    keep = np.ones(1 << n)
    for i, wi in enumerate(w):
        keep *= np.where((masks >> i) & 1, 1.0 - wi, 1.0)
    return 1.0 - keep


def _exact_shapley(table, n):
    from math import factorial
    phi = np.zeros(n)
    for c in range(1 << n):
        s = bin(c).count("1")
        for i in range(n):
            if not (c >> i) & 1:
                wgt = factorial(s) * factorial(n - s - 1) / factorial(n)
                phi[i] += wgt * (table[c | (1 << i)] - table[c])
    return phi


@pytest.mark.parametrize("K", [1, 3])
def test_synthetic_table_topk_certifies_exact_set(K):
    w = [0.02, 0.5, 0.1, 0.3, 0.05, 0.2]
    n = len(w)
    table = _synthetic_table(w)
    phi = _exact_shapley(table, n)
    eng = rb.TableEngine(table)
    kw = e7.topk_kwargs(K, 20260829, n, 1, time_budget=None)
    r, wall = e7.run_sampler(eng, kw)
    assert r.status == "top_k" and r.certified
    rec = e7.result_record(r, wall, kw, engine="table", task_meta={"iid": "synthetic"}, phi_exact=phi, K=K)
    assert rec["certified_set_equals_exact"] is True
    assert rec["separation_recheck"] == {"brief_rule": True, "sampler_rule": True}
    assert rec["n_units"] == sum(rec["batch_sizes"]) and rec["n_perms"] == 2 * rec["n_units"]
    assert rec["n_evals_requested"] == 2 + rec["n_perms"] * (n + 1)
    assert rec["n_evals_unique"] <= 1 << n
    assert np.allclose(np.asarray(rec["L"]), np.asarray(rec["cs_lower"]) - rec["fp_term"])


def test_tied_players_are_not_certified_quickly():
    w = [0.3, 0.3, 0.1, 0.05]
    table = _synthetic_table(w)
    eng = rb.TableEngine(table)
    kw = e7.topk_kwargs(1, 20260829, 4, 1, time_budget=None, max_perms=20_000)
    r, _ = e7.run_sampler(eng, kw)
    assert r.status == "max_perms" and not r.certified


def _first_task(n=16, S=1000, k=50, gs=20260829):
    if not any(rb.E9V3_DIR.glob("e9v3__*.json")):
        pytest.skip("primary experiment reference records not generated yet")
    for t in e7.reference_tasks():
        if (t["n"], t["states_target"], t["k"], t["gen_seed"]) == (n, S, k, gs):
            return t
    raise KeyError


def test_paper_task_topk3_table_oracle():
    t = _first_task()
    if not rb.table_path(t["iid"]).exists():
        pytest.skip("table not cached")
    comp, _, _, _ = rb.build_task(t)
    eng = rb.TableEngine(np.load(rb.table_path(t["iid"])), compiled=comp)
    kw = e7.topk_kwargs(3, 20260829, t["n"], t["k"], time_budget=None)
    r, wall = e7.run_sampler(eng, kw)
    rec = e7.result_record(r, wall, kw, engine="table", task_meta=e7.task_meta(t), phi_exact=t["phi_exact"], K=3)
    assert r.certified and rec["certified_set_equals_exact"]
    assert rec["intervals_contain_exact"]


@pytest.mark.skipif(not _gpu_available(), reason="needs the CUDA engine")
def test_gpu_equals_table_twin_topk3():
    t = _first_task()
    if not rb.table_path(t["iid"]).exists():
        pytest.skip("table not cached")
    comp, _, _, _ = rb.build_task(t)
    kw = e7.topk_kwargs(3, 20260830, t["n"], t["k"], time_budget=e7.E7_GPU_TIME_BUDGET)
    with rb.gpu_lock("E7 test"):
        eng = rb.make_gpu_engine(comp)
        rb.paper_warmup(eng, t["n"], t["k"])
        rg, _ = e7.run_sampler(eng, kw)
        eng.close()
    eng_t = rb.TableEngine(np.load(rb.table_path(t["iid"])), compiled=comp)
    rt, _ = e7.run_sampler(eng_t, dict(kw, time_budget=None))
    assert rg.status == rt.status == "top_k"
    assert np.array_equal(rg.estimates, rt.estimates) and np.array_equal(rg.half_widths, rt.half_widths)
    assert list(rg.batch_sizes) == list(rt.batch_sizes) and rg.n_solves == rt.n_solves
    assert sorted(rg.top_k_players.tolist()) == sorted(rt.top_k_players.tolist())


def test_step_table_engine_indexing():
    from cubcr.masks import step_mask_matrix
    n, k = 2, 3
    table = np.arange(1 << (n * k), dtype=np.float64)
    eng = e7.StepTableEngine(table, n, k)
    rng = np.random.default_rng(0)
    coalitions = [sorted(set(rng.choice(n * k, size=rng.integers(0, n * k + 1), replace=False).tolist()))
                  for _ in range(50)]
    masks = np.stack([step_mask_matrix(c, n, k) for c in coalitions], axis=1)
    vals = eng.solve(masks, k)
    expect = [sum(1 << p for p in c) for c in coalitions]
    assert vals.tolist() == [float(x) for x in expect]


def test_step_table_topk_certifies():
    n, k = 2, 3
    w = [0.4, 0.05, 0.2, 0.1, 0.02, 0.3]
    table = _synthetic_table(w)
    phi = _exact_shapley(table, n * k)
    eng = e7.StepTableEngine(table, n, k)
    kw = dict(e7.DT_SAMPLER_KW, eps=None, top_k=1, seed=20260829, n=n, k=k, time_budget=None, max_perms=200_000)
    r, wall = e7.run_sampler(eng, kw)
    rec = e7.result_record(r, wall, kw, engine="table", task_meta={"iid": "synthetic_step"}, phi_exact=phi, K=1)
    assert r.certified and rec["certified_set_equals_exact"] and rec["top_k_players"] == [0]


def test_tie_aware_set_ok():
    phi = np.array([0.3, 0.3, 0.3, 0.29, 0.1])
    assert e7.tie_aware_set_ok([1], phi, 1) and e7.tie_aware_set_ok([2], phi, 1)
    assert not e7.tie_aware_set_ok([3], phi, 1)
    assert e7.tie_aware_set_ok([2, 0, 1], phi, 3) and not e7.tie_aware_set_ok([0, 1, 3], phi, 3)
    assert not e7.tie_aware_set_ok([0, 0, 1], phi, 3)
