from fractions import Fraction

import numpy as np

import e1_rho


def _comp(weights_rows, P_rows, k=5):


    from cubcr.model import CompiledCSG, DMAX, AMAX
    S = 2
    weights = np.zeros((S, DMAX, AMAX))
    weights[:, :, 0] = 1.0
    radix = np.ones((S, DMAX), dtype=np.uint8)
    agent = np.full((S, DMAX), -1, dtype=np.int16)
    for i, w in enumerate(weights_rows):
        weights[0, i, :] = 0.0
        weights[0, i, :len(w)] = w
        radix[0, i] = len(w)
        agent[0, i] = i
    nrows = int(np.prod([len(w) for w in weights_rows])) if weights_rows else 1
    assert nrows == len(P_rows)
    row_ptr, col, val = [0], [], []
    for pr in P_rows:
        for j, p in enumerate(pr):
            col.append(j % 2)
            val.append(p)
        row_ptr.append(len(col))
    row_ptr.append(len(col) + 1)
    col.append(1)
    val.append(1.0)
    return CompiledCSG(n_agents=max(1, len(weights_rows)), n_states=S, init_state=0, horizon_k=k,
                       state_kind=np.array([0, 1], dtype=np.uint8),
                       state_row_start=np.array([0, nrows, nrows + 1], dtype=np.int64),
                       state_agent=agent, state_radix=radix, weights=weights,
                       row_ptr=np.array(row_ptr, dtype=np.int64), col=np.array(col, dtype=np.int32),
                       val=np.array(val, dtype=np.float64))


def test_fp_policy_sum_rounds_down_exact_sum_does_not():

    assert np.sum(np.array([0.1, 0.1, 0.8])) == 1.0
    assert Fraction(0.1) + Fraction(0.1) + Fraction(0.8) > 1
    comp = _comp([[0.1, 0.1, 0.8]], [[0.5, 0.5]] * 3)
    from cubcr.fp_error import lipschitz_rho
    r = e1_rho.exact_rho(comp)
    assert r["rho_w"] == Fraction(0.1) + Fraction(0.1) + Fraction(0.8)
    assert Fraction(lipschitz_rho(comp)) < r["rho"]


def test_subset_maximum_exceeds_product_of_all_slots():

    w1 = [0.3, 0.3, 0.3999999999999999]
    w2 = [0.1, 0.1, 0.8]
    s1 = sum(Fraction(x) for x in w1)
    s2 = sum(Fraction(x) for x in w2)
    assert s1 < 1 < s2
    comp = _comp([w1, w2], [[0.5, 0.5]] * 9)
    r = e1_rho.exact_rho(comp)

    assert s1 * s2 < 1 and r["rho_w_allslots"] == 1
    assert r["rho_w"] == s2 > r["rho_w_allslots"]
    from cubcr.fp_error import lipschitz_rho
    assert Fraction(lipschitz_rho(comp)) < r["rho"]


def test_exact_max_row_sum():
    rows = [[0.1, 0.2, 0.7], [0.5, 0.5], [0.3, 0.7]]
    rp = np.cumsum([0] + [len(r) for r in rows])
    val = np.concatenate([np.array(r) for r in rows])
    best = e1_rho.exact_max_row_sum(rp, val)
    want = max(sum(Fraction(x) for x in r) for r in rows)
    assert best == want


def test_round_up_down():
    q = Fraction(1, 3)
    up, dn = e1_rho.round_up_float(q), e1_rho.round_down_float(q)
    assert Fraction(dn) <= q <= Fraction(up)
    assert np.nextafter(dn, np.inf) == up
    assert e1_rho.round_up_float(Fraction(1, 4)) == 0.25 == e1_rho.round_down_float(Fraction(1, 4))


def test_outward_bound_dominates_exact_formula():
    comp = _comp([[0.1, 0.1, 0.8]], [[0.1, 0.2, 0.7]] * 3, k=50)
    o = e1_rho.outward_bound(comp, 50)
    r = e1_rho.exact_rho(comp)
    g = e1_rho.gamma_exact(o["M"])
    eta = 50 * g * r["rho"] ** 50 * (1 + g) ** 49
    em = 2 * eta + e1_rho.U64
    epsx = em + e1_rho.gamma_exact(2) * (1 + em) + e1_rho.U64
    assert Fraction(o["eta_v_out"]) >= eta and Fraction(o["eps_x_out"]) >= epsx

    assert Fraction(np.nextafter(o["eps_x_out"], -np.inf)) < epsx

    assert abs(o["eps_x_out"] - o["eps_x_code"]) <= 1e-13 * o["eps_x_code"]

    assert o["_eps_x_exact"] == epsx
    assert o["eps_x_exact_minus_code"] == float(epsx - Fraction(o["eps_x_code"]))
    assert (o["eps_x_exact_minus_code"] > 0) == (not o["code_eps_x_covers_outward"])
