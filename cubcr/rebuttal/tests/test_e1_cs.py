import numpy as np
import pytest

import rb
import e1_cs


def _monotone_table(n, seed):


    rng = np.random.default_rng(seed)
    w = rng.uniform(0.0, 0.3, size=n)
    masks = np.arange(1 << n)
    bits = ((masks[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)
    v = 1.0 - np.prod(np.where(bits, 1.0 - w[None, :], 1.0), axis=1)
    return v.astype(np.float64)


def _run(table, n, eps_div=20, seed=20260829):
    eng = rb.RecordingEngine(rb.TableEngine(table, compiled=None))
    ups = float(table[-1] - table[0])
    r, wall, _ = rb.run_paper_sampler(eng, n, 10, ups / eps_div, seed, record_units=False)
    rep = rb.replay_units(r, eng.calls, n, seed, 10)
    return r, rep["units"]


def test_fp64_replay_bitwise():
    n = 8
    table = _monotone_table(n, 1)
    r, units = _run(table, n)
    lo, hi = r.value_range
    looks, cs = e1_cs.fp64_replay(units, r.batch_sizes, lo, hi, r.fp_term, 0.05, eps=r.eps)
    last = looks[-1]
    assert np.array_equal(last.hw, r.half_widths)
    assert np.array_equal(last.cs_L, r.cs_lower) and np.array_equal(last.cs_U, r.cs_upper)
    assert np.array_equal(last.est, r.estimates)
    assert last.stop and not any(lk.stop for lk in looks[:-1])
    assert len(cs.lams) == len(r.batch_sizes) and all(np.all((l > 0) & (l <= 0.5)) for l in cs.lams)


def test_round_helpers():
    import mpmath
    with mpmath.workdps(40):
        x = mpmath.mpf(1) / 3
        lo, hi = e1_cs.round_down(x), e1_cs.round_up(x)
        assert mpmath.mpf(lo) <= x <= mpmath.mpf(hi)
        assert np.nextafter(lo, np.inf) == hi
        y = mpmath.mpf(0.25)
        assert e1_cs.round_down(y) == 0.25 == e1_cs.round_up(y)


def test_make_params_directions():
    from fractions import Fraction
    prm = e1_cs.make_params(0.3, "1e-14", 20)
    assert Fraction(prm.beta) >= Fraction(1, 10 ** 14)
    assert prm.a <= -prm.beta
    assert Fraction(prm.b) >= Fraction(0.3) + 3 * Fraction(prm.beta)
    assert Fraction(prm.eps) <= (Fraction(0.3) - 2 * Fraction(prm.beta)) / 20


def test_conservative_contains_exact_and_matches_stop():
    from cubcr.shapley_fast import shapley_from_table_fast
    n = 8
    table = _monotone_table(n, 2)
    phi = shapley_from_table_fast(table, n)
    r, units = _run(table, n)
    lo, hi = r.value_range
    prm = e1_cs.make_params(r.v_full - r.v_empty, "1e-14", 20)
    out = e1_cs.conservative_replay(lambda m: units[:m] if m <= units.shape[0] else (_ for _ in ()).throw(RuntimeError()),
                                    r.batch_sizes, n, prm, lo, hi, 0.05, 32, 512)
    lk = out["looks"][out["stop_index"]]
    assert np.all(lk.lower <= lk.upper)
    ok, _ = e1_cs.contained(phi, lk.lower, lk.upper)
    assert ok.all()

    assert np.max(np.abs(out["looks"][len(r.batch_sizes) - 1].hw - r.half_widths)) < 1e-10


def test_units_from_engine_reproduce_replay():
    n = 7
    table = _monotone_table(n, 3)
    r, units = _run(table, n)
    u = e1_cs.units_from_engine(rb.TableEngine(table), n, 20260829, units.shape[0] + 50)
    assert np.array_equal(u[:units.shape[0]], units)


def test_unit_outside_range_raises():
    prm = e1_cs.ConservativeParams(a=0.0, b=1.0, beta=0.0, eps=0.1)
    cs = e1_cs.ConservativeCS(2, prm, 0.0, 1.0)
    with pytest.raises(RuntimeError):
        cs.update(np.array([[1.5, 0.2]]), np.array([0.3, 0.3]), np.array([[0.5, 0.5]]))


def test_strict_xhat_is_predictable_paper_xhat_is_not():


    paper_vals, strict_vals = set(), set()
    for y in (0.7, 0.3, 0.6, 0.9, 0.2):
        cs = e1_cs.TracingPrPlEB(1, 0.05)
        cs.update(np.array([[0.1], [y]]))
        paper_vals.add(float(cs.xhats[-1][1, 0]))
        strict_vals.add(float(cs.xhats_excl[-1][1, 0]))
    assert len(strict_vals) == 1 and strict_vals == {(0.5 + 0.1) / 2.0}
    assert len(paper_vals) >= 2
    assert max(paper_vals) - min(paper_vals) < 1e-16


def test_conservative_replay_xhat_modes():
    from cubcr.shapley_fast import shapley_from_table_fast
    n = 8
    table = _monotone_table(n, 4)
    phi = shapley_from_table_fast(table, n)
    r, units = _run(table, n)
    lo, hi = r.value_range
    prm = e1_cs.make_params(r.v_full - r.v_empty, "1e-14", 20)
    prov = lambda m: units[:m]
    outs = {}
    for mode in ("strict", "paper"):
        out = e1_cs.conservative_replay(prov, r.batch_sizes, n, prm, lo, hi, 0.05, 32, 512, xhat_mode=mode)
        lk = out["looks"][out["stop_index"]]
        assert e1_cs.contained(phi, lk.lower, lk.upper)[0].all()
        assert out["xhat_mode"] == mode and out["xhat_n_total"] == sum(r.batch_sizes) * n
        outs[mode] = out
    assert outs["strict"]["xhat_n_differ"] == outs["paper"]["xhat_n_differ"]
    a = outs["strict"]["looks"][-1]
    b = outs["paper"]["looks"][-1]
    assert np.max(np.abs(a.lower - b.lower)) < 1e-12 and np.max(np.abs(a.upper - b.upper)) < 1e-12
    with pytest.raises(ValueError):
        e1_cs.conservative_replay(prov, r.batch_sizes, n, prm, lo, hi, 0.05, 32, 512, xhat_mode="other")


def test_units_from_engine_tail_equals_full_stream_suffix():

    n = 7
    table = _monotone_table(n, 4)
    r, units = _run(table, n)
    m = units.shape[0]
    full = e1_cs.units_from_engine(rb.TableEngine(table), n, 20260829, m + 40, chunk=16)
    tail = e1_cs.units_from_engine_tail(rb.TableEngine(table), n, 20260829, m, 40, chunk=7)
    assert np.array_equal(full[m:], tail)
    assert np.array_equal(full[:m], units)
