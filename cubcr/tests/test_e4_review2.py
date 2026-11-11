import glob
import hashlib
import json
import math
import os
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from cubcr.engine_cpu import CPUEngine
from cubcr.shapley import shapley_from_table
from cubcr.shapley_sampled import (empirical_bernstein_half_width,
                                   look_schedule_size, sample_shapley)
from experiments import protocol


class _Compiled:
    def __init__(self, n):
        self.n_agents, self.horizon_k = n, 1


class TableEngine:
    def __init__(self, table, n):
        self.table = np.asarray(table, dtype=np.float64)
        self.compiled = _Compiled(n)

    def solve(self, masks, k=None):
        masks = np.asarray(masks, dtype=np.uint64)
        return self.table[masks.astype(np.int64)].copy()


def replay(table, N, seed, batch_perms, max_perms, alpha, antithetic=True):

    rng = np.random.default_rng(seed)
    ppu = 2 if antithetic else 1
    units, n_perms, look_m, m0, next_look = [], 0, [], None, None
    while n_perms < max_perms:
        P = min(batch_perms, -(-(max_perms - n_perms) // ppu))
        for _ in range(P):
            perm = rng.permutation(N)
            u = np.zeros(N)
            for seq in ([perm, perm[::-1]] if antithetic else [perm]):
                S = 0
                for p in seq:
                    u[p] += table[S | (1 << int(p))] - table[S]
                    S |= 1 << int(p)
            units.append(u / ppu)
        n_perms += P * ppu
        m = len(units)
        if m0 is None:
            m0 = next_look = m
        if m >= next_look or n_perms >= max_perms:
            look_m.append(m)
            while next_look <= m:
                next_look *= 2
    U = np.array(units)
    m = U.shape[0]
    est = U.mean(axis=0)
    var = U.var(axis=0, ddof=1)
    L_max = math.ceil(math.log2(max_perms / (batch_perms * ppu)) - 1e-12) + 1
    a_eff = alpha / N / L_max
    Lc = math.log(4.0 / a_eff)
    hw = 2.0 * (np.sqrt(2.0 * (var / 4.0) * Lc / m) + 7.0 * Lc / (3.0 * (m - 1)))
    return est, var, hw, a_eff, L_max, look_m, m, n_perms


@pytest.mark.parametrize("N,batch,max_perms,anti", [(5, 8, 1000, True), (6, 5, 333, False), (4, 16, 4096, True)])
def test_full_allocation_reproduced_by_independent_replay(N, batch, max_perms, anti):
    rng = np.random.default_rng(1000 + N)
    tab = rng.uniform(0, 1, size=1 << N)
    tab[0] = 0.0
    eng = TableEngine(tab, N)
    alpha = 0.05
    r = sample_shapley(eng, N, kind="agent", seed=17, batch_perms=batch, max_perms=max_perms, alpha=alpha,
                       eps=0.0, antithetic=anti)
    est, var, hw, a_eff, L_max, look_m, m, n_perms = replay(tab, N, 17, batch, max_perms, alpha, anti)
    assert r.status == "max_perms" and not r.certified and r.anytime_valid
    assert r.n_samples == m and r.n_perms == n_perms
    assert np.allclose(r.estimates, est, atol=1e-12)
    assert np.allclose(r.sample_var, var, atol=1e-12)
    assert np.allclose(r.half_widths, hw, rtol=1e-12)
    assert r.alpha_eff == pytest.approx(a_eff, rel=1e-12) and r.n_looks_max == L_max == look_schedule_size(max_perms, batch * (2 if anti else 1))
    assert list(r.look_m) == look_m and r.n_looks_used == len(look_m) <= L_max

    phi = shapley_from_table(tab, N)
    assert np.all(np.abs(est - phi) <= hw)


def test_coverage_at_data_dependent_stopping_time_200_games():
    rng = np.random.default_rng(20260829)
    N, alpha, batch, max_perms, eps = 5, 0.05, 8, 4096, 0.5
    fam, n_cert, stop_m = [], 0, []
    L_max = look_schedule_size(max_perms, 2 * batch)
    for g in range(200):
        tab = rng.uniform(0, 1, size=1 << N)
        tab[0] = 0.0
        phi = shapley_from_table(tab, N)
        seen = []
        r = sample_shapley(TableEngine(tab, N), N, kind="agent", seed=g, batch_perms=batch, max_perms=max_perms,
                           alpha=alpha, eps=eps, callback=lambda m, e, h: seen.append((int(m), h.copy())))
        assert r.n_looks_used <= L_max and r.n_looks_max == L_max
        if not r.certified:
            continue
        n_cert += 1
        c = r.certificate
        assert c["kind"] == "eps" and c["m"] == r.n_samples and c["terminal_reason"] in (None, "max_perms")
        m0 = r.look_m[0]
        planned = {m0 * 2 ** j for j in range(L_max + 1)} | {math.ceil(max_perms / 2)}
        assert c["m"] in planned and all(mm in planned for mm in r.look_m)

        first_planned_ok = min(mm for mm, h in seen if mm in planned and np.all(h <= eps))
        assert c["m"] == first_planned_ok
        stop_m.append(c["m"])
        fam.append(bool(np.all(np.abs(c["estimates"] - phi) <= c["half_widths"])))
    assert n_cert >= 150, n_cert
    assert len(set(stop_m)) >= 2, "eps should fire at different looks across games (data-dependent stop)"
    assert float(np.mean(fam)) >= 1 - alpha, float(np.mean(fam))


def test_monotone_marginals_and_cpu_certification_on_e4_n3_instance():
    from generators.ticketbench import build_compiled
    from cubcr.shapley_sampled import exact_player_table
    comp, meta = build_compiled(3, 60, 4, 0.05, "none", 20260829)
    eng = CPUEngine(comp, n_threads=4)
    N = 12
    tab = exact_player_table(eng, N, kind="step", n=3, k=4)
    assert tab.shape == (4096,)
    S = np.arange(4096)
    for p in range(N):
        without = S[(S >> p) & 1 == 0]
        assert np.all(tab[without | (1 << p)] >= tab[without] - 1e-12)
    phi = shapley_from_table(tab, N)
    r = sample_shapley(eng, kind="step", n=3, k=4, seed=20260829, batch_perms=16, max_perms=20000, eps=0.02,
                       alpha=0.05, two_sided=True, bonferroni=True, sequential=True)
    assert r.certified and r.certificate["kind"] == "eps" and r.n_looks_max == 11
    assert r.alpha_eff == pytest.approx(0.05 / 12 / 11)
    assert np.all(np.abs(r.certificate["estimates"] - phi) <= r.certificate["half_widths"])
    assert np.all(r.certificate["half_widths"] <= 0.02)

    assert empirical_bernstein_half_width(r.sample_var, r.n_samples, r.alpha_eff, lo=0.0, hi=1.0, two_sided=True).max() \
        < r.half_widths.max()


def _gpu_available():
    try:
        from cubcr import engine_gpu
        return engine_gpu.device_count() > 0
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA device")
def test_e4_driver_smoke_records_and_censoring(tmp_path):
    from experiments import e4_sampling
    out1 = str(tmp_path / "prod")
    assert e4_sampling.main(["--configs", "smoke", "--out", out1, "--max-perms", "20000", "--time-budget", "600"]) == 0
    recs = {os.path.basename(f).split("__")[2]: json.load(open(f)) for f in glob.glob(os.path.join(out1, "e4__*.json"))}
    assert set(recs) == {"top5", "eps0.02", "eps0.01", "eps0.005", "trajectory"}
    lib = json.load(open(glob.glob(os.path.join(out1, "e4_summary__*.json"))[0]))["provenance"]
    assert lib["libcubcr_gpu_sha256"] == hashlib.sha256(open(lib["libcubcr_gpu_path"], "rb").read()).hexdigest()
    for rule, r in recs.items():
        protocol.validate_record(r)
        for key in ("gpu", "cuda", "driver", "git", "seed", "instance_seed", "dtype", "cap_s", "alpha_allocation",
                    "certificate", "certified", "time_to_certify_s", "censor_reason", "label", "screening"):
            assert key in r, key
        assert r["gpu"] and r["cuda"] and r["driver"] and r["dtype"] == "fp64" and r["cap_s"] == 600.0
        assert r["provenance"]["libcubcr_gpu_sha256"] == lib["libcubcr_gpu_sha256"]
        al = r["alpha_allocation"]
        assert al["n_looks_max"] == 11 and al["alpha_eff"] == pytest.approx(0.05 / 12 / 11) and al["anytime_valid"]
        assert r["status"] == "OK" and r["censored"] is False and r["exact"]["all_inside_ci"]
        c = r["certificate"]
        if r["certified"]:
            assert rule.startswith("eps") and c["kind"] == "eps" and r["time_to_certify_s"] == c["wall_s"]
            assert c["m"] in al["look_m"] and c["look_index"] == al["n_looks_used"] and c["max_half_width"] <= r["rule_params"]["eps"]
            assert np.array_equal(np.array(c["estimates"]).ravel(), np.array(r["sampling"]["estimates"]).ravel())
            assert r["censor_reason"] is None and r["certify_censored"] is False
        else:
            assert c is None and r["time_to_certify_s"] is None
            if rule == "trajectory":
                assert r["label"] == "trajectory" and r["censor_reason"] is None and not r["certify_censored"]
                assert "observed_crossings_not_certificates" in r["sampling"] and al["n_looks_used"] == 0
            else:
                assert r["censor_reason"] == "max_perms" and r["certify_censored"] and r["sampling"]["status"] == "max_perms"
    assert recs["eps0.02"]["certified"] and recs["eps0.01"]["certified"]
    assert recs["eps0.02"]["time_to_certify_s"] <= recs["eps0.01"]["time_to_certify_s"]
    summ = json.load(open(glob.glob(os.path.join(out1, "e4_summary__*.json"))[0]))
    assert summ["per_rule"]["eps0.02"]["n_certified"] == 1 and summ["per_rule"]["eps0.02"]["time_to_certify_median_s"] is not None
    assert summ["per_rule"]["top5"]["n_certified"] == 0 and summ["per_rule"]["top5"]["time_to_certify_median_s"] is None

    out2 = str(tmp_path / "to")
    assert e4_sampling.main(["--configs", "smoke", "--out", out2, "--max-perms", "100000", "--time-budget", "0.001"]) == 0
    for f in glob.glob(os.path.join(out2, "e4__*.json")):
        r = json.load(open(f))
        protocol.validate_record(r)
        assert r["status"] == "TO" and r["censored"] is True and r["wall_s"] >= r["cap_s"] == 0.001
        assert r["coalition_solves_per_s"] is None and r["sampling"]["status"] == "time" and not r["certified"]
        assert r["sampling"]["n_batches"] == 1 and r["sampling"]["n_perms"] == 32
        assert r["censor_reason"] == "time"
        assert r["certify_censored"] == (r["label"] == "certification")
