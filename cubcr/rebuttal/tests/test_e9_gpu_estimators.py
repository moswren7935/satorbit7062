import os as _os_anon
_CUBCR_REPO = _os_anon.environ.get("CUBCR_ROOT") or _os_anon.path.abspath(_os_anon.path.join(
    _os_anon.path.dirname(_os_anon.path.abspath(__file__)), "../.."))
import os
import sys

for _v in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, (_CUBCR_REPO + "/rebuttal/lib"))

import numpy as np
import pytest

import rb
import e9_gpu_estimators as e9

IID16 = "tb_none_n16_S961_k50_pf0.05_C3_s20260829"
IID20 = "tb_none_n20_S961_k50_pf0.05_C3_s20260829"


def _table(iid):
    p = rb.table_path(iid)
    if not p.exists():
        pytest.skip(f"table {iid} not cached")
    return np.load(p)


def _counting(table):

    v0 = float(table[0])
    seen = set()

    def f(masks):
        m = np.asarray(masks, dtype=np.int64).reshape(-1)
        seen.update(m.tolist())
        return table[m] - v0
    return f, seen


class _FakeEngine:

    compiled = None
    dtype = "fp64"

    def __init__(self, table):
        self.table = table
        self.calls = []

    def solve(self, masks, k=None):
        m = np.asarray(masks, dtype=np.uint64).reshape(-1)
        self.calls.append(m.copy())
        return self.table[m.astype(np.int64)]


def test_value_oracle_dedup_cache_prefetch():
    tab = np.cumsum(np.random.default_rng(0).random(1 << 8))
    eng = _FakeEngine(tab)
    orc = e9.ValueOracle(eng, 8, 5)
    v = orc.values(np.array([3, 5, 3, 7, 5], dtype=np.int64))
    assert np.array_equal(v, tab[[3, 5, 3, 7, 5]])
    assert len(eng.calls) == 1 and set(eng.calls[0].tolist()) == {0, 3, 5, 7, 255}
    assert orc.n_requested == 5 and orc.n_engine == 5
    v2 = orc.values(np.array([0, 255, 7], dtype=np.int64))
    assert len(eng.calls) == 1 and np.array_equal(v2, tab[[0, 255, 7]])
    v3 = orc.values(np.array([9, 3], dtype=np.int64))
    assert len(eng.calls) == 2 and eng.calls[1].tolist() == [9] and np.array_equal(v3, tab[[9, 3]])
    assert orc.v_empty == tab[0] and orc.v_full == tab[255]


@pytest.mark.parametrize("T,seed", [(500, 20260829), (3000, 20260830), (15580, 20260829)])
def test_stratsvarm_split_is_bitwise_the_e4_port(T, seed):
    from e4_stratified_svarm import stratified_svarm
    tab = _table(IID16)
    f, seen = _counting(tab)
    ref = stratified_svarm(f, 16, T, seed, mode="fast", count_empty=True)
    plan = e9.stratsvarm_plan(16, T, seed)
    phi = e9.stratsvarm_aggregate(plan, tab[plan["masks"]] - tab[0])
    assert np.array_equal(phi, ref["phi"])
    assert plan["masks"].size == T
    assert np.unique(plan["masks"]).size == len(seen)
    assert e9.stratsvarm_distinct(16, T, seed) == len(seen)


def test_estimator_arms_make_one_engine_call_and_match_e4():


    import e4_estimators as e4e
    tab = _table(IID16)
    phi_exact = np.zeros(16)
    ctx = e4e.TableContext(IID16, tab, phi_exact, k=50)
    for arm, e4name, T in (("kadd2", "shapiq_kADDSHAP", 2000), ("leverage", "LeverageSHAP", 2000),
                           ("stratsvarm", "StratifiedSVARM", 3000)):
        eng = _FakeEngine(tab)
        te = e9.TimedEngine(eng)
        out = e9.run_arm(arm, te, 16, 50, {"T": T}, 20260829)
        ref = e4e.run_estimator(e4name, ctx, T, 20260829)
        assert out["engine_calls"] == 1 and len(eng.calls) == 1, arm
        assert np.array_equal(out["phi_hat"], ref["phi_hat"]), arm
        assert out["n_requested"] == ref["n_total"], arm
        assert out["n_engine"] == ref["n_unique"], arm
        if arm in ("kadd2", "leverage"):
            assert out["n_engine"] == out["n_requested"] == T, arm
        hs = out["host_split"]
        assert all(v >= 0 for v in hs.values())


@pytest.mark.parametrize("iid,n", [(IID16, 16), (IID20, 20)])
def test_paired_distinct_curve_matches_sampler(iid, n):
    tab = _table(iid)
    for seed in (20260829, 20260831):
        curve = e9.paired_distinct_curve(n, seed, 300)
        for m in (1, 17, 96, 300):
            eng = rb.TableEngine(tab)
            te = e9.TimedEngine(eng, keep=False)
            out = e9.run_arm("paired_d", te, n, 50, {"m": m}, seed)
            assert out["n_engine"] == curve[m - 1], (seed, m)
            assert out["n_requested"] == 2 + 2 * m * (n + 1)


def test_paired_arm_equals_e4_fixed_budget_arm():
    import e4_estimators as e4e
    tab = _table(IID16)
    ctx = e4e.TableContext(IID16, tab, np.zeros(16), k=50)
    T = 14128
    m = (T - 2) // (2 * 17)
    ref = e4e.run_estimator("cubcr_paired_perm", ctx, T, 20260830)
    out = e9.run_arm("paired_d", e9.TimedEngine(rb.TableEngine(tab)), 16, 50, {"m": m}, 20260830)
    assert np.array_equal(out["phi_hat"], ref["phi_hat"])
    assert out["n_engine"] == ref["n_unique"] and out["n_requested"] == ref["n_total"]


def test_calibration_helpers():
    r = e9.paired_m_for_distinct(16, 4096, (20260829, 20260830, 20260831))
    curves = [e9.paired_distinct_curve(16, s, r["m"]) for s in (20260829, 20260830, 20260831)]
    med = np.median([c[-1] for c in curves]); med_prev = np.median([c[-2] for c in curves])
    assert med >= 4096 > med_prev
    assert e9.paired_m_requested(16, 4096) == (4096 - 2) // 34
    c = e9.stratsvarm_nominal_for_distinct(16, 4096, (20260829, 20260830, 20260831))
    assert c["median_distinct"] >= 4096
    assert np.median([e9.stratsvarm_distinct(16, c["T"] - 1, s) for s in (20260829, 20260830, 20260831)]) < 4096 or \
        c["T"] == max(4096, 118)


def test_resolve_union_on_table():
    tab = _table(IID16)
    te = e9.TimedEngine(rb.TableEngine(tab))
    e9.run_arm("kadd2", te, 16, 50, {"T": 1000}, 20260829)
    rs = e9.resolve_union(te, 50, n_rep=2)
    assert len(rs) == 2 and all(r["same_values"] for r in rs) and rs[0]["B"] == 1000


def _gpu_available():
    try:
        from cubcr.engine_gpu import device_count
        return device_count() > 0
    except Exception:
        return False


def _holds_gpu_lock(pid: int) -> bool:


    try:
        ino = os.stat(rb.LOCK_DIR / "gpu.lock").st_ino
        for line in open("/proc/locks"):
            parts = line.split()
            if "->" in parts:
                continue
            if len(parts) >= 6 and parts[1] == "FLOCK" and parts[3] == "WRITE" and parts[4] == str(pid) \
                    and parts[5].split(":")[-1] == str(ino):
                return True
    except OSError:
        pass
    return False


def _gpu_lock(tag):


    import contextlib
    holder = os.environ.get("E9_GPU_LOCK_HELD_BY_PARENT")
    if holder and _holds_gpu_lock(int(holder)):
        return contextlib.nullcontext()
    return rb.gpu_lock(tag)


def test_holds_gpu_lock_rejects_non_holders():

    assert not _holds_gpu_lock(os.getpid())
    assert not _holds_gpu_lock(1)


@pytest.mark.skipif(not _gpu_available(), reason="no CUDA device")
def test_gpu_twin_bitwise():

    tab = _table(IID16)
    task = [t for t in rb.paper_tasks(include_records=False) if t["iid"] == IID16][0]
    comp, _, _, _ = rb.build_task(task)
    nominal = {"paired_d": {"m": 120}, "paired_r": {"m": 60}, "kadd2": {"T": 3000}, "leverage": {"T": 3000},
               "stratsvarm": {"T": 4000}}
    with _gpu_lock("E9 pytest twin"):
        for arm in e9.ARMS:
            eng = rb.make_gpu_engine(comp)
            try:
                rb.paper_warmup(eng, 16, 50)
                te = e9.TimedEngine(eng)
                g = e9.run_arm(arm, te, 16, 50, nominal[arm], 20260829)
                rs = e9.resolve_union(te, 50, n_rep=1)
            finally:
                eng.close()
            t = e9.run_arm(arm, e9.TimedEngine(rb.TableEngine(tab, compiled=comp)), 16, 50, nominal[arm], 20260829)
            assert np.array_equal(g["phi_hat"], t["phi_hat"]), arm
            assert g["n_engine"] == t["n_engine"] and g["n_requested"] == t["n_requested"], arm
            assert rs[0]["same_values"], arm
            if g["half_widths"] is not None:
                assert np.array_equal(g["half_widths"], t["half_widths"]), arm
