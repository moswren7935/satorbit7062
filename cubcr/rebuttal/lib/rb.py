from __future__ import annotations

import contextlib
import datetime as _dt
import fcntl
import glob
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[2]
RB = REPO / "rebuttal"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


PAPER_GEN_SEEDS = (20260829, 20260830, 20260831)
PAPER_SAMPLER_SEEDS = (20260829, 20260830, 20260831)
ROUND4_GEN_SEEDS = (20260901, 20260902, 20260903, 20260904, 20260905, 20260906, 20260907)
PAPER_ALPHA = 0.05
PAPER_EPS_DIV = 20
PAPER_PFAIL = 0.05
PAPER_PATTERN = "none"
GRID_N = (16, 20, 24, 28, 32)
GRID_STATES = (1000, 10000)
GRID_K = (50, 100)
E9V3_DIR = REPO / "results" / "e9v3"
LIB_GPU = REPO / "csrc" / "build" / "libcubcr_gpu.so"
LIB_CPU = REPO / "csrc" / "build" / "libcubcr_cpu.so"


PAPER_SAMPLER_KW = dict(kind="agent", batch_perms=32, alpha=PAPER_ALPHA, max_perms=2_000_000, time_budget=900.0,
                        antithetic=True, value_range="monotone", two_sided=True, bonferroni=True, sequential=True,
                        ci="prpl_eb_cs", batch_schedule="geometric")


def paper_sampler_kwargs(eps: float, seed: int, n: int, k: int, **overrides) -> dict:
    kw = dict(PAPER_SAMPLER_KW, eps=float(eps), seed=int(seed), n=int(n), k=int(k))
    kw.update(overrides)
    return kw


def sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_engine_provenance(strict: bool = False) -> dict:


    paths = {"libcubcr_gpu.so": LIB_GPU, "libcubcr_cpu.so": LIB_CPU}
    cur = {name: sha256(path) if path.is_file() else None for name, path in paths.items()}
    ok = all(value is not None for value in cur.values())
    if strict and not ok:
        raise RuntimeError("engine libraries are missing; run make -C csrc all")
    return {"lib_sha256": cur, "local_build_complete": ok}


def git_commit() -> str:
    if not (REPO / ".git").exists():
        return "unknown"
    try:
        result = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=10)
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def load_snapshot(top: int = 6) -> dict:

    out = {"loadavg": list(os.getloadavg()), "time": now_iso(), "nproc": os.cpu_count()}
    try:
        ps = subprocess.run(["ps", "-eo", "pid,pcpu,nlwp,comm", "--sort=-pcpu"], capture_output=True, text=True,
                            timeout=10).stdout.splitlines()[1:top + 1]
        out["top"] = [" ".join(l.split()) for l in ps]
    except Exception:
        out["top"] = None
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10).stdout.strip()
        out["gpu_util_mem"] = q
    except Exception:
        out["gpu_util_mem"] = None
    return out


def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def hardware_tags() -> dict:

    tags = {"host_hash": hashlib.sha256(socket.gethostname().encode()).hexdigest()[:10], "git_commit": git_commit(),
            "python": sys.version.split()[0], "numpy": np.__version__}
    try:
        from cubcr.engine_gpu import device_name
        tags["gpu"] = device_name(0)
    except Exception:
        tags["gpu"] = None
    tags.update(check_engine_provenance(strict=False))
    return tags


LOCK_DIR = RB / "locks"


@contextlib.contextmanager
def _flock(name: str, tag: str = "", shared: bool = False):
    LOCK_DIR.mkdir(parents=True, exist_ok=True)
    path = LOCK_DIR / name
    f = open(path, "a+")
    t0 = time.perf_counter()
    fcntl.flock(f, fcntl.LOCK_SH if shared else fcntl.LOCK_EX)
    waited = time.perf_counter() - t0
    try:
        if not shared:
            f.seek(0); f.truncate(); f.write(f"{os.getpid()} {tag} {now_iso()}\n"); f.flush()
        yield waited
    finally:
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def gpu_lock(tag: str = ""):


    return _flock("gpu.lock", tag)


def host_lock(tag: str = "", shared: bool = False):


    return _flock("host.lock", tag, shared=shared)


def _read(f):
    with open(f) as fh:
        return json.load(fh)


def load_e9v3(arms: Optional[Iterable[str]] = None, paper_only: bool = True) -> List[dict]:

    arms = None if arms is None else set(arms)
    out = []
    for f in sorted(glob.glob(str(E9V3_DIR / "e9v3__*.json"))):
        d = _read(f)
        if not isinstance(d, dict) or "arm" not in d:
            continue
        if paper_only and d.get("gen_seed") not in PAPER_GEN_SEEDS:
            continue
        if arms is not None and d["arm"] not in arms:
            continue
        d["_file"] = os.path.relpath(f, REPO)
        out.append(d)
    return out


def paper_tasks(include_records: bool = True) -> List[dict]:


    recs = load_e9v3(paper_only=True)
    if not recs:
        raise RuntimeError("No primary experiment records; run python experiments/rerun_revised.py --stage primary")
    tasks: Dict[str, dict] = {}
    for r in recs:
        iid = r["instance_id"]
        t = tasks.setdefault(iid, {"iid": iid, "n": int(r["n"]), "states_target": int(r["cell"]["states"]),
                                   "states": int(r["states"]), "k": int(r["k"]), "gen_seed": int(r["gen_seed"]),
                                   "exact_gpu": None, "exact_cpu": None, "cert_gpu": {}, "cert_cpu": {},
                                   "cert_gpu_eblooks": {}, "exact_cpu_genuineTO": None})
        arm = r["arm"]


        if r.get("genuine_capped_run") or str(r.get("_file", "")).endswith("_genuineTO.json"):
            t["exact_cpu_genuineTO"] = r
        elif arm in ("exact_gpu", "exact_cpu"):
            t[arm] = r
        elif arm in ("cert_gpu", "cert_cpu", "cert_gpu_eblooks"):
            t[arm][int(r["seed"])] = r
    out = []
    for t in tasks.values():
        cg = t["cert_gpu"]
        if not cg:
            continue
        r0 = cg[min(cg)]
        t["upsilon"] = float(r0["attributable_value"])
        t["eps20"] = float(r0["eps_abs"])
        phi = None
        for arm in ("exact_gpu", "exact_cpu"):
            r = t[arm]
            if r is not None and r.get("status") == "OK" and "shapley" in r:
                phi = np.asarray(r["shapley"], dtype=np.float64)
                break
        t["phi_exact"] = phi
        t["has_exact_gpu"] = t["exact_gpu"] is not None and t["exact_gpu"].get("status") == "OK"
        t["has_exact_cpu"] = t["exact_cpu"] is not None and t["exact_cpu"].get("status") == "OK"
        if not include_records:
            for arm in ("exact_gpu", "exact_cpu", "cert_gpu", "cert_cpu", "cert_gpu_eblooks", "exact_cpu_genuineTO"):
                t.pop(arm, None)
        out.append(t)
    out.sort(key=lambda t: (t["n"], t["states_target"], t["k"], t["gen_seed"]))
    if not out:
        raise RuntimeError("No primary cert_gpu records; run python experiments/rerun_revised.py --stage primary")
    return out


def cell_of(task) -> tuple:
    return (task["n"], task["states_target"], task["k"])


def cell_label(task_or_cell) -> str:
    n, S, k = cell_of(task_or_cell) if isinstance(task_or_cell, dict) else task_or_cell
    return f"n{n}_S{'1e3' if S == 1000 else '1e4'}_k{k}"


def build_task(task):

    from experiments.grid import build_model
    comp, iid, meta, build_s = build_model("ticketbench", task["n"], task["states_target"], task["k"], task["gen_seed"],
                                           PAPER_PATTERN, PAPER_PFAIL)
    if iid != task["iid"]:
        raise RuntimeError(f"instance id mismatch {iid} != {task['iid']}")
    return comp, iid, meta, build_s


def make_gpu_engine(comp):

    from cubcr.engine_gpu import GPUEngine
    return GPUEngine(comp, dtype="fp64", use_graph=True)


def paper_warmup(eng, n: int, k: int):

    eng.solve(np.random.default_rng(2).integers(0, 1 << n, size=32 * 2 * (n + 1), dtype=np.uint64), k)


class TableEngine:


    dtype = "fp64"

    def __init__(self, table: np.ndarray, compiled=None):
        self.table = np.asarray(table, dtype=np.float64)
        self.compiled = compiled
        self.n_requested = 0
        self.n_calls = 0

    def solve(self, masks, k=None):
        m = np.asarray(masks, dtype=np.uint64).reshape(-1)
        self.n_requested += m.size
        self.n_calls += 1
        return self.table[m.astype(np.int64)]


class RecordingEngine:


    def __init__(self, engine):
        self.engine = engine
        self.compiled = getattr(engine, "compiled", None)
        self.dtype = getattr(engine, "dtype", "fp64")
        self.calls = []

    @property
    def stats(self):
        return getattr(self.engine, "stats", None)

    def solve(self, masks, k=None):
        v = self.engine.solve(masks, k)
        self.calls.append((np.array(masks, dtype=np.uint64, copy=True).reshape(-1), np.array(v, dtype=np.float64, copy=True)))
        return v


class _DictEngine:
    compiled = None
    dtype = "fp64"

    def __init__(self, d):
        self.d = d

    def solve(self, masks, k=None):
        return np.array([self.d[int(x)] for x in np.asarray(masks, dtype=np.uint64).reshape(-1)], dtype=np.float64)


def replay_units(result, calls, N: int, seed: int, k: Optional[int] = None, verify: bool = True):


    from cubcr.shapley_sampled import CoalitionOracle, _perm_marginals
    if not result.antithetic or result.stratify:
        raise ValueError("replay implemented for antithetic, non-stratified runs (the paper's configuration)")
    if seed is None or (result.seed is not None and int(result.seed) != int(seed)):
        raise ValueError("replay needs the integer seed the run was started with (rng overrides are not replayable)")
    if not result.batch_sizes:
        return {"units": np.zeros((0, N)), "batch_bounds": [0], "perms": []}
    table = {}
    for masks, vals in calls:
        for m_, v_ in zip(masks.tolist(), vals.tolist()):
            table[int(m_)] = float(v_)
    orc = CoalitionOracle(_DictEngine(table), N, "agent", n=N, k=k, use_cache=False)
    rng = np.random.default_rng(seed)
    units_l, perms_l, bounds = [], [], [0]
    sum_x = np.zeros(N)
    m = 0
    for P in result.batch_sizes:
        base = np.stack([rng.permutation(N) for _ in range(P)], axis=0)
        perms = np.concatenate([base, base[:, ::-1]], axis=0)
        marg, _, _ = _perm_marginals(orc, perms)
        units = marg.reshape(-1, P, N).mean(axis=0)
        units_l.append(units); perms_l.append(perms)
        sum_x += units.sum(axis=0)
        m += P
        bounds.append(m)
    units = np.concatenate(units_l, axis=0)
    if verify:


        est = sum_x / m
        if m != result.n_samples or not np.array_equal(est, result.estimates):
            raise RuntimeError("replay does not reproduce the sampler's final estimates bitwise")
    return {"units": units, "batch_bounds": bounds, "perms": perms_l}


def run_paper_sampler(engine, n: int, k: int, eps: float, seed: int, record_units: bool = False, **overrides):


    from cubcr.shapley_sampled import sample_shapley
    eng = RecordingEngine(engine) if record_units else engine
    kw = paper_sampler_kwargs(eps, seed, n, k, **overrides)
    t0 = time.perf_counter()
    r = sample_shapley(eng, **kw)
    wall = time.perf_counter() - t0
    rep = replay_units(r, eng.calls, n, seed, k) if record_units else None
    return r, wall, rep


def result_row(r, wall: float) -> dict:

    cert = r.certificate or {}
    return {"status": r.status, "certified": bool(r.certified), "wall_s": float(wall), "n_units": int(r.n_samples),
            "n_perms": int(r.n_perms), "n_solves": int(r.n_solves), "n_requested": int(r.n_coalitions_requested),
            "cache_hits": int(r.cache_hits), "n_batches": int(r.n_batches), "batch_sizes": list(map(int, r.batch_sizes)),
            "engine_s": float(r.engine_time), "v_empty": float(r.v_empty), "v_full": float(r.v_full),
            "eps": None if r.eps is None else float(r.eps), "fp_term": float(r.fp_term),
            "value_range": None if r.value_range is None else [float(x) for x in r.value_range],
            "hw_max": float(np.max(r.half_widths)) if np.all(np.isfinite(r.half_widths)) else None,
            "phi_hat": r.estimates.tolist(), "half_widths": r.half_widths.tolist(),
            "cs_lower": None if r.cs_lower is None else r.cs_lower.tolist(),
            "cs_upper": None if r.cs_upper is None else r.cs_upper.tolist(),
            "cs_anomaly": bool(r.cs_anomaly), "cert_kind": cert.get("kind"),

            "clock_limited": bool(r.status == "time" or (r.time_budget is not None and r.budget_overshoot_s > 0)
                                  or (r.time_budget is not None and wall >= 0.5 * r.time_budget))}


TABLE_DIR = RB / "cache" / "tables"


def table_path(iid: str) -> Path:
    return TABLE_DIR / f"{iid}.npy"


def exact_table(task, comp=None, engine=None, force: bool = False) -> np.ndarray:


    p = table_path(task["iid"])
    if p.exists() and not force:
        return np.load(p)
    n = task["n"]
    if n > 26:
        raise ValueError("table too large")
    own = engine is None
    if comp is None:
        comp, _, _, _ = build_task(task)
    with gpu_lock(f"table {task['iid']}"):
        if p.exists() and not force:
            return np.load(p)
        eng = make_gpu_engine(comp) if own else engine
        vals = eng.solve(np.arange(1 << n, dtype=np.uint64), task["k"])
        if own:
            eng.close()
        vals = np.asarray(vals, dtype=np.float64)
        if task.get("phi_exact") is not None:
            from cubcr.shapley_fast import shapley_from_table_fast
            phi = shapley_from_table_fast(vals, n)
            dev = float(np.max(np.abs(phi - task["phi_exact"])))
            if dev > 1e-12:
                raise RuntimeError(f"table Shapley deviates from stored reference by {dev}")
        TABLE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + f".tmp{os.getpid()}.npy")
        np.save(tmp, vals)
        os.replace(tmp, p)
    return vals


def kendall_tau_a(a: Sequence[float], b: Sequence[float]) -> float:

    a = np.asarray(a, dtype=np.float64); b = np.asarray(b, dtype=np.float64)
    d = len(a)
    if d != len(b):
        raise ValueError("length mismatch")
    if d < 2:
        return 1.0
    iu = np.triu_indices(d, 1)
    sa = np.sign(a[:, None] - a[None, :])[iu]
    sb = np.sign(b[:, None] - b[None, :])[iu]
    conc = int(np.sum((sa * sb) > 0)); disc = int(np.sum((sa * sb) < 0))
    return (conc - disc) / (d * (d - 1) / 2)


def spearman_rho(a, b) -> float:
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


def top_set(values, kk: int) -> np.ndarray:

    return np.argsort(-np.asarray(values, dtype=np.float64), kind="stable")[:kk]


def topk_metrics(phi_hat, phi, lower=None, upper=None, ks=(1, 3), tie_tol: float = 1e-12) -> dict:


    phi_hat = np.asarray(phi_hat, dtype=np.float64); phi = np.asarray(phi, dtype=np.float64)
    out = {}
    amax = np.max(phi)
    out["top1_correct"] = bool(phi[int(np.argmax(phi_hat))] >= amax - tie_tol)
    for kk in ks:
        if kk >= len(phi):
            continue
        Th, T = set(top_set(phi_hat, kk).tolist()), set(top_set(phi, kk).tolist())
        out[f"top{kk}_exact"] = Th == T
        out[f"top{kk}_recall"] = len(Th & T) / kk
        out[f"top{kk}_jaccard"] = len(Th & T) / len(Th | T)
        if lower is not None and upper is not None:
            out[f"sep{kk}"] = cs_separated(lower, upper, kk)
    return out


def cs_separated(lower, upper, kk: int) -> bool:


    lo = np.asarray(lower, dtype=np.float64); hi = np.asarray(upper, dtype=np.float64)
    if lo.shape != hi.shape or lo.ndim != 1:
        raise ValueError("lower/upper must be 1-D arrays of equal length")
    if not (1 <= int(kk)):
        raise ValueError("k must be >= 1")
    if np.any(np.isnan(lo)) or np.any(np.isnan(hi)):
        raise ValueError("NaN interval bound")
    if not (np.all(np.isfinite(lo)) and np.all(np.isfinite(hi))):
        return False
    if np.any(lo > hi):
        raise ValueError("empty interval (lower > upper): cs_anomaly, cannot certify a separation")
    L = np.sort(lo)[::-1]
    U = np.sort(hi)[::-1]
    if kk >= len(L):
        return True
    return bool(L[kk - 1] > U[kk])


def gini(x) -> float:
    x = np.sort(np.clip(np.asarray(x, dtype=np.float64), 0, None))
    if x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.sum((np.arange(1, n + 1)) * x) / (n * x.sum())) - (n + 1) / n)


def results_dir(exp: str) -> Path:
    p = RB / "results" / exp
    (p / "raw").mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def latex_table(df, path, caption: str, label: str, col_format: Optional[str] = None, float_fmt: str = "{:.3g}"):

    cols = list(df.columns)
    col_format = col_format or ("l" * 1 + "r" * (len(cols) - 1))
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{caption}}}", f"\\label{{{label}}}",
             f"\\begin{{tabular}}{{{col_format}}}", "\\toprule", " & ".join(_tex(c) for c in cols) + " \\\\", "\\midrule"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, (float, np.floating)):
                cells.append("--" if not np.isfinite(v) else float_fmt.format(v))
            else:
                cells.append(_tex(str(v)))
        lines.append(" & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    Path(path).write_text("\n".join(lines) + "\n")


def _tex(s: str) -> str:
    s = str(s)
    for a, b in (("\\", "\\textbackslash{}"), ("_", "\\_"), ("%", "\\%"), ("&", "\\&"), ("#", "\\#")):
        s = s.replace(a, b)
    return s


MPLSTYLE = RB / "scripts" / "mplstyle"
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def use_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use(str(MPLSTYLE))
    return plt


def savefig(fig, stem):

    stem = str(stem)
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=200)
