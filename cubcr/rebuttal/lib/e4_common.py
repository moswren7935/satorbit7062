from __future__ import annotations

import functools
import gzip
import io
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import rb

E4 = rb.RB / "results" / "E4"
RAW = E4 / "raw"
LOGS = E4 / "logs"

SEEDS30 = [20260829 + i for i in range(30)]
PAPER_SEEDS = list(rb.PAPER_SAMPLER_SEEDS)
assert SEEDS30[:3] == PAPER_SEEDS

CORE_ESTIMATORS = ["cubcr_paired_perm", "plain_perm", "StratifiedSVARM", "shapiq_UnbiasedKernelSHAP", "LeverageSHAP",
                   "shapiq_kADDSHAP"]
EXTRA_ESTIMATORS = ["shapiq_SVARM", "shapiq_KernelSHAP", "shapiq_PermutationSamplingSV", "shapiq_OwenSamplingSV",
                    "shapiq_StratifiedSamplingSV", "Adalina"]


def set_single_thread_env():
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


@functools.lru_cache(maxsize=1)
def load_games() -> dict:
    d = json.load(open(RAW / "games.json"))
    return {g["table_id"]: g for g in d["games"]}


def game_list(ds=None) -> list:
    gs = list(load_games().values())
    if ds is not None:
        gs = [g for g in gs if g["d"] in ds]
    return sorted(gs, key=lambda g: (g["d"], g["table_id"]))


@functools.lru_cache(maxsize=1)
def load_fp_bounds() -> dict:

    import pickle
    with open(RAW / "fp_bounds.pkl", "rb") as f:
        return pickle.load(f)


@functools.lru_cache(maxsize=4)
def table_context(table_id: str):


    from e4_estimators import TableContext
    g = load_games()[table_id]
    tab = np.load(rb.REPO / g["path"])
    fp = load_fp_bounds()[table_id]
    return TableContext(table_id, tab, np.asarray(g["phi_exact"]), k=int(g["k"]), comp=None, fp=fp)


def write_csv_gz(df: pd.DataFrame, path: Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with gzip.open(tmp, "wt") as f:
        df.to_csv(f, index=False)
    os.replace(tmp, path)


def read_csv_gz(path, **kw) -> pd.DataFrame:


    return pd.read_csv(path, compression="gzip", float_precision="round_trip", **kw)


def read_csv(path, **kw) -> pd.DataFrame:

    return pd.read_csv(path, float_precision="round_trip", **kw)


def nice_worker():
    try:
        os.nice(19 - os.nice(0))
    except Exception:
        pass


def log_line(logfile, msg: str):
    line = f"[{rb.now_iso()}] {msg}"
    print(line, flush=True)
    if logfile is not None:
        Path(logfile).parent.mkdir(parents=True, exist_ok=True)
        with open(logfile, "a") as f:
            f.write(line + "\n")


def run_chunks(chunks, fn, workers: int, tag: str, logfile=None, round_size: int = 40):


    from joblib import Parallel, delayed, parallel_backend
    chunks = list(chunks)
    done = 0
    for r0 in range(0, len(chunks), round_size):
        part = chunks[r0:r0 + round_size]
        snap0 = rb.load_snapshot()
        with rb.host_lock(f"{tag} round {r0 // round_size}", shared=True) as waited:
            log_line(logfile, f"{tag}: round {r0 // round_size} ({len(part)} chunks) start; host-lock wait "
                              f"{waited:.1f}s; loadavg {snap0['loadavg']}")
            with parallel_backend("loky", inner_max_num_threads=1):
                res = Parallel(n_jobs=workers, verbose=0)(delayed(fn)(*c) for c in part)
        done += len(part)
        for x in res:
            if x:
                log_line(logfile, f"{tag}: {x}")
        log_line(logfile, f"{tag}: {done}/{len(chunks)} chunks done; loadavg {rb.load_snapshot()['loadavg']}")
