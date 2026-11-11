from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


REQUIRED_FIELDS = (
    "instance_id", "engine", "dtype", "n", "states", "k", "seed",
    "gpu", "cuda", "git",
    "wall_s", "coalition_solves_per_s", "speedup_vs_cpu",
    "max_abs_err_vs_fp64", "kendall_tau", "timestamp",

    "cap_s", "threads", "censored",
)

AUTO_FIELDS = ("timestamp", "gpu", "cuda", "git", "machine", "env", "schema_version")

OPTIONAL_FIELDS = (
    "n_coalitions",
    "n_repeat", "warmup",
    "all_s", "min_s", "max_s",
    "load_s",
    "status",
    "speedup_is_lower_bound",
    "block", "tpb", "max_batch", "use_graph", "waves",
    "notes",
)
SCHEMA_VERSION = 1

ENGINES = ("cpu_multicore", "gpu_batched", "gpu_unbatched", "prism_games", "naive_python", "mdpax", "storm")
DTYPES = ("fp64", "fp32", "n/a")
STATUSES = ("OK", "TO", "CRASH", "MO", "NOT RUN")


def _run(cmd, timeout=15, cwd=None) -> str | None:

    try:
        if isinstance(cmd, str):
            cmd = cmd.split()
        if shutil.which(cmd[0]) is None:
            return None
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd)
        if out.returncode != 0:
            return None
        return out.stdout.strip()
    except Exception:
        return None


def git_commit(short=False) -> str | None:
    if not (REPO_ROOT / ".git").exists():
        return None
    h = _run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT))
    if h is None:
        return None
    return h[:8] if short else h


def git_dirty() -> bool | None:
    if not (REPO_ROOT / ".git").exists():
        return None
    s = _run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT))
    return None if s is None else bool(s.strip())


def nvidia_smi_info() -> dict:

    info = {"gpu": None, "gpu_mem_mib": None, "driver": None, "cuda": None, "gpu_count": 0}
    q = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
              "--format=csv,noheader,nounits"])
    if q:
        rows = [r.strip() for r in q.splitlines() if r.strip()]
        info["gpu_count"] = len(rows)
        if rows:
            parts = [p.strip() for p in rows[0].split(",")]
            if len(parts) >= 3:
                info["gpu"] = parts[0]
                try:
                    info["gpu_mem_mib"] = int(float(parts[1]))
                except ValueError:
                    info["gpu_mem_mib"] = parts[1]
                info["driver"] = parts[2]
    banner = _run(["nvidia-smi"])
    if banner:
        m = re.search(r"CUDA Version:\s*([0-9.]+)", banner)
        if m:
            info["cuda"] = m.group(1)
    return info


def nvcc_version() -> str | None:
    out = _run(["nvcc", "--version"])
    if not out:
        return None
    m = re.search(r"release\s+([0-9.]+),\s*V([0-9.]+)", out)
    return m.group(2) if m else out.splitlines()[-1].strip()


def cpu_info() -> dict:
    info = {"cpu_model": None, "physical_cores": None, "logical_cpus": os.cpu_count(),
            "sockets": None, "threads_per_core": None, "numa_nodes": None}
    out = _run(["lscpu"])
    if not out:
        info["cpu_model"] = platform.processor() or None
        return info
    kv = {}
    for line in out.splitlines():
        if ":" in line:
            key, val = line.split(":", 1)
            kv[key.strip()] = val.strip()

    def _int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    info["cpu_model"] = kv.get("Model name")
    cores_per_socket = _int(kv.get("Core(s) per socket"))
    sockets = _int(kv.get("Socket(s)"))
    info["sockets"] = sockets
    info["threads_per_core"] = _int(kv.get("Thread(s) per core"))
    info["numa_nodes"] = _int(kv.get("NUMA node(s)"))
    if cores_per_socket is not None and sockets is not None:
        info["physical_cores"] = cores_per_socket * sockets
    elif cores_per_socket is not None:
        info["physical_cores"] = cores_per_socket
    lc = _int(kv.get("CPU(s)"))
    if lc is not None:
        info["logical_cpus"] = lc
    return info


def capture_env() -> dict:

    keys = ("OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES", "OMP_SCHEDULE",
            "CUDA_VISIBLE_DEVICES", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
    return {k: os.environ.get(k) for k in keys}


def machine_tags(use_cubcr_gpu: bool = True) -> dict:


    smi = nvidia_smi_info()
    gpu = smi["gpu"]
    gpu_source = "nvidia-smi" if gpu else None
    if use_cubcr_gpu:
        try:
            sys.path.insert(0, str(REPO_ROOT))
            from cubcr import engine_gpu
            if engine_gpu.device_count() > 0:
                name = engine_gpu.device_name(0)
                if name:
                    gpu, gpu_source = name, "cubcr.engine_gpu.device_name"
        except Exception:
            pass
    cpu = cpu_info()
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "gpu": gpu,
        "gpu_source": gpu_source,
        "gpu_count": smi["gpu_count"],
        "gpu_mem_mib": smi["gpu_mem_mib"],
        "driver": smi["driver"],
        "cuda": smi["cuda"],
        "nvcc": nvcc_version(),
        "cpu_model": cpu["cpu_model"],
        "physical_cores": cpu["physical_cores"],
        "logical_cpus": cpu["logical_cpus"],
        "sockets": cpu["sockets"],
        "numa_nodes": cpu["numa_nodes"],
        "python": platform.python_version(),
        "numpy": np.__version__,
        "git": git_commit(),
        "git_dirty": git_dirty(),
    }


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def timed_run(fn, n_repeat: int = 5, warmup: int = 1, timer=time.perf_counter, sync=None) -> dict:


    if n_repeat < 1:
        raise ValueError("n_repeat must be >= 1")
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    warm = []
    for _ in range(warmup):
        t0 = timer()
        fn()
        if sync is not None:
            sync()
        warm.append(timer() - t0)
    all_s = []
    result = None
    for _ in range(n_repeat):
        t0 = timer()
        result = fn()
        if sync is not None:
            sync()
        all_s.append(timer() - t0)
    arr = np.asarray(all_s, dtype=np.float64)
    return {
        "median_s": float(np.median(arr)),
        "min_s": float(arr.min()),
        "max_s": float(arr.max()),
        "mean_s": float(arr.mean()),
        "all_s": [float(x) for x in all_s],
        "warmup_s": [float(x) for x in warm],
        "n_repeat": int(n_repeat),
        "warmup": int(warmup),
        "result": result,
    }


def _worker_thread(fn, args, kwargs, box):
    try:
        box["result"] = fn(*args, **kwargs)
        box["status"] = "OK"
    except BaseException as exc:
        box["status"] = "ERR"
        box["error"] = f"{type(exc).__name__}: {exc}"


def _worker_process(fn, args, kwargs, q):
    try:
        q.put(("OK", fn(*args, **kwargs), None))
    except BaseException as exc:
        q.put(("ERR", None, f"{type(exc).__name__}: {exc}"))


def run_with_timeout(fn, cap_s: float, *args, mode: str = "thread", ctx: str = "fork", **kwargs) -> dict:


    if cap_s is None or cap_s <= 0:
        raise ValueError("cap_s must be a positive number of seconds")
    t0 = time.perf_counter()
    if mode == "thread":
        box: dict = {"status": "TO", "result": None, "error": None}
        th = threading.Thread(target=_worker_thread, args=(fn, args, kwargs, box), daemon=True)
        th.start()
        th.join(cap_s)
        elapsed = time.perf_counter() - t0
        if th.is_alive():
            return {"status": "TO", "result": None, "elapsed_s": elapsed, "cap_s": float(cap_s),
                    "error": f"time-out after {cap_s} s (worker thread still running)",
                    "censored": True}
        return {"status": box["status"], "result": box.get("result"), "elapsed_s": elapsed,
                "cap_s": float(cap_s), "error": box.get("error"), "censored": False}
    if mode == "process":
        import multiprocessing as mp
        import queue as _queue
        if ctx not in ("fork", "spawn", "forkserver"):
            raise ValueError(f"unknown ctx {ctx!r} (expected 'fork', 'spawn' or 'forkserver')")
        mpctx = mp.get_context(ctx)
        q = mpctx.Queue()
        p = mpctx.Process(target=_worker_process, args=(fn, args, kwargs, q), daemon=True)
        p.start()


        got = None
        while True:
            remaining = cap_s - (time.perf_counter() - t0)
            if remaining <= 0:
                break
            try:
                got = q.get(timeout=min(0.05, remaining))
                break
            except _queue.Empty:
                if not p.is_alive():

                    try:
                        got = q.get(timeout=0.05)
                    except _queue.Empty:
                        got = None
                    break
        elapsed = time.perf_counter() - t0
        if got is None and p.is_alive():
            p.kill()
            p.join(5)
            return {"status": "TO", "result": None, "elapsed_s": elapsed, "cap_s": float(cap_s),
                    "error": f"time-out after {cap_s} s (child process killed)", "censored": True}
        p.join(5)
        if got is None:
            status, result, err = "ERR", None, f"child exited with code {p.exitcode} and no result"
        else:
            status, result, err = got
        return {"status": status, "result": result, "elapsed_s": elapsed, "cap_s": float(cap_s),
                "error": err, "censored": False}
    raise ValueError(f"unknown mode {mode!r} (expected 'thread' or 'process')")


def make_record(**fields) -> dict:

    rec = {
        "schema_version": SCHEMA_VERSION,
        "status": "OK",
        "censored": False,
        "speedup_vs_cpu": None,
        "max_abs_err_vs_fp64": None,
        "kendall_tau": None,
        "coalition_solves_per_s": None,
        "speedup_is_lower_bound": False,
    }
    rec.update(fields)
    return rec


def _is_number(x) -> bool:
    return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, bool)


def validate_record(record: dict) -> None:

    if not isinstance(record, dict):
        raise ValueError("record must be a dict")
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"result record is missing required field(s): {missing}")
    if record["engine"] not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {record['engine']!r}")
    if record["dtype"] not in DTYPES:
        raise ValueError(f"dtype must be one of {DTYPES}, got {record['dtype']!r}")
    for f in ("n", "states", "k", "seed"):
        if not _is_number(record[f]):
            raise ValueError(f"field {f!r} must be numeric, got {record[f]!r}")
    if not _is_number(record["cap_s"]) or record["cap_s"] <= 0:
        raise ValueError("cap_s must be a positive number (the time limit in force for the run)")
    if not _is_number(record["threads"]) or record["threads"] < 0:
        raise ValueError("threads must be a non-negative integer (CPU threads used; 0 for GPU-only)")
    if not isinstance(record["censored"], bool):
        raise ValueError("censored must be a bool")
    status = record.get("status", "OK")
    if status not in STATUSES:
        raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
    if _is_number(record["wall_s"]) and not (record["wall_s"] >= 0 and np.isfinite(record["wall_s"])):
        raise ValueError(f"wall_s must be a finite non-negative number, got {record['wall_s']!r}")
    if status == "TO" and not record["censored"]:
        raise ValueError("status='TO' requires censored=True (a run stopped by its cap is censored)")
    if record["censored"]:
        if status != "TO":
            raise ValueError("censored=True requires status='TO'")
        if not _is_number(record["wall_s"]) or record["wall_s"] < record["cap_s"]:
            raise ValueError("censored run must report wall_s >= cap_s (it is a lower bound)")
        if record.get("coalition_solves_per_s") is not None:
            raise ValueError("censored run must not report a throughput (set to null)")
        if record.get("speedup_vs_cpu") is not None:
            raise ValueError("censored run must not report a speedup (set to null)")
    elif status == "OK" and not _is_number(record["wall_s"]):
        raise ValueError("uncensored OK run must report a numeric wall_s")


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _safe(s) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(s))


def write_result(path_dir, record: dict, filename: str | None = None, tags: dict | None = None) -> Path:


    rec = dict(record)
    if tags is None and any(f not in rec for f in ("gpu", "cuda", "git", "machine")):
        tags = machine_tags()
    if tags is not None:
        rec.setdefault("gpu", tags.get("gpu"))
        rec.setdefault("cuda", tags.get("cuda"))
        rec.setdefault("git", tags.get("git"))
        rec.setdefault("machine", tags)
    rec.setdefault("timestamp", now_iso())
    rec.setdefault("env", capture_env())
    rec.setdefault("schema_version", SCHEMA_VERSION)
    rec.setdefault("status", "TO" if rec.get("censored") else "OK")
    validate_record(rec)
    path_dir = Path(path_dir)
    path_dir.mkdir(parents=True, exist_ok=True)
    if filename is None:
        ts = re.sub(r"[^0-9T]", "", rec["timestamp"].replace("-", "").replace(":", ""))[:15]
        filename = "_".join(_safe(x) for x in (rec["instance_id"], rec["engine"], rec["dtype"],
                                                f"s{rec['seed']}", ts)) + ".json"
        i, base = 1, filename
        while (path_dir / filename).exists():
            filename = base[:-5] + f"_{i}.json"
            i += 1
    path = path_dir / filename
    with open(path, "w") as fh:
        json.dump(rec, fh, indent=1, sort_keys=True, default=_json_default)
    return path


def read_result(path) -> dict:
    with open(path) as fh:
        return json.load(fh)


def format_wall(record: dict) -> str:

    if record.get("censored"):
        return f">= {float(record['cap_s']):g} s"
    st = record.get("status", "OK")
    if st != "OK":
        return st
    return f"{float(record['wall_s']):.4g} s"


def format_speedup(record: dict) -> str:
    sp = record.get("speedup_vs_cpu")
    if sp is None:
        return "n/a"
    return (">= " if record.get("speedup_is_lower_bound") else "") + f"{float(sp):.2f}x"


def throughput(n_solves: int, wall_s: float) -> float:
    return float(n_solves) / float(wall_s) if wall_s > 0 else float("nan")


__all__ = [
    "REQUIRED_FIELDS", "OPTIONAL_FIELDS", "ENGINES", "DTYPES", "STATUSES", "SCHEMA_VERSION",
    "machine_tags", "capture_env", "nvidia_smi_info", "nvcc_version", "cpu_info", "git_commit",
    "timed_run", "run_with_timeout", "make_record", "validate_record", "write_result",
    "read_result", "format_wall", "format_speedup", "throughput", "now_iso",
]


if __name__ == "__main__":
    print(json.dumps({"machine": machine_tags(), "env": capture_env()}, indent=1, default=_json_default))
