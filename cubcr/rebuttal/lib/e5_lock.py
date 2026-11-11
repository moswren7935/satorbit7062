from __future__ import annotations

import contextlib
import fcntl
import os
from pathlib import Path

import rb

ENV_HOLDER = "E5_GPU_LOCK_HELD_BY"
ENV_WAIT = "E5_PARENT_LOCK_WAIT_S"


def verify_parent_holds(lock_path: Path, holder_pid: str, ppid: int) -> None:
    if str(ppid) != str(holder_pid):
        raise RuntimeError(f"{ENV_HOLDER}={holder_pid} but the parent pid is {ppid}: refusing to bypass the GPU lock")
    txt = Path(lock_path).read_text().split()
    if not txt or txt[0] != str(holder_pid):
        raise RuntimeError(f"GPU lock file names {txt[:1]} as holder, not {holder_pid}: refusing to bypass")
    with open(lock_path, "a+") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        fcntl.flock(f, fcntl.LOCK_UN)
        raise RuntimeError("GPU lock is FREE although a parent hold was announced: refusing to run unlocked")


def gpu_hold(tag: str = "", lock_path: Path | None = None):

    holder = os.environ.get(ENV_HOLDER)
    if not holder:
        return rb.gpu_lock(tag)
    verify_parent_holds(lock_path or (rb.LOCK_DIR / "gpu.lock"), holder, os.getppid())
    return contextlib.nullcontext(float(os.environ.get(ENV_WAIT, "0") or 0.0))


def lock_mode() -> str:
    h = os.environ.get(ENV_HOLDER)
    return f"parent hold (pid {h}, rebuttal/scripts/e5_tail_hold.py)" if h else "own hold (rb.gpu_lock)"
