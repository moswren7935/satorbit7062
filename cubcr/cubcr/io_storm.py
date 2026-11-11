from __future__ import annotations

import os
import re
import shutil
import time
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .model import CSG
from .io_prism import REPO_ROOT, _run_killable, _sha256_file, _parse_elapsed, _ELAPSED_RE, _RSS_RE
from . import io_prism

STORM_BIN = os.environ.get("CUBCR_STORM_BIN",
                           os.path.join(REPO_ROOT, "external", "storm", "build", "bin", "storm"))
DEFAULT_TIMEOUT_S = 600.0

_RESULT_RE = re.compile(r"Result \(for initial states\):\s*([-+0-9.eE]+)")
_CONSTR_RE = re.compile(r"Time for model construction:\s*([0-9.eE+-]+)\s*s")
_CHECK_RE = re.compile(r"Time for model checking:\s*([0-9.eE+-]+)\s*s")
_PREPROC_RE = re.compile(r"Time for model preprocessing:\s*([0-9.eE+-]+)\s*s")
_STATES_RE = re.compile(r"^States:\s*(\d+)", re.M)
_TRANS_RE = re.compile(r"^Transitions:\s*(\d+)", re.M)
_CHOICES_RE = re.compile(r"^Choices:\s*(\d+)", re.M)


_ERRLINE_RE = re.compile(r"^\s*(ERROR\b.*|.*\bStormException\b.*|.*terminate called.*|.*what\(\):.*)$", re.M)
_MO_MARKERS = ("bad_alloc", "out of memory", "outofmemory", "memout", "not enough memory")


def resolve_bin(storm_bin: Optional[str] = None) -> str:
    candidate = storm_bin if storm_bin else STORM_BIN
    return shutil.which(candidate) or candidate


def storm_available(storm_bin: Optional[str] = None) -> bool:
    b = resolve_bin(storm_bin)
    return os.path.isfile(b) and os.access(b, os.X_OK)


_TOOL_IDENTITY: Dict[str, Optional[dict]] = {}


def tool_identity(storm_bin: Optional[str] = None, refresh: bool = False) -> dict:


    b = resolve_bin(storm_bin)
    if b in _TOOL_IDENTITY and not refresh:
        return dict(_TOOL_IDENTITY[b])
    ident: dict = {"storm_bin": b, "available": storm_available(b), "storm_version": None,
                   "storm_version_full": None, "storm_bin_sha256": None}
    if ident["available"]:
        ident["storm_bin_sha256"] = _sha256_file(b)
        try:


            out, err, _rc, timed_out = _run_killable([b, "--version"], 30.0)
            txt = (out or err).strip()
            ident["storm_version"] = txt.splitlines()[0] if txt else None
            ident["storm_version_full"] = txt
            if timed_out:
                ident["storm_version"] = "unavailable (timeout on --version)"
        except Exception as e:
            ident["storm_version"] = f"unavailable ({type(e).__name__})"
    _TOOL_IDENTITY[b] = ident
    return dict(ident)


def parse_storm_stdout(out: str, err: str = "") -> dict:


    def _last(rx, text, conv):
        ms = rx.findall(text)
        if not ms:
            return None
        try:
            return conv(ms[-1])
        except ValueError:
            return None
    d = {"storm_construction_s": _last(_CONSTR_RE, out, float),
         "storm_preprocessing_s": _last(_PREPROC_RE, out, float),
         "storm_checking_s": _last(_CHECK_RE, out, float),
         "storm_states": _last(_STATES_RE, out, int),
         "storm_transitions": _last(_TRANS_RE, out, int),
         "storm_choices": _last(_CHOICES_RE, out, int),
         "value": _last(_RESULT_RE, out, float),
         "peak_rss_kb": _last(_RSS_RE, err, int),
         "time_elapsed_s": None}
    m = _ELAPSED_RE.search(err)
    if m:
        d["time_elapsed_s"] = _parse_elapsed(m.group(1))
    c, k = d["storm_construction_s"], d["storm_checking_s"]
    d["storm_reported_time_s"] = (c + k) if (c is not None and k is not None) else None
    return d


def classify_storm_status(timed_out: bool, rc: int, out: str, err: str, value) -> Tuple[str, Optional[str]]:


    text = out + "\n" + err
    low = text.lower()
    if timed_out:
        return "TO", "killed by cuBCR timeout (process group SIGKILL)"
    if any(mk in low for mk in _MO_MARKERS):
        for ln in text.splitlines():
            if any(mk in ln.lower() for mk in _MO_MARKERS):
                return "MO", ln.strip()
        return "MO", "out of memory"
    m = _ERRLINE_RE.search(out) or _ERRLINE_RE.search(err)
    if rc != 0 or value is None or m is not None:
        if m is not None:
            return "ERR", m.group(1).strip()
        if value is None:
            return "ERR", f"no 'Result (for initial states):' line in Storm output (rc={rc})"
        return "ERR", f"non-zero exit status rc={rc}"
    return "OK", None


def skip_result(storm_bin: Optional[str] = None, timeout_s: float = DEFAULT_TIMEOUT_S) -> dict:

    b = resolve_bin(storm_bin)
    return {"value": None, "wall_s": None, "storm_reported_time_s": None,
            "storm_construction_s": None, "storm_preprocessing_s": None, "storm_checking_s": None,
            "storm_states": None, "storm_transitions": None, "storm_choices": None,
            "peak_rss_kb": None, "time_elapsed_s": None,
            "status": "SKIP", "error": f"storm binary missing at {b}",
            "timed_out": False, "timeout_s": float(timeout_s),
            "cmd": None, "cmd_argv": None, "rc": None, "returncode": None,
            "log_path": None, "stdout_path": None, "model_path": None, "props_path": None,
            "tool_identity": tool_identity(b)}


def run_storm(mdp_text: str, props_text: str, timeout_s: float = DEFAULT_TIMEOUT_S,
              workdir: Optional[str] = None, tag: str = "model", storm_bin: Optional[str] = None,
              extra_args: Sequence[str] = (), include_identity: bool = True) -> dict:


    b = resolve_bin(storm_bin)
    if not storm_available(b):
        return skip_result(b, timeout_s)
    if workdir is None:
        workdir = os.path.join(REPO_ROOT, "results", "storm_work")
    os.makedirs(workdir, exist_ok=True)
    model_path = os.path.join(workdir, f"{tag}.nm")
    props_path = os.path.join(workdir, f"{tag}.props")
    log_path = os.path.join(workdir, f"{tag}.log")
    with open(model_path, "w") as f:
        f.write(mdp_text)
    with open(props_path, "w") as f:
        f.write(props_text)
    cmd = [b, "--prism", model_path, "--prop", props_path] + list(extra_args)
    use_time = shutil.which("time") or ("/usr/bin/time" if os.path.exists("/usr/bin/time") else None)
    if use_time:
        cmd = [use_time, "-v"] + cmd
    t0 = time.perf_counter()
    out, err, rc, timed_out = _run_killable(cmd, timeout_s)
    wall = time.perf_counter() - t0
    with open(log_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n=== STDOUT ===\n" + out + "\n=== STDERR ===\n" + err)
    parsed = parse_storm_stdout(out, err)
    value = parsed.pop("value")
    status, error = classify_storm_status(timed_out, rc, out, err, value)
    if status == "OK" and not (value is not None and np.isfinite(value)):
        status, error = "ERR", f"non-finite result {value!r}"
    res = {"value": value, "wall_s": wall, "status": status, "error": error,
           "timed_out": timed_out, "timeout_s": float(timeout_s),
           "cmd": " ".join(cmd), "cmd_argv": list(cmd), "rc": rc, "returncode": rc,
           "log_path": log_path, "stdout_path": log_path,
           "model_path": model_path, "props_path": props_path}
    res.update(parsed)
    if include_identity:
        res["tool_identity"] = tool_identity(b)
    return res


def storm_value(csg: CSG, mask: int, k: Optional[int] = None, **kw) -> dict:

    mt, pr = io_prism.emit_coalition_mdp(csg, mask, k)
    tag = kw.pop("tag", f"coal_{int(mask)}")
    return run_storm(mt, pr, tag=tag, **kw)
