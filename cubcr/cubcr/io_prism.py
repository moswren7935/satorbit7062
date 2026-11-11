from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import shutil
import signal
import subprocess
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from .model import CSG, StateDef, DMAX, AMAX
from . import vi_ref

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PRISM_BIN = os.environ.get("CUBCR_PRISM_BIN",
                           os.path.join(REPO_ROOT, "external", "prism-games", "prism", "bin", "prism"))
PRISM_BIN = shutil.which(PRISM_BIN) or PRISM_BIN
JAVA_HOME = os.environ.get("CUBCR_JAVA_HOME") or os.environ.get("JAVA_HOME", "")


EMPTY_COALITION_PLAYER = 0


def prism_available() -> bool:
    return os.path.isfile(PRISM_BIN) and os.access(PRISM_BIN, os.X_OK)


def _fmt(p: float) -> str:

    s = repr(float(p))
    if "e" in s or "E" in s:
        s = f"{float(p):.17f}"
    return s


def _or_states(states: Sequence[int]) -> str:
    if not states:
        return "false"
    return "(" + "|".join(f"s={s}" for s in states) + ")"


def _player_name(csg: CSG, i: int) -> str:
    return f"p{i}"


def _mixed_rows(csg: CSG, s: int, mask: int):


    d = csg.state_defs[s]
    slots = list(range(len(d.active)))
    free = [i for i in slots if not ((mask >> d.active[i]) & 1)]
    pinned = [i for i in slots if (mask >> d.active[i]) & 1]
    out = []
    for fa in itertools.product(*[range(d.n_actions[i]) for i in free]):
        acc: Dict[int, float] = {}
        for pa in itertools.product(*[range(d.n_actions[i]) for i in pinned]):
            joint = [0] * len(slots)
            for i, a in zip(free, fa):
                joint[i] = a
            w = 1.0
            for i, a in zip(pinned, pa):
                joint[i] = a
                w *= float(csg.observed_policy[(s, d.active[i])][a])
            if w == 0.0:
                continue
            idx = 0
            for i in slots:
                idx = idx * d.n_actions[i] + joint[i]
            _, succ = d.rows[idx]
            for sp, p in succ:
                if p != 0.0:
                    acc[sp] = acc.get(sp, 0.0) + w * p
        out.append(({d.active[i]: a for i, a in zip(free, fa)}, acc))
    return out


def emit_coalition_model(csg: CSG, mask: int, k: Optional[int] = None,
                         comment: str = "") -> Tuple[str, str]:


    csg.validate()
    n, S = csg.n_agents, csg.n_states
    if k is None:
        k = csg.horizon_k
    mask = int(mask)
    fail = set(csg.fail_states)
    avoid = set(csg.avoid_states)
    special = fail | avoid
    pinned = [i for i in range(n) if (mask >> i) & 1]
    free = [i for i in range(n) if not ((mask >> i) & 1)]


    act_states: Dict[Tuple[int, int], List[int]] = {}
    idle_states: Dict[int, List[int]] = {i: [] for i in range(n)}
    for s in range(S):
        d = csg.state_defs[s]
        active = {} if s in special else {a: m for a, m in zip(d.active, d.n_actions)}
        for i in free:
            if i in active:
                for a in range(active[i]):
                    act_states.setdefault((i, a), []).append(s)
            else:
                idle_states[i].append(s)

    def act_name(i: int, a: int) -> str:
        return f"a{i}_{a}"

    lines: List[str] = []
    lines.append(f"// cuBCR coalition model{(': ' + comment) if comment else ''}")
    lines.append(f"// mask={mask} pinned={pinned} free={free} k={k}")
    lines.append("csg")
    lines.append("")
    for i in range(n):
        lines.append(f"player {_player_name(csg, i)} m{i} endplayer")
    lines.append("")
    for i in range(n):
        lines.append(f"module m{i}")
        if i in pinned:
            lines.append(f"\t[pin{i}] true -> true;")
        else:
            for a in range(AMAX):
                st = act_states.get((i, a))
                if st:
                    lines.append(f"\t[{act_name(i, a)}] {_or_states(st)} -> true;")
            if idle_states[i]:
                lines.append(f"\t[idle{i}] {_or_states(idle_states[i])} -> true;")
        lines.append("endmodule")
        lines.append("")
    lines.append("module env")
    lines.append(f"\ts : [0..{S - 1}] init {csg.init_state};")

    def joint_label(s: int, free_joint: Dict[int, int]) -> str:
        parts = []
        for i in range(n):
            if i in pinned:
                parts.append(f"pin{i}")
            elif i in free_joint:
                parts.append(act_name(i, free_joint[i]))
            else:
                parts.append(f"idle{i}")
        return "[" + ",".join(parts) + "]"

    for s in range(S):
        if s in special:
            lines.append(f"\t{joint_label(s, {})} s={s} -> (s'={s});")
            continue
        for fj, acc in _mixed_rows(csg, s, mask):
            upd = " + ".join(f"{_fmt(p)}:(s'={sp})" for sp, p in sorted(acc.items()))
            lines.append(f"\t{joint_label(s, fj)} s={s} -> {upd};")
    lines.append("endmodule")
    lines.append("")
    lines.append(f'label "fail" = {_or_states(sorted(fail))};')
    if avoid:
        lines.append(f'label "avoid" = {_or_states(sorted(avoid))};')
    prism_text = "\n".join(lines) + "\n"

    if free:
        coal = "<<" + ",".join(_player_name(csg, i) for i in free) + ">>"
    else:
        coal = "<<" + _player_name(csg, EMPTY_COALITION_PLAYER) + ">>"
    props_text = f'{coal} Pmin=? [ F<={k} "fail" ]\n'
    return prism_text, props_text


def emit_coalition_mdp(csg: CSG, mask: int, k: Optional[int] = None,
                       comment: str = "") -> Tuple[str, str]:


    csg.validate()
    n, S = csg.n_agents, csg.n_states
    if k is None:
        k = csg.horizon_k
    mask = int(mask)
    fail = set(csg.fail_states)
    avoid = set(csg.avoid_states)
    special = fail | avoid
    pinned = [i for i in range(n) if (mask >> i) & 1]
    free = [i for i in range(n) if not ((mask >> i) & 1)]
    lines: List[str] = []
    lines.append(f"// cuBCR coalition MDP (collapsed CSG){(': ' + comment) if comment else ''}")
    lines.append(f"// mask={mask} pinned={pinned} free={free} k={k}")
    lines.append("// one command per (state, free joint action); pinned agents' observed mixture folded in")
    lines.append("mdp")
    lines.append("")
    lines.append("module env")
    lines.append(f"\ts : [0..{S - 1}] init {csg.init_state};")
    for s in range(S):
        if s in special:
            lines.append(f"\t[] s={s} -> 1.0:(s'={s});")
            continue
        for _fj, acc in _mixed_rows(csg, s, mask):
            upd = " + ".join(f"{_fmt(p)}:(s'={sp})" for sp, p in sorted(acc.items()))
            lines.append(f"\t[] s={s} -> {upd};")
    lines.append("endmodule")
    lines.append("")
    lines.append(f'label "fail" = {_or_states(sorted(fail))};')
    if avoid:
        lines.append(f'label "avoid" = {_or_states(sorted(avoid))};')
    mdp_text = "\n".join(lines) + "\n"
    props_text = f'Pmin=? [ F<={k} "fail" ]\n'
    return mdp_text, props_text


DEFAULT_JAVAMAXMEM = os.environ.get("CUBCR_PRISM_JAVAMAXMEM", "16g")
DEFAULT_MAXITERS = 100000
DEFAULT_TIMEOUT_S = 600.0

_RESULT_RE = re.compile(r"^Result:\s*([-+0-9.eEinfNaIty]+)", re.M)
_RSS_RE = re.compile(r"Maximum resident set size \(kbytes\):\s*(\d+)")
_ELAPSED_RE = re.compile(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([0-9:.]+)")
_CONSTR_RE = re.compile(r"Time for model construction:\s*([0-9.eE+-]+)\s*seconds")
_CHECK_RE = re.compile(r"Time for model checking:\s*([0-9.eE+-]+)\s*seconds")
_STATES_RE = re.compile(r"^States:\s*(\d+)", re.M)
_TRANS_RE = re.compile(r"^Transitions:\s*(\d+)", re.M)
_CHOICES_RE = re.compile(r"^Choices:\s*(\d+)", re.M)
_ERRLINE_RE = re.compile(r"^\s*(Error\b.*|.*\bException\b.*|Exception in thread.*|.*OutOfMemoryError.*)$", re.M)
_MO_MARKERS = ("outofmemoryerror", "out of memory")


def prism_env() -> Dict[str, str]:
    env = dict(os.environ)
    if os.path.isdir(JAVA_HOME):
        env["JAVA_HOME"] = JAVA_HOME
        env["PATH"] = os.path.join(JAVA_HOME, "bin") + os.pathsep + env.get("PATH", "")
    return env


def _run_killable(cmd: Sequence[str], timeout_s: float):


    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env=prism_env(), start_new_session=True)
    try:
        out, err = proc.communicate(timeout=timeout_s)
        return out, err, proc.returncode, False
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            proc.kill()
        out, err = proc.communicate()
        return out or "", err or "", -1, True


def _sha256_file(path: Optional[str]) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_tree(root: str, suffix: str = ".class") -> Tuple[Optional[str], int]:

    if not os.path.isdir(root):
        return None, 0
    h = hashlib.sha256()
    n = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames.sort()
        for fn in sorted(filenames):
            if not fn.endswith(suffix):
                continue
            p = os.path.join(dirpath, fn)
            h.update(os.path.relpath(p, root).encode() + b"\0")
            with open(p, "rb") as f:
                h.update(f.read())
            h.update(b"\0")
            n += 1
    return (h.hexdigest() if n else None), n


_TOOL_IDENTITY: Optional[dict] = None


def tool_identity(refresh: bool = False) -> dict:


    global _TOOL_IDENTITY
    if _TOOL_IDENTITY is not None and not refresh:
        return dict(_TOOL_IDENTITY)
    prism_dir = os.path.dirname(os.path.dirname(os.path.abspath(PRISM_BIN)))
    ident: dict = {"prism_bin": PRISM_BIN, "prism_dir": prism_dir, "prism_version": None,
                   "java_home": JAVA_HOME if os.path.isdir(JAVA_HOME) else None, "java_bin": None,
                   "java_version": None, "prism_launcher_sha256": _sha256_file(PRISM_BIN),
                   "prism_jar": None, "prism_jar_sha256": None, "prism_classes_sha256": None,
                   "prism_classes_n_files": 0, "libprism_so_sha256": None, "prism_git": None}
    env = prism_env()
    if prism_available():
        try:
            r = subprocess.run([PRISM_BIN, "-version"], capture_output=True, text=True, env=env, timeout=120)
            ident["prism_version"] = (r.stdout.strip() or r.stderr.strip()).splitlines()[0] if (r.stdout.strip() or r.stderr.strip()) else None
        except Exception as e:
            ident["prism_version"] = f"unavailable ({type(e).__name__})"
    java = shutil.which("java", path=env.get("PATH"))
    ident["java_bin"] = java
    if java:
        try:
            r = subprocess.run([java, "-version"], capture_output=True, text=True, env=env, timeout=60)
            txt = (r.stderr or r.stdout).strip()
            ident["java_version"] = txt.splitlines()[0] if txt else None
            ident["java_version_full"] = txt
        except Exception as e:
            ident["java_version"] = f"unavailable ({type(e).__name__})"
    jar = os.path.join(prism_dir, "lib", "prism.jar")
    if os.path.isfile(jar):
        ident["prism_jar"] = jar
        ident["prism_jar_sha256"] = _sha256_file(jar)
    digest, n = _sha256_tree(os.path.join(prism_dir, "classes"))
    ident["prism_classes_sha256"], ident["prism_classes_n_files"] = digest, n
    ident["libprism_so_sha256"] = _sha256_file(os.path.join(prism_dir, "lib", "libprism.so"))
    for git_root in (prism_dir, os.path.dirname(prism_dir)):
        if os.path.realpath(git_root) == os.path.realpath(REPO_ROOT) or not os.path.exists(os.path.join(git_root, ".git")):
            continue
        try:
            r = subprocess.run(["git", "-C", git_root, "rev-parse", "HEAD"],
                               capture_output=True, text=True, timeout=15)
            if r.returncode == 0:
                ident["prism_git"] = r.stdout.strip()
        except Exception:
            pass
        break
    _TOOL_IDENTITY = ident
    return dict(ident)


def _parse_elapsed(s: str) -> Optional[float]:
    try:
        parts = [float(x) for x in s.split(":")]
    except ValueError:
        return None
    t = 0.0
    for p in parts:
        t = t * 60.0 + p
    return t


def parse_prism_stdout(out: str, err: str = "") -> dict:


    def _last(rx, text, conv):
        ms = rx.findall(text)
        if not ms:
            return None
        try:
            return conv(ms[-1])
        except ValueError:
            return None
    d = {"prism_construction_s": _last(_CONSTR_RE, out, float),
         "prism_checking_s": _last(_CHECK_RE, out, float),
         "prism_states": _last(_STATES_RE, out, int),
         "prism_transitions": _last(_TRANS_RE, out, int),
         "prism_choices": _last(_CHOICES_RE, out, int),
         "value": _last(_RESULT_RE, out, float),
         "peak_rss_kb": _last(_RSS_RE, err, int),
         "time_elapsed_s": None}
    m = _ELAPSED_RE.search(err)
    if m:
        d["time_elapsed_s"] = _parse_elapsed(m.group(1))
    c, k = d["prism_construction_s"], d["prism_checking_s"]
    d["prism_solver_s"] = (c + k) if (c is not None and k is not None) else None
    return d


def classify_prism_status(timed_out: bool, rc: int, out: str, err: str, value) -> Tuple[str, Optional[str]]:


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
            return "ERR", f"no 'Result:' line in PRISM output (rc={rc})"
        return "ERR", f"non-zero exit status rc={rc}"
    return "OK", None


def run_prism(prism_text: str, props_text: str, timeout_s: float = DEFAULT_TIMEOUT_S,
              javamaxmem: Optional[str] = None, workdir: Optional[str] = None, tag: str = "model",
              extra_args: Sequence[str] = (), consts: Optional[Dict[str, object]] = None,
              keep_files: bool = True, maxiters: Optional[int] = DEFAULT_MAXITERS,
              export_results: bool = True, include_identity: bool = True) -> dict:


    if workdir is None:
        workdir = os.path.join(REPO_ROOT, "results", "prism_work")
    os.makedirs(workdir, exist_ok=True)
    if javamaxmem is None:
        javamaxmem = DEFAULT_JAVAMAXMEM
    model_path = os.path.join(workdir, f"{tag}.prism")
    props_path = os.path.join(workdir, f"{tag}.props")
    stdout_path = os.path.join(workdir, f"{tag}.log")
    results_path = os.path.join(workdir, f"{tag}.results.txt") if export_results else None
    with open(model_path, "w") as f:
        f.write(prism_text)
    with open(props_path, "w") as f:
        f.write(props_text)
    if results_path and os.path.exists(results_path):
        os.remove(results_path)
    cmd = [PRISM_BIN, model_path, props_path, "-prop", "1", "-javamaxmem", str(javamaxmem)]
    if maxiters is not None:
        cmd += ["-maxiters", str(int(maxiters))]
    if results_path:
        cmd += ["-exportresults", results_path]
    if consts:
        cmd += ["-const", ",".join(f"{k}={v}" for k, v in consts.items())]
    cmd += list(extra_args)
    use_time = shutil.which("time") or ("/usr/bin/time" if os.path.exists("/usr/bin/time") else None)
    if use_time:
        cmd = [use_time, "-v"] + cmd
    t0 = time.perf_counter()
    out, err, rc, timed_out = _run_killable(cmd, timeout_s)
    wall = time.perf_counter() - t0
    with open(stdout_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n=== STDOUT ===\n" + out + "\n=== STDERR ===\n" + err)
    parsed = parse_prism_stdout(out, err)
    value = parsed.pop("value")
    value_exported = None
    if results_path and os.path.isfile(results_path):
        try:
            with open(results_path) as f:
                toks = [ln.strip() for ln in f if ln.strip()]
            if len(toks) >= 2 and toks[0].lower().startswith("result"):
                value_exported = float(toks[1].split()[0])
        except (OSError, ValueError):
            value_exported = None
    if value is None and value_exported is not None:
        value = value_exported
    status, error = classify_prism_status(timed_out, rc, out, err, value)
    if status == "OK" and not (value is not None and np.isfinite(value)):
        status, error = "ERR", f"non-finite result {value!r}"
    if not keep_files and status == "OK":
        for p in (model_path, props_path):
            try:
                os.remove(p)
            except OSError:
                pass
    res = {"value": value, "value_exported": value_exported, "wall_s": wall, "status": status, "error": error,
           "timed_out": timed_out, "timeout_s": float(timeout_s), "javamaxmem": str(javamaxmem),
           "maxiters": maxiters, "stdout_path": stdout_path, "log_path": stdout_path,
           "model_path": model_path, "props_path": props_path, "results_path": results_path,
           "cmd": " ".join(cmd), "cmd_argv": list(cmd), "returncode": rc}
    res.update(parsed)
    if include_identity:
        res["tool_identity"] = tool_identity()
    return res


def prism_value(csg: CSG, mask: int, k: Optional[int] = None, **kw) -> dict:

    pt, pr = emit_coalition_model(csg, mask, k)
    tag = kw.pop("tag", f"coal_{int(mask)}")
    return run_prism(pt, pr, tag=tag, **kw)


def crosscheck(csg: CSG, masks: Optional[Iterable[int]] = None, k: Optional[int] = None,
               tol: float = 1e-6, timeout_s: float = DEFAULT_TIMEOUT_S, javamaxmem: Optional[str] = None,
               workdir: Optional[str] = None, verbose: bool = False, n_jobs: int = 1) -> dict:


    if k is None:
        k = csg.horizon_k
    if masks is None:
        masks = range(1 << csg.n_agents)
    masks = [int(m) for m in masks]
    refs = {m: vi_ref.solve(csg, m, k) for m in masks}

    def _one(m):
        return prism_value(csg, m, k, timeout_s=timeout_s, javamaxmem=javamaxmem, workdir=workdir)

    if n_jobs > 1:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=n_jobs) as ex:
            results = list(ex.map(_one, masks))
    else:
        results = [_one(m) for m in masks]
    rows = []
    max_err = 0.0
    n_mis = n_fail = 0
    for m, r in zip(masks, results):
        ref = refs[m]
        err = abs(r["value"] - ref) if r["value"] is not None else None
        row = {"mask": m, "ref": ref, "prism": r["value"], "abs_err": err, "status": r["status"],
               "wall_s": r["wall_s"], "peak_rss_kb": r["peak_rss_kb"], "stdout_path": r["stdout_path"]}
        rows.append(row)
        if r["status"] != "OK":
            n_fail += 1
        else:
            max_err = max(max_err, err)
            if err > tol:
                n_mis += 1
        if verbose:
            print(f"mask={m:>4d} ref={ref:.12f} prism={r['value']} err={err} {r['status']} {r['wall_s']:.2f}s")
    return {"rows": rows, "max_abs_err": max_err, "n_ok": len(rows) - n_fail, "n_mismatch": n_mis,
            "n_fail": n_fail, "all_match": (n_mis == 0 and n_fail == 0), "tol": tol, "k": k}


def _read_lines(path: str) -> List[str]:

    with open(path) as f:
        return [ln.rstrip("\n") for ln in f if ln.strip() and not ln.lstrip().startswith("#")]


def _parse_sta(path: str):
    lines = _read_lines(path)
    var_names = [v.strip() for v in lines[0].strip().strip("()").split(",") if v.strip()]
    states = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        idx, vals = ln.split(":", 1)
        vals = vals.strip().strip("()")
        states[int(idx)] = tuple(v.strip() for v in vals.split(","))
    return var_names, states


def _parse_lab(path: str):
    lines = _read_lines(path)
    names = {}
    for m in re.finditer(r'(\d+)="([^"]*)"', lines[0]):
        names[int(m.group(1))] = m.group(2)
    lab_states: Dict[str, List[int]] = {nm: [] for nm in names.values()}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        st, ids = ln.split(":", 1)
        for i in ids.split():
            lab_states[names[int(i)]].append(int(st))
    return lab_states


_TRA_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*(?:\[([^\]]*)\])?\s*$")


def _parse_tra(path: str):

    lines = _read_lines(path)
    hdr = lines[0]
    first, rest = hdr.split(" ", 1)
    if ":" in first:
        n_states, n_players = (int(x) for x in first.split(":"))
    else:
        n_states, n_players = int(first), None
    choices: Dict[int, Dict[int, Tuple[Tuple[str, ...], Dict[int, float]]]] = {}
    for ln in lines:
        if ln == hdr:
            continue
        m = _TRA_RE.match(ln)
        if not m:
            raise ValueError(f"cannot parse .tra line: {ln!r}")
        s, c, sp, p, act = int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4)), m.group(5)
        joint = tuple(a.strip() for a in act.split(",")) if act is not None else ()
        d = choices.setdefault(s, {})
        if c not in d:
            d[c] = (joint, {})
        elif d[c][0] != joint:
            raise ValueError(f"inconsistent joint action for state {s} choice {c}")
        d[c][1][sp] = d[c][1].get(sp, 0.0) + p
    out = []
    for s in range(n_states):
        d = choices.get(s, {})
        out.append([d[c] for c in sorted(d)])
    return n_states, n_players, out


def export_prism_model(prism_path: str, consts: Optional[Dict[str, object]] = None,
                       out_base: Optional[str] = None, extra_model_text: str = "",
                       timeout_s: float = 600.0, javamaxmem: str = "8g") -> dict:


    if out_base is None:
        out_base = os.path.join(REPO_ROOT, "results", "prism_work", "import",
                                os.path.splitext(os.path.basename(prism_path))[0])
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    model_path = prism_path
    if extra_model_text:
        model_path = out_base + ".cubcr.prism"
        with open(prism_path) as f:
            src = f.read()
        with open(model_path, "w") as f:
            f.write(src + "\n\n// --- appended by cubcr.io_prism ---\n" + extra_model_text + "\n")
    cmd = [PRISM_BIN, model_path, "-javamaxmem", str(javamaxmem),
           "-exportmodel", f"{out_base}.tra,sta,lab:actions"]
    if consts:
        cmd += ["-const", ",".join(f"{k}={v}" for k, v in consts.items())]
    out, err, rc, timed_out = _run_killable(cmd, timeout_s)
    stdout_path = out_base + ".export.log"
    with open(stdout_path, "w") as f:
        f.write(" ".join(cmd) + "\n\n" + out + "\n=== STDERR ===\n" + err)
    if timed_out:
        raise RuntimeError(f"PRISM export timed out after {timeout_s}s (see {stdout_path})")
    ok = rc == 0 and "Error" not in out and os.path.exists(out_base + ".tra")
    if not ok:
        raise RuntimeError(f"PRISM export failed (see {stdout_path})")
    return {"tra": out_base + ".tra", "sta": out_base + ".sta", "lab": out_base + ".lab",
            "model_path": model_path, "stdout_path": stdout_path, "status": "OK"}


PolicyFn = Callable[[int, Tuple[str, ...], int, List[str]], Sequence[float]]


def import_prism_csg(prism_path: str, consts: Optional[Dict[str, object]] = None,
                     policy: "str | PolicyFn" = "uniform", k: int = 1, fail_label: str = "fail",
                     fail_expr: Optional[str] = None, avoid_label: Optional[str] = None,
                     avoid_expr: Optional[str] = None, out_base: Optional[str] = None,
                     javamaxmem: str = "8g", timeout_s: float = 600.0) -> CSG:


    extra = ""
    fl, al = fail_label, avoid_label
    if fail_expr is not None:
        fl = "cubcr_fail"
        extra += f'label "{fl}" = {fail_expr};\n'
    if avoid_expr is not None:
        al = "cubcr_avoid"
        extra += f'label "{al}" = {avoid_expr};\n'
    exp = export_prism_model(prism_path, consts, out_base=out_base, extra_model_text=extra,
                             timeout_s=timeout_s, javamaxmem=javamaxmem)
    var_names, states = _parse_sta(exp["sta"])
    labels = _parse_lab(exp["lab"])
    n_states, n_players, choices = _parse_tra(exp["tra"])
    if fl not in labels:
        raise ValueError(f"label {fl!r} not found in exported labels {sorted(labels)}")
    if al is not None and al not in labels:
        raise ValueError(f"label {al!r} not found in exported labels {sorted(labels)}")
    inits = labels.get("init", [])
    if len(inits) != 1:
        raise NotImplementedError(f"model must have exactly one initial state, got {inits}")
    fail = sorted(set(labels[fl]))
    avoid = sorted(set(labels[al]) - set(fail)) if al is not None else []
    special = set(fail) | set(avoid)
    if n_players is None:
        n_players = max(len(ch[0][0]) for ch in choices if ch)
    n = n_players
    sds: List[StateDef] = []
    obs: Dict[Tuple[int, int], np.ndarray] = {}
    for s in range(n_states):
        if s in special or not choices[s]:
            sds.append(StateDef(active=[], n_actions=[], rows=[((), [(s, 1.0)])]))
            continue
        ch = choices[s]
        acts: List[List[str]] = [[] for _ in range(n)]
        for joint, _ in ch:
            if len(joint) != n:
                raise ValueError(f"state {s}: joint action {joint} does not have {n} entries")
            for i, a in enumerate(joint):
                if a not in acts[i]:
                    acts[i].append(a)
        acts = [sorted(a) for a in acts]
        multi = [i for i in range(n) if len(acts[i]) > 1]
        if len(multi) > DMAX:
            raise NotImplementedError(f"state {s}: {len(multi)} players with >1 action (> DMAX={DMAX})")
        active = list(range(n)) if n <= DMAX else multi
        for i in active:
            if len(acts[i]) > AMAX:
                raise NotImplementedError(f"state {s}: player {i} has {len(acts[i])} actions (> AMAX={AMAX})")
        radix = [len(acts[i]) for i in active]
        lookup = {joint: succ for joint, succ in ch}
        expected = int(np.prod(radix)) if radix else 1
        if len(lookup) != expected:
            raise ValueError(f"state {s}: {len(lookup)} joint actions, expected full product {expected}")
        rows = []
        for digits in itertools.product(*[range(m) for m in radix]):
            joint = [acts[i][0] for i in range(n)]
            for i, dgt in zip(active, digits):
                joint[i] = acts[i][dgt]
            succ = lookup.get(tuple(joint))
            if succ is None:
                raise ValueError(f"state {s}: missing joint action {tuple(joint)}")
            rows.append((tuple(digits), [(sp, p) for sp, p in sorted(succ.items())]))
        sds.append(StateDef(active=active, n_actions=radix, rows=rows,
                            action_labels=[list(acts[i]) for i in active]))
        for i in active:
            m = len(acts[i])
            if policy == "uniform":
                dist = np.full(m, 1.0 / m)
            else:
                dist = np.asarray(policy(s, states.get(s, ()), i, list(acts[i])), dtype=np.float64)
                if dist.shape != (m,):
                    raise ValueError(f"policy for state {s} player {i} must have length {m}")
            obs[(s, i)] = dist
    csg = CSG(n_agents=n, horizon_k=int(k), init_state=int(inits[0]), fail_states=fail,
              state_defs=sds, observed_policy=obs, avoid_states=avoid,
              agent_names=[f"player{i + 1}" for i in range(n)],
              meta={"source": os.path.abspath(prism_path), "consts": dict(consts or {}),
                    "var_names": var_names, "fail_label": fl, "avoid_label": al,
                    "state_tuples": [list(states[s]) for s in range(n_states)],
                    "policy": "uniform" if policy == "uniform" else "custom"})
    csg.validate()
    return csg


def prism_player_names(prism_path: str) -> List[str]:

    with open(prism_path) as f:
        src = f.read()
    src = re.sub(r"//[^\n]*", "", src)
    return re.findall(r"\bplayer\s+([A-Za-z_][A-Za-z0-9_]*)", src)


def crosscheck_import(prism_path: str, consts: Optional[Dict[str, object]], k: int,
                      fail_expr: Optional[str] = None, fail_label: str = "fail",
                      workdir: Optional[str] = None, timeout_s: float = 600.0,
                      javamaxmem: str = "8g") -> dict:


    if workdir is None:
        workdir = os.path.join(REPO_ROOT, "results", "prism_work", "import")
    os.makedirs(workdir, exist_ok=True)
    base = os.path.join(workdir, os.path.splitext(os.path.basename(prism_path))[0])
    csg = import_prism_csg(prism_path, consts, k=k, fail_label=fail_label, fail_expr=fail_expr,
                           out_base=base, javamaxmem=javamaxmem, timeout_s=timeout_s)
    fl = csg.meta["fail_label"]
    orig_model = base + ".cubcr.prism" if fail_expr is not None else prism_path
    with open(orig_model) as f:
        orig_text = f.read()
    players = prism_player_names(prism_path)
    assert len(players) == csg.n_agents, (players, csg.n_agents)
    out = {"n_states": csg.n_states, "n_agents": csg.n_agents, "checks": []}

    props = f'<<{",".join(players)}>> Pmin=? [ F<={k} "{fl}" ]\n'
    r = run_prism(orig_text, props, workdir=workdir, tag=os.path.basename(base) + "_orig_allmin",
                  timeout_s=timeout_s, javamaxmem=javamaxmem, consts=consts)
    ref = vi_ref.solve(csg, 0, k)
    out["checks"].append({"name": "orig_allmin_vs_viref", "prism": r["value"], "ref": ref,
                          "status": r["status"],
                          "abs_err": abs(r["value"] - ref) if r["value"] is not None else None})

    emit_text, _ = emit_coalition_model(csg, 0, k, comment="re-emitted import")
    for i, pl in enumerate(players):
        for op in ("Pmax", "Pmin"):
            p_orig = f'<<{pl}>> {op}=? [ F<={k} "{fl}" ]\n'
            p_emit = f'<<p{i}>> {op}=? [ F<={k} "fail" ]\n'
            r1 = run_prism(orig_text, p_orig, workdir=workdir, consts=consts,
                           tag=f"{os.path.basename(base)}_orig_{op}_{i}", timeout_s=timeout_s,
                           javamaxmem=javamaxmem)
            r2 = run_prism(emit_text, p_emit, workdir=workdir,
                           tag=f"{os.path.basename(base)}_emit_{op}_{i}", timeout_s=timeout_s,
                           javamaxmem=javamaxmem)
            err = abs(r1["value"] - r2["value"]) if (r1["value"] is not None and r2["value"] is not None) else None
            out["checks"].append({"name": f"orig_vs_emit_{op}_{pl}", "prism_orig": r1["value"],
                                  "prism_emit": r2["value"], "abs_err": err,
                                  "status": "OK" if r1["status"] == r2["status"] == "OK" else "ERR"})
    out["csg"] = csg
    out["all_ok"] = all(c["status"] == "OK" and c["abs_err"] is not None and c["abs_err"] <= 1e-6
                        for c in out["checks"])
    return out
