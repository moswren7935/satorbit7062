from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCES = {
    "qest18": "https://www.prismmodelchecker.org/files/qest18/",
    "fm19nash": "https://www.prismmodelchecker.org/files/fm19nash/",
}
KEEP_EXT = (".prism", ".props", ".pm", ".nm")
PRISM_BIN = Path(os.environ.get("CUBCR_PRISM_BIN", str(REPO_ROOT / "external" / "prism-games" / "prism" / "bin" / "prism")))
JAVA_HOME = os.environ.get("CUBCR_JAVA_HOME", os.environ.get("JAVA_HOME", ""))
USER_AGENT = "cuBCR-fetch/1.0 (+python-urllib)"
_HREF_RE = re.compile(r'href="([^"]+)"', re.IGNORECASE)
_UNDEF_CONST_RE = re.compile(r"^\s*const\s+(?:int|double|bool)\s+(\w+)\s*;", re.MULTILINE)
_UNDEF_MSG_RE = re.compile(r"undefined constants?", re.IGNORECASE)


def _get(url: str, timeout: float = 60) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download_to(url: str, path: Path, timeout: float = 300) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    n = 0
    with urllib.request.urlopen(req, timeout=timeout) as r, open(path, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            n += len(chunk)
    return n


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def strip_php_view(page: bytes) -> str:

    txt = page.decode("iso-8859-1", errors="replace")
    m = re.search(r"<pre>(.*?)</pre>", txt, re.DOTALL | re.IGNORECASE)
    body = m.group(1) if m else txt
    body = re.sub(r"<[^>]+>", "", body)
    return html.unescape(body).lstrip("\n")


def list_page(page: str, base: str) -> tuple[list[str], list[str]]:

    listing = _get(base).decode("utf-8", errors="replace")
    files, tarballs = [], []
    for href in _HREF_RE.findall(listing):
        if "://" in href or href.startswith("/"):
            continue
        name = href[:-4] if href.endswith(".php") else href
        if name.endswith(KEEP_EXT):
            if name not in files:
                files.append(name)
        elif href.endswith((".tar.gz", ".tgz")):
            if href not in tarballs:
                tarballs.append(href)
    return files, tarballs


def fetch_file(base: str, name: str, dest: Path) -> dict:

    dest.mkdir(parents=True, exist_ok=True)
    out = dest / name
    url = base + name
    origin = "direct"
    try:
        data = _get(url)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        url = base + name + ".php"
        data = strip_php_view(_get(url)).encode("utf-8")
        origin = "php-stripped"
    out.write_bytes(data)
    return {"file": str(out), "size": len(data), "sha256": sha256_of(out), "url": url, "origin": origin}


TARBALL_FILTER_DEFAULT = r"(^|/)examples-csg/|(^|/)examples-distr/games/csg/"


def _safe_member(m: tarfile.TarInfo, member_re: re.Pattern | None = None) -> bool:
    p = Path(m.name)
    if not (m.isfile() and not p.is_absolute() and ".." not in p.parts and m.name.endswith(KEEP_EXT)):
        return False
    if p.name.startswith("._"):
        return False
    return member_re is None or bool(member_re.search(m.name))


def _common_prefix(names: list[str]) -> str:

    if not names:
        return ""
    parts = [n.split("/")[:-1] for n in names]
    pref = parts[0]
    for q in parts[1:]:
        i = 0
        while i < min(len(pref), len(q)) and pref[i] == q[i]:
            i += 1
        pref = pref[:i]
    return "/".join(pref) + ("/" if pref else "")


def fetch_tarball(base: str, tar_name: str, dest: Path, root_for_rel: Path,
                  member_filter: str | None = TARBALL_FILTER_DEFAULT) -> list[dict]:


    url = base + tar_name
    member_re = re.compile(member_filter) if member_filter else None
    stem = tar_name
    for suf in (".tar.gz", ".tgz"):
        if stem.endswith(suf):
            stem = stem[: -len(suf)]
    target = dest / stem
    entries = []
    with tempfile.TemporaryDirectory(prefix="cubcr_fetch_") as td:
        tmp = Path(td) / tar_name
        n = _download_to(url, tmp)
        print(f"  downloaded {tar_name} ({n/1e6:.1f} MB) -> extracting {KEEP_EXT} only")
        with tarfile.open(tmp, "r:gz") as tf:
            all_members = tf.getmembers()
            members = [m for m in all_members if _safe_member(m, member_re)]
            prefix = _common_prefix([m.name for m in members])
            n_all = sum(1 for m in all_members if _safe_member(m, None))
            print(f"  {len(all_members)} members; {n_all} with kept extensions; {len(members)} match filter "
                  f"{member_filter!r}; stripping prefix {prefix!r}")
            for m in members:
                rel = Path(m.name[len(prefix):] if m.name.startswith(prefix) else m.name)
                out = target / rel
                out.parent.mkdir(parents=True, exist_ok=True)
                src = tf.extractfile(m)
                if src is None:
                    continue
                with src, open(out, "wb") as fh:
                    shutil.copyfileobj(src, fh)
                out = out.resolve()
                recorded_path = out.relative_to(root_for_rel) if root_for_rel in out.parents else out
                entries.append({"file": str(recorded_path), "size": out.stat().st_size,
                                "sha256": sha256_of(out), "url": url, "origin": f"tarball:{tar_name}",
                                "member": m.name})

    print(f"  extracted {len(entries)} files from {tar_name} into {target}")
    return entries


def prism_env() -> dict:
    env = dict(os.environ)
    if JAVA_HOME and Path(JAVA_HOME).is_dir():
        env["JAVA_HOME"] = JAVA_HOME
        env["PATH"] = f"{JAVA_HOME}/bin:" + env.get("PATH", "")
    return env


def prism_available() -> bool:
    return PRISM_BIN.is_file() and os.access(PRISM_BIN, os.X_OK)


def _run_prism(args: list[str], timeout: float) -> tuple[str, str, int | None, float]:
    t0 = time.perf_counter()
    try:
        p = subprocess.run([str(PRISM_BIN), *args], capture_output=True, text=True, timeout=timeout,
                           env=prism_env(), cwd=str(REPO_ROOT))
        return "OK", p.stdout + p.stderr, p.returncode, time.perf_counter() - t0
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"") if isinstance(exc.stdout, (bytes, bytearray)) else (exc.stdout or "")
        if isinstance(out, (bytes, bytearray)):
            out = out.decode(errors="replace")
        return "TO", out, None, time.perf_counter() - t0


def _error_message(output: str) -> str:
    lines = [ln.strip() for ln in output.splitlines() if ln.strip()]
    errs = [ln for ln in lines if ln.lower().startswith("error")]
    if errs:
        return errs[-1][:500]
    return (lines[-1] if lines else "no output")[:500]


def verify_prism_file(path: Path, timeout: float = 60, try_build: bool = True) -> dict:

    res: dict = {"prism_status": None, "prism_msg": "", "parse_s": None,
                 "undefined_consts": [], "build_status": None, "build_msg": "", "build_s": None}
    src = path.read_text(errors="replace")
    res["undefined_consts"] = sorted(set(_UNDEF_CONST_RE.findall(src)))
    st, out, rc, dt = _run_prism([str(path), "-nobuild"], timeout)
    res["parse_s"] = round(dt, 3)
    if st == "TO":
        res["prism_status"] = "ERROR"
        res["prism_msg"] = f"parse timed out after {timeout} s"
        return res
    if rc != 0 or "Error" in out:
        msg = _error_message(out)
        if _UNDEF_MSG_RE.search(msg):


            res["prism_status"] = "NEEDS_CONSTS"
            res["prism_msg"] = msg
            res["undefined_consts"] = sorted(set(res["undefined_consts"]) | set(re.findall(r'"(\w+)"', msg)))
        else:
            res["prism_status"] = "ERROR"
            res["prism_msg"] = msg
        return res
    if res["undefined_consts"]:
        res["prism_status"] = "NEEDS_CONSTS"
        res["prism_msg"] = "undefined constants: " + ", ".join(res["undefined_consts"])
        return res
    res["prism_status"] = "PARSE_OK"
    res["prism_msg"] = "parsed with -nobuild"
    if try_build:
        st, out, rc, dt = _run_prism([str(path)], timeout)
        res["build_s"] = round(dt, 3)
        if st == "TO":
            res["build_status"], res["build_msg"] = "TO", f"build timed out after {timeout} s"
        elif rc != 0 or "Error" in out:
            msg = _error_message(out)
            if _UNDEF_MSG_RE.search(msg):
                res["prism_status"] = "NEEDS_CONSTS"
                res["prism_msg"] = msg
            res["build_status"], res["build_msg"] = "ERROR", msg
        else:
            m = re.search(r"States:\s+(\d+)", out)
            res["build_status"] = "BUILD_OK"
            res["build_msg"] = f"states={m.group(1)}" if m else "built"
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default=str(REPO_ROOT / "datasets" / "prism_csg"))
    ap.add_argument("--only", choices=sorted(SOURCES), default=None)
    ap.add_argument("--no-tarball", action="store_true", help="skip linked .tar.gz archives")
    ap.add_argument("--tarball-filter", default=TARBALL_FILTER_DEFAULT,
                    help="regex on tarball member paths to keep (default: CSG example dirs only; '.' = all)")
    ap.add_argument("--skip-verify", action="store_true", help="do not run PRISM-games parse checks")
    ap.add_argument("--no-build", action="store_true", help="parse-only verification (skip build attempt)")
    ap.add_argument("--timeout", type=float, default=60.0, help="per-PRISM-invocation cap in seconds")
    ap.add_argument("--manifest", default=None, help="manifest path (default <dest>/manifest.json)")
    a = ap.parse_args(argv)

    dest = Path(a.dest).expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(a.manifest) if a.manifest else dest / "manifest.json"
    entries: list[dict] = []
    for page, base in SOURCES.items():
        if a.only and page != a.only:
            continue
        print(f"== {page}: {base}")
        try:
            files, tarballs = list_page(page, base)
        except Exception as exc:
            print(f"  ERROR listing {base}: {exc}", file=sys.stderr)
            entries.append({"file": None, "page": page, "url": base, "origin": "listing",
                            "error": f"{type(exc).__name__}: {exc}"})
            continue
        print(f"  {len(files)} model/property files, {len(tarballs)} tarball(s) linked")
        for name in files:
            try:
                e = fetch_file(base, name, dest / page)
                out = (dest / page / name).resolve()
                e["file"] = str(out.relative_to(REPO_ROOT)) if REPO_ROOT in out.parents else str(out)
                e["page"] = page
                entries.append(e)
                print(f"  {name:36s} {e['size']:7d} B  [{e['origin']}]")
            except Exception as exc:
                print(f"  {name:36s} FAILED: {exc}", file=sys.stderr)
                entries.append({"file": name, "page": page, "url": base + name, "origin": "direct",
                                "error": f"{type(exc).__name__}: {exc}"})
        if not a.no_tarball:
            for tb in tarballs:
                try:
                    for e in fetch_tarball(base, tb, dest / page, REPO_ROOT, a.tarball_filter):
                        e["page"] = page
                        entries.append(e)
                except Exception as exc:
                    print(f"  tarball {tb} FAILED: {exc}", file=sys.stderr)
                    entries.append({"file": tb, "page": page, "url": base + tb, "origin": "tarball",
                                    "error": f"{type(exc).__name__}: {exc}"})

    prism_meta = {"bin": str(PRISM_BIN), "available": prism_available(), "java_home": JAVA_HOME,
                  "timeout_s": a.timeout, "version": None}
    if not a.skip_verify and prism_available():
        st, out, rc, _ = _run_prism(["-version"], a.timeout)
        prism_meta["version"] = out.strip().splitlines()[-1] if out.strip() else None
        todo = [e for e in entries if e.get("file") and e["file"].endswith(".prism") and "error" not in e]
        print(f"== verifying {len(todo)} .prism files with PRISM-games ({prism_meta['version']}), cap {a.timeout}s each")
        for e in todo:
            p = REPO_ROOT / e["file"] if not Path(e["file"]).is_absolute() else Path(e["file"])
            r = verify_prism_file(p, a.timeout, try_build=not a.no_build)
            e.update(r)
            print(f"  {e['file']:70s} {r['prism_status']:12s} {r['prism_msg'][:60]}"
                  + (f"  build={r['build_status']}" if r["build_status"] else ""))
    elif not a.skip_verify:
        print(f"== PRISM-games binary not found at {PRISM_BIN}; skipping verification", file=sys.stderr)

    counts: dict = {}
    for e in entries:
        s = e.get("prism_status") or ("FETCH_ERROR" if "error" in e else "n/a")
        counts[s] = counts.get(s, 0) + 1
    manifest = {
        "generated": _dt.datetime.now(_dt.timezone.utc).astimezone().isoformat(timespec="seconds"),
        "sources": SOURCES, "keep_ext": list(KEEP_EXT), "prism": prism_meta,
        "summary": {"files": len([e for e in entries if e.get("file") and "error" not in e]),
                    "bytes": sum(e.get("size", 0) for e in entries), "status_counts": counts},
        "files": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1))
    print(f"== manifest written to {manifest_path}: {manifest['summary']}")
    return 1 if any("error" in e for e in entries) else 0


if __name__ == "__main__":
    sys.exit(main())
