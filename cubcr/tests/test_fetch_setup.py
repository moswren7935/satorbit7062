import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile

from datasets import fetch


def test_failed_download_returns_failure_with_manifest(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("network unavailable")

    monkeypatch.setattr(fetch, "list_page", fail)
    assert fetch.main(["--dest", str(tmp_path), "--skip-verify"]) == 1
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["summary"]["files"] == 0
    assert manifest["summary"]["status_counts"]["FETCH_ERROR"] == 2


def test_external_destination_manifest_resolves_archive_files(tmp_path, monkeypatch):
    archive = tmp_path / "models.tgz"
    body = b"csg\n"
    with tarfile.open(archive, "w:gz") as tf:
        member = tarfile.TarInfo("models/examples-csg/example.prism")
        member.size = len(body)
        tf.addfile(member, io.BytesIO(body))

    def download(url, target):
        shutil.copyfile(archive, target)
        return archive.stat().st_size

    monkeypatch.setattr(fetch, "list_page", lambda *args: ([], ["models.tgz"]))
    monkeypatch.setattr(fetch, "_download_to", download)
    dest = tmp_path / "download"
    assert fetch.main(["--dest", str(dest), "--only", "qest18", "--skip-verify"]) == 0
    manifest = json.loads((dest / "manifest.json").read_text())
    assert len(manifest["files"]) == 1
    path = fetch.REPO_ROOT / manifest["files"][0]["file"]
    assert path.is_file()
    assert path.read_bytes() == body


def _external_script(tmp_path, curl_body):
    root = Path(__file__).resolve().parents[1]
    script = tmp_path / "rebuttal" / "scripts" / "e4_fetch_external.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(root / "rebuttal" / "scripts" / script.name, script)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("#!/usr/bin/env bash\n" + curl_body)
    curl.chmod(0o755)
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"])
    return script, script.parent.parent / "env" / "ext", env


def test_partial_external_checkout_does_not_count_as_complete(tmp_path):
    script, dest, env = _external_script(tmp_path, "exit 22\n")
    for name in ("leverageshap-9702c2cc6386e13fca20b9caeb8747dedf253cef",
                 "adalina-19554d00e8fe9338c43e66ec439eb5c4ab747135",
                 "svarm-c4e30395b14b8bf9a8687d839d5b406f825bb23d"):
        (dest / name).mkdir(parents=True)
    result = subprocess.run(["bash", str(script), str(dest)], env=env, capture_output=True)
    assert result.returncode != 0
    assert not (dest / "FETCHED.txt").exists()
    assert not list(dest.glob(".fetch-*"))


def test_corrupt_external_archive_leaves_no_checkout(tmp_path):
    script, dest, env = _external_script(tmp_path, "while [ $# -gt 0 ]; do if [ \"$1\" = -o ]; then printf broken > \"$2\"; exit 0; fi; shift; done\nexit 2\n")
    result = subprocess.run(["bash", str(script), str(dest)], env=env, capture_output=True)
    assert result.returncode != 0
    assert not list(dest.glob("leverageshap-*"))
    assert not list(dest.glob(".fetch-*"))


def test_tool_environment_uses_reviewers_java_and_path(tmp_path):
    import json
    import os
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for name in ("review-prism", "review-storm"):
        binary = tmp_path / name
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    env = dict(os.environ, JAVA_HOME=str(tmp_path),
               PATH=str(tmp_path) + os.pathsep + os.environ["PATH"],
               CUBCR_PRISM_BIN="review-prism", CUBCR_STORM_BIN="review-storm")
    env.pop("CUBCR_JAVA_HOME", None)
    code = "from cubcr import io_prism,io_storm; import json; print(json.dumps([io_prism.JAVA_HOME,io_prism.prism_available(),io_storm.storm_available()]))"
    actual = subprocess.check_output([sys.executable, "-c", code], cwd=root, env=env, text=True)
    assert json.loads(actual) == [str(tmp_path), True, True]
    env["CUBCR_JAVA_HOME"] = str(tmp_path / "override")
    actual = subprocess.check_output([sys.executable, "-c", code], cwd=root, env=env, text=True)
    assert json.loads(actual)[0] == env["CUBCR_JAVA_HOME"]
