#!/usr/bin/env bash


set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -gt 1 ]; then
  echo "usage: $0 [destination]" >&2
  exit 2
fi
EXT="${1:-${CUBCR_EXTERNAL_DIR:-$HERE/../env/ext}}"
for program in curl tar sha256sum; do
  if ! command -v "$program" >/dev/null 2>&1; then
    echo "required command not found: $program" >&2
    exit 1
  fi
done
mkdir -p "$EXT"
complete () {
  local dst="$1" required
  shift
  [ -s "$dst/.tarball.sha256" ] || return 1
  for required in "$@"; do
    [ -s "$dst/$required" ] || return 1
  done
}
fetch () (
  local name="$1" repo="$2" sha="$3"
  shift 3
  local dst="$EXT/$name-$sha"
  if complete "$dst" "$@"; then echo "exists $dst"; return; fi
  local tmp; tmp="$(mktemp -d "$EXT/.fetch-$name.XXXXXXXX")"
  trap 'rm -rf "$tmp"' EXIT
  curl --fail --show-error --silent --location --retry 3 --connect-timeout 30 --max-time 300 "https://codeload.github.com/$repo/tar.gz/$sha" -o "$tmp/src.tgz"
  sha256sum "$tmp/src.tgz" | awk '{print $1}' > "$tmp/tarball.sha256"
  mkdir "$tmp/source"
  tar xzf "$tmp/src.tgz" -C "$tmp/source" --strip-components=1
  cp "$tmp/tarball.sha256" "$tmp/source/.tarball.sha256"
  if ! complete "$tmp/source" "$@"; then
    echo "downloaded archive is missing required files: $repo $sha" >&2
    exit 1
  fi
  if [ -e "$dst" ]; then mv "$dst" "$tmp/previous"; fi
  mv "$tmp/source" "$dst"
  echo "$repo $sha $(cat "$tmp/tarball.sha256") fetched $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$EXT/FETCHED.txt"
)
fetch leverageshap rtealwitter/leverageshap 9702c2cc6386e13fca20b9caeb8747dedf253cef leverageshap/estimators/sampling.py leverageshap/estimators/leverage_shap.py
fetch adalina watml/adalina 19554d00e8fe9338c43e66ec439eb5c4ab747135 estimators/Adalina.py utilFuncs/utilTemplate.py
fetch svarm kolpaczki/Approximating-the-Shapley-Value-without-Marginal-Contributions c4e30395b14b8bf9a8687d839d5b406f825bb23d ApproxMethods/SVARM/SVARM.py ApproxMethods/SVARM/StratifiedSVARM.py
