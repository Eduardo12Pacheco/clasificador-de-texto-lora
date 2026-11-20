#!/usr/bin/env bash
# Attach the verified model bundle to an existing GitHub release (D13/S19).
#
# Usage: scripts/publish_model.sh v0.1.0
#
# The workflow is deliberately manual: CI never uploads weights. This helper
# verifies the release exists, checks the bundle checksum against the committed
# registry and only then uploads the assets with `gh release upload`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
REGISTRY="$REPO_ROOT/src/stc/model_registry.json"
MODEL_DIR="$REPO_ROOT/dist-model"
SUMS="$MODEL_DIR/SHA256SUMS-model.txt"

usage() {
  cat <<'USAGE'
Usage: scripts/publish_model.sh <tag>

  <tag>  Existing release tag, for example v0.1.0.
USAGE
}

if [ "$#" -ne 1 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
  usage
  exit 2
fi
TAG="$1"

for command in gh sha256sum python3; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "error: required command not found: $command" >&2
    exit 1
  fi
done

if ! gh auth status >/dev/null 2>&1; then
  echo "error: gh is not authenticated; run: gh auth login" >&2
  exit 1
fi

if ! gh release view "$TAG" >/dev/null 2>&1; then
  echo "error: release $TAG does not exist; push the tag and let CI create it first" >&2
  exit 1
fi

expected="$(python3 - "$REGISTRY" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    registry = json.load(handle)
models = [entry for entry in registry["models"] if entry["name"] == "stc-es-intent"]
if not models:
    raise SystemExit("stc-es-intent is not registered")
latest = max(models, key=lambda entry: entry["version"])
print(latest["version"], latest["sha256"])
PY
)"
version="${expected%% *}"
expected_sha="${expected#* }"
bundle="$MODEL_DIR/stc-es-intent-$version.zip"
if [ ! -f "$bundle" ]; then
  echo "error: missing bundle $bundle (see the release notes for the export command)" >&2
  exit 1
fi

actual="$(sha256sum "$bundle" | awk '{print $1}')"
if [ "$expected_sha" != "$actual" ]; then
  echo "error: bundle sha256 $actual does not match the registry $expected_sha" >&2
  exit 1
fi

if [ ! -f "$SUMS" ]; then
  (cd "$MODEL_DIR" && sha256sum ./*.zip > SHA256SUMS-model.txt)
fi

echo "==> uploading verified model assets to $TAG"
gh release upload "$TAG" "$MODEL_DIR"/*.zip "$SUMS" --clobber
echo "MODEL ASSETS PUBLISHED"
