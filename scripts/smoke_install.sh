#!/usr/bin/env bash
# End-to-end installation smoke test for the published wheel (S19).
#
# Usage:
#   scripts/smoke_install.sh              build, install, exercise and keep state
#   scripts/smoke_install.sh --check      only validate prerequisites
#   scripts/smoke_install.sh --uninstall  remove the last smoke installation
#
# The installation lives under a temporary directory (never the user's tool
# environment) and is recorded in ${TMPDIR}/stc-smoke-last-<uid> so --uninstall
# can find it. Running the script twice replaces the previous installation.
set -euo pipefail

STATE_FILE="${TMPDIR:-/tmp}/stc-smoke-last-$(id -u)"
TOOL_NAME="spanish-text-classifier-lora"

usage() {
  cat <<'USAGE'
Usage: scripts/smoke_install.sh [--check | --uninstall | --keep]

  --check      Validate prerequisites only.
  --uninstall  Remove the tool environment left by the previous run.
  --keep       Keep the previous successful installation instead of replacing it.
  -h, --help   Show this help.
USAGE
}

check_prerequisites() {
  local missing=0
  for command in uv python3; do
    if ! command -v "$command" >/dev/null 2>&1; then
      echo "error: required command not found: $command" >&2
      missing=1
    fi
  done
  if [ "$missing" -ne 0 ]; then
    return 1
  fi
  echo "prerequisites OK (uv $(uv --version | awk '{print $2}'), python3 $(python3 --version | awk '{print $2}'))"
}

uninstall() {
  if [ ! -f "$STATE_FILE" ]; then
    echo "no previous smoke installation recorded; nothing to uninstall"
    return 0
  fi
  local root tool_dir bin_dir
  root="$(cat "$STATE_FILE")"
  tool_dir="$root/tool"
  bin_dir="$root/bin"
  if [ -d "$tool_dir" ]; then
    UV_TOOL_DIR="$tool_dir" UV_TOOL_BIN_DIR="$bin_dir" \
      uv tool uninstall "$TOOL_NAME" >/dev/null 2>&1 || true
  fi
  rm -rf "$root"
  rm -f "$STATE_FILE"
  echo "uninstalled the smoke installation at $root"
}

MODE="run"
KEEP=0
while [ "$#" -gt 0 ]; do
  case "$1" in
    --check) MODE="check" ;;
    --uninstall) MODE="uninstall" ;;
    --keep) KEEP=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "error: unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

if [ "$MODE" = "check" ]; then
  check_prerequisites
  exit 0
fi

if [ "$MODE" = "uninstall" ]; then
  uninstall
  exit 0
fi

check_prerequisites

if [ "$KEEP" -eq 1 ] && [ -f "$STATE_FILE" ]; then
  previous="$(cat "$STATE_FILE")"
  case "$previous" in
    "${TMPDIR:-/tmp}"/stc-smoke.*) ROOT="$previous" ;;
  esac
fi
if [ -z "${ROOT:-}" ]; then
  if [ -f "$STATE_FILE" ]; then
    previous="$(cat "$STATE_FILE")"
    case "$previous" in
      "${TMPDIR:-/tmp}"/stc-smoke.*) rm -rf "$previous" ;;
    esac
  fi
  ROOT="$(mktemp -d "${TMPDIR:-/tmp}/stc-smoke.XXXXXX")"
fi
echo "$ROOT" > "$STATE_FILE"
TOOL_DIR="$ROOT/tool"
BIN_DIR="$ROOT/bin"
STC_HOME="$ROOT/home"
WORK="$ROOT/work"
mkdir -p "$TOOL_DIR" "$BIN_DIR" "$STC_HOME" "$WORK"
export STC_HOME
export UV_TOOL_DIR="$TOOL_DIR"
export UV_TOOL_BIN_DIR="$BIN_DIR"

SERVE_PID=""
cleanup() {
  local status=$?
  if [ -n "$SERVE_PID" ]; then
    kill "$SERVE_PID" >/dev/null 2>&1 || true
    wait "$SERVE_PID" >/dev/null 2>&1 || true
  fi
  if [ "$status" -ne 0 ]; then
    echo "smoke installation failed; artifacts kept in $ROOT" >&2
  fi
  exit "$status"
}
trap cleanup EXIT

echo "==> building the wheel"
uv build --wheel --out-dir "$WORK/dist" --no-build-logs >/dev/null
wheel="$(find "$WORK/dist" -maxdepth 1 -name '*.whl' -print -quit)"
if [ -z "$wheel" ]; then
  echo "error: uv build did not produce a wheel" >&2
  exit 1
fi

echo "==> installing $wheel into a throw-away tool environment"
uv tool install --python 3.12 --force "$wheel" >/dev/null
STC="$BIN_DIR/stc"
if [ ! -x "$STC" ]; then
  echo "error: stc console script was not installed" >&2
  exit 1
fi

echo "==> stc --version"
"$STC" --version

echo "==> stc doctor"
"$STC" doctor >/dev/null

echo "==> stc data fixture info"
"$STC" data fixture info

echo "==> stc train baseline --dataset fixture --quick"
"$STC" train baseline --dataset fixture --quick --out "$WORK/baseline"

echo "==> stc predict"
prediction="$("$STC" predict --model "$WORK/baseline" "enciende la luz" --json)"
python3 - "$prediction" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
assert payload["schema_version"] == 1
assert len(payload["predictions"]) == 1
prediction = payload["predictions"][0]
assert isinstance(prediction["intent"], str) and prediction["intent"]
assert isinstance(prediction["confidence"], float)
print(f"predicted {prediction['intent']} ({prediction['confidence']:.4f})")
PY

echo "==> stc evaluate --split dev --dataset fixture"
"$STC" evaluate --split dev --dataset fixture --model "$WORK/baseline" --out "$WORK/eval" >/dev/null
test -s "$WORK/eval/evaluation.json"

echo "==> the tool environment must not contain torch"
tool_python="$(head -n 1 "$STC" | sed 's/^#!//')"
if [ ! -x "$tool_python" ]; then
  tool_python="$TOOL_DIR/$TOOL_NAME/bin/python"
fi
if "$tool_python" -c "import torch" >/dev/null 2>&1; then
  echo "error: torch is importable inside the installed tool environment" >&2
  exit 1
fi
echo "torch is not importable (S12)"

echo "==> stc serve /healthz"
port="$("$tool_python" - <<'PY'
import socket

for candidate in range(19000, 20000):
    with socket.socket() as sock:
        try:
            sock.bind(("127.0.0.1", candidate))
        except OSError:
            continue
        print(candidate)
        break
else:
    raise SystemExit("no free port in 19000-19999")
PY
)"
STC_API_PORT="$port" STC_MODEL="$WORK/baseline" \
  "$STC" serve --port "$port" >"$WORK/serve.log" 2>&1 &
SERVE_PID=$!
"$tool_python" - "$port" <<'PY'
import json
import sys
import time
import urllib.request

port = int(sys.argv[1])
url = f"http://127.0.0.1:{port}/healthz"
for _attempt in range(60):
    try:
        with urllib.request.urlopen(url, timeout=2) as response:
            payload = json.load(response)
        assert payload["status"] == "ok"
        print(f"healthz OK: {payload}")
        break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("serve did not answer /healthz within 60 seconds")
PY
kill "$SERVE_PID" >/dev/null 2>&1 || true
wait "$SERVE_PID" >/dev/null 2>&1 || true
SERVE_PID=""

echo "SMOKE OK"
