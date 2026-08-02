#!/usr/bin/env bash
# GoEye launcher — runs the floating Go analysis overlay.
#
# One-time setup (on the user's Mac):
#   brew install katago                 # provides the katago binary (Metal backend)
#   PIP=$(python3 -m pip --version >/dev/null 2>&1 && echo python3 -m pip || echo pip)
#   $PIP install -r requirements.txt    # pyqt6, opencv-python-headless, mss, numpy
#   (The two KataGo networks in models/ are already bundled — no download needed.)
#
# Usage:
#   ./run.sh          # start the app
#   ./run.sh --test   # run the headless recognition + engine self-tests

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/Users/tommydu/.workbuddy/binaries/python/envs/default/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

cd "$SCRIPT_DIR"

if [[ "${1:-}" == "--test" ]]; then
  "$PYTHON" tests/test_vision.py
  "$PYTHON" tests/test_engine.py
  exit 0
fi

exec "$PYTHON" -m goeye
