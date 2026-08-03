#!/usr/bin/env bash
# Build a standalone GoEye.app for macOS using PyInstaller.
#
# One-time setup (you handle installs, as agreed):
#   PIP=/Users/tommydu/.workbuddy/binaries/python/envs/default/bin/pip
#   $PIP install pyinstaller
#   brew install katago          # optional: stronger engine; GNU Go works too
#
# Usage:
#   ./build_app.sh            # builds dist/GoEye.app
#   ./build_app.sh --dmg      # also wraps it into dist/GoEye.dmg
#
# The bundled KataGo weights in models/ (git-ignored, ~60 MB) are copied into
# the .app automatically when present, so engine analysis works out of the box.
# If models/ is absent the app still runs in recognition-only mode (or via the
# GNU Go fallback) and you can drop weights in later.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="/Users/tommydu/.workbuddy/binaries/python/envs/default/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  PYTHON="python3"
fi

# Fail fast with a clear message if PyInstaller is not installed.
if ! "$PYTHON" -c "import PyInstaller" 2>/dev/null; then
  echo "PyInstaller not found." >&2
  echo "Install it once with:" >&2
  echo "  $PYTHON -m pip install pyinstaller" >&2
  exit 1
fi

# Bundle the KataGo weights if they are present locally.
EXTRA_DATA=()
if [[ -d models ]]; then
  EXTRA_DATA+=( "--add-data" "models:models" )
fi

rm -rf build dist

"$PYTHON" -m PyInstaller \
  --name GoEye \
  --windowed \
  --noconfirm \
  --osx-bundle-identifier com.goeye.app \
  --hidden-import goeye \
  "${EXTRA_DATA[@]}" \
  macos_main.py

echo "Built: dist/GoEye.app"

if [[ "${1:-}" == "--dmg" ]]; then
  if [[ -f dist/GoEye.dmg ]]; then rm -f dist/GoEye.dmg; fi
  hdiutil create -volname GoEye -srcfolder dist/GoEye.app -ov -format UDZO dist/GoEye.dmg
  echo "Built: dist/GoEye.dmg"
fi

echo
echo "First launch (unsigned app):"
echo "  xattr -cr dist/GoEye.app && open dist/GoEye.app"
echo "Then grant Screen Recording access to GoEye in System Settings."
