#!/usr/bin/env bash
# Build vLLM Manager for macOS (Apple Silicon / Intel) — PyInstaller .app bundle
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="vLLM Manager"
ENTRY="vllm_manager.py"

echo "=== Building $APP_NAME for macOS ==="

# Ensure venv
if [ ! -d "$SCRIPT_DIR/.venv-build" ]; then
    python3 -m venv "$SCRIPT_DIR/.venv-build"
fi
source "$SCRIPT_DIR/.venv-build/bin/activate"

pip install --upgrade pip pyinstaller

# Build .app bundle
pyinstaller \
    --onefile \
    --name "vLLM Manager" \
    --windowed \
    --noconfirm \
    --osx-bundle-identifier "com.vllm-manager.app" \
    "$SCRIPT_DIR/$ENTRY"

echo ""
echo "=== Build complete ==="
echo "App bundle: $SCRIPT_DIR/dist/vLLM Manager.app"
echo ""
echo "Per installare, copia 'dist/vLLM Manager.app' nella cartella Applicazioni."
