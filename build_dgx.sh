#!/usr/bin/env bash
# Build vLLM Manager for DGX Spark (ARM64 / aarch64) — PyInstaller
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="vLLM Manager"
ENTRY="vllm_manager.py"

echo "=== Building $APP_NAME for DGX Spark (ARM64) ==="

# Verify architecture
ARCH="$(uname -m)"
if [ "$ARCH" != "aarch64" ]; then
    echo "WARNING: Current arch is $ARCH, not aarch64."
    echo "Il binario risultante sara' per $ARCH, non per DGX Spark."
    echo "Per un build nativo, esegui questo script direttamente sul DGX Spark."
fi

# Ensure venv
if [ ! -d "$SCRIPT_DIR/.venv-build" ]; then
    python3 -m venv "$SCRIPT_DIR/.venv-build"
fi
source "$SCRIPT_DIR/.venv-build/bin/activate"

pip install --upgrade pip pyinstaller

# Build
pyinstaller \
    --onefile \
    --name "vllm-manager-dgx" \
    --windowed \
    --noconfirm \
    "$SCRIPT_DIR/$ENTRY"

echo ""
echo "=== Build complete ==="
echo "Binary: $SCRIPT_DIR/dist/vllm-manager-dgx"
echo ""
echo "Per eseguire sul DGX Spark:"
echo "  chmod +x dist/vllm-manager-dgx"
echo "  ./dist/vllm-manager-dgx"
