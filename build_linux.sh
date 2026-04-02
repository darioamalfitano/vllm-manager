#!/usr/bin/env bash
# Build vLLM Manager for Linux (x86_64) — PyInstaller + optional AppImage
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
APP_NAME="vLLM Manager"
ENTRY="vllm_manager.py"

echo "=== Building $APP_NAME for Linux ==="

# Ensure venv
if [ ! -d "$SCRIPT_DIR/.venv-build" ]; then
    python3 -m venv "$SCRIPT_DIR/.venv-build"
fi
source "$SCRIPT_DIR/.venv-build/bin/activate"

pip install --upgrade pip pyinstaller

# Build with PyInstaller
pyinstaller \
    --onefile \
    --name "vllm-manager" \
    --windowed \
    --noconfirm \
    "$SCRIPT_DIR/$ENTRY"

echo ""
echo "=== Build complete ==="
echo "Binary: $SCRIPT_DIR/dist/vllm-manager"

# Optional: create AppImage if appimagetool is available
if command -v appimagetool &>/dev/null; then
    echo "=== Creating AppImage ==="
    APPDIR="$SCRIPT_DIR/dist/vllm-manager.AppDir"
    mkdir -p "$APPDIR/usr/bin"
    cp "$SCRIPT_DIR/dist/vllm-manager" "$APPDIR/usr/bin/"

    cat > "$APPDIR/AppRun" << 'EOF'
#!/bin/bash
SELF="$(readlink -f "$0")"
HERE="${SELF%/*}"
exec "${HERE}/usr/bin/vllm-manager" "$@"
EOF
    chmod +x "$APPDIR/AppRun"

    cat > "$APPDIR/vllm-manager.desktop" << EOF
[Desktop Entry]
Name=vLLM Manager
Exec=vllm-manager
Icon=vllm-manager
Type=Application
Categories=Development;
EOF

    # Placeholder icon
    touch "$APPDIR/vllm-manager.png"

    appimagetool "$APPDIR" "$SCRIPT_DIR/dist/vllm-manager-x86_64.AppImage"
    echo "AppImage: $SCRIPT_DIR/dist/vllm-manager-x86_64.AppImage"
else
    echo "(appimagetool non trovato, AppImage non creato)"
fi
