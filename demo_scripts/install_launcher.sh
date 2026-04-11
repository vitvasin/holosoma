#!/usr/bin/env bash
# install_launcher.sh — Install the Holosoma Workflow Launcher desktop icon
#
# Run once per workstation:
#   bash demo_scripts/install_launcher.sh
#
# What it does:
#   1. Resolves the project root (works from any working directory / symlink)
#   2. Installs required Python dependencies (PySide6, matplotlib, tbparse)
#   3. Creates ~/.local/share/applications/holosoma-launcher.desktop
#   4. Creates a Desktop shortcut (if ~/Desktop exists)
#   5. Marks the launcher script as executable

set -e

# ── Resolve project root ──────────────────────────────────────────────────────
SOURCE="${BASH_SOURCE[0]:-${(%):-%x}}"
while [ -h "$SOURCE" ]; do
  DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
  SOURCE="$(readlink "$SOURCE")"
  [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
done
SCRIPT_DIR="$( cd -P "$( dirname "$SOURCE" )" >/dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
LAUNCHER="$SCRIPT_DIR/workflow_launcher.py"

echo "Project root : $PROJECT_ROOT"
echo "Launcher     : $LAUNCHER"

# ── Make launcher executable ──────────────────────────────────────────────────
chmod +x "$LAUNCHER"

# ── Detect python ─────────────────────────────────────────────────────────────
PYTHON="$(which python3 2>/dev/null || which python)"
if [ -z "$PYTHON" ]; then
  echo "ERROR: python3 not found on PATH."
  exit 1
fi
echo "Python       : $PYTHON ($($PYTHON --version 2>&1))"

# ── Install Python dependencies ───────────────────────────────────────────────
echo ""
echo "Installing Python dependencies..."
"$PYTHON" -m pip install --user --quiet 'PySide6>=6.5' 'matplotlib>=3.8' tbparse psutil
echo "Dependencies OK."

# ── Pick an icon ─────────────────────────────────────────────────────────────
# Prefer SVG (supported by all modern DEs); fall back to PNG or system icon
ICON_SVG="$PROJECT_ROOT/demo_scripts/holosoma_icon.svg"
ICON_PNG="$PROJECT_ROOT/demo_scripts/holosoma_icon.png"
if [ -f "$ICON_SVG" ]; then
  ICON="$ICON_SVG"
elif [ -f "$ICON_PNG" ]; then
  ICON="$ICON_PNG"
else
  ICON="utilities-terminal"
fi
echo "Icon         : $ICON"

# ── Write .desktop file ───────────────────────────────────────────────────────
APPS_DIR="$HOME/.local/share/applications"
mkdir -p "$APPS_DIR"
DESKTOP_FILE="$APPS_DIR/holosoma-launcher.desktop"

cat > "$DESKTOP_FILE" <<EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=Holosoma Launcher
Comment=Holosoma Whole-Body Tracking Workflow Launcher
Exec=$PYTHON $LAUNCHER
Icon=$ICON
Terminal=false
Categories=Science;Robotics;
StartupWMClass=workflow_launcher
EOF

chmod +x "$DESKTOP_FILE"
echo "Desktop entry: $DESKTOP_FILE"

# ── Add Desktop shortcut if ~/Desktop exists ──────────────────────────────────
DESKTOP_DIR="${XDG_DESKTOP_DIR:-$HOME/Desktop}"
if [ -d "$DESKTOP_DIR" ]; then
  cp "$DESKTOP_FILE" "$DESKTOP_DIR/holosoma-launcher.desktop"
  chmod +x "$DESKTOP_DIR/holosoma-launcher.desktop"
  # Some DEs require trust validation
  gio set "$DESKTOP_DIR/holosoma-launcher.desktop" metadata::trusted true 2>/dev/null || true
  echo "Desktop icon : $DESKTOP_DIR/holosoma-launcher.desktop"
fi

# ── Refresh application database ──────────────────────────────────────────────
update-desktop-database "$APPS_DIR" 2>/dev/null || true

echo ""
echo "Done! You can now launch Holosoma from the desktop icon or run:"
echo "  $PYTHON $LAUNCHER"
