#!/usr/bin/env bash
# =============================================================================
# Video Translator — Linux/macOS Installer
# =============================================================================
# Usage:
#   chmod +x scripts/install.sh
#   ./scripts/install.sh              # Auto-detect
#   ./scripts/install.sh --cpu        # CPU-only
#   ./scripts/install.sh --cuda 12.4  # Force CUDA version
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/venv"

echo "=== Video Translator Installer ==="
echo "Project root: $PROJECT_ROOT"

# Check Python version
PYTHON=""
for candidate in python3.10 python3.11 python3.12 python3; do
    if command -v "$candidate" &>/dev/null; then
        version=$("$candidate" --version 2>&1 | grep -oP '\d+\.\d+')
        major=$(echo "$version" | cut -d. -f1)
        minor=$(echo "$version" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 10 ] && [ "$minor" -le 12 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10-3.12 is required but not found."
    echo "Install it with:"
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "  brew install python@3.10"
    else
        echo "  sudo apt-get install python3.10 python3.10-venv"
    fi
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create venv if needed
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# Activate and run the installer
source "$VENV_DIR/bin/activate"
python "$SCRIPT_DIR/install.py" "$@"
