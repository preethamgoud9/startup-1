#!/usr/bin/env bash
# Download and install the antelopev2 (ResNet100) model for higher accuracy.
# This is optional — the system works with buffalo_l (ResNet50) out of the box.
set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

MODEL_DIR="$HOME/.insightface/models/antelopev2"
RELEASE_URL="https://github.com/preethamgoud2912/production_1/releases/download/v1.0.0/antelopev2.zip"
TMP_ZIP="/tmp/antelopev2.zip"

if [ -d "$MODEL_DIR" ] && [ "$(ls -1 "$MODEL_DIR"/*.onnx 2>/dev/null | wc -l)" -ge 5 ]; then
    info "antelopev2 model is already installed at $MODEL_DIR"
    exit 0
fi

info "Downloading antelopev2 model (~344MB)..."
info "This is a one-time download for the high-accuracy ResNet100 model."
echo ""

if command -v curl &>/dev/null; then
    curl -L --progress-bar -o "$TMP_ZIP" "$RELEASE_URL"
elif command -v wget &>/dev/null; then
    wget --show-progress -O "$TMP_ZIP" "$RELEASE_URL"
else
    error "Neither curl nor wget found. Install one and try again."
fi

info "Extracting model files..."
mkdir -p "$HOME/.insightface/models"
unzip -o "$TMP_ZIP" -d "$HOME/.insightface/models/" > /dev/null

rm -f "$TMP_ZIP"

# Verify
if [ "$(ls -1 "$MODEL_DIR"/*.onnx 2>/dev/null | wc -l)" -ge 5 ]; then
    info "antelopev2 model installed successfully!"
    info "Restart the backend to use the upgraded model."
else
    error "Installation failed. Model files not found at $MODEL_DIR"
fi
