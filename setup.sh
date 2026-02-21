#!/usr/bin/env bash
# Setup script for Face Recognition Attendance System (macOS / Linux)
set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[INFO]${NC} $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; exit 1; }

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ── Check Python 3.11 ──────────────────────────────────────────────
info "Checking Python 3.11..."
PYTHON=""
for cmd in python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        version=$("$cmd" --version 2>&1 | grep -oE '3\.[0-9]+')
        if [ "$version" = "3.11" ]; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.11 is required but not found.
  macOS:   brew install python@3.11
  Ubuntu:  sudo apt install python3.11 python3.11-venv
  Windows: https://www.python.org/downloads/release/python-3119/"
fi
info "Found Python 3.11 ($PYTHON)"

# ── Check Node.js ──────────────────────────────────────────────────
info "Checking Node.js..."
if ! command -v node &>/dev/null; then
    error "Node.js is required but not found.
  Install from https://nodejs.org (v20 or higher)"
fi

NODE_MAJOR=$(node -v | grep -oE '[0-9]+' | head -1)
if [ "$NODE_MAJOR" -lt 20 ]; then
    error "Node.js 20+ is required. Found $(node -v).
  Update from https://nodejs.org"
fi
info "Found Node.js $(node -v)"

# ── Backend setup ──────────────────────────────────────────────────
info "Setting up backend..."
cd "$ROOT_DIR/backend"

if [ ! -d "venv" ]; then
    info "Creating virtual environment..."
    $PYTHON -m venv venv
fi

info "Activating virtual environment..."
source venv/bin/activate

info "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r requirements.txt -q

# ── Create data directories ───────────────────────────────────────
mkdir -p data/attendance/exports data/embeddings

# ── Download face recognition model (antelopev2) ─────────────────
ANTELOPE_DIR="$HOME/.insightface/models/antelopev2"

if [ -d "$ANTELOPE_DIR" ] && [ "$(ls -1 "$ANTELOPE_DIR"/*.onnx 2>/dev/null | wc -l)" -ge 5 ]; then
    info "antelopev2 model already installed"
else
    $PYTHON "$ROOT_DIR/download_model.py"
fi

deactivate
cd "$ROOT_DIR"

# ── Frontend setup ────────────────────────────────────────────────
info "Setting up frontend..."
cd "$ROOT_DIR/frontend"

if [ ! -f ".env" ]; then
    info "Creating frontend .env from example..."
    cp .env.example .env
fi

info "Installing Node.js dependencies..."
npm install --silent

cd "$ROOT_DIR"

echo ""
info "Setup complete!"
echo ""
echo "  To start the app, run:"
echo ""
echo "    ./start.sh"
echo ""
echo "  Or start each server manually:"
echo ""
echo "    # Terminal 1 - Backend"
echo "    cd backend && source venv/bin/activate && python main.py"
echo ""
echo "    # Terminal 2 - Frontend"
echo "    cd frontend && npm run dev"
echo ""
echo "  Then open http://localhost:5173 in your browser."
echo ""
