#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Fashion MNIST MLOps — Local Run Script (no Docker required)
#
# Starts the FastAPI backend and the React dashboard's dev server, with Vite
# proxying the API paths so the browser sees one origin — the same shape the
# deployed single-container image has.
#
#   ./run_local.sh            backend + dashboard dev server (hot reload)
#   ./run_local.sh --build    build the dashboard and let FastAPI serve it,
#                             which is exactly what production does
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"
PYTHON="python3.11"
API_PORT="${API_PORT:-8000}"
WEB_PORT="${WEB_PORT:-5173}"
BUILD_MODE=false
[ "${1:-}" = "--build" ] && BUILD_MODE=true

echo "=================================================="
echo "  Fashion MNIST MLOps — Local Development Server"
echo "=================================================="
echo ""

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[1/4] Creating virtual environment with $PYTHON ..."
    "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Python dependencies ────────────────────────────────────────────────────
echo "[2/4] Checking / installing Python dependencies ..."
pip install --quiet --upgrade pip

# api/requirements.txt pins tensorflow-cpu, which publishes no wheels for macOS
# on Apple Silicon. Swap in the plain tensorflow package there — same version,
# same API, and it is the only build that exists for that platform.
if [ "$(uname -s)" = "Darwin" ] && [ "$(uname -m)" = "arm64" ]; then
    grep -v '^tensorflow-cpu' "$ROOT/api/requirements.txt" > /tmp/fmnist-reqs.txt
    echo "tensorflow==2.17.0" >> /tmp/fmnist-reqs.txt
    pip install --quiet -r /tmp/fmnist-reqs.txt
else
    pip install --quiet -r "$ROOT/api/requirements.txt"
fi

# ── 3. Model check ────────────────────────────────────────────────────────────
MODEL_PATH="$ROOT/models/fashion_model.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "  WARNING: No trained model found at models/fashion_model.h5"
    echo "  Train it first (from the project root):"
    echo "    source venv/bin/activate && python -m src.train"
    echo ""
fi

# ── 4. Dashboard ──────────────────────────────────────────────────────────────
if ! command -v npm >/dev/null 2>&1; then
    echo "  ERROR: npm not found. Install Node 20+ to build the dashboard."
    exit 1
fi

echo "[3/4] Installing dashboard dependencies ..."
(cd "$ROOT/frontend" && npm install --silent --no-audit --no-fund)

FRONTEND_PID=""

if [ "$BUILD_MODE" = true ]; then
    echo "[4/4] Building the dashboard — FastAPI will serve it ..."
    (cd "$ROOT/frontend" && npm run build)
else
    echo "[4/4] Starting services ..."
fi

# ── Start FastAPI ─────────────────────────────────────────────────────────────
cd "$ROOT"
PYTHONPATH="$ROOT" uvicorn api.main:app \
    --host 0.0.0.0 --port "$API_PORT" --reload \
    --log-level info &
BACKEND_PID=$!

# The model load takes a few seconds; starting the dev server against a backend
# that isn't listening yet just means a wake-up screen for the first moments.
sleep 4

if [ "$BUILD_MODE" = false ]; then
    # VITE_API_TARGET lets the proxy follow API_PORT when 8000 is already taken.
    (cd "$ROOT/frontend" && VITE_API_TARGET="http://127.0.0.1:$API_PORT" \
        npm run dev -- --port "$WEB_PORT" --host) &
    FRONTEND_PID=$!
    sleep 3
fi

echo ""
if [ "$BUILD_MODE" = true ]; then
    echo "  Dashboard + API : http://localhost:$API_PORT"
else
    echo "  Dashboard : http://localhost:$WEB_PORT   (hot reload)"
    echo "  API       : http://localhost:$API_PORT"
fi
echo "  API Docs  : http://localhost:$API_PORT/docs"
echo ""
echo "  Press Ctrl+C to stop."

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$BACKEND_PID" 2>/dev/null || true
    [ -n "$FRONTEND_PID" ] && kill "$FRONTEND_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

wait
