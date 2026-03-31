#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Fashion MNIST MLOps — Local Run Script (no Docker required)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="$ROOT/venv"
PYTHON="python3.11"

echo "=================================================="
echo "  Fashion MNIST MLOps — Local Development Server"
echo "=================================================="
echo ""

# ── 1. Virtual environment ────────────────────────────────────────────────────
if [ ! -d "$VENV" ]; then
    echo "[1/3] Creating virtual environment with $PYTHON ..."
    "$PYTHON" -m venv "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"

# ── 2. Install dependencies ───────────────────────────────────────────────────
echo "[2/3] Checking / installing dependencies ..."
pip install --quiet --upgrade pip
pip install --quiet \
    "fastapi==0.115.0" "uvicorn[standard]==0.30.6" "python-multipart==0.0.9" \
    "pydantic==2.9.2" "tensorflow-cpu==2.17.0" "numpy==1.26.4" "pandas==2.2.3" \
    "scikit-learn==1.5.2" "Pillow==10.4.0" \
    "streamlit==1.39.0" "requests==2.32.3" \
    "matplotlib==3.9.2" "seaborn==0.13.2"

# ── 3. Check model ────────────────────────────────────────────────────────────
MODEL_PATH="$ROOT/models/fashion_model.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo ""
    echo "  WARNING: No trained model found at models/fashion_model.h5"
    echo "  Train the model first (from the project root):"
    echo "    source venv/bin/activate"
    echo "    python -m src.train"
    echo ""
fi

echo "[3/3] Starting services ..."

# ── 4. Start FastAPI backend ──────────────────────────────────────────────────
cd "$ROOT"
PYTHONPATH="$ROOT" uvicorn api.main:app \
    --host 0.0.0.0 --port 8000 --reload \
    --log-level info &
BACKEND_PID=$!

sleep 4

# ── 5. Start Streamlit frontend ───────────────────────────────────────────────
API_URL=http://localhost:8000 streamlit run "$ROOT/frontend/app.py" \
    --server.port 8501 --server.address 0.0.0.0 \
    --server.headless true \
    --browser.gatherUsageStats false &
FRONTEND_PID=$!

echo ""
echo "  Backend   : http://localhost:8000"
echo "  API Docs  : http://localhost:8000/docs"
echo "  Frontend  : http://localhost:8501"
echo ""
echo "  Press Ctrl+C to stop all services."

# ── Cleanup on exit ───────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Stopping services..."
    kill "$BACKEND_PID"  2>/dev/null || true
    kill "$FRONTEND_PID" 2>/dev/null || true
    echo "Done."
}
trap cleanup EXIT INT TERM

wait
