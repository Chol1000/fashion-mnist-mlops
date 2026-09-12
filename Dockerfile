# ─────────────────────────────────────────────────────────────────────────────
# Single-container build: stage 1 compiles the React dashboard, stage 2 serves
# it and the API from one FastAPI process.
#
# One container rather than two is the point. When the dashboard and the API
# are separate deployments, each sleeps on its own idle timer, so opening the
# dashboard finds an API that is still asleep — the worst version of the
# free-tier cold start, because the thing the user opened looks up while the
# thing it needs is not. Here there is exactly one service: if the page loads
# at all, the API behind it is already running.
#
# Runs unchanged on Hugging Face Spaces, Render and Cloud Run.
# ─────────────────────────────────────────────────────────────────────────────

FROM node:20-slim AS frontend-build
WORKDIR /build

# Dependencies first, so a change to application source doesn't re-run npm ci.
COPY frontend/package*.json ./
RUN npm ci

COPY frontend/ ./
# Same-origin by default: FastAPI serves this bundle, so the API is wherever
# the page is. Only override VITE_API_BASE when deploying the dashboard away
# from the API, which reintroduces the two-services-sleeping-apart problem.
ARG VITE_API_BASE=""
ARG VITE_SPACE_URL=""
ENV VITE_API_BASE=$VITE_API_BASE
ENV VITE_SPACE_URL=$VITE_SPACE_URL
RUN npm run build


FROM python:3.11-slim

LABEL maintainer="Fashion MNIST MLOps"
LABEL description="FastAPI backend and React dashboard for Fashion MNIST classification"

# Spaces run the container as uid 1000, so anything written at runtime has to
# live somewhere that user owns — the SQLite database and the metrics file
# that a retraining run updates both do. Creating the user here and chowning
# once is what keeps a Space that builds fine from dying on its first upload.
RUN useradd -m -u 1000 appuser

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/appuser \
    TF_CPP_MIN_LOG_LEVEL=3 \
    CUDA_VISIBLE_DEVICES="" \
    TF_NUM_INTRAOP_THREADS=2 \
    TF_NUM_INTEROP_THREADS=2 \
    OMP_NUM_THREADS=2

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY --chown=appuser:appuser src/            /app/src/
COPY --chown=appuser:appuser api/            /app/api/
COPY --chown=appuser:appuser models/         /app/models/
COPY --chown=appuser:appuser outputs/figures/ /app/figures/
COPY --chown=appuser:appuser data/           /data/
COPY --chown=appuser:appuser --from=frontend-build /build/dist /app/frontend/dist

# SQLite lives here; the directory must exist and be writable before uid 1000
# takes over.
RUN mkdir -p /app/api/data && chown -R appuser:appuser /app/api/data /data

USER appuser

# Hosts disagree on which port to serve: Render and Cloud Run inject $PORT,
# while Spaces expects the app_port declared in the Space README frontmatter.
# Honour $PORT when set and fall back to 7860, so one image runs on any of them.
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -f http://localhost:${PORT:-7860}/health || exit 1

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-7860} --workers 1"]
