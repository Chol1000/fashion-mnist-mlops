FROM python:3.11-slim

LABEL maintainer="Fashion MNIST MLOps"
LABEL description="FastAPI backend for Fashion MNIST classification"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        gcc \
    && rm -rf /var/lib/apt/lists/*

COPY api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY src/    /app/src/
COPY api/    /app/api/
COPY models/ /app/models/
COPY data/   /data/

RUN mkdir -p /app/api/data

ENV TF_CPP_MIN_LOG_LEVEL=3
ENV CUDA_VISIBLE_DEVICES=""
ENV TF_NUM_INTRAOP_THREADS=2
ENV TF_NUM_INTEROP_THREADS=2
ENV OMP_NUM_THREADS=2

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
