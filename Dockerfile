FROM python:3.11-slim

LABEL maintainer="Fashion MNIST MLOps"
LABEL description="Streamlit frontend for Fashion MNIST MLOps dashboard"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

COPY frontend/          /app/
COPY outputs/figures/  /app/figures/
COPY data/test/        /data/test/

ENV API_URL=https://cholatemgiet-fashion-mnist-api.hf.space

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
