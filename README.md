---
title: Fashion MNIST API
emoji: 👔
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

# Fashion MNIST MLOps Pipeline

End-to-end MLOps system for clothing image classification using the Fashion MNIST dataset.

**Video Demo:** [YouTube Link](https://youtu.be/YOUR_VIDEO_ID)

**GitHub:** https://github.com/Chol1000/fashion-mnist-mlops

**Live Frontend:** https://fashion-mnist-frontend.onrender.com

**Live API:** https://cholatemgiet-fashion-mnist-api.hf.space

**API Docs:** https://cholatemgiet-fashion-mnist-api.hf.space/docs

---

## Project Overview

This project builds a complete MLOps pipeline for a 10-class clothing image classifier. It covers the full lifecycle from raw data to a deployed, monitored, and retrainable system.

The model is a MobileNetV2 transfer learning architecture trained on Fashion MNIST, achieving 93.17% test accuracy. Training follows a two-phase strategy: feature extraction on a frozen base followed by fine-tuning of the last 80 layers. The trained model is served through a FastAPI backend, visualised in a Streamlit dashboard, containerised with Docker, load-balanced with Nginx, and load-tested with Locust.

**Dataset:** Fashion MNIST — 70,000 greyscale 28×28 images across 10 balanced clothing categories.

**Model:** MobileNetV2 (ImageNet pre-trained) with a custom head — input resized from 28×28 to 128×128×3.

---

## Repository Structure

```
Summative_MLOP/
|
|-- notebook/
|   `-- fashion_mnist_mlops.ipynb    # Self-contained training notebook (run on Colab)
|
|-- src/
|   |-- preprocessing.py             # Data loading, cleaning, tf.data pipeline
|   |-- model.py                     # MobileNetV2 transfer learning, two-phase training
|   |-- prediction.py                # Inference wrapper used by the API
|   `-- train.py                     # Standalone CLI training script
|
|-- api/
|   |-- main.py                      # FastAPI endpoints (predict, retrain, metrics, insights)
|   |-- database.py                  # SQLite storage for uploaded samples and retrain logs
|   `-- requirements.txt
|
|-- frontend/
|   |-- app.py                       # Streamlit dashboard
|   `-- requirements.txt
|
|-- locust/
|   `-- locustfile.py                # Load testing scenarios
|
|-- data/
|   |-- train/                       # Place fashion-mnist_train.csv here (downloaded by notebook)
|   `-- test/                        # Place fashion-mnist_test.csv here (downloaded by notebook)
|
|-- models/
|   |-- fashion_model.h5             # Trained model — download from Colab, place here
|   `-- training_metrics.json        # Accuracy, F1, per-class metrics, training history
|
|-- outputs/
|   `-- figures/                     # EDA plots, training curves, confusion matrix
|
|-- nginx.conf                       # Nginx load balancer config
|-- docker-compose.yml               # Full stack: api, frontend, locust, nginx
|-- Dockerfile.api
|-- Dockerfile.frontend
|-- requirements.txt
`-- run_local.sh                     # One-command local start without Docker
```

> **Note:** `data/` CSV files are stored via Git LFS (148 MB total) — clone the repo normally and they download automatically. `models/fashion_model.h5` is included in the repo. Both are ready to use without any manual steps.

---

## Setup and Running

### Option A — Docker (Recommended)

Requires Docker Desktop installed and running.

```bash
git clone <repo-url>
cd Summative_MLOP

# Build and start all services
docker compose up --build

# Scale the API to multiple replicas for load testing
docker compose up --scale backend=3
```

Once running:

- Frontend dashboard: http://localhost:8501
- API: http://localhost:8000
- Interactive API docs: http://localhost:8000/docs
- Locust load testing: http://localhost:8089

---

### Option B — Local (No Docker)

Requires Python 3.11.

```bash
git clone <repo-url>
cd Summative_MLOP

bash run_local.sh
```

---

### Option C — Manual Setup

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

# Start API (from project root)
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

# Start frontend (new terminal)
API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501
```

---

## Training the Model

The model must be trained before the API can serve predictions.

### Option A — Jupyter Notebook (recommended)

Open `notebook/fashion_mnist_mlops.ipynb`. The notebook is fully self-contained — no external imports required.

To run on Google Colab with GPU:

1. Upload `notebook/fashion_mnist_mlops.ipynb` to Colab
2. Set `Runtime > Change runtime type > T4 GPU`
3. Run all cells — Fashion MNIST data downloads automatically via Keras, no uploads or Drive needed
4. The final cell downloads `fashion_model.h5` and `training_metrics.json` directly to your machine
5. Place both files in the local `models/` folder before running the API

### Option B — Standalone script

```bash
source venv/bin/activate
python -m src.train
```

Saves `models/fashion_model.h5` and `models/training_metrics.json`.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service information |
| GET | `/health` | Model readiness and database stats |
| POST | `/predict` | Predict from 784 pixel values |
| POST | `/predict/image` | Predict from uploaded PNG or JPG |
| POST | `/upload-data` | Upload CSV for retraining |
| POST | `/retrain` | Trigger fine-tuning on uploaded data |
| GET | `/retrain/status` | Poll current training progress |
| GET | `/retrain/history` | Past retraining run logs |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/insights` | Dataset statistics |

Full interactive documentation: http://localhost:8000/docs

---

## Load Testing

```bash
pip install locust

locust -f locust/locustfile.py --host http://localhost:8000
# Open http://localhost:8089 and configure users, spawn rate, and run time

# Headless run with CSV output
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 60s --headless \
       --csv locust/results/run_50u
```

---

## Model Architecture

```
Input (128 x 128 x 3)
  MobileNetV2 base  [ImageNet weights, include_top=False]
    output: (4, 4, 1280)
    GlobalAveragePooling2D  ->  (1280,)
      Dense(256, relu, L2=1e-4)
        BatchNormalization
          Dropout(0.5)
            Dense(10, softmax)
```

Total parameters: approximately 2.6M. Trainable in Phase 1: approximately 330K (head only).

Training strategy:

- Phase 1 — Feature extraction: MobileNetV2 base fully frozen, Adam lr=1e-3, EarlyStopping patience=5
- Phase 2 — Fine-tuning: last 80 layers unfrozen, Adam lr=1e-5, EarlyStopping patience=5

---

## Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | 93.17% |
| Test Loss | 0.2377 |
| Macro F1 | 0.9320 |
| Macro Precision | 0.9324 |
| Macro Recall | 0.9317 |

The most frequently confused classes are Shirt, T-shirt/top, and Coat due to overlapping silhouettes in greyscale. Bag and Trouser are the easiest to classify due to visually distinct shapes.

---

## Retraining Workflow

1. Open the **Upload & Retrain** tab in the Streamlit dashboard
2. Upload a CSV with columns: `label, pixel1, ..., pixel784`
3. Uploaded samples are stored in SQLite for traceability
4. Click **Start Retraining** — the model is fine-tuned with Adam lr=1e-4
5. Updated metrics are displayed in the dashboard and logged to the database

> **Sample data for testing:** A ready-to-use 100-row sample CSV is available at `data/sample_retrain.csv`. Download it and upload it directly in the Upload & Retrain tab to test the full retraining pipeline.

---

## Load Testing Results (Locust)

Flood simulation using Locust with two user classes:
- **FashionAPIUser** (25 users): realistic mix of `/health`, `/predict`, `/metrics`, `/insights`
- **HeavyPredictUser** (25 users): rapid-fire `/predict` calls only

```bash
# 1 container — 10 users
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 30s --headless \
       --csv locust/results/run_10u_local

# 1 container — 50 users
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 30s --headless \
       --csv locust/results/run_50u_local

# 3 containers behind Nginx — 50 users
docker compose up -d --scale backend=3
locust -f locust/locustfile.py --host http://localhost:80 \
       --users 50 --spawn-rate 5 --run-time 30s --headless \
       --csv locust/results/run_50u_3replicas
```

### Overall throughput — all endpoints

| Containers | Users | Total Requests | Requests/s | Median Latency | 95th pct | Failures |
|-----------|-------|---------------|-----------|----------------|----------|---------|
| 1         | 10    | 569           | 21.79/s   | 46 ms          | 78 ms    | 0 (0%)  |
| 1         | 50    | 1,601         | 55.56/s   | 250 ms         | 470 ms   | 0 (0%)  |
| 3 (Nginx) | 50    | 1,357         | 46.70/s   | 340 ms         | 600 ms   | 0 (0%)  |

### `/predict` endpoint

| Containers | Users | Avg Latency | Median | 95th pct | Req/s |
|-----------|-------|------------|--------|----------|-------|
| 1         | 10    | 63 ms      | 51 ms  | 78 ms    | 2.37  |
| 1         | 50    | 261 ms     | 260 ms | 490 ms   | 6.49  |
| 3 (Nginx) | 50    | 388 ms     | 370 ms | 920 ms   | 5.82  |

> **Zero failures across all runs.** The 3-replica run routes through Nginx (port 80) adding overhead vs direct port 8000 access, which accounts for the slightly higher latency. All configurations handle 50 concurrent users with 100% success rate, demonstrating the stability of the pipeline under concurrent load.

### Screenshots

**Statistics (all endpoints, 50 users through Nginx)**

![Locust Statistics](locust/results/statistics.png)

**Request rate and response time over time**

![Locust Charts](locust/results/charts.png)

**Failures — zero failures across all runs**

![Locust Failures](locust/results/failures.png)
