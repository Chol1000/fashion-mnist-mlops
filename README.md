---
title: Fashion MNIST API
emoji: 👔
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
---

<div align="center">

# Fashion MNIST MLOps Pipeline

**End-to-end Machine Learning pipeline for clothing image classification**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.17-orange?logo=tensorflow)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![Streamlit](https://img.shields.io/badge/Streamlit-1.39-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow?logo=huggingface)

| Resource | Description | Link |
|----------|-------------|------|
| Video Demo | Full walkthrough — prediction, retraining, live deployment (camera on) | [Watch on YouTube](https://youtu.be/Nw2GPJmoh20) |
| Live Dashboard (Public) | Streamlit UI — predict, retrain, view metrics and insights | [Open Dashboard](https://cholatemgiet-fashion-mnist-frontend.hf.space) |
| Live API (Public) | FastAPI backend — hosted on Hugging Face Spaces (16 GB RAM, zero cost) | [Open API](https://cholatemgiet-fashion-mnist-api.hf.space) |
| Interactive API Docs | Swagger UI — test all endpoints directly in the browser | [Open Swagger](https://cholatemgiet-fashion-mnist-api.hf.space/docs) |
| GitHub Repository | Full source code — notebook, API, frontend, Docker, Locust | [View on GitHub](https://github.com/Chol1000/fashion-mnist-mlops) |

</div>

---

## Project Overview

This project implements a production-grade MLOps pipeline for classifying 10 categories of clothing images using the Fashion MNIST dataset. It covers the complete machine learning lifecycle:

| Component | Technology |
|-----------|-----------|
| Data preprocessing | TensorFlow tf.data, NumPy, Pandas |
| Model training | MobileNetV2 transfer learning |
| API serving | FastAPI + Uvicorn |
| Dashboard | Streamlit |
| Database | SQLite |
| Containerisation | Docker + Nginx |
| Load testing | Locust |
| Cloud deployment | Hugging Face Spaces |

**Dataset:** Fashion MNIST — 70,000 greyscale 28×28 images across 10 balanced clothing categories (60,000 train / 10,000 test).

**Model:** MobileNetV2 pre-trained on ImageNet with a custom classification head. Input images are resized from 28×28 to 128×128×3 (RGB).

---

## Model Performance

| Metric | Score |
|--------|-------|
| Test Accuracy | **93.17%** |
| Test Loss | 0.2377 |
| Macro F1 Score | **0.9320** |
| Macro Precision | 0.9324 |
| Macro Recall | 0.9317 |

The most frequently confused classes are Shirt, T-shirt/top, and Coat due to overlapping silhouettes in greyscale. Bag and Trouser are the easiest to classify due to visually distinct shapes.

---

## Dataset Visualisations

### Class Distribution
![Class Distribution](outputs/figures/eda_01_class_distribution.png)

The dataset is perfectly balanced — each of the 10 categories contains exactly 6,000 training samples, eliminating class imbalance as a source of bias.

### Sample Images per Class
![Sample Images](outputs/figures/eda_02_sample_images.png)

Visual inspection reveals why certain classes are harder to distinguish — Pullovers, Shirts, and Coats share similar collar and sleeve shapes in greyscale.

### Pixel Intensity Distribution
![Pixel Intensity](outputs/figures/eda_03_pixel_intensity.png)

Most pixel values cluster at 0 (black background), with foreground clothing pixels spread across the 100–255 range. This bimodal distribution informs our normalisation strategy.

### Mean Images per Class
![Mean Images](outputs/figures/eda_04_mean_images.png)

Averaging all images per class reveals the archetypal shape of each garment — Bags and Trousers have the clearest silhouettes while Shirts blend with T-shirts.

---

## Model Architecture

```
Input (128 × 128 × 3)
└── MobileNetV2 base [ImageNet weights, include_top=False]
    └── GlobalAveragePooling2D → (1280,)
        └── Dense(256, relu, L2=1e-4)
            └── BatchNormalization
                └── Dropout(0.5)
                    └── Dense(10, softmax)
```

**Total parameters:** ~2.6M | **Trainable in Phase 1:** ~330K (head only)

**Training strategy:**
- **Phase 1 — Feature extraction:** MobileNetV2 base frozen, Adam lr=1e-3, EarlyStopping patience=5
- **Phase 2 — Fine-tuning:** Last 80 layers unfrozen, Adam lr=1e-5, EarlyStopping patience=5

### Training History
![Training History](outputs/figures/training_history.png)

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Per-Class Metrics
![Per Class Metrics](outputs/figures/per_class_metrics.png)

### Preprocessing Pipeline
![Preprocessing Pipeline](outputs/figures/preprocessing_pipeline.png)

---

## Repository Structure

```
fashion-mnist-mlops/
│
├── notebook/
│   └── fashion_mnist_mlops.ipynb    # Full training notebook (Colab-ready)
│
├── src/
│   ├── preprocessing.py             # Data loading, normalisation, tf.data pipeline
│   ├── model.py                     # MobileNetV2 architecture, two-phase training
│   ├── prediction.py                # Inference wrapper used by the API
│   └── train.py                     # Standalone CLI training script
│
├── api/
│   ├── main.py                      # FastAPI endpoints (predict, retrain, metrics, insights)
│   ├── database.py                  # SQLite storage for uploaded samples and retrain logs
│   └── requirements.txt
│
├── frontend/
│   ├── app.py                       # Streamlit dashboard
│   └── requirements.txt
│
├── locust/
│   ├── locustfile.py                # Load testing scenarios
│   └── results/                     # CSV outputs and screenshots
│
├── data/
│   ├── train/                       # fashion-mnist_train.csv (Git LFS)
│   ├── test/                        # fashion-mnist_test.csv (Git LFS)
│   └── sample_retrain.csv           # 100-row sample for testing retraining
│
├── models/
│   ├── fashion_model.h5             # Trained MobileNetV2 model
│   └── training_metrics.json        # Accuracy, F1, per-class metrics
│
├── outputs/
│   └── figures/                     # EDA plots, training curves, confusion matrix
│
├── nginx.conf                       # Nginx load balancer config
├── docker-compose.yml               # Full stack: api, frontend, locust, nginx
├── Dockerfile.api                   # API container
├── Dockerfile.frontend              # Frontend container
└── run_local.sh                     # One-command local start without Docker
```

> **Note:** `data/` CSV files are stored via Git LFS — clone normally and they download automatically. `models/fashion_model.h5` is included in the repo. No manual steps required.

---

## Quick Start

### Option A — Docker (Recommended)

Requires Docker Desktop installed and running.

```bash
git clone https://github.com/Chol1000/fashion-mnist-mlops.git
cd fashion-mnist-mlops

# Build and start all services
docker compose up --build
```

| Service | URL |
|---------|-----|
| Frontend Dashboard | http://localhost:8501 |
| API | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |
| Locust Load Testing | http://localhost:8089 |

Scale API to multiple replicas:
```bash
docker compose up --scale backend=3
```

---

### Option B — Local (No Docker)

Requires Python 3.11.

```bash
git clone https://github.com/Chol1000/fashion-mnist-mlops.git
cd fashion-mnist-mlops
bash run_local.sh
```

---

### Option C — Manual Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — Start API
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Start Frontend
API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501
```

---

## Training the Model

### Option A — Google Colab (Recommended)

1. Upload `notebook/fashion_mnist_mlops.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set `Runtime > Change runtime type > T4 GPU`
3. Run all cells — Fashion MNIST downloads automatically via Keras
4. The final cell downloads `fashion_model.h5` and `training_metrics.json` to your machine
5. Place both files in the `models/` folder

### Option B — Standalone Script

```bash
source venv/bin/activate
python -m src.train
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Service information and uptime |
| GET | `/health` | Model readiness and database stats |
| POST | `/predict` | Predict from 784 raw pixel values |
| POST | `/predict/image` | Predict from uploaded PNG/JPG image |
| POST | `/upload-data` | Upload labelled CSV for retraining |
| POST | `/retrain` | Trigger fine-tuning on uploaded data |
| GET | `/retrain/status` | Poll live training progress |
| GET | `/retrain/history` | View past retraining run logs |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/insights` | Dataset statistics and class distribution |

Full interactive documentation: https://cholatemgiet-fashion-mnist-api.hf.space/docs

---

## Retraining Workflow

1. Open the **Upload & Retrain** tab in the dashboard
2. Upload a CSV with columns: `label, pixel1, pixel2, ..., pixel784`
3. Samples are validated, cleaned, and stored in SQLite
4. Click **Start Retraining** — the model is fine-tuned using Adam lr=1e-4
5. Live epoch progress is displayed during training
6. Updated metrics are shown in the dashboard and logged to the database

### Sample Data for Testing

A ready-to-use file is included at `data/sample_retrain.csv`.

| Property | Detail |
|----------|--------|
| Rows | 100 samples |
| Source | Fashion MNIST test set (randomly sampled) |
| Format | `label` (0–9) + `pixel1`…`pixel784` (values 0–255) |

**Steps:**
1. Download `data/sample_retrain.csv` from the GitHub repo
2. Open the dashboard → **Upload & Retrain** tab
3. Upload the file → validated and stored in database
4. Click **Start Retraining** → watch live epoch progress
5. View updated accuracy and F1 score once complete

---

## Load Testing Results

Load tests were run against the **live production API** on Hugging Face Spaces using Locust with two concurrent user classes:

- **FashionAPIUser** — realistic mix of `GET /health`, `POST /predict`, `GET /metrics`, `GET /insights` (wait: 0.5–2 s)
- **HeavyPredictUser** — rapid-fire `POST /predict` only (wait: 0.1–0.3 s)

**Host tested:** `https://cholatemgiet-fashion-mnist-api.hf.space` | **Run time:** 45 s each

### Run Commands

```bash
# 10 users
locust -f locust/locustfile.py \
       --host https://cholatemgiet-fashion-mnist-api.hf.space \
       --users 10 --spawn-rate 2 --run-time 45s --headless \
       --csv locust/results/run_10u_live

# 50 users
locust -f locust/locustfile.py \
       --host https://cholatemgiet-fashion-mnist-api.hf.space \
       --users 50 --spawn-rate 5 --run-time 45s --headless \
       --csv locust/results/run_50u_live

# 100 users
locust -f locust/locustfile.py \
       --host https://cholatemgiet-fashion-mnist-api.hf.space \
       --users 100 --spawn-rate 10 --run-time 45s --headless \
       --csv locust/results/run_100u_live
```

### Overall Throughput — All Endpoints

| Users | Total Requests | Req/s | Median Latency | 95th pct | Failures |
|-------|---------------|-------|----------------|----------|----------|
| 10 | 410 | 9.29/s | 400 ms | 1,000 ms | 0 (0%) |
| 50 | 731 | 16.99/s | 1,600 ms | 2,300 ms | 1 (0.1%) |
| 100 | 549 | 12.45/s | 5,200 ms | 7,700 ms | 0 (0%) |

### `/predict` Endpoint

| Users | Total Requests | Avg Latency | Median | 95th pct | Req/s |
|-------|---------------|------------|--------|----------|-------|
| 10 | 68 | 615 ms | 410 ms | 2,800 ms | 1.54 |
| 50 | 150 | 1,651 ms | 1,700 ms | 2,400 ms | 3.49 |
| 100 | 139 | 4,916 ms | 5,200 ms | 7,700 ms | 3.15 |

> All tests ran against a **free-tier shared CPU** on Hugging Face Spaces — latency increases at high concurrency are expected and reflect the shared infrastructure, not the application code. At 10 users the API handles predictions in under 500 ms median with zero failures. At 100 concurrent users the API remains stable (zero failures) despite higher latency under the shared CPU constraint.

### Screenshots

**Statistics — all endpoints, 50 users, live HF Space**

![Locust Statistics](locust/results/statistics.png)

**Request rate and response time over time**

![Locust Charts](locust/results/charts.png)

**Failures — near-zero across all runs**

![Locust Failures](locust/results/failures.png)

---

## Dataset

**Fashion MNIST** is a dataset of Zalando's article images — a drop-in replacement for the original MNIST handwritten digits dataset.

| Property | Detail |
|----------|--------|
| Source | Zalando Research |
| Classes | 10 clothing categories |
| Training samples | 60,000 |
| Test samples | 10,000 |
| Image size | 28 × 28 greyscale |
| Format | CSV (pixel values 0–255) |

**Class labels:**

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

**Download:** [Fashion MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)

---

## References

- **Dataset:** Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* Zalando Research. [GitHub](https://github.com/zalandoresearch/fashion-mnist) · [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)
- **Model:** Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [Paper](https://arxiv.org/abs/1801.04381)
- **Framework:** Abadi, M., et al. (2015). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.* [tensorflow.org](https://www.tensorflow.org)
- **Load Testing:** [Locust — An open source load testing tool](https://locust.io)
- **Deployment:** [Hugging Face Spaces](https://huggingface.co/spaces)
