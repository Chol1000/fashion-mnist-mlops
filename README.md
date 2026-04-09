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
| Video Demo | Full walkthrough — prediction, retraining, deployment (camera on) | [Watch on YouTube](https://youtu.be/Nw2GPJmoh20) |
| Live Dashboard | Streamlit UI — predict, retrain, view metrics and insights | [Open Dashboard](https://huggingface.co/spaces/CholatemGiet/fashion-mnist-frontend) |
| Live API | FastAPI backend hosted on Hugging Face Spaces | [Open API](https://cholatemgiet-fashion-mnist-api.hf.space) |
| API Docs | Swagger UI — test all endpoints directly in the browser | [Open Swagger](https://cholatemgiet-fashion-mnist-api.hf.space/docs) |
| GitHub | Full source code — notebook, API, frontend, Docker, Locust | [View on GitHub](https://github.com/Chol1000/fashion-mnist-mlops) |

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Training the Model](#training-the-model)
- [Model Architecture & Performance](#model-architecture--performance)
- [Dataset Visualisations](#dataset-visualisations)
- [API Endpoints](#api-endpoints)
- [Retraining Workflow](#retraining-workflow)
- [Load Testing with Locust](#load-testing-with-locust)
- [Dataset](#dataset)
- [References](#references)

---

## Project Overview

This project implements a production-grade MLOps pipeline for classifying clothing images using the **Fashion MNIST** dataset — 70,000 greyscale 28×28 images across 10 balanced clothing categories.

The pipeline covers the full ML lifecycle: data acquisition and preprocessing, model training with transfer learning, API serving, interactive dashboard, database-backed retraining, Docker containerisation, and cloud deployment on Hugging Face Spaces. Load testing with Locust validates the system under concurrent traffic.

**Model:** MobileNetV2 pre-trained on ImageNet, fine-tuned on Fashion MNIST. Input images are resized from 28×28 to 128×128×3 before inference.

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
│   ├── main.py                      # FastAPI endpoints
│   ├── database.py                  # SQLite — uploaded samples and retrain logs
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
│   └── training_metrics.json        # Saved evaluation metrics
│
├── outputs/figures/                 # EDA plots, training curves, confusion matrix
├── nginx.conf                       # Nginx load balancer config
├── docker-compose.yml               # Full stack: api, frontend, locust, nginx
├── Dockerfile.api                   # API container
├── Dockerfile.frontend              # Frontend container
└── run_local.sh                     # One-command local start without Docker
```

> `data/` CSV files are tracked via Git LFS and download automatically on clone. The trained model `fashion_model.h5` is included in the repo — no manual steps required.

---

## Quick Start

### Option A — Docker (Recommended)

```bash
git clone https://github.com/Chol1000/fashion-mnist-mlops.git
cd fashion-mnist-mlops
docker compose up --build
```

Once running, open:
- **Dashboard:** http://localhost:8501
- **API:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **Locust UI:** http://localhost:8089

To scale the API across multiple replicas:
```bash
docker compose up --scale backend=3
```

### Option B — Local (No Docker)

```bash
git clone https://github.com/Chol1000/fashion-mnist-mlops.git
cd fashion-mnist-mlops
bash run_local.sh
```

### Option C — Manual Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Terminal 1 — API
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — Frontend
API_URL=http://localhost:8000 streamlit run frontend/app.py --server.port 8501
```

---

## Training the Model

### Google Colab (Recommended)

1. Upload `notebook/fashion_mnist_mlops.ipynb` to [Google Colab](https://colab.research.google.com)
2. Set `Runtime > Change runtime type > T4 GPU`
3. Run all cells — Fashion MNIST downloads automatically via Keras
4. The final cell downloads `fashion_model.h5` and `training_metrics.json` to your machine
5. Place both files in the `models/` folder

### Standalone Script

```bash
source venv/bin/activate
python -m src.train
```

---

## Model Architecture & Performance

The model uses a two-phase transfer learning strategy on MobileNetV2:

```
Input (128 × 128 × 3)
└── MobileNetV2 [ImageNet weights, frozen in Phase 1]
    └── GlobalAveragePooling2D
        └── Dense(256, relu) + BatchNorm + Dropout(0.5)
            └── Dense(10, softmax)
```

- **Phase 1 — Feature extraction:** base frozen, Adam lr=1e-3, EarlyStopping patience=5
- **Phase 2 — Fine-tuning:** last 80 layers unfrozen, Adam lr=1e-5, EarlyStopping patience=5

| Metric | Score |
|--------|-------|
| Test Accuracy | **93.17%** |
| Test Loss | 0.2377 |
| Macro F1 Score | **0.9320** |
| Macro Precision | 0.9324 |
| Macro Recall | 0.9317 |

Shirt, T-shirt/top, and Coat are the most confused classes due to overlapping greyscale silhouettes. Bag and Trouser are the easiest to classify.

---

## Dataset Visualisations

### Class Distribution
![Class Distribution](outputs/figures/eda_01_class_distribution.png)

The dataset is perfectly balanced — 6,000 training samples per class, eliminating class imbalance as a source of bias.

### Sample Images per Class
![Sample Images](outputs/figures/eda_02_sample_images.png)

Pullovers, Shirts, and Coats share similar collar and sleeve shapes in greyscale — explaining the model's confusion between these classes.

### Pixel Intensity Distribution
![Pixel Intensity](outputs/figures/eda_03_pixel_intensity.png)

Most pixels cluster at 0 (black background), with clothing pixels spread across the 100–255 range. This bimodal distribution informed the normalisation strategy.

### Mean Images per Class
![Mean Images](outputs/figures/eda_04_mean_images.png)

Averaging all images per class reveals each garment's archetypal shape — Bags and Trousers have the clearest silhouettes.

### Training History
![Training History](outputs/figures/training_history.png)

### Confusion Matrix
![Confusion Matrix](outputs/figures/confusion_matrix.png)

### Per-Class Metrics
![Per Class Metrics](outputs/figures/per_class_metrics.png)

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
| GET | `/retrain/history` | Past retraining run logs |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/insights` | Dataset statistics and class distribution |

Full interactive docs: https://cholatemgiet-fashion-mnist-api.hf.space/docs

---

## Retraining Workflow

1. Open the **Upload & Retrain** tab in the dashboard
2. Upload a CSV with columns: `label, pixel1, pixel2, ..., pixel784`
3. Samples are validated, cleaned, and stored in SQLite
4. Click **Start Retraining** — the model is fine-tuned using Adam lr=1e-4
5. Live epoch progress is streamed during training
6. Updated metrics are logged to the database and displayed in the dashboard

A ready-to-use sample file is included at `data/sample_retrain.csv` — 100 randomly sampled rows from the Fashion MNIST test set in the correct format.

---

## Load Testing with Locust

Two concurrent user classes were used:

- **FashionAPIUser** — realistic mix of health checks, predictions, metrics, and insights (wait: 0.5–2 s)
- **HeavyPredictUser** — rapid-fire `/predict` calls to stress-test inference (wait: 0.1–0.3 s)

---

### Local Docker — Different Container Counts

Tests run locally against the Dockerised stack using Nginx as a load balancer.

```bash
# 1 container — 10 users
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 10 --spawn-rate 2 --run-time 30s --headless \
       --csv locust/results/run_10u_local

# 1 container — 50 users
locust -f locust/locustfile.py --host http://localhost:8000 \
       --users 50 --spawn-rate 5 --run-time 30s --headless \
       --csv locust/results/run_50u_local

# Scale to 3 containers behind Nginx
docker compose up -d --scale backend=3

# 3 containers — 50 users
locust -f locust/locustfile.py --host http://localhost:80 \
       --users 50 --spawn-rate 5 --run-time 30s --headless \
       --csv locust/results/run_50u_3replicas
```

#### Overall Throughput

| Containers | Users | Total Requests | Req/s | Median Latency | 95th pct | Failures |
|-----------|-------|---------------|-------|----------------|----------|----------|
| 1 | 10 | 566 | 21.79/s | 46 ms | 78 ms | 0 (0%) |
| 1 | 50 | 1,553 | 55.35/s | 250 ms | 480 ms | 0 (0%) |
| 3 (Nginx) | 50 | 1,255 | 43.21/s | 370 ms | 630 ms | 0 (0%) |

#### `/predict` Endpoint

| Containers | Users | Requests | Median | 95th pct | Req/s |
|-----------|-------|----------|--------|----------|-------|
| 1 | 10 | 62 | 51 ms | 78 ms | 2.39 |
| 1 | 50 | 183 | 250 ms | 490 ms | 6.52 |
| 3 (Nginx) | 50 | 139 | 390 ms | 680 ms | 4.79 |

**Zero failures across all runs.** The 3-replica Nginx setup adds routing overhead on local hardware — all configurations handled load with 100% success rate.

### Statistics — 50 users, 3 containers (Nginx)
![Docker Statistics](locust/results/docker_statistics.png)

### Request Rate & Response Time Over Time
![Docker Charts](locust/results/docker_charts.png)

### Failures — Zero Across All Runs
![Docker Failures](locust/results/docker_failures.png)

---

### Live Deployment — Hugging Face Spaces

Tests run against the **live production API** on Hugging Face Spaces.

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

#### Overall Throughput

| Users | Total Requests | Req/s | Median Latency | 95th pct | Failures |
|-------|---------------|-------|----------------|----------|----------|
| 10 | 410 | 9.29/s | 400 ms | 1,000 ms | 0 (0%) |
| 50 | 1,050 | 17.00/s | 1,500 ms | 2,100 ms | 0 (0%) |
| 100 | 549 | 12.45/s | 5,200 ms | 7,700 ms | 0 (0%) |

#### `/predict` Endpoint

| Users | Requests | Avg Latency | Median | 95th pct |
|-------|----------|-------------|--------|----------|
| 10 | 68 | 615 ms | 410 ms | 2,800 ms |
| 50 | 174 | 1,635 ms | 1,600 ms | 2,300 ms |
| 100 | 139 | 4,916 ms | 5,200 ms | 7,700 ms |

**Zero failures across all runs.** Latency increases at higher concurrency reflect the shared free-tier CPU on Hugging Face Spaces.

### Statistics — 50 users, live HF Space
![Locust Statistics](locust/results/statistics.png)

### Request Rate & Response Time Over Time
![Locust Charts](locust/results/charts.png)

### Failures — Zero Across All Runs
![Locust Failures](locust/results/failures.png)

---

## Dataset

**Fashion MNIST** — Zalando's article images, a drop-in replacement for the original MNIST handwritten digits dataset.

60,000 training samples and 10,000 test samples across 10 classes:

| Label | Class | Label | Class |
|-------|-------|-------|-------|
| 0 | T-shirt/top | 5 | Sandal |
| 1 | Trouser | 6 | Shirt |
| 2 | Pullover | 7 | Sneaker |
| 3 | Dress | 8 | Bag |
| 4 | Coat | 9 | Ankle boot |

Download: [Fashion MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)

---

## References

- Xiao, H., Rasul, K., & Vollgraf, R. (2017). *Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms.* Zalando Research. [GitHub](https://github.com/zalandoresearch/fashion-mnist) · [Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist?resource=download)
- Sandler, M., et al. (2018). *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018. [Paper](https://arxiv.org/abs/1801.04381)
- Abadi, M., et al. (2015). *TensorFlow: Large-Scale Machine Learning on Heterogeneous Systems.* [tensorflow.org](https://www.tensorflow.org)
- [Locust — Open source load testing tool](https://locust.io)
- [Hugging Face Spaces](https://huggingface.co/spaces)
