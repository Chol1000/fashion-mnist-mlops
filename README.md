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
![React](https://img.shields.io/badge/React-19-61dafb?logo=react)
![Ant Design](https://img.shields.io/badge/Ant%20Design-6-0170fe?logo=antdesign)
![Docker](https://img.shields.io/badge/Docker-Containerised-blue?logo=docker)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Deployed-yellow?logo=huggingface)

| Resource | Description | Link |
|----------|-------------|------|
| Video Demo | Full walkthrough — prediction, retraining, deployment (camera on) | [Watch on YouTube](https://youtu.be/Nw2GPJmoh20) |
| Live Dashboard | React dashboard — classify, retrain, inspect metrics and dataset | [Open Dashboard](https://cholatemgiet-fashion-mnist-api.hf.space) |
| Live API | FastAPI backend — same container as the dashboard | [Open API](https://cholatemgiet-fashion-mnist-api.hf.space/info) |
| API Docs | Swagger UI — test all endpoints directly in the browser | [Open Swagger](https://cholatemgiet-fashion-mnist-api.hf.space/docs) |
| GitHub | Full source code — notebook, API, dashboard, Docker, Locust | [View on GitHub](https://github.com/Chol1000/fashion-mnist-mlops) |
| Deployment | How it ships, and why the API no longer sleeps | [deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md) |

</div>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [The Dashboard](#the-dashboard)
- [Deployment](#deployment)
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

The pipeline covers the full ML lifecycle: data acquisition and preprocessing, model training with transfer learning, API serving, an interactive dashboard, database-backed retraining, Docker containerisation, and cloud deployment on Hugging Face Spaces. Load testing with Locust validates the system under concurrent traffic.

The dashboard and the API ship as **one container**: Vite builds the React frontend and FastAPI serves it alongside its own routes. That is a deliberate choice rather than a convenience — see [Deployment](#deployment).

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
├── frontend/                        # React 19 + Vite + Ant Design dashboard
│   ├── src/
│   │   ├── pages/                   # Overview, Classify, Retrain, Insights, Metrics, System, About
│   │   ├── components/              # Layout, wake-up gate, draw pad, result panels
│   │   ├── context/                 # Theme (light/dark) and backend health polling
│   │   ├── api.ts                   # Typed client for every endpoint
│   │   └── ui.tsx                   # Shared design primitives
│   ├── index.html
│   └── package.json
│
├── deploy/
│   ├── DEPLOYMENT.md                # How it ships, and the sleeping-Space fix
│   ├── push-to-space.sh             # One-command deploy to a Hugging Face Space
│   └── space-frontmatter.md         # Space README frontmatter, prepended at deploy time
│
├── .github/workflows/
│   └── keep-spaces-awake.yml        # Scheduled ping + restart so the API never sleeps
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
├── nginx.conf                       # Serves the dashboard, load balances the API
├── docker-compose.yml               # Full stack: nginx, backend replicas, locust
├── Dockerfile                       # Deployed image — dashboard + API in one container
├── Dockerfile.api                   # API only, used for the scalable Compose backend
├── Dockerfile.lb                    # Nginx + built dashboard, the Compose edge
├── Dockerfile.frontend              # Dashboard only (separate-deployment option)
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
- **Dashboard:** http://localhost
- **API:** http://localhost/health — same origin as the dashboard
- **API Docs:** http://localhost/docs
- **Locust UI:** http://localhost:8089

Nginx serves the built dashboard and proxies the API paths to the backend pool,
so the browser sees a single origin — the same shape as the deployed image.

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

`run_local.sh` starts the API and the dashboard's dev server with hot reload.
Pass `--build` to build the dashboard instead and have FastAPI serve it, which
is exactly what the deployed container does:

```bash
bash run_local.sh --build     # everything on http://localhost:8000
```

### Option C — Manual Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r api/requirements.txt

# Terminal 1 — API
PYTHONPATH=. uvicorn api.main:app --host 0.0.0.0 --port 8000

# Terminal 2 — dashboard (Node 20+)
cd frontend && npm install && npm run dev
```

The dev server proxies the API paths to port 8000; set `VITE_API_TARGET` if the
API is bound somewhere else.

> On Apple Silicon, `tensorflow-cpu` has no wheels — install `tensorflow==2.17.0`
> instead. `run_local.sh` does this substitution automatically.

---

## The Dashboard

Seven pages, built with React 19, Ant Design and Recharts. Light and dark themes
follow the system preference and can be toggled from the header. Everything on
every page is a live call against the running API — nothing is mocked.

| Page | What it does |
|------|--------------|
| **Overview** | Headline accuracy and F1 of whatever model is currently live, per-class F1, the training curve, and a jump-off into the rest. Shows a comparison banner when the model has been fine-tuned since the baseline. |
| **Classify** | Four ways to get an image in: upload a photo, pull a random held-out test image (with its true label, so you can see whether the model was right), draw a garment on a canvas, or paste 784 raw pixel values. Every result shows the full 10-class probability distribution, not just the winner. |
| **Upload & Retrain** | The complete loop — upload a labelled CSV into SQLite, watch it validated and previewed, then fine-tune the model and follow each pipeline step and epoch as they land. Includes a sample-CSV generator so the loop can be exercised without hunting for data. |
| **Dataset Insights** | Class balance, pixel-intensity distributions, per-class sample images and the preprocessing pipeline. |
| **Model Metrics** | Architecture and training configuration, baseline test-set scores, per-class precision/recall/F1, the confusion matrix, and the full 30-epoch history. |
| **API & System** | Live health, deployment topology, retraining state, and the endpoint reference. |
| **About** | How the pieces fit together. |

### Two details worth knowing

**The model is closed-set.** It emits a softmax over exactly ten garment
classes, with no "none of these" option. Upload a photo of a car and it will not
refuse — it will return one of the ten labels, sometimes at 90%+ confidence. The
Classify page says so before you upload, because a confident answer is not
evidence that the input was a garment.

**Dashboard routes avoid API routes.** Both live at the root of the same origin,
so the pages are at `/classify`, `/training`, `/dataset`, `/evaluation`,
`/status` and `/about` — not `/predict` or `/metrics`, which would have been
shadowed by the endpoints of the same name and returned JSON on every refresh or
shared link.

---

## Deployment

Full detail in **[deploy/DEPLOYMENT.md](deploy/DEPLOYMENT.md)**. The short
version:

A free Hugging Face CPU Space sleeps after 48 hours without traffic, and waking
it costs 60–90 seconds while the container boots and TensorFlow loads the model.
When the dashboard and the API were **separate Spaces**, opening the dashboard
woke only the dashboard — the API had its own idle timer and was still asleep,
which is why every panel failed and the API Space had to be restarted by hand.

Three changes fix it, and they stack:

1. **One container.** The root `Dockerfile` builds the dashboard and has FastAPI
   serve it alongside its own routes. One Space, one URL, one idle timer — if
   the page loads, the API behind it is already running.
2. **Scheduled pings.** `.github/workflows/keep-spaces-awake.yml` calls
   `/health` every 20 minutes, so the 48-hour idle timer never expires and
   nobody pays a cold start. No secrets required.
3. **Automatic restart.** A ping wakes a *sleeping* Space but can do nothing for
   a *paused* or crashed one. Add an `HF_TOKEN` repository secret and the same
   workflow calls the Hub's restart endpoint when a ping fails.

And if a visitor still arrives mid-wake — right after a rebuild, say — the
dashboard opens on a wake-up screen that explains what is happening, retries
every four seconds (which is itself what wakes the Space), and offers a manual
restart link if it drags on. It blocks only the first connection of a session.

```bash
bash deploy/push-to-space.sh     # builds the Space README, pushes, triggers a rebuild
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
| GET | `/` | The dashboard (JSON service info moved to `/info`) |
| GET | `/info` | Service name, version and uptime |
| GET | `/health` | Model readiness and database stats |
| POST | `/predict` | Predict from 784 raw pixel values |
| POST | `/predict/image` | Predict from an uploaded PNG/JPG/WebP/AVIF image |
| GET | `/sample/random` | A random held-out test image, optionally filtered by class |
| GET | `/sample/csv` | Download a correctly-shaped slice of the dataset |
| POST | `/upload-data` | Upload labelled CSV for retraining |
| POST | `/retrain` | Trigger fine-tuning on uploaded data |
| GET | `/retrain/status` | Poll live training progress |
| GET | `/retrain/history` | Past retraining run logs |
| DELETE | `/uploaded-data` | Clear stored samples |
| GET | `/metrics` | Model evaluation metrics |
| GET | `/insights` | Dataset statistics and class distribution |
| GET | `/figures/*` | EDA and evaluation PNGs from the training run |

Endpoints live at the **root** of the origin — there is no `/api` prefix — which
is why the dashboard's own pages are named `/classify`, `/evaluation` and so on
rather than `/predict` and `/metrics`. In an API-only image (`Dockerfile.api`)
`/` keeps its original JSON payload, since there is no dashboard to serve.

`/sample/random` and `/sample/csv` are new: the dashboard used to ship the whole
10,000-row test CSV to the browser so it could pick a random image client-side.
Serving one row from the API instead keeps the dataset on the server, where it
already lives for training.

Full interactive docs: https://cholatemgiet-fashion-mnist-api.hf.space/docs

---

## Retraining Workflow

1. Open the **Upload & Retrain** page in the dashboard (`/training`)
2. Upload a CSV with columns: `label, pixel1, pixel2, ..., pixel784`
3. Samples are validated, cleaned, and stored in SQLite
4. Click **Start Retraining** — the model is fine-tuned using Adam lr=1e-4
5. Live epoch progress is streamed during training
6. Updated metrics are logged to the database and displayed in the dashboard

A ready-to-use sample file is included at `data/sample_retrain.csv` — 100 randomly sampled rows from the Fashion MNIST test set in the correct format. The page can also generate one on demand at any size via `GET /sample/csv`, so the loop can be exercised without hunting for data.

The CSV is validated twice: the browser previews the rows, the class distribution and the column count before you upload, and the API re-validates and cleans independently. What neither can catch is a *mislabelled* row — the shape can be perfect while the label is wrong, and fine-tuning on wrong labels makes the model measurably worse.

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
