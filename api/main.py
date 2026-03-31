"""
FastAPI backend for the Fashion MNIST MLOps pipeline.

Endpoints:
    GET  /                  Service info
    GET  /health            Model readiness and database stats
    POST /predict           Predict from 784 pixel values
    POST /predict/image     Predict from uploaded PNG or JPG
    POST /upload-data       Upload labelled CSV to SQLite
    POST /retrain           Trigger background fine-tuning
    GET  /retrain/status    Poll retraining progress
    GET  /retrain/history   Past retraining run logs
    GET  /metrics           Model evaluation metrics
    GET  /insights          Dataset statistics
"""

from __future__ import annotations

import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Path setup ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.preprocessing import FashionMNISTPreprocessor, PIXEL_COLUMNS, CLASS_NAMES, NUM_CLASSES
from src.prediction import Predictor
from src.model import MobileNetTransfer
import api.database as db

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
log = logging.getLogger(__name__)

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Fashion MNIST MLOps API",
    description=(
        "Production-ready API for Fashion MNIST clothing classification. "
        "Supports prediction, batch CSV upload, and model retraining."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Startup ───────────────────────────────────────────────────────────────────
predictor    = Predictor()
preprocessor = FashionMNISTPreprocessor()
cnn_builder  = MobileNetTransfer()

_retrain_state: dict = {
    "running": False,
    "last_result": None,
    "started_at": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "epoch_logs": [],
    "phase": "",
    "n_samples": 0,
    "steps": [],          # completed steps: [{msg, elapsed}]
    "current_step": "",   # what is happening right now
}


import tensorflow as tf

class _EpochProgressCallback(tf.keras.callbacks.Callback):
    """Writes per-epoch metrics into _retrain_state so the API can stream them."""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        _retrain_state["current_epoch"] = epoch + 1
        _retrain_state["epoch_logs"].append({
            "epoch":        epoch + 1,
            "loss":         round(float(logs.get("loss", 0)), 4),
            "accuracy":     round(float(logs.get("accuracy", 0)) * 100, 2),
            "val_loss":     round(float(logs.get("val_loss", 0)), 4) if "val_loss" in logs else None,
            "val_accuracy": round(float(logs.get("val_accuracy", 0)) * 100, 2) if "val_accuracy" in logs else None,
        })
_start_time = time.time()


@app.on_event("startup")
def startup():
    db.init_db()
    log.info("API ready — model loaded: %s", predictor.model_ready)


# ── Schemas ───────────────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    pixels: List[float] = Field(
        ...,
        min_length=784,
        max_length=784,
        description="784 raw pixel values in range [0, 255]",
    )


class RetrainRequest(BaseModel):
    epochs:     Optional[int] = Field(5, ge=1, le=30, description="Max fine-tuning epochs")
    batch_size: Optional[int] = Field(64, description="Training batch size")
    clear_after: Optional[bool] = Field(
        False, description="Clear uploaded samples from DB after retraining"
    )


class UploadResponse(BaseModel):
    message:       str
    samples_added: int
    total_in_db:   int


# ── Health & Status ───────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
def root():
    return {
        "status":      "ok",
        "service":     "Fashion MNIST MLOps API",
        "version":     "2.0.0",
        "uptime_sec":  round(time.time() - _start_time, 1),
    }


@app.get("/health", tags=["Health"])
def health():
    return {
        "status":        "ok",
        "model_ready":   predictor.model_ready,
        "uptime_sec":    round(time.time() - _start_time, 1),
        "db_samples":    db.count_uploaded_samples(),
    }


# ── Prediction ────────────────────────────────────────────────────────────────
@app.post("/predict", tags=["Prediction"])
def predict_pixels(request: PredictRequest):
    """
    Classify a garment from 784 raw pixel values.

    Input: JSON body with `pixels` array of 784 numbers (0-255).
    Output: predicted class, label, confidence, and per-class probabilities.
    """
    try:
        result = predictor.predict_from_pixels(request.pixels)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception("Prediction error")
        raise HTTPException(500, f"Prediction failed: {exc}")


@app.post("/predict/image", tags=["Prediction"])
async def predict_image(file: UploadFile = File(...)):
    """
    Classify a garment from an uploaded PNG or JPG image.

    The image is resized to 28×28 greyscale before inference.
    """
    if file.content_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/avif"):
        raise HTTPException(415, "Only PNG, JPG, WebP, and AVIF images are supported")
    try:
        image_bytes = await file.read()
        result = predictor.predict_from_image_bytes(image_bytes)
        return result
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc))
    except Exception as exc:
        log.exception("Image prediction error")
        raise HTTPException(500, f"Image prediction failed: {exc}")


# ── Data Upload ───────────────────────────────────────────────────────────────
@app.post("/upload-data", response_model=UploadResponse, tags=["Retraining"])
async def upload_data(file: UploadFile = File(...)):
    """
    Upload a CSV file of labelled samples to be used for retraining.

    CSV must contain columns: label, pixel1, pixel2, ..., pixel784
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Only CSV files are accepted")

    contents = await file.read()
    try:
        df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    # Validate
    if "label" not in df.columns:
        raise HTTPException(400, "CSV must contain a 'label' column")
    missing_cols = [c for c in PIXEL_COLUMNS if c not in df.columns]
    if missing_cols:
        raise HTTPException(
            400, f"CSV is missing pixel columns: {missing_cols[:5]}..."
        )

    # Clean
    df = df[["label"] + PIXEL_COLUMNS].dropna()
    df["label"] = df["label"].astype(int)
    df = df[df["label"].isin(range(NUM_CLASSES))]
    df[PIXEL_COLUMNS] = df[PIXEL_COLUMNS].clip(0, 255).astype(int)

    if len(df) == 0:
        raise HTTPException(400, "No valid rows found after cleaning")

    count = db.insert_samples(df)
    total = db.count_uploaded_samples()

    return UploadResponse(
        message=f"Uploaded and stored {count} samples successfully",
        samples_added=count,
        total_in_db=total,
    )


# ── Retraining ────────────────────────────────────────────────────────────────
def _run_retrain(epochs: int, batch_size: int, clear_after: bool):
    """Background task: preprocess uploaded data → fine-tune → log results."""
    global _retrain_state
    _retrain_state.update({
        "running": True,
        "started_at": time.time(),
        "current_epoch": 0,
        "total_epochs": epochs,
        "epoch_logs": [],
        "phase": "loading data",
        "last_result": None,
        "n_samples": 0,
        "steps": [],
        "current_step": "",
    })

    def _add_step(msg):
        elapsed = round(time.time() - _retrain_state["started_at"], 1)
        _retrain_state["steps"].append({"msg": msg, "elapsed": elapsed})
        _retrain_state["current_step"] = msg
        log.info("[retrain step] %s", msg)

    try:
        # 1. Fetch raw data from database
        _retrain_state["phase"] = "loading data"
        _add_step("Connecting to SQLite database (api/data/fashion_data.db)")
        df = db.fetch_uploaded_samples()
        if df is None or len(df) == 0:
            _retrain_state["last_result"] = {"error": "No uploaded data in database"}
            return

        n_samples = len(df)
        _retrain_state["n_samples"] = n_samples
        unique_labels = sorted(df["label"].unique().tolist())
        _add_step(f"Fetched {n_samples} labelled samples — {len(unique_labels)} classes: {unique_labels}")

        # 2. Extract raw pixel arrays
        _retrain_state["phase"] = "preprocessing"
        X_new = df[PIXEL_COLUMNS].values.astype("float32")
        y_new = df["label"].values.astype("int32")
        _add_step(f"Extracted pixel arrays — shape: {list(X_new.shape)} (samples × 784 pixels)")

        # 3. Fine-tune the saved MobileNetV2 model on the new samples
        _retrain_state["phase"] = "training"
        result_metrics = cnn_builder.retrain(
            X_new, y_new, epochs=epochs, batch_size=batch_size,
            extra_callbacks=[_EpochProgressCallback()],
            step_callback=_add_step,
        )
        _retrain_state["phase"] = "done"

        acc    = result_metrics["accuracy"]
        f1     = result_metrics["f1_score"]
        prec   = result_metrics["precision"]
        rec    = result_metrics["recall"]
        ep_ran = result_metrics["epochs_ran"]

        # 4. Reload cached predictor
        predictor.reload()

        # 6. Persist metrics
        _retrain_state["phase"] = "saving"
        metrics_path = ROOT / "models" / "training_metrics.json"
        existing = {}
        if metrics_path.exists():
            with open(metrics_path) as f:
                existing = json.load(f)
        existing["retrain"] = {
            "accuracy":   acc,
            "f1_score":   f1,
            "precision":  prec,
            "recall":     rec,
            "epochs_ran": ep_ran,
            "samples":    n_samples,
        }
        with open(metrics_path, "w") as f:
            json.dump(existing, f, indent=2)

        # 7. Log to DB
        db.log_retrain(
            samples_used=n_samples,
            accuracy=acc,
            f1=f1,
            precision=prec,
            recall=rec,
            epochs_ran=ep_ran,
            notes=f"Fine-tuned for {ep_ran} epochs",
        )

        if clear_after:
            db.clear_uploaded_samples()

        _retrain_state["last_result"] = {
            "success":    True,
            "accuracy":   acc,
            "f1_score":   f1,
            "precision":  prec,
            "recall":     rec,
            "epochs_ran": ep_ran,
            "samples":    n_samples,
        }
        log.info("Retraining complete — acc=%.4f  f1=%.4f", acc, f1)

    except Exception as exc:
        log.exception("Retraining failed")
        _retrain_state["last_result"] = {"error": str(exc)}
    finally:
        _retrain_state["running"] = False


@app.post("/retrain", tags=["Retraining"])
def trigger_retrain(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger background retraining of the model on uploaded data.

    The model is fine-tuned (lower LR) on the stored samples.
    Poll `/retrain/status` to check progress.
    """
    if _retrain_state["running"]:
        raise HTTPException(409, "Retraining is already in progress")

    count = db.count_uploaded_samples()
    if count == 0:
        raise HTTPException(400, "No uploaded data. Use /upload-data first.")

    background_tasks.add_task(
        _run_retrain, request.epochs, request.batch_size, request.clear_after
    )
    return {
        "message":  f"Retraining started on {count} samples",
        "samples":  count,
        "status":   "running",
    }


@app.get("/retrain/status", tags=["Retraining"])
def retrain_status():
    """Check whether retraining is currently running and get the latest result."""
    elapsed = None
    if _retrain_state["started_at"]:
        elapsed = round(time.time() - _retrain_state["started_at"], 1)
    return {
        "running":       _retrain_state["running"],
        "elapsed_sec":   elapsed,
        "phase":         _retrain_state.get("phase", ""),
        "current_epoch": _retrain_state.get("current_epoch", 0),
        "total_epochs":  _retrain_state.get("total_epochs", 0),
        "epoch_logs":    _retrain_state.get("epoch_logs", []),
        "n_samples":     _retrain_state.get("n_samples", 0),
        "steps":         _retrain_state.get("steps", []),
        "current_step":  _retrain_state.get("current_step", ""),
        "last_result":   _retrain_state["last_result"],
    }


@app.get("/retrain/history", tags=["Retraining"])
def retrain_history(limit: int = 20):
    """Return the last N retraining events from the database."""
    return {"history": db.get_retrain_history(limit=limit)}


@app.delete("/uploaded-data", tags=["Retraining"])
def clear_uploaded_data():
    """Clear all uploaded samples from the database."""
    db.clear_uploaded_samples()
    return {"message": "Uploaded samples cleared", "remaining": db.count_uploaded_samples()}


# ── Metrics & Insights ────────────────────────────────────────────────────────
@app.get("/metrics", tags=["Monitoring"])
def get_metrics():
    """Return saved model evaluation metrics (accuracy, F1, precision, recall)."""
    metrics_path = ROOT / "models" / "training_metrics.json"
    if not metrics_path.exists():
        raise HTTPException(
            404, "No metrics file found. Train the model via the notebook first."
        )
    with open(metrics_path) as f:
        return json.load(f)


@app.get("/insights", tags=["Monitoring"])
def get_insights():
    """Return dataset statistics and class distribution."""
    # Try Docker-mounted path first, then local path
    train_path = Path("/data/train/fashion-mnist_train.csv")
    if not train_path.exists():
        train_path = ROOT / "data" / "train" / "fashion-mnist_train.csv"

    # If CSV is available, compute live stats
    if train_path.exists():
        df = pd.read_csv(train_path)
        label_counts = df["label"].value_counts().sort_index()
        pixel_vals   = df.iloc[:, 1:].values
        total_samples = int(len(df))
        class_counts  = {str(i): int(label_counts.get(i, 0)) for i in range(NUM_CLASSES)}
        pixel_stats   = {
            "mean": float(pixel_vals.mean()),
            "std":  float(pixel_vals.std()),
            "min":  float(pixel_vals.min()),
            "max":  float(pixel_vals.max()),
        }
    else:
        # Use known Fashion MNIST training set statistics (60,000 samples, balanced)
        total_samples = 60000
        class_counts  = {str(i): 6000 for i in range(NUM_CLASSES)}
        pixel_stats   = {"mean": 72.94, "std": 90.02, "min": 0.0, "max": 255.0}

    return {
        "total_train_samples": total_samples,
        "num_classes":         NUM_CLASSES,
        "image_size":          "28x28",
        "classes": {
            str(i): {
                "name":  CLASS_NAMES[i],
                "count": class_counts[str(i)],
            }
            for i in range(NUM_CLASSES)
        },
        "pixel_statistics":    pixel_stats,
        "db_uploaded_samples": db.count_uploaded_samples(),
        "model_ready":         predictor.model_ready,
        "uptime_sec":          round(time.time() - _start_time, 1),
    }
