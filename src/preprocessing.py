"""
Data preprocessing pipeline for Fashion MNIST (MobileNetV2 input).

Stages:
    1. Load CSV and validate schema
    2. Clean — drop nulls, clamp labels and pixel values
    3. Build tf.data pipeline — resize 28→128, grayscale→RGB, normalise to [-1, 1]
    4. Inference helpers — single sample and image bytes
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

log = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
TARGET_SIZE   = 128                        # MobileNetV2 — 128×128 gives 4×4 spatial output (vs 3×3 at 96)
INPUT_SHAPE   = (TARGET_SIZE, TARGET_SIZE, 3)
NUM_PIXELS    = 784                        # 28 × 28
PIXEL_COLUMNS = [f"pixel{i}" for i in range(1, NUM_PIXELS + 1)]
NUM_CLASSES   = 10

CLASS_NAMES = [
    "T-shirt/top", "Trouser",  "Pullover", "Dress",      "Coat",
    "Sandal",      "Shirt",    "Sneaker",  "Bag",         "Ankle boot",
]


class FashionMNISTPreprocessor:
    """
    End-to-end preprocessing pipeline for Fashion MNIST (CSV format).

    Usage
    -----
    prep = FashionMNISTPreprocessor()
    X_raw, y = prep.load_csv_to_arrays("data/train/fashion-mnist_train.csv")
    train_ds = prep.make_tf_dataset(X_raw, y, batch_size=32, augment=True)
    """

    def __init__(self, val_size: float = 0.10, random_state: int = 42):
        self.val_size     = val_size
        self.random_state = random_state

    # ── 1. Load ────────────────────────────────────────────────────────────────
    def load_csv(self, path: str | Path) -> pd.DataFrame:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        df = pd.read_csv(path)
        if "label" not in df.columns:
            raise ValueError("CSV must contain a 'label' column")
        missing = [c for c in PIXEL_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"CSV missing pixel columns (e.g. {missing[:3]})")
        log.info("Loaded %d rows from %s", len(df), path.name)
        return df

    # ── 2. Clean ───────────────────────────────────────────────────────────────
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(subset=["label"] + PIXEL_COLUMNS).copy()
        df["label"] = df["label"].astype(int)
        df = df[df["label"].isin(range(NUM_CLASSES))]
        df[PIXEL_COLUMNS] = df[PIXEL_COLUMNS].clip(0, 255).astype("float32")
        return df

    # ── Combined load + clean → raw arrays ────────────────────────────────────
    def load_csv_to_arrays(
        self, path: str | Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return raw pixel arrays (N, 784) float32 in [0,255] and int32 labels."""
        df = self.clean(self.load_csv(path))
        X  = df[PIXEL_COLUMNS].values.astype("float32")   # (N, 784)  [0, 255]
        y  = df["label"].values.astype("int32")
        log.info("Arrays: X=%s  y=%s", X.shape, y.shape)
        return X, y

    # ── 3. tf.data pipeline ────────────────────────────────────────────────────
    @staticmethod
    def _preprocess_fn(
        x: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """x: (784,) float32 [0,255]  →  (TARGET_SIZE, TARGET_SIZE, 3) float32 [-1,1]"""
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
        x = tf.reshape(x, (28, 28, 1))
        x = tf.image.resize(x, (TARGET_SIZE, TARGET_SIZE))
        x = tf.image.grayscale_to_rgb(x)          # (TARGET_SIZE, TARGET_SIZE, 3) still [0,255]
        x = preprocess_input(x)                   # → [-1, 1]
        return x, label

    @staticmethod
    def _augment_fn(
        x: tf.Tensor, label: tf.Tensor
    ) -> Tuple[tf.Tensor, tf.Tensor]:
        """Random flip, brightness, contrast, and zoom applied during training only."""
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.05)
        x = tf.image.random_contrast(x, lower=0.9, upper=1.1)
        crop = tf.random.uniform([], 0.85, 1.0)
        h = tf.cast(tf.cast(tf.shape(x)[0], tf.float32) * crop, tf.int32)
        w = tf.cast(tf.cast(tf.shape(x)[1], tf.float32) * crop, tf.int32)
        x = tf.image.random_crop(x, size=[h, w, 3])
        x = tf.image.resize(x, (TARGET_SIZE, TARGET_SIZE))
        x = tf.clip_by_value(x, -1.0, 1.0)
        return x, label

    def make_tf_dataset(
        self,
        X_flat: np.ndarray,
        y: np.ndarray,
        batch_size: int = 32,
        augment: bool = False,
        shuffle: bool = True,
    ) -> tf.data.Dataset:
        """
        Build a tf.data.Dataset from flat pixel arrays.

        Parameters
        ----------
        X_flat    : (N, 784) raw pixel values [0, 255]
        y         : (N,) int32 labels
        batch_size: mini-batch size
        augment   : apply random flip & brightness
        """
        ds = tf.data.Dataset.from_tensor_slices(
            (X_flat.astype("float32"), y.astype("int32"))
        )
        ds = ds.map(self._preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if augment:
            ds = ds.map(self._augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
        if shuffle:
            ds = ds.shuffle(buffer_size=min(len(X_flat), 10_000), seed=42)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    def train_val_split(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return train_test_split(
            X, y,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=y,
        )

    # ── 4. Inference helpers ───────────────────────────────────────────────────
    @staticmethod
    def preprocess_single(pixel_values: list) -> np.ndarray:
        """
        Preprocess 784 raw pixel values for MobileNetV2 inference.

        Returns np.ndarray shape (1, TARGET_SIZE, TARGET_SIZE, 3) float32 in [-1, 1].
        """
        from PIL import Image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        arr = np.array(pixel_values, dtype="float32").reshape(28, 28)
        arr_uint8 = np.clip(arr, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr_uint8, mode="L")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
        img_rgb = img.convert("RGB")
        arr_rgb = np.array(img_rgb, dtype="float32")
        arr_rgb = preprocess_input(arr_rgb)
        return arr_rgb.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)

    @staticmethod
    def preprocess_image_bytes(image_bytes: bytes) -> np.ndarray:
        """
        Preprocess a PNG/JPG image (as bytes) for MobileNetV2 inference.

        Returns np.ndarray shape (1, TARGET_SIZE, TARGET_SIZE, 3) float32 in [-1, 1].
        """
        from PIL import Image
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = img.resize((TARGET_SIZE, TARGET_SIZE), Image.Resampling.BILINEAR)
        arr = np.array(img, dtype="float32")
        arr = preprocess_input(arr)
        return arr.reshape(1, TARGET_SIZE, TARGET_SIZE, 3)
