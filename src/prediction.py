"""
Inference wrapper for the Fashion MNIST classifier.

Loads the trained model once and caches it for the lifetime of the process.
Supports prediction from pixel arrays, image bytes, and batches.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np

from src.preprocessing import FashionMNISTPreprocessor, CLASS_NAMES

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_PATH = ROOT / "models" / "fashion_model.h5"


class Predictor:
    """
    Thread-safe predictor that caches the loaded TF model.

    Parameters
    ----------
    model_path : path to the .h5 model file. Defaults to ROOT/models/fashion_model.h5.
    """

    def __init__(self, model_path: Optional[str | Path] = None):
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._model = None  # lazy-loaded

    # ── Model lifecycle ───────────────────────────────────────────────────────
    def _get_model(self):
        """Return cached model, loading from disk if necessary."""
        if self._model is None:
            import tensorflow as tf
            if not self._model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at {self._model_path}. "
                    "Run the training notebook first."
                )
            log.info("Loading model from %s", self._model_path)
            self._model = tf.keras.models.load_model(str(self._model_path))
            log.info("Model loaded — %d parameters", self._model.count_params())
        return self._model

    def reload(self):
        """Force reload the model from disk (call after retraining)."""
        self._model = None
        return self._get_model()

    @property
    def model_ready(self) -> bool:
        return self._model_path.exists()

    # ── Core inference ────────────────────────────────────────────────────────
    def _run_inference(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Run a single forward pass and return structured results.

        Parameters
        ----------
        X : np.ndarray  shape (1, TARGET_SIZE, TARGET_SIZE, 3), float32 in [-1, 1]

        Returns
        -------
        dict with predicted_class, predicted_label, confidence, probabilities
        """
        model      = self._get_model()
        probs      = model.predict(X, verbose=0)[0]
        pred_idx   = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        return {
            "predicted_class": pred_idx,
            "predicted_label": CLASS_NAMES[pred_idx],
            "confidence":      confidence,
            "probabilities": {
                CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
            },
        }

    # ── Public API ────────────────────────────────────────────────────────────
    def predict_from_pixels(self, pixel_values: List[float]) -> Dict[str, Any]:
        """
        Predict the class of a garment from 784 raw pixel values.

        Parameters
        ----------
        pixel_values : list of 784 numbers in range [0, 255]
        """
        X = FashionMNISTPreprocessor.preprocess_single(pixel_values)
        return self._run_inference(X)

    def predict_from_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Predict the class of a garment from raw PNG / JPG bytes.
        Image is converted to greyscale and resized to 28×28.
        """
        X = FashionMNISTPreprocessor.preprocess_image_bytes(image_bytes)
        return self._run_inference(X)

    def predict_batch(self, pixel_matrix: np.ndarray) -> List[Dict[str, Any]]:
        """
        Predict multiple samples at once.

        Parameters
        ----------
        pixel_matrix : np.ndarray  shape (N, 784) raw pixel values [0, 255]

        Returns
        -------
        list of prediction dicts
        """
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

        n = pixel_matrix.shape[0]
        X_flat = pixel_matrix.astype("float32")
        ds = tf.data.Dataset.from_tensor_slices(X_flat)

        from src.preprocessing import TARGET_SIZE

        def _prep(x):
            x = tf.reshape(x, (28, 28, 1))
            x = tf.image.resize(x, (TARGET_SIZE, TARGET_SIZE))
            x = tf.image.grayscale_to_rgb(x)
            x = preprocess_input(x)
            return x

        ds = ds.map(_prep, num_parallel_calls=tf.data.AUTOTUNE).batch(64)
        model = self._get_model()
        probs_all = model.predict(ds, verbose=0)

        results = []
        for probs in probs_all:
            pred_idx = int(np.argmax(probs))
            results.append({
                "predicted_class": pred_idx,
                "predicted_label": CLASS_NAMES[pred_idx],
                "confidence":      float(probs[pred_idx]),
                "probabilities": {
                    CLASS_NAMES[i]: float(probs[i]) for i in range(len(CLASS_NAMES))
                },
            })
        return results
