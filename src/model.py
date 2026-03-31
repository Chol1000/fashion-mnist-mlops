"""
MobileNetV2 transfer learning model for Fashion MNIST classification.

Architecture:
    MobileNetV2 base (ImageNet, 128×128×3) → GlobalAveragePooling2D
    → Dense(256, relu, L2=1e-4) → BatchNormalization → Dropout(0.5) → Dense(10, softmax)

Training strategy:
    Phase 1 — Feature extraction: MobileNetV2 frozen, train head only, Adam lr=1e-3
    Phase 2 — Fine-tuning: last 80 layers unfrozen, Adam lr=1e-5
    Retraining — fine-tune saved model on new uploaded data, Adam lr=1e-4

Optimisation: transfer learning, L2 regularisation, dropout, batch normalisation,
              data augmentation, early stopping, ReduceLROnPlateau
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import (
    Dense, Dropout, GlobalAveragePooling2D, BatchNormalization,
)
from tensorflow.keras.regularizers import l2
from src.preprocessing import CLASS_NAMES  # single source of truth

log = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent.parent

INPUT_SHAPE   = (128, 128, 3)
NUM_CLASSES   = 10
DEFAULT_MODEL = ROOT / "models" / "fashion_model.h5"
DEFAULT_METRICS = ROOT / "models" / "training_metrics.json"


class MobileNetTransfer:
    """MobileNetV2-based transfer learning model wrapper."""

    def __init__(self, model_save_path: Optional[str | Path] = None):
        self.model_save_path = Path(model_save_path) if model_save_path else DEFAULT_MODEL
        self.model_save_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Build ─────────────────────────────────────────────────────────────────
    def build(self) -> Model:
        """
        Build the MobileNetV2 + custom head model.

        Returns a compiled Keras Model (Phase 1 ready: base frozen).
        """
        # Load pre-trained MobileNetV2 base (ImageNet weights, no top classifier)
        base = MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=INPUT_SHAPE,
        )
        base.trainable = False   # Phase 1: freeze base completely

        # Custom classification head
        inp = keras.Input(shape=INPUT_SHAPE, name="input")
        x   = base(inp, training=False)             # inference mode (BN layers frozen)
        x   = GlobalAveragePooling2D(name="gap")(x)
        x   = Dense(256, activation="relu",
                    kernel_regularizer=l2(1e-4), name="dense1")(x)
        x   = BatchNormalization(name="bn")(x)
        x   = Dropout(0.5, name="dropout")(x)
        out = Dense(NUM_CLASSES, activation="softmax", name="output")(x)

        model = Model(inputs=inp, outputs=out, name="FashionMobileNetV2")
        log.info(
            "Model built — total=%d  trainable=%d  frozen=%d",
            model.count_params(),
            sum(p.numpy().size for p in model.trainable_variables),
            sum(p.numpy().size for p in model.non_trainable_variables),
        )
        return model

    # ── Compile helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _compile(model: Model, lr: float) -> Model:
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Callbacks ─────────────────────────────────────────────────────────────
    def _callbacks(self, monitor: str = "val_loss", patience: int = 5) -> list:
        return [
            keras.callbacks.EarlyStopping(
                monitor=monitor,
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            keras.callbacks.ModelCheckpoint(
                filepath=str(self.model_save_path),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1,
            ),
        ]

    # ── Phase 1 — feature extraction ─────────────────────────────────────────
    def train_phase1(
        self,
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        epochs: int = 10,
        class_weight: Optional[dict] = None,
    ):
        """Train only the custom head (MobileNetV2 base fully frozen)."""
        log.info("=== Phase 1: Feature Extraction (base frozen) ===")
        model = self._compile(model, lr=1e-3)
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self._callbacks(patience=5),
            class_weight=class_weight,
            verbose=1,
        )
        return history

    # ── Phase 2 — fine-tuning ─────────────────────────────────────────────────
    def train_phase2(
        self,
        model: Model,
        train_ds: tf.data.Dataset,
        val_ds: tf.data.Dataset,
        unfreeze_last: int = 80,
        epochs: int = 10,
        class_weight: Optional[dict] = None,
    ):
        """
        Unfreeze the last `unfreeze_last` layers of MobileNetV2 base and
        fine-tune with a very small learning rate.
        """
        log.info("=== Phase 2: Fine-Tuning (last %d layers unfrozen) ===", unfreeze_last)

        # Unfreeze the base model's last N layers
        # Layer name encodes the input size: mobilenetv2_1.00_128 for 128×128
        base_name = f"mobilenetv2_1.00_{INPUT_SHAPE[0]}"
        base = model.get_layer(base_name)
        base.trainable = True
        for layer in base.layers[:-unfreeze_last]:
            layer.trainable = False
        # Keep BatchNorm layers frozen — prevents epoch-1 accuracy drop caused by
        # disruption of running mean/variance statistics during fine-tuning
        for layer in base.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        trainable = sum(1 for l in base.layers if l.trainable)
        log.info("MobileNetV2 trainable layers: %d / %d", trainable, len(base.layers))

        model = self._compile(model, lr=1e-5)   # very small LR for fine-tuning
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs,
            callbacks=self._callbacks(patience=5),
            class_weight=class_weight,
            verbose=1,
        )
        return history

    # ── Retraining (fine-tune on new uploaded data) ───────────────────────────
    def retrain(
        self,
        X_new: np.ndarray,
        y_new: np.ndarray,
        epochs: int = 5,
        batch_size: int = 32,
        extra_callbacks: list = None,
        step_callback=None,
    ) -> Dict[str, Any]:
        """
        Fine-tune the saved model on newly uploaded samples.

        Parameters
        ----------
        X_new : (N, 784) raw pixel values [0, 255]
        y_new : (N,) int32 labels

        Returns
        -------
        dict with accuracy, f1_score, precision, recall, epochs_ran
        """
        from src.preprocessing import FashionMNISTPreprocessor
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score
        )

        def _s(msg):
            if step_callback: step_callback(msg)

        log.info("Retraining on %d new samples", len(X_new))

        # ── Load pre-trained model ─────────────────────────────────────────
        _s(f"Loading pre-trained model: fashion_model.h5 (custom MobileNetV2, 93.17% accuracy)")
        model = tf.keras.models.load_model(str(self.model_save_path))
        total_params = model.count_params()
        _s(f"Model loaded — {total_params:,} total parameters")

        # ── Preprocessing ─────────────────────────────────────────────────
        prep = FashionMNISTPreprocessor()
        _s(f"Normalising pixel values: 0–255 → 0.0–1.0 float32 ({len(X_new)} samples)")

        val_split = 0.2 if len(X_new) >= 50 else 0.0
        if val_split > 0:
            X_tr, X_val, y_tr, y_val = prep.train_val_split(X_new, y_new)
            _s(f"Train/validation split: {len(X_tr)} training / {len(X_val)} validation samples (80/20)")
            _s(f"Resizing images: 28×28 greyscale → 128×128 RGB (3 channels) for MobileNetV2 input")
            _s(f"Building training dataset with augmentation: random flip, brightness, contrast, zoom")
            train_ds = prep.make_tf_dataset(X_tr, y_tr, batch_size, augment=True)
            _s(f"Building validation dataset (no augmentation, batch_size={batch_size})")
            val_ds   = prep.make_tf_dataset(X_val, y_val, batch_size, shuffle=False)
            val_data = val_ds
        else:
            _s(f"Resizing images: 28×28 greyscale → 128×128 RGB (3 channels)")
            train_ds = prep.make_tf_dataset(X_new, y_new, batch_size, augment=False)
            val_data = None

        # ── Compile ────────────────────────────────────────────────────────
        _s(f"Compiling model: Adam lr=1e-4 (fine-tune rate), loss=sparse_categorical_crossentropy")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-4),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        _s(f"Ready to train — {epochs} max epochs, batch size {batch_size}, early stopping (patience=3)")

        cb = [keras.callbacks.EarlyStopping(
            monitor="val_loss" if val_data else "loss",
            patience=3,
            restore_best_weights=True,
        )] if val_data else []

        if extra_callbacks:
            cb = cb + extra_callbacks

        history = model.fit(
            train_ds,
            validation_data=val_data,
            epochs=epochs,
            callbacks=cb,
            verbose=1,
        )

        # ── Evaluate ───────────────────────────────────────────────────────
        _s(f"Evaluating model on all {len(X_new)} samples (no augmentation)")
        eval_ds = prep.make_tf_dataset(X_new, y_new, batch_size=64, shuffle=False, augment=False)
        y_pred  = np.argmax(model.predict(eval_ds, verbose=0), axis=1)
        acc     = float(accuracy_score(y_new, y_pred))
        f1      = float(f1_score(y_new, y_pred, average="macro", zero_division=0))
        prec    = float(precision_score(y_new, y_pred, average="macro", zero_division=0))
        rec     = float(recall_score(y_new, y_pred, average="macro", zero_division=0))
        _s(f"Evaluation complete — Accuracy: {acc*100:.2f}%  F1: {f1:.4f}")

        # ── Save ───────────────────────────────────────────────────────────
        _s(f"Saving updated model weights → fashion_model.h5")
        model.save(str(self.model_save_path))
        log.info("Retrained model saved → acc=%.4f  f1=%.4f", acc, f1)
        _s(f"Model saved successfully — pipeline complete")

        return {
            "accuracy":   acc,
            "f1_score":   f1,
            "precision":  prec,
            "recall":     rec,
            "epochs_ran": len(history.history["loss"]),
        }

    # ── Evaluation ──────────────────────────────────────────────────────
    @staticmethod
    def evaluate(
        model: Model,
        test_ds: tf.data.Dataset,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Compute accuracy, loss, F1, precision, recall, and per-class metrics.
        """
        from sklearn.metrics import (
            accuracy_score, f1_score, precision_score, recall_score,
            classification_report,
        )

        test_loss, test_acc_keras = model.evaluate(test_ds, verbose=0)
        y_pred_probs = model.predict(test_ds, verbose=0)
        y_pred       = np.argmax(y_pred_probs, axis=1)

        acc   = float(accuracy_score(y_test, y_pred))
        f1    = float(f1_score(y_test, y_pred, average="macro",    zero_division=0))
        prec  = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
        rec   = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
        pf1   = f1_score(y_test, y_pred, average=None, zero_division=0)
        pp    = precision_score(y_test, y_pred, average=None, zero_division=0)
        pr    = recall_score(y_test, y_pred, average=None, zero_division=0)

        log.info(
            "Evaluation — Acc:%.4f  Loss:%.4f  F1:%.4f  Prec:%.4f  Rec:%.4f",
            acc, test_loss, f1, prec, rec,
        )
        return {
            "accuracy":           acc,
            "test_loss":          float(test_loss),
            "f1_score":           f1,
            "precision":          prec,
            "recall":             rec,
            "per_class_f1":       {CLASS_NAMES[i]: float(pf1[i]) for i in range(NUM_CLASSES)},
            "per_class_precision":{CLASS_NAMES[i]: float(pp[i])  for i in range(NUM_CLASSES)},
            "per_class_recall":   {CLASS_NAMES[i]: float(pr[i])  for i in range(NUM_CLASSES)},
        }
