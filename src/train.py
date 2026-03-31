"""
Standalone training script for the Fashion MNIST MobileNetV2 model.

Usage:
    python -m src.train
    python -m src.train --train data/train/fashion-mnist_train.csv \\
                        --test  data/test/fashion-mnist_test.csv  \\
                        --out   models/fashion_model.h5

Output:
    models/fashion_model.h5          trained Keras model
    models/training_metrics.json     accuracy, F1, per-class metrics, training history
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Suppress TF info/warning logs before importing TF ─────────────────────────
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train")


def parse_args():
    p = argparse.ArgumentParser(description="Train MobileNetV2 on Fashion MNIST")
    p.add_argument("--train",   default="data/train/fashion-mnist_train.csv")
    p.add_argument("--test",    default="data/test/fashion-mnist_test.csv")
    p.add_argument("--out",     default="models/fashion_model.h5")
    p.add_argument("--metrics", default="models/training_metrics.json")
    p.add_argument("--epochs1", type=int, default=10,
                   help="Max epochs for Phase 1 (feature extraction)")
    p.add_argument("--epochs2", type=int, default=10,
                   help="Max epochs for Phase 2 (fine-tuning)")
    p.add_argument("--batch",   type=int, default=32)
    p.add_argument("--val",     type=float, default=0.10,
                   help="Validation split fraction")
    return p.parse_args()


def main():
    args = parse_args()

    train_csv   = ROOT / args.train
    test_csv    = ROOT / args.test
    out_path    = ROOT / args.out
    metrics_path = ROOT / args.metrics

    # ── 1. Load & preprocess ──────────────────────────────────────────────────
    log.info("Loading data …")
    from src.preprocessing import FashionMNISTPreprocessor, TARGET_SIZE
    prep = FashionMNISTPreprocessor(val_size=args.val)

    X_train_all, y_train_all = prep.load_csv_to_arrays(train_csv)
    X_test,      y_test      = prep.load_csv_to_arrays(test_csv)

    X_tr, X_val, y_tr, y_val = prep.train_val_split(X_train_all, y_train_all)
    log.info("Split — train=%d  val=%d  test=%d", len(X_tr), len(X_val), len(X_test))

    log.info("Building tf.data pipelines (resize 28→%d, grayscale→RGB, MobileNetV2 scale) …", TARGET_SIZE)
    train_ds = prep.make_tf_dataset(X_tr,   y_tr,   args.batch, augment=True,  shuffle=True)
    val_ds   = prep.make_tf_dataset(X_val,  y_val,  args.batch, augment=False, shuffle=False)
    test_ds  = prep.make_tf_dataset(X_test, y_test, 64,         augment=False, shuffle=False)

    # ── 2. Build model ────────────────────────────────────────────────────────
    log.info("Building MobileNetV2 transfer learning model …")
    from src.model import MobileNetTransfer
    trainer = MobileNetTransfer(model_save_path=out_path)
    model   = trainer.build()
    model.summary(print_fn=log.info)

    # ── 3. Phase 1 — feature extraction ──────────────────────────────────────
    log.info("=== Phase 1: Feature Extraction ===")
    history1 = trainer.train_phase1(model, train_ds, val_ds, epochs=args.epochs1)

    # ── 4. Phase 2 — fine-tuning ──────────────────────────────────────────────
    log.info("=== Phase 2: Fine-Tuning ===")
    history2 = trainer.train_phase2(model, train_ds, val_ds, epochs=args.epochs2)

    # ── 5. Final evaluation ───────────────────────────────────────────────────
    log.info("Evaluating on test set …")
    eval_metrics = MobileNetTransfer.evaluate(model, test_ds, y_test)
    log.info(
        "Test — Acc=%.4f  Loss=%.4f  F1=%.4f  Prec=%.4f  Rec=%.4f",
        eval_metrics["accuracy"],
        eval_metrics["test_loss"],
        eval_metrics["f1_score"],
        eval_metrics["precision"],
        eval_metrics["recall"],
    )

    # ── 6. Combine training history (phase1 + phase2) ─────────────────────────
    def _concat(h1, h2, key):
        return h1.history.get(key, []) + h2.history.get(key, [])

    history = {
        "accuracy":     _concat(history1, history2, "accuracy"),
        "val_accuracy": _concat(history1, history2, "val_accuracy"),
        "loss":         _concat(history1, history2, "loss"),
        "val_loss":     _concat(history1, history2, "val_loss"),
    }

    # ── 7. Save metrics ────────────────────────────────────────────────────────
    out_json = {
        "training_config": {
            "model":          "MobileNetV2 Transfer Learning",
            "input_shape":    [TARGET_SIZE, TARGET_SIZE, 3],
            "head":           "GAP → Dense(256,relu,L2) → BN → Dropout(0.5) → Dense(10,softmax)",
            "phase1_epochs":  len(history1.history["loss"]),
            "phase2_epochs":  len(history2.history["loss"]),
            "batch_size":     args.batch,
            "val_split":      args.val,
            "optimizer_p1":   "Adam(lr=1e-3)",
            "optimizer_p2":   "Adam(lr=1e-5)",
            "augmentation":   ["random_flip_left_right", "random_brightness(0.05)"],
        },
        "initial_training": eval_metrics,
        "history": history,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(out_json, f, indent=2)
    log.info("Metrics saved → %s", metrics_path)

    # ── 8. Save final model ───────────────────────────────────────────────────
    model.save(str(out_path))
    log.info("Model saved → %s", out_path)

    log.info("Training complete!")
    log.info(
        "  Test Accuracy : %.4f (%.2f%%)",
        eval_metrics["accuracy"],
        eval_metrics["accuracy"] * 100,
    )
    log.info("  Macro F1      : %.4f", eval_metrics["f1_score"])
    log.info("  Macro Prec    : %.4f", eval_metrics["precision"])
    log.info("  Macro Recall  : %.4f", eval_metrics["recall"])


if __name__ == "__main__":
    main()
