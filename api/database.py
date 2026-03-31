"""
Fashion MNIST MLOps — Database Layer
======================================
SQLite-backed storage for:
  • Uploaded training samples (for retraining)
  • Retraining history (logs per retrain run)
  • Model uptime events
"""

from __future__ import annotations

import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd

from src.preprocessing import PIXEL_COLUMNS

log = logging.getLogger(__name__)

# ── Resolve DB path ───────────────────────────────────────────────────────────
# Works both inside Docker (/app/api/data) and locally (api/data/)
_HERE   = Path(__file__).resolve().parent
DB_PATH = _HERE / "data" / "fashion_data.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Connection ────────────────────────────────────────────────────────────────
@contextmanager
def _get_conn():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


# ── Schema ────────────────────────────────────────────────────────────────────
def init_db():
    """Create all tables if they do not exist."""
    with _get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS uploaded_samples (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                label       INTEGER NOT NULL,
                pixels      TEXT    NOT NULL,
                source      TEXT    DEFAULT 'csv_upload',
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS retrain_history (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                samples_used INTEGER,
                accuracy     REAL,
                f1_score     REAL,
                precision    REAL,
                recall       REAL,
                epochs_ran   INTEGER,
                notes        TEXT,
                trained_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_samples_label
                ON uploaded_samples(label);
        """)
    log.info("Database initialised at %s", DB_PATH)


# ── Uploaded samples ──────────────────────────────────────────────────────────
def insert_samples(df: pd.DataFrame) -> int:
    """
    Insert rows from a DataFrame (columns: label, pixel1..pixel784).
    Returns the number of rows inserted.
    """
    rows = []
    for _, row in df.iterrows():
        label  = int(row["label"])
        pixels = ",".join(str(int(row[c])) for c in PIXEL_COLUMNS)
        rows.append((label, pixels, "csv_upload"))

    with _get_conn() as conn:
        conn.executemany(
            "INSERT INTO uploaded_samples (label, pixels, source) VALUES (?, ?, ?)",
            rows,
        )
    log.info("Inserted %d samples into database", len(rows))
    return len(rows)


def fetch_uploaded_samples() -> Optional[pd.DataFrame]:
    """Return all uploaded samples as a DataFrame (label + pixel columns)."""
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT label, pixels FROM uploaded_samples ORDER BY uploaded_at"
        ).fetchall()

    if not rows:
        return None

    records = []
    for row in rows:
        record = {"label": row["label"]}
        vals   = list(map(int, row["pixels"].split(",")))
        for i, v in enumerate(vals, 1):
            record[f"pixel{i}"] = v
        records.append(record)

    return pd.DataFrame(records)


def count_uploaded_samples() -> int:
    with _get_conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM uploaded_samples"
        ).fetchone()[0]


def clear_uploaded_samples():
    with _get_conn() as conn:
        conn.execute("DELETE FROM uploaded_samples")
    log.info("Cleared all uploaded samples")


# ── Retrain history ───────────────────────────────────────────────────────────
def log_retrain(
    samples_used: int,
    accuracy: float,
    f1: float,
    precision: float = 0.0,
    recall: float = 0.0,
    epochs_ran: int = 0,
    notes: str = "",
):
    with _get_conn() as conn:
        conn.execute(
            """INSERT INTO retrain_history
               (samples_used, accuracy, f1_score, precision, recall, epochs_ran, notes)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (samples_used, accuracy, f1, precision, recall, epochs_ran, notes),
        )


def get_retrain_history(limit: int = 20) -> List[Dict[str, Any]]:
    with _get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM retrain_history ORDER BY trained_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
