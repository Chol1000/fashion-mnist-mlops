"""
Load testing scenarios for the Fashion MNIST MLOps API.

Two user classes:
    FashionAPIUser    — realistic mix of /health, /predict, /metrics, /insights
    HeavyPredictUser  — rapid-fire /predict calls to stress-test inference

Run:
    locust -f locust/locustfile.py --host http://localhost:8000
    # Open http://localhost:8089 to configure users and spawn rate

Headless example:
    locust -f locust/locustfile.py --host http://localhost:8000 \
           --users 50 --spawn-rate 5 --run-time 60s --headless \
           --csv locust/results/run_50u
"""

import csv
import json
import random
from pathlib import Path

from locust import HttpUser, between, task, events

# ── Load test pixel data ──────────────────────────────────────────────────────
_HERE    = Path(__file__).resolve().parent
ROOT     = _HERE.parent  # project root (local) or /mnt (Docker — handled below)

# Support both local layout and Docker mount (./data:/mnt/locust/data)
_POSSIBLE_TEST_CSVS = [
    _HERE / "data" / "test" / "fashion-mnist_test.csv",   # Docker mount
    ROOT / "data"  / "test" / "fashion-mnist_test.csv",   # local
]
TEST_CSV = next((p for p in _POSSIBLE_TEST_CSVS if p.exists()), None)

_TEST_SAMPLES: list = []


def _load_samples():
    """Load up to 500 random test samples into memory for reuse."""
    global _TEST_SAMPLES
    if _TEST_SAMPLES:
        return

    if TEST_CSV is None or not TEST_CSV.exists():
        print("WARNING: test CSV not found — generating random pixel data for load test")
        _TEST_SAMPLES = [
            [random.randint(0, 255) for _ in range(784)] for _ in range(500)
        ]
        return
    path = TEST_CSV

    pixel_cols = [f"pixel{i}" for i in range(1, 785)]
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    sampled = random.sample(rows, min(500, len(rows)))
    _TEST_SAMPLES = [
        [float(row[c]) for c in pixel_cols]
        for row in sampled
    ]
    print(f"Loaded {len(_TEST_SAMPLES)} test samples for load testing")


@events.init.add_listener
def on_locust_init(environment, **kwargs):
    _load_samples()


# ── User Behaviour ────────────────────────────────────────────────────────────
class FashionAPIUser(HttpUser):
    """
    Simulates a typical user of the Fashion MNIST prediction API.

    Wait time: 0.5–2 seconds between requests (realistic think time).
    """
    wait_time = between(0.5, 2.0)

    def on_start(self):
        """Called once per user when they start."""
        _load_samples()

    # ── Task weights: higher = more frequent ─────────────────────────────────
    @task(10)
    def predict_pixel_values(self):
        """Most frequent: single-sample inference (core use case)."""
        pixels = random.choice(_TEST_SAMPLES) if _TEST_SAMPLES else [random.randint(0, 255) for _ in range(784)]
        payload = {"pixels": pixels}
        with self.client.post(
            "/predict",
            json=payload,
            catch_response=True,
            name="POST /predict",
        ) as resp:
            if resp.status_code == 200:
                data = resp.json()
                if "predicted_label" not in data:
                    resp.failure("Response missing 'predicted_label'")
                else:
                    resp.success()
            elif resp.status_code == 503:
                resp.failure("Model not loaded (503)")
            else:
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(3)
    def health_check(self):
        """Frequent: lightweight health probe."""
        with self.client.get("/health", catch_response=True, name="GET /health") as resp:
            if resp.status_code == 200:
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(2)
    def get_metrics(self):
        """Occasional: fetch model performance metrics."""
        with self.client.get("/metrics", catch_response=True, name="GET /metrics") as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")

    @task(1)
    def get_insights(self):
        """Least frequent: fetch dataset insights."""
        with self.client.get("/insights", catch_response=True, name="GET /insights") as resp:
            if resp.status_code in (200, 404):
                resp.success()
            else:
                resp.failure(f"Status {resp.status_code}")


class HeavyPredictUser(HttpUser):
    """
    Aggressive load tester — fires predictions as fast as possible.
    Use this to stress-test the model serving layer.
    """
    wait_time = between(0.1, 0.3)

    @task
    def rapid_predict(self):
        pixels = random.choice(_TEST_SAMPLES) if _TEST_SAMPLES else [random.randint(0, 255) for _ in range(784)]
        self.client.post("/predict", json={"pixels": pixels}, name="POST /predict [heavy]")
