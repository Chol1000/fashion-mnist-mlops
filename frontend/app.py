"""
Fashion MNIST MLOps — Streamlit Dashboard
==========================================
Tabs:
  1. Predict       — single-image or pixel-value inference
  2. Retrain       — upload CSV data + trigger model fine-tuning
  3. Insights      — dataset EDA with 3 feature interpretations
  4. Model Metrics — training curves, confusion matrix info, per-class F1
  5. API Status    — live model uptime and endpoint health
"""

from __future__ import annotations

import gc
import io
import os
import time

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

# ── Page config ───────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://backend:8000")

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]
PIXEL_COLS = [f"pixel{i}" for i in range(1, 785)]

st.set_page_config(
    page_title="Fashion MNIST MLOps",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ── */
[data-testid="stAppViewContainer"] > .main {
    background: #f5f6fa;
}

/* ── Main content block — white card ── */
[data-testid="block-container"] {
    background: #ffffff;
    border-radius: 16px;
    padding: 28px 36px !important;
    box-shadow: 0 1px 6px rgba(0,0,0,0.05);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #ffffff;
    border-right: 1px solid #ececf0;
    padding-top: 0 !important;
}
[data-testid="stSidebarContent"] {
    padding: 0 16px 24px 16px;
}

/* ── Sidebar brand ── */
.sb-brand {
    background: #5b6af5;
    margin: 0 -16px 24px -16px;
    padding: 26px 20px 22px 20px;
}
.sb-brand-title {
    font-size: 1.1rem; font-weight: 800;
    color: #ffffff; letter-spacing: -0.3px;
    margin: 0 0 4px 0;
}
.sb-brand-sub {
    font-size: 0.75rem; color: rgba(255,255,255,0.72);
    font-weight: 400; margin: 0;
}

/* ── Sidebar section heading ── */
.sb-section {
    font-size: 0.65rem; font-weight: 700;
    letter-spacing: 1.1px; text-transform: uppercase;
    color: #b0b4c1; margin: 20px 0 8px 2px;
}

/* ── Status row ── */
.sb-status-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 12px; border-radius: 9px;
    background: #fafafa; border: 1px solid #ececf0;
    margin-bottom: 5px;
}
.sb-status-key { font-size: 0.77rem; color: #6b7280; font-weight: 500; }
.dot-green  { display:inline-block; width:7px; height:7px; border-radius:50%; background:#22c55e; margin-right:5px; vertical-align:middle; }
.dot-red    { display:inline-block; width:7px; height:7px; border-radius:50%; background:#ef4444; margin-right:5px; vertical-align:middle; }
.sb-badge   { font-size: 0.73rem; font-weight: 600; }
.sb-badge.green  { color: #16a34a; }
.sb-badge.red    { color: #dc2626; }
.sb-badge.slate  { color: #5b6af5; }

/* ── Model stat rows ── */
.sb-stat-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 8px 12px; border-radius: 9px;
    background: #fafafa; border: 1px solid #ececf0;
    margin-bottom: 5px;
}
.sb-stat-lbl { font-size: 0.76rem; color: #6b7280; }
.sb-stat-val { font-size: 0.85rem; font-weight: 700; color: #1a1f36; }
.sb-stat-val.accent { color: #5b6af5; }

/* ── Class list ── */
.sb-class-grid {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 4px; margin-top: 2px;
}
.sb-class-item {
    font-size: 0.74rem; color: #374151; font-weight: 500;
    padding: 5px 8px; border-radius: 7px;
    background: #f3f4f6; border: 1px solid #e5e7eb;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

/* ── Page header ── */
.page-header {
    background: #ffffff;
    border: 1px solid #ececf0;
    border-left: 4px solid #5b6af5;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
.page-header h1 {
    margin: 0 0 4px 0; font-size: 1.35rem;
    font-weight: 700; color: #1a1f36;
}
.page-header p { margin: 0; font-size: 0.88rem; color: #6b7280; }

/* ── KPI card ── */
.kpi-card {
    background: #ffffff;
    border: 1px solid #ececf0;
    border-radius: 12px;
    padding: 18px 16px 16px;
    text-align: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04);
}
.kpi-value { font-size: 1.55rem; font-weight: 800; color: #5b6af5; line-height: 1.1; }
.kpi-label { font-size: 0.76rem; color: #9ca3af; margin-top: 5px; font-weight: 500; }

/* ── Prediction result cards ── */
.result-card {
    border-radius: 14px;
    padding: 24px 20px;
    text-align: center;
    border-width: 1.5px;
    border-style: solid;
}
.result-tag  { font-size: 0.68rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }
.result-name { font-size: 1.55rem; font-weight: 800; color: #1a1f36; margin: 4px 0 8px; line-height: 1.2; }
.result-conf-bar {
    height: 6px; border-radius: 99px;
    margin: 10px auto 8px; max-width: 180px;
}
.result-conf-text { font-size: 0.85rem; font-weight: 600; }

/* high ≥85% */
.conf-high { background:#f0fdf4; border-color:#86efac; }
.conf-high .result-tag  { color:#16a34a; }
.conf-high .result-conf-bar { background:#86efac; }
.conf-high .result-conf-text { color:#15803d; }

/* mid 50–84% */
.conf-mid  { background:#fffbeb; border-color:#fcd34d; }
.conf-mid .result-tag  { color:#b45309; }
.conf-mid .result-conf-bar { background:#fcd34d; }
.conf-mid .result-conf-text { color:#92400e; }

/* low <50% */
.conf-low  { background:#fff1f2; border-color:#fda4af; }
.conf-low .result-tag  { color:#be123c; }
.conf-low .result-conf-bar { background:#fda4af; }
.conf-low .result-conf-text { color:#9f1239; }

/* ── Primary button ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: #5b6af5 !important;
    border: none !important;
    border-radius: 8px !important;
    color: white !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 9px 22px !important;
    box-shadow: 0 2px 6px rgba(91,106,245,0.35) !important;
    transition: all 0.18s ease !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    background: #4a58e8 !important;
    box-shadow: 0 4px 10px rgba(91,106,245,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.84rem !important;
    color: #6b7280 !important;
    background: transparent !important;
    padding: 10px 18px !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #5b6af5 !important;
    border-bottom: 2px solid #5b6af5 !important;
    background: transparent !important;
}
[data-testid="stTabs"] button:hover {
    color: #5b6af5 !important;
    background: #f0f4ff !important;
    border-radius: 6px 6px 0 0 !important;
}
/* Tab content area */
[data-testid="stTabsContent"] {
    padding-top: 20px !important;
}

/* ── Sidebar expanders ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent !important;
    border: none !important;
    border-bottom: 1px solid #ececf0 !important;
    border-radius: 0 !important;
    margin-bottom: 2px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    font-size: 0.76rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.6px !important;
    color: #374151 !important;
    padding: 10px 2px !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary:hover {
    color: #5b6af5 !important;
}
[data-testid="stSidebar"] [data-testid="stExpanderContent"] {
    padding: 4px 0 12px 0 !important;
}

/* ── Crisp pixel rendering — only for small Fashion MNIST thumbnails ── */
[data-testid="stImage"] img[width="220"],
[data-testid="stImage"] img[width="224"] {
    image-rendering: pixelated;
    image-rendering: crisp-edges;
}

/* ── Charts and full-width images — smooth rendering ── */
[data-testid="stImage"] img {
    border-radius: 8px;
}

/* ── Divider ── */
hr { border-color: #ececf0 !important; margin: 22px 0 !important; }

/* ── Prevent grey dimming overlay on rerun ── */
[data-testid="stApp"] { opacity: 1 !important; }
[data-testid="stStatusWidget"] { display: none !important; }
.stApp > iframe { opacity: 1 !important; }
div[class*="stSpinner"] > div { border-top-color: #5b6af5 !important; }
</style>
""", unsafe_allow_html=True)


# ── Helpers ───────────────────────────────────────────────────────────────────
def api_get(path: str, timeout: int = 60, _retries: int = 3):
    for attempt in range(_retries):
        try:
            r = requests.get(f"{API_URL}{path}", timeout=timeout)
            if r.status_code == 502 and attempt < _retries - 1:
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json(), None
        except requests.ConnectionError:
            if attempt < _retries - 1:
                time.sleep(5)
                continue
            return None, "Cannot connect to the API. The service may be waking up — please refresh in 30 seconds."
        except requests.Timeout:
            return None, "Request timed out. The API may be waking up from sleep — please refresh in 30 seconds."
        except requests.HTTPError as e:
            return None, f"HTTP {e.response.status_code}"
        except Exception as e:
            return None, str(e)
    return None, "API unavailable after retries — please refresh in 30 seconds."


def api_post(path: str, json_body=None, files=None, timeout: int = 120, _retries: int = 3):
    for attempt in range(_retries):
        try:
            if files:
                r = requests.post(f"{API_URL}{path}", files=files, timeout=timeout)
            else:
                r = requests.post(f"{API_URL}{path}", json=json_body, timeout=timeout)
            if r.status_code == 502 and attempt < _retries - 1:
                time.sleep(5)
                continue
            r.raise_for_status()
            return r.json(), None
        except requests.ConnectionError:
            if attempt < _retries - 1:
                time.sleep(5)
                continue
            return None, "Cannot connect to the API. The service may be waking up — please retry."
        except requests.Timeout:
            return None, "Request timed out. The API may be waking up — please retry."
        except requests.HTTPError as e:
            try:
                detail = e.response.json().get("detail", str(e))
            except Exception:
                detail = str(e)
            return None, detail
        except Exception as e:
            return None, str(e)
    return None, "API unavailable after retries — please retry in 30 seconds."


@st.cache_data(ttl=600)
def load_test_data():
    for p in ["/data/test/fashion-mnist_test.csv",
              "../data/test/fashion-mnist_test.csv",
              "data/test/fashion-mnist_test.csv"]:
        if os.path.exists(p):
            return pd.read_csv(p)
    return None


@st.cache_data(ttl=300)
def _fetch_health():
    data, err = api_get("/health")
    if err:
        raise RuntimeError(err)
    return data


@st.cache_data(ttl=300)
def _fetch_insights():
    data, err = api_get("/insights")
    if err:
        raise RuntimeError(err)
    return data


@st.cache_data(ttl=300)
def _fetch_metrics():
    data, err = api_get("/metrics")
    if err:
        raise RuntimeError(err)
    return data


def _conf_level(confidence: float) -> str:
    """Return css class based on confidence."""
    if confidence >= 0.85:
        return "conf-high"
    elif confidence >= 0.50:
        return "conf-mid"
    return "conf-low"


def _conf_bar_color(confidence: float) -> str:
    if confidence >= 0.85: return "#22c55e"
    if confidence >= 0.50: return "#f59e0b"
    return "#f43f5e"


def _conf_bar_bg(confidence: float) -> str:
    if confidence >= 0.85: return "#dcfce7"
    if confidence >= 0.50: return "#fef3c7"
    return "#ffe4e6"


def render_result_card(label: str, confidence: float, tag: str = "Predicted Class"):
    """Render a confidence-colored prediction result card."""
    level     = _conf_level(confidence)
    bar_color = _conf_bar_color(confidence)
    bar_bg    = _conf_bar_bg(confidence)
    bar_w     = int(confidence * 100)
    pct       = f"{confidence * 100:.1f}%"
    st.markdown(f"""
    <div class="result-card {level}">
        <div class="result-tag">{tag}</div>
        <div class="result-name">{label}</div>
        <div style="background:{bar_bg};border-radius:99px;height:6px;max-width:200px;margin:0 auto 6px">
            <div style="background:{bar_color};width:{bar_w}%;height:6px;border-radius:99px;transition:width 0.4s ease"></div>
        </div>
        <div class="result-conf-text" style="color:{'#15803d' if confidence>=0.85 else '#92400e' if confidence>=0.50 else '#9f1239'}">
            {pct} confidence
        </div>
    </div>
    """, unsafe_allow_html=True)


def plot_probs(probs: dict, confidence: float = 1.0):
    """All-class probability bar chart sorted descending, winner colored by confidence."""
    sorted_items = sorted(probs.items(), key=lambda x: x[1], reverse=True)
    labels  = [item[0] for item in sorted_items]
    values  = [item[1] for item in sorted_items]
    top_col = _conf_bar_color(confidence)
    colors  = [top_col if i == 0 else "#e5e7eb" for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(7, 3.8))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("#fafafa")
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1], height=0.52, edgecolor="none")
    ax.set_xlim(0, 1.12)
    ax.set_xlabel("Probability", fontsize=9, color="#9ca3af")
    ax.set_title("All Class Probabilities", fontsize=11, fontweight="600", color="#1a1f36", pad=10)
    ax.tick_params(labelsize=8.5, colors="#6b7280")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#e5e7eb")
    ax.spines["bottom"].set_color("#e5e7eb")
    rev_values = values[::-1]
    rev_colors = colors[::-1]
    for bar, v, c in zip(bars, rev_values, rev_colors):
        if v > 0.001:
            is_top = c == top_col
            ax.text(v + 0.012, bar.get_y() + bar.get_height() / 2,
                    f"{v*100:.1f}%", va="center", fontsize=8.5,
                    color=top_col if is_top else "#9ca3af",
                    fontweight="bold" if is_top else "normal")
    plt.tight_layout()
    _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight"); _buf.seek(0)
    st.image(_buf, use_column_width="always")
    plt.close("all"); gc.collect()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand — text only, no icons
    st.markdown("""
    <div class="sb-brand">
        <p class="sb-brand-title">Fashion MNIST</p>
        <p class="sb-brand-sub">MLOps Pipeline Dashboard</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        health = _fetch_health()
        api_online  = True
        model_ready = health.get("model_ready", False)
        uptime      = health.get("uptime_sec", 0)
        db_samples  = health.get("db_samples", 0)
    except Exception:
        api_online = model_ready = False
        uptime = db_samples = 0

    api_dot = "dot-green" if api_online  else "dot-red"
    api_cls = "green"     if api_online  else "red"
    api_txt = "Online"    if api_online  else "Offline"
    mdl_dot = "dot-green" if model_ready else "dot-red"
    mdl_cls = "green"     if model_ready else "red"
    mdl_txt = "Ready"     if model_ready else "Not Ready"

    with st.expander("System Status", expanded=True):
        st.markdown(f"""
        <div class="sb-status-row">
            <span class="sb-status-key">API Server</span>
            <span class="sb-badge {api_cls}"><span class="{api_dot}"></span>{api_txt}</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-key">ML Model</span>
            <span class="sb-badge {mdl_cls}"><span class="{mdl_dot}"></span>{mdl_txt}</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-key">Uptime</span>
            <span class="sb-badge slate">{int(uptime)}s</span>
        </div>
        <div class="sb-status-row">
            <span class="sb-status-key">DB Samples</span>
            <span class="sb-badge slate">{db_samples}</span>
        </div>
        """, unsafe_allow_html=True)

    with st.expander("Model Performance", expanded=True):
        try:
            _sb_metrics = _fetch_metrics()
        except Exception:
            _sb_metrics = None
        _base_acc  = 93.17
        _base_f1   = 0.9297
        _base_prec = 0.9318
        _base_rec  = 0.9288
        if _sb_metrics:
            _ev = _sb_metrics.get("evaluation", {})
            _base_acc  = round(_ev.get("accuracy",  _base_acc  / 100) * 100, 2)
            _base_f1   = _ev.get("f1_score",   _base_f1)
            _base_prec = _ev.get("precision",  _base_prec)
            _base_rec  = _ev.get("recall",     _base_rec)
        _rt = (_sb_metrics or {}).get("retrain", {})

        _perf_html = f"""
        <div style="font-size:0.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px">Original Model</div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Test Accuracy</span>
            <span class="sb-stat-val accent">{_base_acc:.2f}%</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Macro F1</span>
            <span class="sb-stat-val accent">{_base_f1:.4f}</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Precision</span>
            <span class="sb-stat-val">{_base_prec:.4f}</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Recall</span>
            <span class="sb-stat-val">{_base_rec:.4f}</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Architecture</span>
            <span class="sb-stat-val">MobileNetV2</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Input Size</span>
            <span class="sb-stat-val">128 x 128 x 3</span>
        </div>
        """
        if _rt:
            _rt_acc  = round(_rt.get("accuracy",  0) * 100, 2)
            _rt_f1   = _rt.get("f1_score",  0)
            _rt_prec = _rt.get("precision", 0)
            _rt_rec  = _rt.get("recall",    0)
            _rt_ep   = _rt.get("epochs_ran", "?")
            _rt_samp = _rt.get("samples", "?")
            _acc_delta = _rt_acc - _base_acc
            _delta_color = "#10b981" if _acc_delta >= 0 else "#ef4444"
            _delta_sign  = "+" if _acc_delta >= 0 else ""
            _perf_html += f"""
        <div style="margin-top:10px;padding-top:8px;border-top:1px solid #e5e7eb">
        <div style="font-size:0.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                    letter-spacing:.06em;margin-bottom:4px">After Retraining</div>
        <div style="font-size:0.7rem;color:#6b7280;margin-bottom:6px">{_rt_samp} samples · {_rt_ep} epochs</div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Accuracy</span>
            <span class="sb-stat-val" style="color:{_delta_color}">{_rt_acc:.2f}%
                <small>({_delta_sign}{_acc_delta:.2f}%)</small></span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">F1 Score</span>
            <span class="sb-stat-val">{_rt_f1:.4f}</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Precision</span>
            <span class="sb-stat-val">{_rt_prec:.4f}</span>
        </div>
        <div class="sb-stat-row">
            <span class="sb-stat-lbl">Recall</span>
            <span class="sb-stat-val">{_rt_rec:.4f}</span>
        </div>
        </div>
            """
        st.markdown(_perf_html, unsafe_allow_html=True)

    with st.expander("Clothing Classes", expanded=True):
        items = "".join(f'<div class="sb-class-item">{n}</div>' for n in CLASS_NAMES)
        st.markdown(f'<div class="sb-class-grid">{items}</div>', unsafe_allow_html=True)

    st.markdown(f"<br><small style='color:#b0b4c1;font-size:0.7rem'>v2.0 &nbsp;·&nbsp; {API_URL}</small>", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "Predict",
    "Upload & Retrain",
    "Dataset Insights",
    "Model Metrics",
    "API Status",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PREDICT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("""
    <div class="page-header">
        <h1>Garment Classification</h1>
        <p>Upload a clothing image, load a random test sample, or paste raw pixel values to classify it.</p>
    </div>
    """, unsafe_allow_html=True)

    method = st.radio("Input method", ["Upload Image", "Random Test Sample", "Paste Pixel Values"], horizontal=True)
    st.divider()

    # ── Upload image ──────────────────────────────────────────────────────────
    if method == "Upload Image":
        uploaded = st.file_uploader(
            "Upload a clothing image",
            type=["png", "jpg", "jpeg", "webp", "avif"],
            help="Any size — will be auto-resized to 28x28 greyscale",
        )
        if uploaded:
            raw_bytes = uploaded.read()
            uploaded.seek(0)
            try:
                original = Image.open(io.BytesIO(raw_bytes))
                img = original.convert("L").resize((28, 28))
                preview_bytes = raw_bytes
            except Exception:
                # AVIF or unsupported format — convert via API directly, no local preview
                original = None
                img = None
                preview_bytes = raw_bytes
            c1, c2 = st.columns([1, 2])
            with c1:
                if original:
                    st.image(original, caption="Uploaded image", width=220)
                else:
                    st.info("Preview not available for this format — image will be sent to API for classification.")

            if st.button("Classify Image", type="primary", key="btn_img"):
                with st.spinner("Running inference..."):
                    result, err = api_post("/predict/image", files={"file": (uploaded.name, preview_bytes, uploaded.type)})
                if err:
                    st.error(f"Error: {err}")
                else:
                    with c2:
                        render_result_card(result["predicted_label"], result["confidence"])
                        st.markdown("<br>", unsafe_allow_html=True)
                        plot_probs(result["probabilities"], result["confidence"])

    # ── Random test sample ────────────────────────────────────────────────────
    elif method == "Random Test Sample":
        test_df = load_test_data()
        if test_df is None:
            st.warning("Test data not found.")
        else:
            col_f, col_b = st.columns([2, 1])
            with col_f:
                class_filter = st.selectbox("Filter by class (optional)", ["All"] + CLASS_NAMES)
            with col_b:
                st.markdown("<br>", unsafe_allow_html=True)
                load_btn = st.button("Load Random Sample", type="primary", key="btn_random")

            if load_btn:
                filtered = test_df if class_filter == "All" else test_df[test_df["label"] == CLASS_NAMES.index(class_filter)]
                st.session_state["sample_row"] = filtered.sample(1, random_state=None).iloc[0].to_dict()

            if "sample_row" in st.session_state:
                row        = st.session_state["sample_row"]
                true_label = CLASS_NAMES[int(row["label"])]
                pixels     = [float(row[c]) for c in PIXEL_COLS]
                arr        = np.array(pixels).reshape(28, 28)

                st.divider()
                c1, c2 = st.columns([1, 2])
                with c1:
                    # Upscale 28x28 → 224x224 with nearest neighbour (sharp, no blur)
                    pil_img = Image.fromarray(arr.astype(np.uint8), mode="L").resize((224, 224), Image.NEAREST)
                    st.image(pil_img, caption=f"True label: {true_label}", width=220)
                    st.markdown("<br>", unsafe_allow_html=True)
                    st.markdown("<br>", unsafe_allow_html=True)
                    classify_btn = st.button("Classify", type="primary", key="btn_sample", use_container_width=True)

                with c2:
                    if classify_btn:
                        with st.spinner("Running inference..."):
                            result, err = api_post("/predict", json_body={"pixels": pixels})
                        if err:
                            st.error(err)
                        else:
                            correct = result["predicted_label"] == true_label
                            tag = "Correct Prediction" if correct else "Incorrect Prediction"
                            render_result_card(result["predicted_label"], result["confidence"], tag)
                            st.markdown("<br>", unsafe_allow_html=True)
                            plot_probs(result["probabilities"], result["confidence"])

    # ── Paste pixels ──────────────────────────────────────────────────────────
    else:
        st.markdown("Paste **784 comma-separated pixel values** (range 0–255).")
        pixel_text = st.text_area("Pixel values", height=100, placeholder="0,0,0,12,45,200,...")
        if pixel_text and st.button("Classify", type="primary", key="btn_paste"):
            try:
                pixels = [float(v.strip()) for v in pixel_text.split(",")]
                if len(pixels) != 784:
                    st.error(f"Expected 784 values, got {len(pixels)}")
                else:
                    with st.spinner("Running inference..."):
                        result, err = api_post("/predict", json_body={"pixels": pixels})
                    if err:
                        st.error(err)
                    else:
                        arr = np.array(pixels).reshape(28, 28)
                        st.divider()
                        c1, c2 = st.columns([1, 2])
                        with c1:
                            pil_img = Image.fromarray(arr.astype(np.uint8), mode="L").resize((224, 224), Image.NEAREST)
                            st.image(pil_img, caption="28 x 28 greyscale", width=220)
                        with c2:
                            render_result_card(result["predicted_label"], result["confidence"])
                            st.markdown("<br>", unsafe_allow_html=True)
                            plot_probs(result["probabilities"], result["confidence"])
            except ValueError:
                st.error("Invalid values — enter comma-separated numbers.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — UPLOAD & RETRAIN
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("""
    <div class="page-header">
        <h1>Upload Data & Retrain</h1>
        <p>Three-stage MLOps pipeline: upload labelled samples to the database, preprocess, then fine-tune the model.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Pipeline overview ──────────────────────────────────────────────────────
    st.markdown("""
    <div style="display:flex;gap:0;margin-bottom:24px">
        <div style="flex:1;background:#eef2ff;border:1px solid #c7d2fe;border-radius:8px 0 0 8px;
                    padding:14px 16px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:#5b6af5">1</div>
            <div style="font-size:0.82rem;font-weight:600;color:#374151;margin-top:2px">Upload CSV</div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:2px">Save to SQLite DB</div>
        </div>
        <div style="flex:1;background:#f0fdf4;border-top:1px solid #bbf7d0;border-bottom:1px solid #bbf7d0;
                    padding:14px 16px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:#10b981">2</div>
            <div style="font-size:0.82rem;font-weight:600;color:#374151;margin-top:2px">Preprocess</div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:2px">Normalise · Resize · Augment</div>
        </div>
        <div style="flex:1;background:#fff7ed;border:1px solid #fed7aa;border-radius:0 8px 8px 0;
                    padding:14px 16px;text-align:center">
            <div style="font-size:1.3rem;font-weight:700;color:#f97316">3</div>
            <div style="font-size:0.82rem;font-weight:600;color:#374151;margin-top:2px">Fine-tune</div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:2px">Custom MobileNetV2 base</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — UPLOAD
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:28px;height:28px;border-radius:50%;background:#5b6af5;color:#fff;
                    font-weight:700;font-size:0.85rem;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0">1</div>
        <span style="font-size:1.05rem;font-weight:700;color:#1f2937">Upload Training Data</span>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Expected CSV format", expanded=False):
        st.markdown("""
| Column | Type | Description |
|--------|------|-------------|
| `label` | Integer 0–9 | Clothing class (0=T-shirt, 1=Trouser, 2=Pullover, 3=Dress, 4=Coat, 5=Sandal, 6=Shirt, 7=Sneaker, 8=Bag, 9=Ankle boot) |
| `pixel1` … `pixel784` | Integer 0–255 | Greyscale pixel values of the 28×28 image (784 total) |
        """)
        st.code("label,pixel1,pixel2,...,pixel784\n0,0,0,12,45,200,...\n9,255,128,0,...", language="csv")

    # Quick sample generator — prefer training set (60k), fall back to test set (10k)
    _gen_df = None
    for _p in ["/data/train/fashion-mnist_train.csv", "../data/train/fashion-mnist_train.csv", "data/train/fashion-mnist_train.csv"]:
        if os.path.exists(_p):
            _gen_df = pd.read_csv(_p)
            break
    if _gen_df is None:
        _gen_df = load_test_data()
    _gen_source = "training set (60,000 samples)" if _gen_df is not None and len(_gen_df) > 10000 else "test set (10,000 samples)"

    if _gen_df is not None:
        _gen_max = len(_gen_df)
        st.markdown(f"<div style='font-size:0.85rem;color:#6b7280;margin-bottom:6px'>No CSV? Generate one instantly from the Fashion MNIST {_gen_source}:</div>", unsafe_allow_html=True)
        sc1, sc2 = st.columns([2, 1])
        with sc1:
            n_samples = st.slider("Sample size", 60, _gen_max, 60, step=10)
        with sc2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Generate Sample CSV", key="btn_gen_sample"):
                sample_df = _gen_df.sample(n_samples, random_state=None)
                csv_bytes  = sample_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {n_samples}-row CSV",
                    data=csv_bytes,
                    file_name="retrain_sample.csv",
                    mime="text/csv",
                    key="btn_dl_sample",
                )

    # Database status bar
    db_info, _ = api_get("/health")
    cur = db_info.get("db_samples", 0) if db_info else 0
    sb_bg  = "#d1fae5" if cur > 0 else "#f3f4f6"
    sb_bdr = "#6ee7b7" if cur > 0 else "#e5e7eb"
    sb_clr = "#065f46" if cur > 0 else "#6b7280"
    st.markdown(f"""
    <div style="background:{sb_bg};border:1px solid {sb_bdr};border-radius:8px;
                padding:11px 16px;margin:10px 0">
        <span style="font-size:0.82rem;font-weight:600;color:{sb_clr}">SQLite Database</span>
        <span style="font-size:0.82rem;color:{sb_clr};margin-left:8px">
            {"— <b>" + str(cur) + " sample(s)</b> stored and ready for retraining" if cur > 0 else "— empty, upload a CSV to begin"}
        </span>
    </div>
    """, unsafe_allow_html=True)

    upcol1, upcol2 = st.columns([4, 1])
    with upcol1:
        upload_file = st.file_uploader("Choose a CSV file", type=["csv"], key="upload_csv")
    with upcol2:
        if cur > 0:
            st.markdown("<br><br>", unsafe_allow_html=True)
            if st.button("Clear DB", key="btn_clear_db", help="Remove all uploaded samples"):
                try:
                    requests.delete(f"{API_URL}/uploaded-data", timeout=10)
                    st.success("Cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if upload_file:
        try:
            preview_df = pd.read_csv(io.BytesIO(upload_file.read()))
            upload_file.seek(0)
            n_rows, n_cols = len(preview_df), len(preview_df.columns)
            st.success(f"File loaded: **{n_rows} samples**, {n_cols} columns (1 label + {n_cols-1} pixel features)")

            # Clean preview — label renamed, first 5 pixels shown
            pcols = ["label"] + [c for c in preview_df.columns if c.startswith("pixel")][:5]
            disp  = preview_df[pcols].head(8).copy()
            disp.columns = ["Label (class)"] + [f"Pixel {i+1}" for i in range(len(disp.columns)-1)]
            st.markdown("<div style='font-size:0.78rem;color:#6b7280;margin-bottom:4px'>Preview — first 8 rows, first 5 of 784 pixel columns:</div>", unsafe_allow_html=True)
            st.dataframe(disp, use_container_width=True, hide_index=True)

            lc = preview_df["label"].value_counts().sort_index()
            st.markdown(
                "<div style='font-size:0.75rem;color:#6b7280;margin-top:4px'>Class distribution: " +
                " · ".join([f"{CLASS_NAMES[int(k)]}: {v}" for k, v in lc.items()]) +
                "</div>", unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Could not read file: {e}")

        if st.button("Upload to Database", type="primary", key="btn_upload"):
            upload_file.seek(0)
            with st.spinner("Saving samples to SQLite database..."):
                result, err = api_post("/upload-data", files={"file": (upload_file.name, upload_file.read(), "text/csv")})
            if err:
                st.error(f"Upload failed: {err}")
            else:
                c1, c2 = st.columns(2)
                c1.metric("Samples Added", result["samples_added"])
                c2.metric("Total in DB",   result["total_in_db"])
                st.success("Saved to database — proceed to Stage 3 to retrain.")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — PREPROCESSING (informational)
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:28px;height:28px;border-radius:50%;background:#10b981;color:#fff;
                    font-weight:700;font-size:0.85rem;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0">2</div>
        <span style="font-size:1.05rem;font-weight:700;color:#1f2937">Data Preprocessing</span>
    </div>
    <div style="font-size:0.82rem;color:#6b7280;margin-bottom:10px;margin-left:38px">
        Applied automatically to every uploaded sample before training begins (FashionMNISTPreprocessor).
    </div>
    """, unsafe_allow_html=True)

    p1, p2, p3, p4 = st.columns(4)
    for col, title, desc in [
        (p1, "Normalise",  "Pixels 0–255 → 0.0–1.0 float32"),
        (p2, "Resize",     "28×28 greyscale → 128×128 RGB"),
        (p3, "Augment",    "Random flip, brightness, contrast, zoom"),
        (p4, "Split",      "80% train · 20% validation"),
    ]:
        col.markdown(f"""
        <div style="background:#f9fafb;border:1px solid #e5e7eb;border-radius:8px;
                    padding:12px;text-align:center">
            <div style="font-size:0.82rem;font-weight:700;color:#374151">{title}</div>
            <div style="font-size:0.72rem;color:#6b7280;margin-top:4px">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — FINE-TUNE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <div style="width:28px;height:28px;border-radius:50%;background:#f97316;color:#fff;
                    font-weight:700;font-size:0.85rem;display:flex;align-items:center;
                    justify-content:center;flex-shrink:0">3</div>
        <span style="font-size:1.05rem;font-weight:700;color:#1f2937">Fine-tune the Model</span>
    </div>
    <div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:10px 14px;
                margin-bottom:12px;margin-left:38px;font-size:0.82rem;color:#92400e">
        Loads <strong>fashion_model.h5</strong> — your custom-trained MobileNetV2 (93.17% accuracy on 60K samples)
        as the pre-trained base — then fine-tunes the final layers on uploaded samples using
        <strong>Adam lr=1e-4</strong> with early stopping.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1: epochs     = st.slider("Max epochs", 1, 20, 5)
    with c2: batch_size = st.selectbox("Batch size", [32, 64, 128], index=1)
    with c3:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_after = st.checkbox("Clear DB after retraining")

    if st.button("Start Retraining", type="primary", key="btn_retrain"):
        result, err = api_post("/retrain", json_body={"epochs": epochs, "batch_size": batch_size, "clear_after": clear_after})
        if err:
            st.error(f"Could not start: {err}")
        else:
            st.success(result["message"])
            progress_bar = st.progress(0)

            # Two-column live dashboard: left=steps checklist, right=epoch chart
            left_col, right_col = st.columns([1, 1])
            with left_col:
                st.markdown("**Pipeline Steps**")
                steps_slot = st.empty()
            with right_col:
                st.markdown("**Training Progress**")
                epoch_slot  = st.empty()
                chart_slot  = st.empty()

            while True:
                time.sleep(1)
                status, _ = api_get("/retrain/status")
                if not status:
                    continue

                phase       = status.get("phase", "")
                cur_epoch   = status.get("current_epoch", 0)
                tot_epochs  = status.get("total_epochs", epochs)
                epoch_logs  = status.get("epoch_logs", [])
                elapsed     = status.get("elapsed_sec", 0) or 0
                steps       = status.get("steps", [])
                current_step= status.get("current_step", "")

                # ── Progress bar ────────────────────────────────────────────
                if phase in ("loading data", "preprocessing"):
                    pct = 8
                elif phase == "training" and tot_epochs > 0:
                    pct = 10 + min(int(cur_epoch / tot_epochs * 85), 85)
                elif phase == "done":
                    pct = 100
                else:
                    pct = 96
                progress_bar.progress(pct)

                # ── LEFT: step-by-step checklist ─────────────────────────
                steps_html = "<div style='font-size:0.78rem;line-height:1.8'>"
                for s in steps:
                    steps_html += (
                        f"<div style='display:flex;gap:6px;align-items:flex-start;"
                        f"margin-bottom:3px'>"
                        f"<span style='color:#10b981;font-weight:700;flex-shrink:0'>✓</span>"
                        f"<span style='color:#374151'>{s['msg']}"
                        f" <span style='color:#9ca3af;font-size:0.7rem'>({s['elapsed']}s)</span>"
                        f"</span></div>"
                    )
                if current_step and (not steps or steps[-1]["msg"] != current_step):
                    steps_html += (
                        f"<div style='display:flex;gap:6px;align-items:flex-start;margin-bottom:3px'>"
                        f"<span style='color:#f97316;font-weight:700;flex-shrink:0'>▶</span>"
                        f"<span style='color:#6b7280;font-style:italic'>{current_step}</span>"
                        f"</div>"
                    )
                steps_html += "</div>"
                steps_slot.markdown(steps_html, unsafe_allow_html=True)

                # ── RIGHT: epoch counter + chart ─────────────────────────
                if epoch_logs:
                    epoch_slot.markdown(
                        f"<div style='font-size:0.82rem;color:#374151'>"
                        f"Epoch <b>{cur_epoch}</b> / <b>{tot_epochs}</b> &nbsp;·&nbsp; {int(elapsed)}s elapsed</div>",
                        unsafe_allow_html=True
                    )
                    has_val = any(e.get("val_accuracy") is not None for e in epoch_logs)
                    fig, ax = plt.subplots(figsize=(5, 2.6))
                    xs = [e["epoch"] for e in epoch_logs]
                    ax.plot(xs, [e["accuracy"] for e in epoch_logs],
                            marker="o", color="#5b6af5", label="Train Acc %", linewidth=2)
                    if has_val:
                        ax.plot(xs, [e.get("val_accuracy") for e in epoch_logs],
                                marker="s", color="#10b981", label="Val Acc %",
                                linewidth=2, linestyle="--")
                    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy %")
                    ax.set_xticks(xs); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
                    plt.tight_layout()
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
                    buf.seek(0)
                    chart_slot.image(buf, use_column_width="always")
                    plt.close("all"); gc.collect()

                if not status["running"]:
                    progress_bar.progress(100)
                    break

            # ── Final results ────────────────────────────────────────────────
            status, _ = api_get("/retrain/status")
            if status and status.get("last_result"):
                res = status["last_result"]
                if "error" in res:
                    st.error(f"Failed: {res['error']}")
                else:
                    st.balloons()
                    ep_ran = res.get("epochs_ran", "?")
                    st.success(f"Retraining complete! Ran {ep_ran} epoch(s).")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy",  f"{res['accuracy']*100:.2f}%")
                    m2.metric("F1 Score",  f"{res['f1_score']:.4f}")
                    m3.metric("Precision", f"{res['precision']:.4f}")
                    m4.metric("Recall",    f"{res['recall']:.4f}")

                    # Final epoch table + chart
                    final_logs = status.get("epoch_logs", [])
                    if final_logs:
                        st.markdown("#### Epoch-by-Epoch Results")
                        has_val = any(e.get("val_accuracy") is not None for e in final_logs)
                        rows = []
                        for e in final_logs:
                            row = {"Epoch": e["epoch"], "Train Loss": e["loss"], "Train Acc %": e["accuracy"]}
                            if has_val:
                                row["Val Loss"]  = e.get("val_loss")
                                row["Val Acc %"] = e.get("val_accuracy")
                            rows.append(row)
                        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

                        fig2, ax2 = plt.subplots(figsize=(7, 3))
                        xs2 = [e["epoch"] for e in final_logs]
                        ax2.plot(xs2, [e["accuracy"] for e in final_logs],
                                 marker="o", color="#5b6af5", label="Train Acc %", linewidth=2)
                        if has_val:
                            ax2.plot(xs2, [e.get("val_accuracy") for e in final_logs],
                                     marker="s", color="#10b981", label="Val Acc %",
                                     linewidth=2, linestyle="--")
                        ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy %")
                        ax2.set_title("Final Training Curve")
                        ax2.set_xticks(xs2); ax2.legend(); ax2.grid(True, alpha=0.3)
                        plt.tight_layout()
                        buf2 = io.BytesIO()
                        fig2.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                        buf2.seek(0)
                        st.image(buf2, use_column_width="always")
                        plt.close("all"); gc.collect()

    st.divider()

    st.divider()
    st.markdown("### Retraining History")
    history, _ = api_get("/retrain/history")
    if history and history.get("history"):
        h_df = pd.DataFrame(history["history"])
        cols = [c for c in ["trained_at", "samples_used", "accuracy", "f1_score", "precision", "recall", "epochs_ran", "notes"] if c in h_df.columns]
        st.dataframe(h_df[cols], use_container_width=True)
    else:
        st.info("No retraining runs yet.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DATASET INSIGHTS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown("""
    <div class="page-header">
        <h1>Dataset Insights</h1>
        <p>Exploratory analysis of Fashion MNIST — 70,000 images across 10 clothing categories.</p>
    </div>
    """, unsafe_allow_html=True)

    try:
        with st.spinner("Loading dataset insights..."):
            insights = _fetch_insights()
    except Exception as e:
        st.error(str(e))
        st.stop()

    # KPIs
    k1, k2, k3, k4 = st.columns(4)
    kpi_data = [
        (k1, f"{insights['total_train_samples']:,}", "Training Samples"),
        (k2, str(insights["num_classes"]),           "Clothing Classes"),
        (k3, insights["image_size"],                 "Image Size (px)"),
        (k4, f"{insights['pixel_statistics']['mean']:.1f}", "Mean Pixel Value"),
    ]
    for col, val, lbl in kpi_data:
        with col:
            st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature 1 — Class distribution
    st.markdown("### Feature 1 — Class Distribution")
    st.markdown(
        "The dataset is perfectly balanced — each of the 10 categories has exactly **6,000 training samples**. "
        "This eliminates class imbalance as a source of bias."
    )
    with st.spinner("Rendering chart..."):
        classes = insights["classes"]
        names   = [v["name"]  for v in classes.values()]
        counts  = [v["count"] for v in classes.values()]
        palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(names)))

        fig, ax = plt.subplots(figsize=(11, 4))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("#fafafa")
        bars = ax.bar(names, counts, color=palette, edgecolor="none", width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=10)
        ax.set_ylabel("Samples", fontsize=10)
        ax.set_title("Training Samples per Class", fontsize=13, fontweight="600", pad=12)
        ax.set_ylim(0, max(counts) * 1.18)
        ax.axhline(np.mean(counts), color="#ef4444", linestyle="--", linewidth=1.2, label=f"Mean: {int(np.mean(counts)):,}")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#e5e7eb")
        ax.spines["bottom"].set_color("#e5e7eb")
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 60,
                    f"{cnt:,}", ha="center", va="bottom", fontsize=8, color="#6b7280")
        plt.tight_layout()
        _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight"); _buf.seek(0)
        st.image(_buf, use_column_width="always")
        plt.close("all"); gc.collect()

    st.divider()

    # Feature 2 — Pixel intensity KDE
    st.markdown("### Feature 2 — Pixel Intensity Distribution per Class")
    st.markdown(
        "The brightness distribution shows how distinct each category is visually. "
        "Trousers and Bags skew dark; Shirts and Coats show broader, lighter distributions. "
        "High overlap between Pullover and Coat explains the model's most common confusion pair."
    )
    st.image("figures/eda_03_pixel_intensity.png", use_column_width="always")

    st.divider()

    # Feature 3 — Sample images per class
    st.markdown("### Feature 3 — Sample Images per Class")
    st.markdown(
        "Visual inspection explains per-class difficulty. Pullovers and Shirts share collar shapes; "
        "Sneakers and Ankle Boots share shoe contours. Bags and Trousers are easiest to classify due to "
        "visually distinct silhouettes."
    )
    st.image("figures/eda_02_sample_images.png", use_column_width="always")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MODEL METRICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown("""
    <div class="page-header">
        <h1>Model Evaluation Metrics</h1>
        <p>Full evaluation results for the MobileNetV2 classifier — accuracy, loss, F1, precision, recall,
           per-class breakdown, and 30-epoch training history.</p>
    </div>
    """, unsafe_allow_html=True)

    _metrics_ph = st.empty()
    _metrics_ph.info("⏳ Loading model metrics and building charts...")
    try:
        metrics = _fetch_metrics()
        err = None
    except Exception as e:
        metrics = None
        err = str(e)
    _metrics_ph.empty()
    if err:
        st.warning(err)
    elif metrics:
        with st.spinner("Rendering charts..."):
            initial = metrics.get("evaluation", {}) or metrics.get("initial_training", {})
            retrain = metrics.get("retrain", {})
            cfg     = metrics.get("training_config", {})

            # ── Model architecture info ────────────────────────────────────────
            if cfg:
                st.markdown("""
                <div style="background:#f8faff;border:1px solid #e0e7ff;border-radius:10px;
                            padding:14px 20px;margin-bottom:16px">
                <div style="font-size:0.72rem;font-weight:700;color:#9ca3af;text-transform:uppercase;
                            letter-spacing:.06em;margin-bottom:8px">Model Architecture & Training Config</div>
                """, unsafe_allow_html=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.markdown(f"**Model**<br><span style='color:#5b6af5'>{cfg.get('model','MobileNetV2')}</span>", unsafe_allow_html=True)
                c2.markdown(f"**Input Shape**<br><span style='color:#5b6af5'>{' × '.join(str(x) for x in cfg.get('input_shape',[128,128,3]))}</span>", unsafe_allow_html=True)
                c3.markdown(f"**Epochs Ran**<br><span style='color:#5b6af5'>{cfg.get('phase1_epochs_ran',0) + cfg.get('phase2_epochs_ran',0)}</span>", unsafe_allow_html=True)
                c4.markdown(f"**Batch Size**<br><span style='color:#5b6af5'>{cfg.get('batch_size',32)}</span>", unsafe_allow_html=True)

                opts = []
                if cfg.get("optimizer_phase1"): opts.append(f"Adam (phase1: {cfg['optimizer_phase1']})")
                if cfg.get("optimizer_phase2"): opts.append(f"Adam fine-tune (phase2: {cfg['optimizer_phase2']})")
                cbs = cfg.get("callbacks", [])
                if any("Early" in c for c in cbs): opts.append("Early Stopping")
                if any("ReduceLR" in c for c in cbs): opts.append("ReduceLROnPlateau")
                if any("Checkpoint" in c for c in cbs): opts.append("ModelCheckpoint (best only)")
                aug = cfg.get("augmentation", [])
                if aug: opts.append(f"Data Augmentation ({len(aug)} transforms)")
                st.markdown("<br>**Optimization Techniques:** " + " &nbsp;·&nbsp; ".join(f"<code>{o}</code>" for o in opts), unsafe_allow_html=True)
                st.markdown("**Regularization:** L2 weight decay on Dense(256) head &nbsp;·&nbsp; Dropout(0.5) &nbsp;·&nbsp; BatchNormalization", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            # ── Initial training metrics ───────────────────────────────────────
            if initial:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin:16px 0 10px">
                    <div style="width:4px;height:20px;background:#5b6af5;border-radius:2px"></div>
                    <span style="font-size:1rem;font-weight:700;color:#1f2937">Initial Training — Test Set Results (60,000 train / 10,000 test)</span>
                </div>
                """, unsafe_allow_html=True)
                m1, m2, m3, m4, m5 = st.columns(5)
                for col, lbl, val in [
                    (m1, "Test Accuracy",  f"{initial.get('accuracy', 0)*100:.2f}%"),
                    (m2, "Test Loss",      f"{initial.get('test_loss', 0):.4f}"),
                    (m3, "Macro F1",       f"{initial.get('f1_score', 0):.4f}"),
                    (m4, "Precision",      f"{initial.get('precision', 0):.4f}"),
                    (m5, "Recall",         f"{initial.get('recall', 0):.4f}"),
                ]:
                    with col:
                        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

            # ── Retrain metrics ────────────────────────────────────────────────
            if retrain:
                rt_acc  = retrain.get("accuracy", 0) * 100
                base_acc = initial.get("accuracy", 0) * 100
                delta   = rt_acc - base_acc
                dcolor  = "#10b981" if delta >= 0 else "#ef4444"
                dsign   = "+" if delta >= 0 else ""
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin:20px 0 10px">
                    <div style="width:4px;height:20px;background:#10b981;border-radius:2px"></div>
                    <span style="font-size:1rem;font-weight:700;color:#1f2937">After Retraining
                    — {retrain.get('samples','?')} new samples · {retrain.get('epochs_ran','?')} epochs</span>
                    <span style="font-size:0.82rem;color:{dcolor};font-weight:600">{dsign}{delta:.2f}% vs baseline</span>
                </div>
                """, unsafe_allow_html=True)
                m1, m2, m3, m4 = st.columns(4)
                for col, lbl, val in [
                    (m1, "Accuracy",  f"{rt_acc:.2f}%"),
                    (m2, "Macro F1",  f"{retrain.get('f1_score', 0):.4f}"),
                    (m3, "Precision", f"{retrain.get('precision', 0):.4f}"),
                    (m4, "Recall",    f"{retrain.get('recall', 0):.4f}"),
                ]:
                    with col:
                        st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)

            st.divider()

            # ── Per-class breakdown table + F1 chart ──────────────────────────
            pf1  = initial.get("per_class_f1", {})
            ppr  = initial.get("per_class_precision", {})
            prec = initial.get("per_class_recall", {})

            if pf1:
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                    <div style="width:4px;height:20px;background:#f97316;border-radius:2px"></div>
                    <span style="font-size:1rem;font-weight:700;color:#1f2937">Per-Class Evaluation — All 10 Clothing Categories</span>
                </div>
                """, unsafe_allow_html=True)

                per_class_rows = []
                for cls in pf1:
                    per_class_rows.append({
                        "Class": cls,
                        "F1 Score":  round(pf1.get(cls, 0), 4),
                        "Precision": round(ppr.get(cls, 0), 4) if ppr else "—",
                        "Recall":    round(prec.get(cls, 0), 4) if prec else "—",
                    })
                st.dataframe(pd.DataFrame(per_class_rows), use_container_width=True, hide_index=True)

                labels = list(pf1.keys())
                values = list(pf1.values())
                colors = plt.cm.RdYlGn(np.array(values))
                fig, ax = plt.subplots(figsize=(11, 4))
                fig.patch.set_facecolor("white")
                ax.set_facecolor("#fafafa")
                bars = ax.bar(labels, values, color=colors, edgecolor="none", width=0.6)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=10)
                ax.set_ylabel("F1 Score", fontsize=10)
                ax.set_title("Per-Class F1 Score — Test Set (10,000 samples)", fontsize=13, fontweight="600", pad=12)
                ax.set_ylim(0, 1.1)
                ax.axhline(np.mean(values), color="#6366f1", linestyle="--",
                           linewidth=1.2, label=f"Mean F1: {np.mean(values):.4f}")
                ax.legend(fontsize=9)
                ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#e5e7eb"); ax.spines["bottom"].set_color("#e5e7eb")
                for bar, v in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012,
                            f"{v:.3f}", ha="center", va="bottom", fontsize=9, color="#374151")
                plt.tight_layout()
                _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight"); _buf.seek(0)
                st.image(_buf, use_column_width="always")
                plt.close("all"); gc.collect()

            # ── Training history ───────────────────────────────────────────────
            if metrics.get("history"):
                h = metrics["history"]
                st.divider()
                st.markdown("""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
                    <div style="width:4px;height:20px;background:#8b5cf6;border-radius:2px"></div>
                    <span style="font-size:1rem;font-weight:700;color:#1f2937">30-Epoch Training History — Accuracy & Loss</span>
                </div>
                """, unsafe_allow_html=True)

                fig, axes = plt.subplots(1, 2, figsize=(13, 4))
                fig.patch.set_facecolor("white")
                for ax in axes:
                    ax.set_facecolor("#fafafa")
                    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
                    ax.spines["left"].set_color("#e5e7eb"); ax.spines["bottom"].set_color("#e5e7eb")
                    ax.grid(alpha=0.3, linestyle="--")

                if "accuracy" in h:
                    axes[0].plot(h["accuracy"], label="Train", color="#4f46e5", linewidth=2)
                if "val_accuracy" in h:
                    axes[0].plot(h["val_accuracy"], label="Validation", color="#10b981", linestyle="--", linewidth=2)
                axes[0].set_title("Accuracy over Epochs", fontsize=12, fontweight="600")
                axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Accuracy"); axes[0].legend()

                if "loss" in h:
                    axes[1].plot(h["loss"], label="Train", color="#4f46e5", linewidth=2)
                if "val_loss" in h:
                    axes[1].plot(h["val_loss"], label="Validation", color="#10b981", linestyle="--", linewidth=2)
                axes[1].set_title("Loss over Epochs", fontsize=12, fontweight="600")
                axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss"); axes[1].legend()

                plt.tight_layout()
                _buf = io.BytesIO(); fig.savefig(_buf, format="png", dpi=110, bbox_inches="tight"); _buf.seek(0)
                st.image(_buf, use_column_width="always")
                plt.close("all"); gc.collect()
    else:
        st.info("No metrics found. Run the training notebook first.")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — API STATUS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown("""
    <div class="page-header">
        <h1>API & System Status</h1>
        <p>Live service health, retraining status, and full endpoint reference.</p>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Refresh", type="primary"):
        st.rerun()

    st.markdown("### Service Health")
    _health_ph = st.empty()
    _health_ph.info("⏳ Fetching service health...")
    health, err = api_get("/health")
    _health_ph.empty()
    if health:
        h1, h2, h3, h4 = st.columns(4)
        for col, lbl, val in [
            (h1, "Status",      "Online"),
            (h2, "Model Ready", "Yes" if health["model_ready"] else "No"),
            (h3, "Uptime",      f"{health['uptime_sec']:.0f}s"),
            (h4, "DB Samples",  str(health["db_samples"])),
        ]:
            with col:
                st.markdown(f'<div class="kpi-card"><div class="kpi-value">{val}</div><div class="kpi-label">{lbl}</div></div>', unsafe_allow_html=True)
    else:
        st.error(f"API unreachable: {err}")

    st.divider()

    st.markdown("### Retraining Status")
    _rt_ph = st.empty()
    _rt_ph.info("⏳ Fetching retraining status...")
    rtrain, _ = api_get("/retrain/status")
    _rt_ph.empty()
    if rtrain:
        if rtrain.get("running"):
            st.warning(f"Retraining in progress — phase: {rtrain.get('phase','')} · elapsed: {rtrain.get('elapsed_sec',0):.0f}s")
        else:
            st.success("No retraining in progress")
        lr = rtrain.get("last_result")
        if lr and "error" not in lr:
            r1, r2, r3, r4, r5 = st.columns(5)
            for col, lbl, val in [
                (r1, "Accuracy",   f"{lr.get('accuracy',0)*100:.2f}%"),
                (r2, "F1 Score",   f"{lr.get('f1_score',0):.4f}"),
                (r3, "Precision",  f"{lr.get('precision',0):.4f}"),
                (r4, "Recall",     f"{lr.get('recall',0):.4f}"),
                (r5, "Epochs Ran", str(lr.get("epochs_ran","—"))),
            ]:
                col.metric(lbl, val)
        elif lr and "error" in lr:
            st.error(f"Last retrain failed: {lr['error']}")

    st.divider()

    st.markdown("### API Endpoints")
    endpoints = [
        ("GET",  "/",                "Service info & uptime"),
        ("GET",  "/health",          "Model readiness & DB stats"),
        ("POST", "/predict",         "Predict from 784 pixel values"),
        ("POST", "/predict/image",   "Predict from PNG / JPG / WebP / AVIF"),
        ("POST", "/upload-data",     "Upload CSV of labelled samples"),
        ("POST", "/retrain",         "Trigger model fine-tuning"),
        ("GET",  "/retrain/status",  "Poll retraining progress"),
        ("GET",  "/retrain/history", "Retraining run logs"),
        ("GET",  "/metrics",         "Model evaluation metrics"),
        ("GET",  "/insights",        "Dataset statistics"),
        ("GET",  "/docs",            "Swagger / OpenAPI docs"),
    ]
    st.dataframe(
        pd.DataFrame(endpoints, columns=["Method", "Path", "Description"]),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(f"Backend URL: `{API_URL}`")
