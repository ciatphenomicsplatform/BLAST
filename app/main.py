"""
BLAST MLOps Platform — Home / Entry Point
Run with:  streamlit run app/main.py
"""

import os
import glob
import streamlit as st

st.set_page_config(
    page_title="BLAST MLOps Platform",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from components.shared import inject_css, BLAST_ROOT

inject_css()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div class="hero-banner">
        <p class="hero-title">🌾 BLAST MLOps Platform</p>
        <p class="hero-sub">
            Unified interface for rice blast severity classification<br>
            Deep Learning · Machine Learning · MLflow Experiment Tracking
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Pipeline cards ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        """
        <div class="blast-card">
            <h3 style="color:#58a6ff;margin-top:0">🧠 Deep Learning</h3>
            <p style="color:#8b949e;font-size:.9rem">
                Transfer learning on drone imagery.<br>
                25+ architectures: EfficientNet, ResNet,
                ViT, Swin, ConvNeXt and more.
            </p>
            <span class="badge-blue">PyTorch</span>
            <span class="badge-blue">GradCAM</span>
            <span class="badge-green">MLflow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        """
        <div class="blast-card">
            <h3 style="color:#3fb950;margin-top:0">📊 Machine Learning</h3>
            <p style="color:#8b949e;font-size:.9rem">
                Tabular spectral indices classification.<br>
                AutoML with PyCaret: RF, XGBoost,
                LightGBM, ExtraTrees, DecisionTree.
            </p>
            <span class="badge-green">PyCaret</span>
            <span class="badge-green">SHAP</span>
            <span class="badge-green">MLflow</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    st.markdown(
        """
        <div class="blast-card">
            <h3 style="color:#e3b341;margin-top:0">📈 Experiment Tracker</h3>
            <p style="color:#8b949e;font-size:.9rem">
                Browse every training run across both
                pipelines. Compare metrics, download
                artifacts, and track model evolution.
            </p>
            <span class="badge-orange">MLflow UI</span>
            <span class="badge-orange">SQLite</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ── Quick stats ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📁 Dataset Overview")

ML_DATA_DIR   = os.path.join(BLAST_ROOT, "ML_Part", "datasets", "input")
DL_DATA_DIR   = os.path.join(BLAST_ROOT, "DL_Part", "smaple_data", "input")
DL_MODELS_DIR = os.path.join(BLAST_ROOT, "DL_Part", "saved_models")

ml_files  = glob.glob(os.path.join(ML_DATA_DIR,   "*.xlsx")) if os.path.isdir(ML_DATA_DIR)   else []
dl_images = glob.glob(os.path.join(DL_DATA_DIR,   "*.png"))  if os.path.isdir(DL_DATA_DIR)   else []
dl_saved  = (
    glob.glob(os.path.join(DL_MODELS_DIR, "*.pt")) +
    glob.glob(os.path.join(DL_MODELS_DIR, "*.pth"))
) if os.path.isdir(DL_MODELS_DIR) else []
ml_saved  = glob.glob(os.path.join(BLAST_ROOT, "*.pkl"))

c1, c2, c3, c4 = st.columns(4)
c1.metric("📋 ML Datasets",     len(ml_files))
c2.metric("🖼️ DL Images",      len(dl_images))
c3.metric("💾 Saved DL Models", len(dl_saved))
c4.metric("💾 Saved ML Models", len(ml_saved))

# ── Navigation hint ────────────────────────────────────────────────────────────
st.markdown("---")
st.info(
    "👈 Use the **sidebar** to navigate to the Deep Learning pipeline, "
    "Machine Learning pipeline, or the MLflow Run Browser.",
    icon="🗺️",
)

st.markdown(
    "<p style='text-align:center;color:#484f58;font-size:.8rem;margin-top:32px'>"
    "BLAST MLOps Platform · CIAT · Rice Blast Severity Classification</p>",
    unsafe_allow_html=True,
)
