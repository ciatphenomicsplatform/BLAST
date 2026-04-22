"""
Page 2 — Machine Learning Pipeline
Train with PyCaret AutoML OR run inference on new tabular data.
"""

import sys
import os
import queue
import threading
import time
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.shared import inject_css, BLAST_ROOT
from components.ml_runner import (
    get_dataset_files,
    run_ml_training,
    run_ml_inference,
    ML_MODELS_AVAILABLE,
    FEATURES_ALL,
    TARGET_COLUMN,
    ML_DATA_DIR,
)

st.set_page_config(page_title="ML Pipeline · BLAST", page_icon="📊", layout="wide")

inject_css("""
    .stButton > button {
        background: linear-gradient(135deg, #145a2e, #2ea043) !important;
    }
    div[data-testid="stMetricValue"] { color: #3fb950 !important; }
""")

st.markdown(
    "<h1 style='margin-bottom:4px'>📊 Machine Learning Pipeline</h1>"
    "<p style='color:#8b949e'>AutoML with PyCaret on spectral vegetation index datasets</p>",
    unsafe_allow_html=True,
)

mode = st.radio(
    "Pipeline mode",
    ["🏋️  Train & compare models", "🔍  Run inference"],
    horizontal=True,
    key="ml_mode",
)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN MODE
# ══════════════════════════════════════════════════════════════════════════════
if "Train" in mode:
    st.markdown("### ⚙️ Training Configuration")

    dataset_files = get_dataset_files()
    if not dataset_files:
        st.error(
            f"❌ No `.xlsx` files found in `{ML_DATA_DIR}`.\n\n"
            "Please place your dataset Excel files in that directory.",
        )
        st.stop()

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Dataset Selection**")
        dataset_file = st.selectbox("Dataset file", dataset_files, key="ml_dataset_sel")
        use_oversampling = st.toggle(
            "Apply oversampling (balance classes)", value=True, key="ml_oversample"
        )

        st.markdown("**Run / Model Name**")
        custom_run_name = st.text_input(
            "Custom name (leave blank for auto)",
            placeholder="e.g. RF_2025_oversampled",
            key="ml_run_name",
        )
        st.caption("Used as the MLflow run label and the saved `.pkl` filename.")

        st.markdown("**Feature Columns**")
        selected_features = st.multiselect(
            "Spectral features to include",
            FEATURES_ALL,
            default=FEATURES_ALL,
            key="ml_features",
        )
        st.caption(f"Target column: **{TARGET_COLUMN}** (fixed)")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Models to Compare**")

        model_labels = {
            "et":       "Extra Trees",
            "rf":       "Random Forest",
            "xgboost":  "XGBoost",
            "lightgbm": "LightGBM",
            "dt":       "Decision Tree",
        }
        selected_models = st.multiselect(
            "Classifiers",
            ML_MODELS_AVAILABLE,
            default=ML_MODELS_AVAILABLE,
            format_func=lambda m: f"{model_labels[m]} ({m})",
            key="ml_models_sel",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if dataset_file:
            file_path = os.path.join(ML_DATA_DIR, dataset_file)
            try:
                df_preview = pd.read_excel(file_path, nrows=3)
                st.markdown('<div class="panel">', unsafe_allow_html=True)
                st.markdown("**Dataset Preview** (first 3 rows)")
                st.dataframe(df_preview, use_container_width=True)
                full_df = pd.read_excel(file_path)
                st.caption(f"Full shape: {full_df.shape[0]:,} rows × {full_df.shape[1]} cols")
                st.markdown("</div>", unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Could not preview: {e}")

    if not selected_models:
        st.warning("Please select at least one model.")
    if not selected_features:
        st.warning("Please select at least one feature.")

    st.markdown("---")
    start_btn = st.button(
        "🚀 Start Training",
        disabled=(
            not selected_models
            or not selected_features
            or st.session_state.get("ml_train_running", False)
        ),
        key="ml_start",
    )

    # Only launch a thread when the button is pressed and one isn't already running
    if start_btn and not st.session_state.get("ml_train_running", False):
        log_q       = queue.Queue()
        result_hold = {}

        thread = threading.Thread(
            target=run_ml_training,
            kwargs=dict(
                dataset_filename=dataset_file,
                selected_models=selected_models,
                use_oversampling=use_oversampling,
                selected_features=selected_features,
                log_queue=log_q,
                result_holder=result_hold,
                custom_run_name=custom_run_name.strip() or None,
            ),
            daemon=True,
        )
        thread.start()

        st.session_state["ml_train_running"] = True
        st.session_state["ml_result"]        = None
        st.session_state["ml_error"]         = None
        st.session_state["_ml_thread"]       = thread
        st.session_state["_ml_log_q"]        = log_q
        st.session_state["_ml_result_hold"]  = result_hold

    if st.session_state.get("ml_train_running"):
        thread      = st.session_state["_ml_thread"]
        log_q       = st.session_state["_ml_log_q"]
        result_hold = st.session_state["_ml_result_hold"]

        st.markdown("### 📡 Training Log")
        log_placeholder = st.empty()
        progress_bar    = st.progress(0, text="PyCaret training in progress…")

        log_lines = []
        done      = False

        while thread.is_alive() or not log_q.empty():
            while not log_q.empty():
                msg = log_q.get()
                if msg == "__DONE__":
                    done = True
                    break
                log_lines.append(msg)

            log_placeholder.markdown(
                "<div class='log-box'>" +
                "<br>".join(log_lines[-60:]) +
                "</div>",
                unsafe_allow_html=True,
            )
            if done:
                break
            time.sleep(0.3)

        thread.join()

        # Drain any remaining messages after thread finishes
        while not log_q.empty():
            msg = log_q.get()
            if msg != "__DONE__":
                log_lines.append(msg)

        log_placeholder.markdown(
            "<div class='log-box'>" + "<br>".join(log_lines[-60:]) + "</div>",
            unsafe_allow_html=True,
        )
        progress_bar.empty()
        st.session_state["ml_train_running"] = False

        if "result" in result_hold:
            r = result_hold["result"]
            st.session_state["ml_result"] = r
            st.success("✅ Training complete!")
            st.info(f"💾 Model saved to: `{r['save_path']}.pkl`")

            st.markdown("### 🏆 Model Comparison Results")
            results_df = r["comparison_df"]

            styled = (
                results_df.head(10)
                .style
                .background_gradient(subset=["Accuracy", "F1"], cmap="Greens")
                .format(precision=4)
            )
            st.dataframe(styled, use_container_width=True)

            import matplotlib.pyplot as plt
            top5      = results_df.head(5)
            model_col = results_df.columns[0]

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor("#161b22")
            for ax in axes:
                ax.set_facecolor("#0d1117")
                for sp in ax.spines.values():
                    sp.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e")

            axes[0].barh(top5[model_col], top5["Accuracy"], color="#3fb950")
            axes[0].set_title("Accuracy", color="#e6edf3")
            axes[0].set_xlim(0, 1)
            axes[0].invert_yaxis()

            axes[1].barh(top5[model_col], top5["F1"], color="#58a6ff")
            axes[1].set_title("F1 Score", color="#e6edf3")
            axes[1].set_xlim(0, 1)
            axes[1].invert_yaxis()

            st.pyplot(fig)
            plt.close(fig)

            if r.get("complexity"):
                st.info(f"🔎 Best model complexity: **{r['complexity']:,}** nodes/parameters")

        elif "error" in result_hold:
            st.error("❌ Training failed.")
            with st.expander("Error traceback"):
                st.code(result_hold["error"])

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    import glob as _glob

    st.markdown("### 🔍 Inference Configuration")

    saved_pkls    = _glob.glob(os.path.join(BLAST_ROOT, "*.pkl"))
    model_options = saved_pkls

    if not model_options:
        st.warning("⚠️ No saved `.pkl` model found. Train a model first.", icon="⚠️")

    c1, c2 = st.columns([1, 2])
    with c1:
        if model_options:
            model_path = st.selectbox(
                "Saved model (.pkl)",
                model_options,
                format_func=os.path.basename,
                key="ml_inf_model",
            )
        else:
            model_path = None

        selected_features_inf = st.multiselect(
            "Feature columns in your file",
            FEATURES_ALL,
            default=FEATURES_ALL,
            key="ml_inf_features",
        )

    with c2:
        uploaded_xlsx = st.file_uploader(
            "Upload new dataset (.xlsx)",
            type=["xlsx"],
            key="ml_inf_upload",
        )

        if uploaded_xlsx:
            import tempfile
            tmp = tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False)
            tmp.write(uploaded_xlsx.read())
            tmp.close()
            df_inf = pd.read_excel(tmp.name)
            os.unlink(tmp.name)
            st.markdown("**Preview (first 5 rows)**")
            st.dataframe(df_inf.head(), use_container_width=True)
            st.caption(f"Shape: {df_inf.shape[0]:,} rows × {df_inf.shape[1]} cols")

    inf_btn = st.button(
        "🔮 Run Inference",
        disabled=(not model_options or not uploaded_xlsx),
        key="ml_inf_run",
    )

    if inf_btn and model_path and uploaded_xlsx:
        with st.spinner("Running inference…"):
            preds_df = run_ml_inference(df_inf, model_path, selected_features_inf)

        st.success(f"✅ Predictions generated for {len(preds_df):,} rows!")
        st.markdown("### 📋 Prediction Results")
        st.dataframe(preds_df, use_container_width=True)

        pred_col = [c for c in preds_df.columns if "predict" in c.lower() or "Label" in c]
        if pred_col:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 3))
            fig.patch.set_facecolor("#161b22")
            ax.set_facecolor("#0d1117")
            preds_df[pred_col[0]].value_counts().plot(
                kind="bar", ax=ax, color="#3fb950", edgecolor="#161b22"
            )
            ax.set_title("Prediction Distribution", color="#e6edf3")
            ax.tick_params(colors="#8b949e")
            for sp in ax.spines.values():
                sp.set_edgecolor("#30363d")
            st.pyplot(fig)
            plt.close(fig)

        csv_bytes = preds_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="blast_ml_predictions.csv",
            mime="text/csv",
        )
