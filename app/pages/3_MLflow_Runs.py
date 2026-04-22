"""
Page 3 — MLflow Run Browser
Browse all experiments from both DL and ML pipelines.
"""

import sys
import os
import streamlit as st

st.set_page_config(page_title="MLflow Runs · BLAST", page_icon="📈", layout="wide")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.shared import inject_css, MLFLOW_URI

inject_css("""
    .stButton > button {
        background: linear-gradient(135deg, #6e40c9, #8957e5) !important;
    }
    div[data-testid="stMetricValue"] { color: #e3b341 !important; }
""")

st.markdown(
    "<h1 style='margin-bottom:4px'>📈 MLflow Experiment Browser</h1>"
    "<p style='color:#8b949e'>All training runs from the DL and ML pipelines — tracked via SQLite</p>",
    unsafe_allow_html=True,
)

import mlflow

try:
    mlflow.set_tracking_uri(MLFLOW_URI)
    experiments = mlflow.search_experiments()
except Exception as e:
    st.error(f"Could not connect to MLflow database: {e}")
    st.stop()

if not experiments:
    st.info("No experiments found yet. Run a training job first.")
    st.stop()

# ── Experiment selector ────────────────────────────────────────────────────────
exp_names = [e.name for e in experiments if e.name != "Default"]
if not exp_names:
    exp_names = [e.name for e in experiments]

selected_exp = st.selectbox("Experiment", exp_names, key="mlf_exp_sel")
exp_obj = next((e for e in experiments if e.name == selected_exp), None)

if exp_obj is None:
    st.warning("Experiment not found.")
    st.stop()

# ── Load runs ──────────────────────────────────────────────────────────────────
import pandas as pd

try:
    runs_df = mlflow.search_runs(
        experiment_ids=[exp_obj.experiment_id],
        order_by=["start_time DESC"],
    )
except Exception as e:
    st.error(f"Could not load runs: {e}")
    st.stop()

if runs_df.empty:
    st.info(f"No runs found for experiment **{selected_exp}**.")
    st.stop()

# ── Summary metrics ────────────────────────────────────────────────────────────
total_runs    = len(runs_df)
finished_runs = (runs_df["status"] == "FINISHED").sum()

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Total Runs",   total_runs)
mc2.metric("Finished",     finished_runs)
mc3.metric("Failed/Other", total_runs - finished_runs)

st.markdown("---")
st.markdown("### 📋 Run History")

# ── Filter & search ────────────────────────────────────────────────────────────
search_term = st.text_input(
    "🔍 Filter by run name", placeholder="e.g. efficientnet", key="mlf_search"
)

metric_cols  = [c for c in runs_df.columns if c.startswith("metrics.")]
param_cols   = [c for c in runs_df.columns if c.startswith("params.")]

display_cols = [
    "tags.mlflow.runName",
    "tags.dataset_file",
    "status",
    "start_time",
    # ML pipeline model path (logged as param)
    "params.save_path",
    # DL pipeline paths (logged as tags)
    "tags.data_folder",
    "tags.save_path",
] + metric_cols[:8]
display_cols = [c for c in display_cols if c in runs_df.columns]

filtered = runs_df.copy()
if search_term:
    name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in filtered.columns else "run_id"
    filtered = filtered[
        filtered[name_col].str.contains(search_term, case=False, na=False)
    ]

rename_map = {
    "tags.mlflow.runName": "Run Name",
    "tags.dataset_file":   "Dataset",
    "status":              "Status",
    "start_time":          "Started",
    "params.save_path":    "Saved Model (.pkl)",
    "tags.data_folder":    "Training Data Path",
    "tags.save_path":      "Saved Model (.pt)",
}
rename_map.update({c: c.replace("metrics.", "") for c in metric_cols})
disp_df = filtered[display_cols].rename(columns=rename_map)


def style_status(val):
    if val == "FINISHED":
        return "color: #3fb950"
    elif val == "FAILED":
        return "color: #f78166"
    return "color: #e3b341"


if "Status" in disp_df.columns:
    styled_df = disp_df.style.applymap(style_status, subset=["Status"])
    st.dataframe(styled_df, use_container_width=True, height=300)
else:
    st.dataframe(disp_df, use_container_width=True, height=300)

st.caption(f"Showing {len(filtered)} / {total_runs} runs")

# ── Run detail expander ────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🔎 Inspect a Run")

run_name_col = "tags.mlflow.runName" if "tags.mlflow.runName" in runs_df.columns else "run_id"
run_options  = filtered[run_name_col].dropna().tolist() or filtered["run_id"].tolist()

if run_options:
    selected_run_name = st.selectbox("Select run", run_options, key="mlf_run_detail")
    sel_row = filtered[filtered[run_name_col] == selected_run_name].iloc[0]

    det1, det2 = st.columns(2)

    with det1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**📊 Metrics**")
        metrics = {
            c.replace("metrics.", ""): sel_row[c]
            for c in metric_cols if not pd.isna(sel_row[c])
        }
        if metrics:
            mc_cols = st.columns(min(len(metrics), 3))
            for i, (k, v) in enumerate(metrics.items()):
                mc_cols[i % 3].metric(k, f"{v:.4f}" if isinstance(v, float) else v)
        else:
            st.caption("No metrics logged.")
        st.markdown("</div>", unsafe_allow_html=True)

    with det2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**⚙️ Parameters**")
        params = {
            c.replace("params.", ""): sel_row[c]
            for c in param_cols if not pd.isna(sel_row.get(c, float("nan")))
        }
        if params:
            params_df = pd.DataFrame(list(params.items()), columns=["Parameter", "Value"])
            st.dataframe(params_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No parameters logged.")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("### 🗑️ Manage Run")
    if st.button("Delete this run", type="primary", key="del_mlf_run"):
        mlflow.delete_run(sel_row["run_id"])
        st.success(f"Run **{selected_run_name}** deleted! Refreshing...")
        st.rerun()

    # Metric comparison chart
    st.markdown("### 📉 Metric Comparison (top 10 runs)")
    import matplotlib.pyplot as plt

    key_metrics   = ["final_accuracy", "final_f1", "final_precision", "final_recall",
                     "val_accuracy", "val_f1", "Accuracy", "AUC"]
    avail_metrics = [m for m in key_metrics if f"metrics.{m}" in runs_df.columns]

    if avail_metrics and len(filtered) > 1:
        metric_choice = st.selectbox("Metric to compare", avail_metrics, key="mlf_metric_cmp")
        col_name = f"metrics.{metric_choice}"
        top10  = filtered.dropna(subset=[col_name]).head(10)
        labels = top10[run_name_col].tolist()

        fig, ax = plt.subplots(figsize=(10, 4))
        fig.patch.set_facecolor("#161b22")
        ax.set_facecolor("#0d1117")
        bars = ax.barh(labels, top10[col_name].tolist(), color="#e3b341", edgecolor="#161b22")
        ax.set_xlabel(metric_choice, color="#8b949e")
        ax.set_title(f"{metric_choice} across runs", color="#e6edf3")
        ax.tick_params(colors="#8b949e")
        for sp in ax.spines.values():
            sp.set_edgecolor("#30363d")
        ax.set_xlim(0, max(top10[col_name].max() * 1.1, 1.0))
        ax.invert_yaxis()
        for bar, val in zip(bars, top10[col_name].tolist()):
            ax.text(val + 0.002, bar.get_y() + bar.get_height() / 2,
                    f"{val:.4f}", va="center", color="#e6edf3", fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    else:
        st.caption("Not enough runs or metrics to plot comparison.")

# ── MLflow UI link ─────────────────────────────────────────────────────────────
st.markdown("---")
st.info(
    "💡 For the full MLflow UI, run this command in your terminal:\n"
    f"```\nmlflow ui --backend-store-uri {MLFLOW_URI}\n```\n"
    "Then open http://localhost:5000",
    icon="🔗",
)
