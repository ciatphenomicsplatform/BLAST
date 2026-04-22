"""
Page 1 — Deep Learning Pipeline
Train a transfer-learning model OR run inference with GradCAM.
"""

import sys
import os
import queue
import threading
import time
import glob
import streamlit as st
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.shared import inject_css
from components.dl_runner import (
    DL_MODELS, FINE_TUNE_MODES, BIN_LABELS,
    run_dl_training, run_dl_inference, get_saved_models,
)

st.set_page_config(page_title="DL Pipeline · BLAST", page_icon="🧠", layout="wide")

inject_css()  # blue buttons — matches DL theme

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='margin-bottom:4px'>🧠 Deep Learning Pipeline</h1>"
    "<p style='color:#8b949e'>Transfer learning on drone imagery for blast severity classification</p>",
    unsafe_allow_html=True,
)

# ── Mode selector ──────────────────────────────────────────────────────────────
mode = st.radio(
    "Pipeline mode",
    ["🏋️  Train a new model", "🔍  Run inference"],
    horizontal=True,
    key="dl_mode",
)

# ══════════════════════════════════════════════════════════════════════════════
#  TRAIN MODE
# ══════════════════════════════════════════════════════════════════════════════
if "Train" in mode:
    st.markdown("### ⚙️ Training Configuration")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Model & Strategy**")
        model_name = st.selectbox(
            "Architecture", DL_MODELS,
            index=DL_MODELS.index("efficientnet_v2_s"),
            key="dl_train_arch",
        )
        fine_tune = st.selectbox(
            "Fine-tune mode",
            FINE_TUNE_MODES,
            format_func=lambda m: {
                "full":    "Full (train all layers)",
                "freeze":  "Freeze (head only)",
                "partial": "Partial (top 50% layers)",
            }[m],
            key="dl_ft_mode",
        )
        num_classes = st.selectbox(
            "Number of output classes",
            options=[2, 3],
            format_func=lambda n: {
                2: "2 — Low (Index 1–5) vs High (Index 6–9)",
                3: "3 — Low / Medium / High",
            }[n],
            index=1,   # default 3
            key="dl_nclasses",
        )
        st.markdown("**Run / Model Name**")
        custom_run_name = st.text_input(
            "Custom name (leave blank for auto)",
            placeholder=f"e.g. EfficientNet_2025_full",
            key="dl_run_name",
        )
        st.caption("Used as the MLflow run label and the `.pt` save filename.")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**Hyperparameters**")
        max_epochs = st.slider("Max epochs", 1, 100, 10, key="dl_epochs")
        batch_size = st.select_slider(
            "Batch size", [8, 16, 32, 64, 128], value=32, key="dl_bs"
        )
        lr = st.select_slider(
            "Learning rate",
            [0.0001, 0.0003, 0.001, 0.003, 0.01],
            value=0.001,
            format_func=lambda v: f"{v:.4f}",
            key="dl_lr",
        )
        patience = st.slider("Early stopping patience", 1, 20, 5, key="dl_patience")
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Dataset path pickers ────────────────────────────────────────────────────
    st.markdown("### 📂 Dataset Paths")

    # macOS requires AppKit (NSWindow) on the main thread.
    # Streamlit callbacks run on a worker thread, so tk.Tk() in an on_click
    # crashes with NSInternalInconsistencyException.
    # Fix: spawn a *subprocess* — it gets its own main thread for tkinter.
    _FOLDER_SCRIPT = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', True)
path = filedialog.askdirectory(title='Select image folder')
root.destroy()
print(path, end='')
"""

    _CSV_SCRIPT = """
import tkinter as tk
from tkinter import filedialog
root = tk.Tk(); root.withdraw(); root.wm_attributes('-topmost', True)
path = filedialog.askopenfilename(
    title='Select CSV label file',
    filetypes=[('CSV files', '*.csv'), ('All files', '*.*')]
)
root.destroy()
print(path, end='')
"""

    def _run_picker(script: str, state_key: str):
        import subprocess, sys as _sys
        try:
            result = subprocess.run(
                [_sys.executable, "-c", script],
                capture_output=True, text=True, timeout=60,
            )
            path = result.stdout.strip()
            if path:
                st.session_state[state_key] = path
        except Exception as e:
            st.session_state["_dl_path_err"] = str(e)

    def _pick_folder():
        _run_picker(_FOLDER_SCRIPT, "dl_img_folders")

    def _pick_csv():
        _run_picker(_CSV_SCRIPT, "dl_csv_files")

    if err := st.session_state.get("_dl_path_err"):
        st.warning(f"⚠️ Path picker error: {err}  \nType the path manually below.")
        st.session_state.pop("_dl_path_err", None)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**🖼️ Image Folder**")
        ba, bb = st.columns([3, 1])
        img_folders_raw = ba.text_input(
            "Image folder path",
            value=st.session_state.get("dl_img_folders", ""),
            placeholder="/path/to/images",
            key="dl_img_folders",
            label_visibility="collapsed",
        )
        bb.button("📁 Browse", on_click=_pick_folder, key="dl_browse_folder",
                  use_container_width=True)
        st.caption("Folder containing the training images.")
        st.markdown("</div>", unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown("**📄 CSV Label File**")
        ca, cb = st.columns([3, 1])
        csv_files_raw = ca.text_input(
            "CSV label file path",
            value=st.session_state.get("dl_csv_files", ""),
            placeholder="/path/to/labels.csv",
            key="dl_csv_files",
            label_visibility="collapsed",
        )
        cb.button("📁 Browse", on_click=_pick_csv, key="dl_browse_csv",
                  use_container_width=True)
        st.caption("CSV with columns: `filename`, `label` (or equivalent).")
        st.markdown("</div>", unsafe_allow_html=True)

    img_folders = [p.strip() for p in img_folders_raw.split(",") if p.strip()]
    csv_files   = [p.strip() for p in csv_files_raw.split(",")   if p.strip()]

    # ── CSV preview + column validation ────────────────────────────────────────
    REQUIRED_COLS = {"PLOT", "Index"}
    csv_errors    = []

    if csv_files and all(os.path.isfile(p) for p in csv_files):
        import pandas as _pd
        st.markdown("#### 📄 CSV Preview")
        for csv_path in csv_files:
            fname = os.path.basename(csv_path)
            try:
                if os.path.getsize(csv_path) == 0:
                    st.error(f"❌ **{fname}** — file is **empty** (0 bytes). Select the correct CSV.")
                    csv_errors.append(fname)
                    continue
                df_prev = _pd.read_csv(csv_path)
                missing = REQUIRED_COLS - set(df_prev.columns)
                has_ok  = len(missing) == 0
                status  = "✅" if has_ok else "❌"
                with st.expander(f"{status} {fname}  —  {df_prev.shape[0]:,} rows × {df_prev.shape[1]} cols", expanded=not has_ok):
                    if missing:
                        st.error(
                            f"Missing required columns: **{', '.join(sorted(missing))}**  \n"
                            f"Found: `{list(df_prev.columns)}`  \n"
                            f"The dataset loader expects **PLOT** (plot ID) and **Index** (severity 1–8)."
                        )
                        csv_errors.append(fname)
                    else:
                        st.dataframe(df_prev.head(5), use_container_width=True)
                        st.caption(f"Columns: {list(df_prev.columns)}")
            except Exception as _e:
                st.error(f"❌ **{fname}** — could not read: {_e}")
                csv_errors.append(fname)

    def _validate_paths(folders, csvs):
        if not folders or not csvs:
            return False, "Please enter at least one image folder and one CSV file."
        if len(folders) != len(csvs):
            return False, f"Count mismatch: {len(folders)} folder(s) vs {len(csvs)} CSV(s)."
        bad_folders = [p for p in folders if not os.path.isdir(p)]
        bad_csvs    = [p for p in csvs    if not os.path.isfile(p)]
        if bad_folders:
            return False, f"Folders not found: {bad_folders}"
        if bad_csvs:
            return False, f"CSV files not found: {bad_csvs}"
        if csv_errors:
            return False, f"Fix CSV issues before training: {csv_errors}"
        return True, ""

    paths_ok, path_msg = _validate_paths(img_folders, csv_files)
    if (img_folders_raw or csv_files_raw) and not paths_ok and not csv_errors:
        st.warning(f"⚠️ {path_msg}")

    # ── START button ────────────────────────────────────────────────────────────
    st.markdown("---")
    start_btn = st.button(
        "🚀 Start Training",
        disabled=(not paths_ok or st.session_state.get("dl_train_running", False)),
        key="dl_start_train",
    )

    # Only launch a thread when the button is pressed and one isn't already running
    if start_btn and not st.session_state.get("dl_train_running", False):
        log_q       = queue.Queue()
        result_hold = {}

        thread = threading.Thread(
            target=run_dl_training,
            kwargs=dict(
                image_folders=img_folders,
                csv_files=csv_files,
                model_name=model_name,
                fine_tune_mode=fine_tune,
                max_epochs=max_epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                num_classes=num_classes,
                log_queue=log_q,
                result_holder=result_hold,
                custom_run_name=custom_run_name.strip() or None,
            ),
            daemon=True,
        )
        thread.start()

        # Store in session_state so rerenders don't spawn a second thread
        st.session_state["dl_train_running"]     = True
        st.session_state["dl_train_log"]         = []
        st.session_state["dl_train_result"]      = None
        st.session_state["dl_train_error"]       = None
        st.session_state["_dl_thread"]           = thread
        st.session_state["_dl_log_q"]            = log_q
        st.session_state["_dl_result_hold"]      = result_hold

    if st.session_state.get("dl_train_running"):
        thread      = st.session_state["_dl_thread"]
        log_q       = st.session_state["_dl_log_q"]
        result_hold = st.session_state["_dl_result_hold"]

        st.markdown("### 📡 Training Log")
        log_placeholder = st.empty()
        progress_bar    = st.progress(0, text="Training in progress…")

        log_lines = st.session_state.get("dl_train_log", [])
        done      = False

        while thread.is_alive() or not log_q.empty():
            while not log_q.empty():
                msg = log_q.get()
                if msg == "__DONE__":
                    done = True
                    break
                elif msg.startswith("__EPOCH__"):
                    # Parse "1/10" style token for real progress
                    try:
                        cur, total = msg.split()[1].split("/")
                        progress_bar.progress(
                            int(cur) / int(total),
                            text=f"Epoch {cur} / {total}",
                        )
                    except Exception:
                        pass
                else:
                    log_lines.append(msg)

            log_placeholder.markdown(
                "<div class='log-box'>" +
                "<br>".join(log_lines[-60:]) +
                "</div>",
                unsafe_allow_html=True,
            )
            if done:
                break
            time.sleep(0.25)

        thread.join()

        # Drain any remaining messages after thread finishes
        while not log_q.empty():
            msg = log_q.get()
            if msg not in ("__DONE__",) and not msg.startswith("__EPOCH__"):
                log_lines.append(msg)

        log_placeholder.markdown(
            "<div class='log-box'>" + "<br>".join(log_lines[-60:]) + "</div>",
            unsafe_allow_html=True,
        )
        progress_bar.empty()
        st.session_state["dl_train_log"]     = log_lines
        st.session_state["dl_train_running"] = False

        if "result" in result_hold:
            r = result_hold["result"]
            st.session_state["dl_train_result"] = r
            st.success("✅ Training complete!")

            # ── Save / data path summary ───────────────────────────────────
            sp = r.get("save_path", "")
            dfs = r.get("data_folders", [])
            if sp:
                st.info(f"💾 **Model saved to:** `{sp}`")
            if dfs:
                st.info(f"📂 **Training data:** `{'; '.join(dfs)}`")

            # ── Final metric tiles ─────────────────────────────────────────
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Accuracy",  f"{r['final_accuracy']:.4f}")
            mc2.metric("F1 Score",  f"{r['final_f1']:.4f}")
            mc3.metric("Precision", f"{r['final_precision']:.4f}")
            mc4.metric("Recall",    f"{r['final_recall']:.4f}")

            # ── Loss + validation metric curves ───────────────────────────
            import matplotlib.pyplot as plt
            import numpy as np

            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            fig.patch.set_facecolor("#161b22")
            for ax in axes:
                ax.set_facecolor("#0d1117")
                for sp_ax in ax.spines.values():
                    sp_ax.set_edgecolor("#30363d")
                ax.tick_params(colors="#8b949e")

            axes[0].plot(r["train_losses"], color="#58a6ff", label="Train Loss")
            axes[0].plot(r["val_losses"],   color="#f78166", label="Val Loss", linestyle="--")
            axes[0].set_title("Loss Curves", color="#e6edf3")
            axes[0].legend(facecolor="#161b22", labelcolor="#e6edf3")
            axes[0].set_xlabel("Epoch", color="#8b949e")

            ep_x = len(r["val_accuracies"]) - 1   # index of last epoch
            axes[1].plot(r["val_accuracies"], color="#3fb950", label="Val Accuracy / epoch")
            axes[1].plot(r["val_f1_scores"],  color="#e3b341", label="Val F1 / epoch", linestyle="--")
            # Mark the final evaluation point so users can verify consistency
            axes[1].scatter([ep_x], [r["final_accuracy"]], color="#3fb950",
                            zorder=5, s=80, marker="*",
                            label=f"Final Acc = {r['final_accuracy']:.4f}")
            axes[1].scatter([ep_x], [r["final_f1"]], color="#e3b341",
                            zorder=5, s=80, marker="*",
                            label=f"Final F1 = {r['final_f1']:.4f}")
            axes[1].set_title("Validation Metrics", color="#e6edf3")
            axes[1].legend(facecolor="#161b22", labelcolor="#e6edf3", fontsize=7)
            axes[1].set_xlabel("Epoch", color="#8b949e")

            st.pyplot(fig)
            plt.close(fig)

            # ── Confusion matrix + per-class breakdown ────────────────────
            import seaborn as sns
            import pandas as pd

            cm = np.array(r["confusion_matrix"])
            n_cls   = cm.shape[0]
            # Semantic class labels (Low / Medium / High with fallback)
            _bin_labels = {0: "Low (Bin 0)", 1: "Medium (Bin 1)", 2: "High (Bin 2)"}
            cls_names = [_bin_labels.get(i, f"Bin {i}") for i in range(n_cls)]

            # Build annotated heatmap: "N\n(row %)"
            row_sums  = cm.sum(axis=1, keepdims=True).clip(min=1)
            cm_pct    = cm / row_sums * 100
            annots    = np.array(
                [[f"{cm[i,j]}\n({cm_pct[i,j]:.1f}%)" for j in range(n_cls)]
                 for i in range(n_cls)]
            )

            st.markdown("### 🧩 Confusion Matrix & Per-class Metrics")
            cm_col, tbl_col = st.columns([1, 1])

            with cm_col:
                fig2, ax2 = plt.subplots(figsize=(5, 4))
                fig2.patch.set_facecolor("#161b22")
                ax2.set_facecolor("#161b22")
                sns.heatmap(
                    cm, annot=annots, fmt="", cmap="Blues",
                    ax=ax2, cbar=False,
                    xticklabels=cls_names,
                    yticklabels=cls_names,
                )
                ax2.set_title(
                    f"Confusion Matrix\n"
                    f"Overall Accuracy = {cm.diagonal().sum() / cm.sum():.4f}",
                    color="#e6edf3", fontsize=9,
                )
                ax2.set_xlabel("Predicted", color="#8b949e")
                ax2.set_ylabel("True", color="#8b949e")
                ax2.tick_params(colors="#8b949e")
                st.pyplot(fig2)
                plt.close(fig2)
                st.caption(
                    "Each cell: **count** (% of true-class row).  "
                    "Diagonal = correct predictions.  "
                    "Overall Accuracy = diagonal sum ÷ total."
                )

            with tbl_col:
                # Compute per-class Precision, Recall, F1 from CM
                rows = []
                for i, name in enumerate(cls_names):
                    tp = cm[i, i]
                    fp = cm[:, i].sum() - tp
                    fn = cm[i, :].sum() - tp
                    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
                    rows.append({
                        "Class":     name,
                        "TP":        int(tp),
                        "FP":        int(fp),
                        "FN":        int(fn),
                        "Precision": f"{prec:.4f}",
                        "Recall":    f"{rec:.4f}",
                        "F1":        f"{f1:.4f}",
                    })
                # Overall row
                total = cm.sum()
                correct = cm.diagonal().sum()
                rows.append({
                    "Class":     "✅ Overall",
                    "TP":        int(correct),
                    "FP":        "",
                    "FN":        "",
                    "Precision": f"{r['final_precision']:.4f}",
                    "Recall":    f"{r['final_recall']:.4f}",
                    "F1":        f"{r['final_f1']:.4f}",
                })
                cm_df = pd.DataFrame(rows)
                st.dataframe(cm_df, use_container_width=True, hide_index=True)
                st.caption(
                    f"Total samples: **{int(total)}**  ·  "
                    f"Correct: **{int(correct)}**  ·  "
                    f"Accuracy from CM: **{correct/total:.4f}**  "
                    f"(should match the metric tile above)"
                )

            with st.expander("📋 Full Classification Report"):
                st.code(r["classification_report"])

        elif "error" in result_hold:
            st.error("❌ Training failed.")
            with st.expander("Error traceback"):
                st.code(result_hold["error"])

# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE MODE
# ══════════════════════════════════════════════════════════════════════════════
else:
    import matplotlib.pyplot as plt
    import tempfile
    import shutil
    import pandas as pd

    st.markdown("### 🔍 Inference with Saved Checkpoints")

    saved = get_saved_models()

    if not saved:
        st.warning(
            "⚠️ No `.pth` or `.pt` checkpoints found in `DL_Part/saved_models/`.\n\n"
            "Train a model first, or place checkpoints there.",
            icon="⚠️",
        )
        st.stop()

    st.success(f"✅ **{len(saved)} trained checkpoints** found in `DL_Part/saved_models/`")

    c1, c2 = st.columns([2, 1])
    with c1:
        model_options  = {m["display"]: m for m in saved}
        chosen_display = st.selectbox(
            "Select checkpoint",
            list(model_options.keys()),
            key="dl_inf_model_sel",
        )
        chosen = model_options[chosen_display]

        if chosen["arch"]:
            st.markdown(
                f"<span style='color:#8b949e;font-size:.85rem'>"
                f"📐 Architecture resolved: <code>{chosen['arch']}</code> &nbsp;·&nbsp; "
                f"File: <code>{chosen['filename']}</code></span>",
                unsafe_allow_html=True,
            )
        else:
            st.warning("⚠️ Architecture not auto-detected — please select manually below.")
            chosen["arch"] = st.selectbox(
                "Architecture (manual)", DL_MODELS, key="dl_inf_arch_manual"
            )

    with c2:
        inf_classes = st.number_input(
            "Number of output classes", 2, 3, 3, key="dl_inf_nclasses"
        )

    st.markdown("---")
    uploaded = st.file_uploader(
        "📤 Upload image(s) for inference (.png / .jpg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="dl_inf_upload",
    )
    if uploaded:
        st.caption(f"{len(uploaded)} image(s) ready.")

    inf_btn = st.button(
        "🔮 Run Inference + GradCAM",
        disabled=(not uploaded),
        key="dl_inf_run",
    )

    if inf_btn and uploaded:
        tmp_dir   = tempfile.mkdtemp()
        tmp_paths = []
        for uf in uploaded:
            dest = os.path.join(tmp_dir, uf.name)
            with open(dest, "wb") as fout:
                fout.write(uf.read())
            tmp_paths.append(dest)

        with st.spinner(
            f"Running {chosen['display']} inference + GradCAM on {len(tmp_paths)} image(s)…"
        ):
            try:
                results = run_dl_inference(
                    tmp_paths, chosen["path"], chosen["arch"], inf_classes
                )
                st.session_state["dl_inf_results"]    = results
                st.session_state["dl_inf_model_name"] = chosen["display"]
            except Exception as e:
                import traceback
                st.error(f"❌ Inference failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                results = []

        shutil.rmtree(tmp_dir, ignore_errors=True)

    results        = st.session_state.get("dl_inf_results", [])
    inf_model_name = st.session_state.get("dl_inf_model_name", "")

    if results:
        st.success(
            f"✅ Inference complete — **{len(results)} image(s)** — model: **{inf_model_name}**"
        )
        st.markdown("---")

        severity_colors = {0: "#3fb950", 1: "#e3b341", 2: "#f78166"}
        summary_rows    = []

        for r in results:
            st.markdown(
                f"<h4 style='color:#58a6ff'>🖼️ {r['filename']}</h4>",
                unsafe_allow_html=True,
            )

            img_cols = st.columns(4)
            img_cols[0].image(r["original_np"],    caption="Original",        use_container_width=True)
            img_cols[1].image(r["heatmap_np"],     caption="GradCAM Heatmap", use_container_width=True)
            img_cols[2].image(r["superimposed_np"],caption="Overlay",         use_container_width=True)

            fig_cam, ax_cam = plt.subplots(figsize=(3, 3))
            fig_cam.patch.set_facecolor("#161b22")
            ax_cam.imshow(r["cam_np"], cmap="jet")
            ax_cam.axis("off")
            ax_cam.set_title("Attention Map", color="#e6edf3", fontsize=9)
            img_cols[3].pyplot(fig_cam)
            plt.close(fig_cam)

            pred_col, bar_col = st.columns([1, 2])
            color = severity_colors.get(r["predicted_bin"], "#58a6ff")

            with pred_col:
                st.markdown(
                    f"<div style='background:#161b22;border:1px solid #30363d;"
                    f"border-radius:10px;padding:16px;text-align:center'>"
                    f"<p style='color:#8b949e;font-size:.85rem;margin:0'>Prediction</p>"
                    f"<p style='color:{color};font-size:1.5rem;font-weight:700;margin:4px 0'>"
                    f"{r['label']}</p>"
                    f"<p style='color:#8b949e;font-size:.85rem;margin:0'>"
                    f"Confidence: <b style='color:#e6edf3'>{r['confidence']:.1%}</b></p>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with bar_col:
                if "class_confidences" in r:
                    labels     = list(r["class_confidences"].keys())
                    vals       = list(r["class_confidences"].values())
                    bar_colors = [severity_colors.get(i, "#58a6ff") for i in range(len(labels))]

                    fig_bar, ax_bar = plt.subplots(figsize=(5, 2.2))
                    fig_bar.patch.set_facecolor("#161b22")
                    ax_bar.set_facecolor("#0d1117")
                    bars = ax_bar.barh(labels, vals, color=bar_colors, edgecolor="#161b22", height=0.5)
                    ax_bar.set_xlim(0, 100)
                    ax_bar.set_xlabel("Confidence (%)", color="#8b949e", fontsize=8)
                    ax_bar.tick_params(colors="#8b949e", labelsize=8)
                    for sp in ax_bar.spines.values():
                        sp.set_edgecolor("#30363d")
                    for bar, val in zip(bars, vals):
                        ax_bar.text(
                            min(val + 1, 96), bar.get_y() + bar.get_height() / 2,
                            f"{val:.1f}%", va="center", color="#e6edf3", fontsize=8,
                        )
                    ax_bar.invert_yaxis()
                    st.pyplot(fig_bar)
                    plt.close(fig_bar)

            summary_rows.append({
                "Image":      r["filename"],
                "Prediction": r["label"],
                "Confidence": f"{r['confidence']:.2%}",
                **{k: f"{v:.1f}%" for k, v in r.get("class_confidences", {}).items()},
            })
            st.markdown("---")

        st.markdown("### 📋 Summary Table")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        csv_bytes = summary_df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download predictions as CSV",
            data=csv_bytes,
            file_name="blast_dl_predictions.csv",
            mime="text/csv",
            key="dl_inf_download",
        )
