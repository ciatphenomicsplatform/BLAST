"""
ML Runner — thin wrapper around ML_Part/src/ that captures stdout
and streams it back to a Streamlit log box.
"""

import sys
import os
import queue
import traceback
from datetime import datetime

# ── Import shared utilities ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.shared import StreamCapture, MLFLOW_URI, BLAST_ROOT  # noqa: F401

# ── Make ML_Part/src importable ────────────────────────────────────────────────
ML_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ML_Part", "src")
)
if ML_SRC not in sys.path:
    sys.path.insert(0, ML_SRC)

ML_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ML_Part", "datasets", "input")
)

ML_MODELS_AVAILABLE = ["et", "rf", "xgboost", "lightgbm", "dt"]

FEATURES_ALL = [
    "NDRE_MEAN", "NDVI_MEAN", "GNDVI_MEAN", "BNDVI_MEAN",
    "NDREI_MEAN", "NPCI_MEAN", "GRVI_MEAN", "NGBDI_MEAN",
    "CH_MEAN", "CV_MEAN", "CC_%", "CC_mt2",
]
TARGET_COLUMN = "BL"


def get_dataset_files() -> list[str]:
    """Return list of .xlsx filenames found in ML datasets/input/."""
    if not os.path.isdir(ML_DATA_DIR):
        return []
    return sorted(f for f in os.listdir(ML_DATA_DIR) if f.endswith(".xlsx"))


def run_ml_training(
    dataset_filename: str,
    selected_models: list[str],
    use_oversampling: bool,
    selected_features: list[str],
    log_queue: queue.Queue,
    result_holder: dict,
    save_path: str = None,
    custom_run_name: str = None,
):
    """
    Runs ML training in a background thread.
    Appends log strings to `log_queue`, stores results in `result_holder`.
    Model is saved with a timestamp suffix to avoid overwriting previous runs.
    """
    import mlflow

    with StreamCapture(log_queue):
        try:
            import pandas as pd
            from pycaret.classification import setup, compare_models, pull, save_model

            # Force reload from ML_SRC to avoid importing the DL_Part cache
            sys.modules.pop("dataset", None)
            sys.modules.pop("utils", None)
            
            if ML_SRC not in sys.path:
                sys.path.insert(0, ML_SRC)
            from dataset import delete_columns, filter_features_no_canopy, balance_dataset_oversampling
            from utils import model_complexity

            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("ML_AgriBlast")
            print(f"📊 MLflow → {MLFLOW_URI}")

            file_path = os.path.join(ML_DATA_DIR, dataset_filename)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Dataset not found: {file_path}")

            print(f"📂 Loading {dataset_filename}...")
            df = pd.read_excel(file_path)

            print("🔧 Preprocessing data...")
            df = delete_columns(df)

            feature_cols = [f for f in selected_features if f in df.columns] + [TARGET_COLUMN]
            df = df[[c for c in feature_cols if c in df.columns]].copy()

            if use_oversampling:
                print("⚖️  Applying oversampling...")
                df = balance_dataset_oversampling(
                    df, target_column=TARGET_COLUMN, oversampling_factor="auto"
                )

            print(f"📐 Dataset shape: {df.shape}")

            print("⚙️  Setting up PyCaret experiment...")

            # Disable PyCaret's built-in MLflow logging so it does NOT create
            # a separate "Session Initialized" parent run.  We own the run below.
            exp = setup(
                data=df,
                target=TARGET_COLUMN,
                normalize=True,
                session_id=42,
                log_experiment=False,   # <-- prevents duplicate "Session Initialized" run
                verbose=False,
            )

            print(f"🤖 Training models: {selected_models}...")
            best_model = compare_models(include=selected_models, verbose=False)
            results_df = pull()

            print("\n🏆 Top model comparison:")
            print(results_df.head(5).to_string())

            # Build save path: prefer custom name, fall back to timestamped default
            if save_path is None:
                if custom_run_name:
                    safe_name = custom_run_name.replace(" ", "_").replace("/", "-")
                    save_path = os.path.join(BLAST_ROOT, safe_name)
                else:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(
                        BLAST_ROOT, f"best_ml_blast_model_{timestamp}"
                    )
            save_model(best_model, save_path)
            print(f"\n💾 Best model saved to: {save_path}.pkl")

            try:
                complexity = model_complexity(best_model)
                print(f"🔎 Model complexity: {complexity:,} nodes/params")
            except Exception:
                complexity = None

            # ── Log everything into ONE unified MLflow run ──────────────────
            best_model_name = type(best_model).__name__
            auto_name = f"{best_model_name} — {dataset_filename}"
            run_name  = custom_run_name if custom_run_name else auto_name
            top_row   = results_df.iloc[0] if not results_df.empty else {}

            with mlflow.start_run(run_name=run_name):
                # Dataset + config tags
                mlflow.set_tags({
                    "dataset_file":    dataset_filename,
                    "best_model":      best_model_name,
                    "oversampling":    str(use_oversampling),
                    "features":        ",".join(selected_features),
                    "models_compared": ",".join(selected_models),
                    "custom_run_name": custom_run_name or "",
                })

                # Hyperparameters
                mlflow.log_params({
                    "session_id":     42,
                    "normalize":      True,
                    "use_oversampling": use_oversampling,
                    "n_rows":         df.shape[0],
                    "n_features":     df.shape[1] - 1,
                })

                # Metrics from the leaderboard top row
                metric_map = {
                    "Accuracy":  "accuracy",
                    "AUC":       "auc",
                    "Recall":    "recall",
                    "Prec.":     "precision",
                    "F1":        "f1",
                    "Kappa":     "kappa",
                    "MCC":       "mcc",
                }
                for col, key in metric_map.items():
                    if col in results_df.columns and not pd.isna(top_row.get(col)):
                        mlflow.log_metric(key, float(top_row[col]))

                if complexity is not None:
                    mlflow.log_metric("model_complexity", complexity)

                mlflow.log_param("save_path", save_path + ".pkl")

            print(f"✅ Single MLflow run logged: '{run_name}'")
            # ────────────────────────────────────────────────────────────────

            result_holder["result"] = {
                "comparison_df": results_df,
                "best_model":    best_model,
                "complexity":    complexity,
                "save_path":     save_path,
            }
            log_queue.put("__DONE__")

        except Exception as exc:
            result_holder["error"] = traceback.format_exc()
            log_queue.put(f"❌ ERROR: {exc}")
            log_queue.put("__DONE__")


def run_ml_inference(
    df_input,
    model_path: str,
    selected_features: list[str],
) -> "pd.DataFrame":
    """
    Load a saved PyCaret .pkl pipeline and predict on df_input.
    Returns df_input with a 'prediction' column added.
    """
    from pycaret.classification import load_model, predict_model
    import pandas as pd

    pipeline = load_model(model_path.replace(".pkl", ""))

    available = [f for f in selected_features if f in df_input.columns]
    df_feat   = df_input[available].copy()

    predictions = predict_model(pipeline, data=df_feat)
    return predictions
