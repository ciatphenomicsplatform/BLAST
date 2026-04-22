import os
import pandas as pd
from pycaret.classification import setup, compare_models, pull, save_model
import mlflow

from config import DATA_DIR, DATASET_FILES, TARGET_COLUMN, FEATURES_NO_CANOPY, MLFLOW_TRACKING_URI
from dataset import delete_columns, filter_features_no_canopy, balance_dataset_oversampling
from utils import model_complexity

def main():
    # 1. Setup MLflow Tracking Server URI directly. 
    # PyCaret will respect this configuration when logging experiments.
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"MLflow configured to track to: {MLFLOW_TRACKING_URI}")
    
    # 2. Load Data
    # For demonstration, we use one of the main unified datsets as listed in config.
    # The user can configure this to point to any loaded data slice.
    file_name = DATASET_FILES[4] # example: "df_blast_2022_2023_2024_BL2_class.xlsx"
    file_path = os.path.join(DATA_DIR, file_name)
    
    if not os.path.exists(file_path):
        print(f"Dataset file not found: {file_path}")
        print("Please place the raw datasets (e.g. .xlsx fields) inside")
        print(f"  {DATA_DIR}")
        print("or update DATA_DIR in config.py to point to your existing location.")
        return
        
    print(f"\nLoading {file_name}...")
    df = pd.read_excel(file_path)
    
    # 3. Preprocess
    print("Preprocessing data...")
    # Clean unwanted columns
    df = delete_columns(df)
    # Select feature boundaries
    df = filter_features_no_canopy(df, FEATURES_NO_CANOPY)
    
    # Oversampling
    print("Applying oversampling to handle imbalanced classes...")
    df_balanced = balance_dataset_oversampling(df, target_column=TARGET_COLUMN, oversampling_factor='auto')
    print(f"Data shape after preprocessing: {df_balanced.shape}")
    
    # 4. PyCaret Setup
    print("\nSetting up PyCaret Experiment...")
    # This automatically tracks parameters and models to the local MLflow sqlite set above. 
    exp = setup(data=df_balanced,
                target=TARGET_COLUMN,
                normalize=True,
                session_id=42,
                log_experiment=True,
                experiment_name='ML_AgriBlast',
                verbose=False)
                
    # 5. Model Training & Comparison
    print("\nTraining and comparing models (Random Forest, LightGBM, XGBoost, etc.)...")
    # For a standardized production ML pipeline, we typically restrict to the known best tree-based models.
    best_model = compare_models(include=['et', 'rf', 'xgboost', 'lightgbm', 'dt'])
    
    # Pull the results grid
    results_df = pull()
    print("\nTop Models Comparison Results:")
    print(results_df.head(5))
    
    # 6. Analytics & Saving
    try:
        complexity = model_complexity(best_model)
        print(f"\nBest model node/param complexity: {complexity}")
    except Exception as e:
        print(f"Could not compute complexity for chosen model format: {e}")
        
    # Save best model representation to binary
    save_model(best_model, 'best_ml_blast_model')
    print("\nPipeline complete! Best model saved to disk and fully tracked via MLflow.")
    print("To view runs natively, execute:")
    print(f"  mlflow ui --backend-store-uri {MLFLOW_TRACKING_URI}")

if __name__ == "__main__":
    main()
