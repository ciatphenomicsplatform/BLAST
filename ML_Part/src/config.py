import os

# Base directory for the ML pipeline
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# MLflow SQLite Tracking URI — derived from this file's location so it works
# on any machine, regardless of where the repo is cloned.
_BLAST_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MLFLOW_TRACKING_URI = "sqlite:///" + os.path.join(_BLAST_ROOT, "mlruns.db")

# Directory where raw datasets should be placed
# We map the old absolute Windows path to a 'datasets/input' folder locally.
DATA_DIR = os.path.join(BASE_DIR, "datasets", "input")

# List of expected filenames, based on BL_Models.ipynb
DATASET_FILES = [
    "df_blast_2022_BL2_class.xlsx",
    "df_blast_2023_BL2_class.xlsx",
    "df_blast_2024_BL2_class.xlsx",
    "df_blast_2025_BL2_class.xlsx",
    "df_blast_2022_2023_2024_BL2_class.xlsx",
    "df_blast_2022_2023_2024_oversampled_BL2_class.xlsx",
    "df_blast_2022_2023_2024_2025_BL2_class.xlsx",
]

TARGET_COLUMN = "BL"

# Numeric features often used during classification (no canopy variables)
FEATURES_NO_CANOPY = [
    'BL', 'NDRE_MEAN', 'NDVI_MEAN', 'GNDVI_MEAN', 'BNDVI_MEAN', 
    'NDREI_MEAN', 'NPCI_MEAN', 'GRVI_MEAN', 'NGBDI_MEAN'
]
