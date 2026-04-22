import torch

# --- Global Configuration ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of classes (Index values 1-8 mapped to 0-7)
NUM_CLASSES = 3

# Mapping raw severity indices to bins
BIN_MAPPING = {
    0: 0,        # bin1
    1: 0, 2: 0,  # bin2
    3: 0, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2  # bin3
}

# Default sample paths — relative to this file so the script works on any machine.
# The Streamlit app always overrides these at runtime.
_DL_PART_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

IMAGE_FOLDER = [
    os.path.join(_DL_PART_ROOT, "smaple_data", "input")
]  # str or list[str]

CSV_FILE = [
    os.path.join(_DL_PART_ROOT, "smaple_data", "output", "b3_bl1_2025.csv")
]  # str or list[str]
