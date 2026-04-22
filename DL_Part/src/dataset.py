import os
import glob
import random
import importlib.util
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# Load DL config by absolute path to avoid sys.path collisions with ML_Part/src/config.py
_cfg_spec = importlib.util.spec_from_file_location(
    "dl_config",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
BIN_MAPPING = _cfg.BIN_MAPPING

class ImageClassificationDataset(Dataset):
    def __init__(self, image_folder, csv_file, transform=None, target_col="Index"):
        assert len(image_folder) == len(csv_file), "image_folders and csv_files must have the same length"

        self.transform = transform

        # Temporary storage for all samples before balancing
        temp_image_paths = []
        temp_targets = []

        for img_folder, csv_path in zip(image_folder, csv_file):
            df = pd.read_csv(csv_path)
            df['PLOT'] = pd.to_numeric(df['PLOT'], errors='coerce')
            df = df.replace([float('inf'), float('-inf')], pd.NA)
            df = df.dropna(subset=['PLOT'])

            df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
            df = df.replace([float('inf'), float('-inf')], pd.NA)
            df = df.dropna(subset=[target_col])

            df['PLOT'] = df['PLOT'].astype(int)
            df[target_col] = df[target_col].astype(float)

            # Convert Index values (1.0–8.0) to original class labels (0–7)
            df['class_label'] = (df[target_col] - 1).astype(int)

            # Filter out invalid classes
            valid_mask = (df['class_label'] >= 0) & (df['class_label'] < 9)
            df = df[valid_mask]

            # Apply binning
            df['binned_label'] = df['class_label'].map(BIN_MAPPING)

            id_to_target = dict(zip(df['PLOT'], df['binned_label']))
            image_files = glob.glob(os.path.join(img_folder, "*.png"))

            for img_path in image_files:
                filename = os.path.basename(img_path)
                try:
                    plot_id = int(filename.split('_')[-1].split('.')[0])
                except Exception as e:
                    raise ValueError(f"Filename format error for {filename}: {e}")

                if plot_id in id_to_target:
                    temp_image_paths.append(img_path)
                    temp_targets.append(id_to_target[plot_id])

            print(f"✅ Loaded {len(image_files)} images from {img_folder}")

        print(f"✅ Total images loaded before balancing: {len(temp_image_paths)}")

        # Print original class distribution
        unique, counts = np.unique(temp_targets, return_counts=True)
        print(f"\n📊 Original Binned Class Distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"   Bin {cls}: {cnt} samples ({cnt/len(temp_targets)*100:.1f}%)")

        # Balance the dataset by selecting first n samples from each class
        # where n = min(samples per class)
        min_samples = min(counts)
        print(f"\n⚖️ Balancing to {min_samples} samples per class...")

        # Group samples by class
        class_samples = {cls: [] for cls in unique}
        for img_path, target in zip(temp_image_paths, temp_targets):
            class_samples[target].append((img_path, target))

        # Select random n samples from each class
        self.image_paths = []
        self.targets = []

        for cls in sorted(class_samples.keys()):
            selected_samples = random.sample(class_samples[cls], min_samples)
            for img_path, target in selected_samples:
                self.image_paths.append(img_path)
                self.targets.append(target)

        # Print balanced class distribution
        unique, counts = np.unique(self.targets, return_counts=True)
        print(f"\n📊 Balanced Class Distribution:")
        for cls, cnt in zip(unique, counts):
            print(f"   Bin {cls}: {cnt} samples ({cnt/len(self.targets)*100:.1f}%)")

        print(f"✅ Final dataset size after balancing: {len(self.image_paths)} samples")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        target = self.targets[idx]

        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error opening image {img_path}: {e}. Skipping.")
            return None

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(target, dtype=torch.long)


def collate_fn_skip_none(batch):
    batch = list(filter(lambda x: x is not None, batch))
    if len(batch) == 0:
        return None, None
    return torch.utils.data.dataloader.default_collate(batch)
