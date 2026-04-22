"""
DL Runner — thin wrapper around DL_Part/src/ that captures stdout
and streams it back to a Streamlit log box.
"""

import sys
import os
import glob
import threading
import queue
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split

# ── Import shared utilities ────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from components.shared import StreamCapture, MLFLOW_URI, BLAST_ROOT

# ── Make DL_Part/src importable ────────────────────────────────────────────────
DL_SRC = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "DL_Part", "src")
)
if DL_SRC not in sys.path:
    sys.path.insert(0, DL_SRC)

# ── Saved-models directory ─────────────────────────────────────────────────────
SAVED_MODELS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "DL_Part", "saved_models")
)

# ── Mapping from .pth / .pt filename stem → models.py architecture key ────────
FILENAME_TO_ARCH = {
    "ConvNeXt-Small":     "convnext_small",
    "DenseNet121":        "densenet121",
    "EfficientNet-B1":    "efficientnet_b1",
    "EfficientNet-V2-S":  "efficientnet_v2_s",
    "MaxViT-T":           "maxvit_t",
    "MobileNet-V3-Large": "mobilenet_v3_large",
    "RegNet-Y-800MF":     "regnet_y_800mf",
    "ResNeXt50":          "resnext50",
    "ResNet50":           "resnet50",
    "Swin-T":             "swin_t",
    "ViT-B_16":           "vit_b_16",
}

# Friendly display labels for the UI dropdown
ARCH_DISPLAY = {
    "convnext_small":     "ConvNeXt-Small",
    "densenet121":        "DenseNet-121",
    "efficientnet_b1":    "EfficientNet-B1",
    "efficientnet_v2_s":  "EfficientNet-V2-S  ★ best",
    "maxvit_t":           "MaxViT-T",
    "mobilenet_v3_large": "MobileNet-V3-Large",
    "regnet_y_800mf":     "RegNet-Y-800MF",
    "resnext50":          "ResNeXt-50",
    "resnet50":           "ResNet-50",
    "swin_t":             "Swin-T",
    "vit_b_16":           "ViT-B/16",
}

# ── Available model names (mirror models.py) ───────────────────────────────────
DL_MODELS = [
    # EfficientNet
    "efficientnet_v2_s", "efficientnet_v2_m",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2",
    # ResNet
    "resnet18", "resnet34", "resnet50", "resnet101",
    # ResNeXt
    "resnext50", "resnext101",
    # DenseNet
    "densenet121", "densenet161", "densenet169",
    # MobileNet
    "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
    # ConvNeXt
    "convnext_tiny", "convnext_small", "convnext_base",
    # Vision Transformers
    "vit_b_16", "vit_b_32", "vit_l_16",
    # Swin
    "swin_t", "swin_s", "swin_b",
    # MaxViT
    "maxvit_t",
    # RegNet
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf",
]

FINE_TUNE_MODES = ["full", "freeze", "partial"]

# ── Severity bin labels ────────────────────────────────────────────────────────
BIN_LABELS = {0: "Low (Bin 0)", 1: "Medium (Bin 1)", 2: "High (Bin 2)"}


def get_saved_models() -> list[dict]:
    """
    Scan SAVED_MODELS_DIR for .pth and .pt files.
    Returns a list of dicts:
      { 'path': str, 'filename': str, 'stem': str, 'arch': str, 'display': str }
    """
    results = []
    patterns = ["*.pth", "*.pt"]
    seen = set()
    for pat in patterns:
        for fpath in sorted(glob.glob(os.path.join(SAVED_MODELS_DIR, pat))):
            if fpath in seen:
                continue
            seen.add(fpath)
            stem = os.path.splitext(os.path.basename(fpath))[0]
            arch = None
            for k, v in FILENAME_TO_ARCH.items():
                if k.lower() == stem.lower():
                    arch = v
                    break
            if arch is None and stem in DL_MODELS:
                arch = stem
            display = ARCH_DISPLAY.get(arch, stem) if arch else stem
            results.append({
                "path":     fpath,
                "filename": os.path.basename(fpath),
                "stem":     stem,
                "arch":     arch,
                "display":  display,
            })
    return results


def run_dl_training(
    image_folders: list[str],
    csv_files: list[str],
    model_name: str,
    fine_tune_mode: str,
    max_epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    num_classes: int,
    log_queue: queue.Queue,
    result_holder: dict,
    custom_run_name: str = None,
):
    """
    Runs DL training in a background thread.
    Appends log strings to `log_queue` and stores the results dict
    in `result_holder['result']` when done (or error in `result_holder['error']`).
    """
    import mlflow
    import mlflow.pytorch

    with StreamCapture(log_queue):
        try:
            # ── Guarantee DL_Part/src is first in sys.path ──────────────────
            # ml_runner may have prepended ML_SRC earlier in this process;
            # we must evict the cached ML modules before importing DL ones.
            for _mod in ("dataset", "utils", "config", "train", "models",
                         "gradcam", "dl_config"):
                sys.modules.pop(_mod, None)
            if DL_SRC in sys.path:
                sys.path.remove(DL_SRC)
            sys.path.insert(0, DL_SRC)
            # ───────────────────────────────────────────────────────────────

            import importlib
            import importlib.util as _ilu

            # Load DL config by path to avoid name collision with ML config
            _cfg_spec = _ilu.spec_from_file_location(
                "dl_config", os.path.join(DL_SRC, "config.py")
            )
            dl_cfg = _ilu.module_from_spec(_cfg_spec)
            _cfg_spec.loader.exec_module(dl_cfg)
            dl_cfg.IMAGE_FOLDER = image_folders
            dl_cfg.CSV_FILE     = csv_files
            dl_cfg.NUM_CLASSES  = num_classes

            # ── BIN_MAPPING must match num_classes ──────────────────────────
            # class_label = Index - 1  (so 0–8 for Index 1–9)
            # 2-class:  Low [0-4] → 0,  High [5-8] → 1
            # 3-class:  Low [0-3] → 0,  Med [4-5] → 1,  High [6-8] → 2
            _BIN_MAPS = {
                2: {0:0, 1:0, 2:0, 3:0, 4:0, 5:1, 6:1, 7:1, 8:1},
                3: {0:0, 1:0, 2:0, 3:0, 4:1, 5:1, 6:2, 7:2, 8:2},
            }
            if num_classes not in _BIN_MAPS:
                raise ValueError(
                    f"num_classes={num_classes} has no defined BIN_MAPPING. "
                    f"Supported values: {list(_BIN_MAPS.keys())}"
                )
            dl_cfg.BIN_MAPPING = _BIN_MAPS[num_classes]
            print(f"🗂️  BIN_MAPPING for {num_classes} classes: {dl_cfg.BIN_MAPPING}")
            # ───────────────────────────────────────────────────────────────

            sys.modules["config"] = dl_cfg   # so train.py `import config` gets DL version

            # Import as module objects so we can patch dataset.BIN_MAPPING.
            # dataset.py loads BIN_MAPPING from the real config.py file at import time
            # (bypassing sys.modules["config"]), so the only reliable way to override
            # it is to patch the module's global dict directly before any instance is made.
            import dataset as _ds_mod
            import models  as _models_mod
            from train import train_and_evaluate_model

            _ds_mod.BIN_MAPPING = _BIN_MAPS[num_classes]
            print(f"🗂️  Patched dataset.BIN_MAPPING for {num_classes} classes: {_BIN_MAPS[num_classes]}")

            ImageClassificationDataset = _ds_mod.ImageClassificationDataset
            collate_fn_skip_none       = _ds_mod.collate_fn_skip_none
            get_classification_model   = _models_mod.get_classification_model

            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment("AgriBlast_Transfer_Learning")

            img_size  = 224
            transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            print("📂 Loading dataset...")
            full_dataset = ImageClassificationDataset(
                image_folders, csv_files, transform=transform
            )

            train_size = int(0.8 * len(full_dataset))
            val_size   = len(full_dataset) - train_size
            train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

            train_loader = DataLoader(
                train_ds, batch_size=batch_size, shuffle=True,
                num_workers=0, collate_fn=collate_fn_skip_none,
            )
            val_loader = DataLoader(
                val_ds, batch_size=batch_size, shuffle=False,
                num_workers=0, collate_fn=collate_fn_skip_none,
            )

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"🖥️  Device: {device}")
            print(f"🔧 Initialising {model_name} ({fine_tune_mode} fine-tune)...")

            model = get_classification_model(
                model_name, num_classes=num_classes, fine_tune_mode=fine_tune_mode
            )

            run_label = custom_run_name or model_name
            extra_tags = {
                "custom_run_name":  custom_run_name or "",
                "data_folder":      "; ".join(image_folders),
                "csv_files":        "; ".join(csv_files),
                "fine_tune_mode":   fine_tune_mode,
                "num_classes":      str(num_classes),
            }

            results = train_and_evaluate_model(
                model_name=model_name,
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                max_epochs=max_epochs,
                patience=patience,
                lr=lr,
                run_name=run_label,
                extra_tags=extra_tags,
            )

            save_dir  = os.path.join(BLAST_ROOT, "DL_Part", "saved_models")
            os.makedirs(save_dir, exist_ok=True)
            # Use custom run name for the file, falling back to model name
            file_stem = (
                custom_run_name.replace(" ", "_").replace("/", "-")
                if custom_run_name else model_name
            )
            save_path = os.path.join(save_dir, f"{file_stem}.pt")
            torch.save(results["model"].state_dict(), save_path)
            print(f"💾 Model weights saved to: {save_path}")

            # Patch save_path onto the CLOSED run using MlflowClient (safe — no new run created)
            run_id = results.get("mlflow_run_id")
            if run_id:
                from mlflow.tracking import MlflowClient
                _client = MlflowClient(tracking_uri=MLFLOW_URI)
                _client.set_tag(run_id, "save_path", save_path)

            result_holder["result"] = {
                **results,
                "save_path":    save_path,
                "data_folders": image_folders,
                "csv_files":    csv_files,
            }
            log_queue.put("__DONE__")

        except Exception as exc:
            import traceback
            result_holder["error"] = traceback.format_exc()
            log_queue.put(f"❌ ERROR: {exc}")
            log_queue.put("__DONE__")


def run_dl_inference(
    image_paths: list[str],
    model_path: str,
    model_name: str,
    num_classes: int,
) -> list[dict]:
    """
    Load a saved .pth/.pt checkpoint and run inference + GradCAM on each image.
    model_name must be a valid key from models.py (auto-resolved by get_saved_models).
    Returns a list of dicts per image.
    """
    import warnings
    import torch.nn.functional as F
    from PIL import Image
    import numpy as np
    warnings.filterwarnings("ignore")

    if DL_SRC not in sys.path:
        sys.path.insert(0, DL_SRC)

    from models import get_classification_model
    from gradcam import GradCAM, get_target_layer_name, apply_colormap_on_image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load model ─────────────────────────────────────────────────────────────
    model = get_classification_model(
        model_name, num_classes=num_classes, fine_tune_mode="full"
    )
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    target_layer = get_target_layer_name(model_name)
    grad_cam     = GradCAM(model, target_layer)

    results = []
    try:
        for img_path in image_paths:
            img_pil = Image.open(img_path).convert("RGB")
            tensor  = transform(img_pil).unsqueeze(0).to(device)

            cam, pred_class, model_output = grad_cam.generate_cam(tensor)
            probs      = F.softmax(model_output, dim=1)
            confidence = probs[0, pred_class].item()
            all_probs  = probs[0].detach().cpu().numpy().tolist()

            class_confidences = {
                BIN_LABELS.get(i, f"Bin {i}"): round(p * 100, 2)
                for i, p in enumerate(all_probs)
            }

            img_np    = tensor[0].cpu().permute(1, 2, 0).numpy()
            mean      = np.array([0.485, 0.456, 0.406])
            std       = np.array([0.229, 0.224, 0.225])
            img_np    = np.clip(img_np * std + mean, 0, 1)
            img_np_u8 = (img_np * 255).astype(np.uint8)

            superimposed, heatmap = apply_colormap_on_image(img_np_u8, cam)

            results.append({
                "filename":          os.path.basename(img_path),
                "predicted_bin":     pred_class,
                "label":             BIN_LABELS.get(pred_class, f"Bin {pred_class}"),
                "confidence":        confidence,
                "class_confidences": class_confidences,
                "original_np":       img_np_u8,
                "heatmap_np":        heatmap,
                "superimposed_np":   superimposed,
                "cam_np":            cam,
            })
    finally:
        # Always clean up hooks to prevent accumulation across multiple calls
        grad_cam.cleanup()

    return results
