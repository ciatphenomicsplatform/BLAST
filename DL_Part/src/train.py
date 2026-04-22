import os
import importlib.util
import time
import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms

from dataset import ImageClassificationDataset, collate_fn_skip_none
from models import get_classification_model, count_parameters
from utils import EarlyStopping

# ── Load DL config by absolute path (avoids sys.path collision with ML config) ──
_cfg_spec = importlib.util.spec_from_file_location(
    "dl_config",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.py"),
)
_cfg = importlib.util.module_from_spec(_cfg_spec)
_cfg_spec.loader.exec_module(_cfg)
device       = _cfg.device
NUM_CLASSES  = _cfg.NUM_CLASSES
IMAGE_FOLDER = _cfg.IMAGE_FOLDER
CSV_FILE     = _cfg.CSV_FILE

# ── MLflow URI resolved relative to project root ──────────────────────────────
_BLAST_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
)
_MLFLOW_URI = "sqlite:///" + os.path.join(_BLAST_ROOT, "mlruns.db")

def train_and_evaluate_model(model_name, model, train_loader, val_loader, device,
                           max_epochs=2, patience=1, lr=0.001,
                           run_name=None, extra_tags=None):
    """Train and evaluate a classification model with MLflow Tracking"""
    print(f"\n--- Training {model_name} ---")
    print(f"Trainable parameters: {count_parameters(model):,}")

    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)
    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    # Track metrics locally
    train_losses, val_losses, val_accuracies = [], [], []
    val_f1_scores, val_precisions, val_recalls = [], [], []

    start_time = time.time()

    # --- MLFLOW START ---
    with mlflow.start_run(run_name=run_name or model_name):
        if extra_tags:
            mlflow.set_tags(extra_tags)
        # Log all your hyperparameters
        mlflow.log_params({
            "model_name": model_name,
            "max_epochs": max_epochs,
            "patience": patience,
            "learning_rate": lr,
            "trainable_parameters": count_parameters(model)
        })

        for epoch in range(max_epochs):
            # Training phase
            model.train()
            running_loss = 0.0
            for images, targets in train_loader:
                if images is None or targets is None:
                    continue
                images, targets = images.to(device), targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            avg_train_loss = running_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # Validation phase
            model.eval()
            val_loss = 0.0
            all_preds, all_targets = [], []

            with torch.no_grad():
                for images, targets in val_loader:
                    if images is None or targets is None:
                        continue
                    images, targets = images.to(device), targets.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs, 1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_targets.extend(targets.cpu().numpy())

            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = accuracy_score(all_targets, all_preds)
            val_f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
            val_recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)

            val_losses.append(avg_val_loss)
            val_accuracies.append(val_accuracy)
            val_f1_scores.append(val_f1)
            val_precisions.append(val_precision)
            val_recalls.append(val_recall)

            # --- MLFLOW LOG METRICS PER EPOCH ---
            mlflow.log_metrics({
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "val_accuracy": val_accuracy,
                "val_f1": val_f1,
                "val_precision": val_precision,
                "val_recall": val_recall
            }, step=epoch)

            # Emit a machine-readable progress token (parsed by the Streamlit page)
            print(f"__EPOCH__ {epoch+1}/{max_epochs}")

            if epoch % 5 == 0 or epoch == max_epochs - 1:
                print(f"Epoch [{epoch+1}/{max_epochs}]  "
                      f"Train Loss: {avg_train_loss:.4f}  "
                      f"Val Loss: {avg_val_loss:.4f}  "
                      f"Acc: {val_accuracy:.4f}  "
                      f"F1: {val_f1:.4f}")

            if early_stopping(avg_val_loss, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        training_time = time.time() - start_time
        mlflow.log_metric("total_training_time_seconds", training_time)

        # Final evaluation
        model.eval()
        final_preds, final_targets = [], []
        with torch.no_grad():
            for images, targets in val_loader:
                if images is None or targets is None:
                    continue
                images, targets = images.to(device), targets.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                final_preds.extend(predicted.cpu().numpy())
                final_targets.extend(targets.cpu().numpy())

        final_accuracy = accuracy_score(final_targets, final_preds)
        final_f1 = f1_score(final_targets, final_preds, average='weighted', zero_division=0)
        final_precision = precision_score(final_targets, final_preds, average='weighted', zero_division=0)
        final_recall = recall_score(final_targets, final_preds, average='weighted', zero_division=0)
        conf_matrix = confusion_matrix(final_targets, final_preds)

        # --- MLFLOW LOG FINAL MODEL & METRICS ---
        mlflow.log_metrics({
            "final_accuracy":  final_accuracy,
            "final_f1":        final_f1,
            "final_precision": final_precision,
            "final_recall":    final_recall,
        })
        
        # Save the actual PyTorch model weights directly into MLflow!
        mlflow.pytorch.log_model(model, "model")

        return {
            'model': model,
            'final_accuracy': final_accuracy,
            'final_f1': final_f1,
            'final_precision': final_precision,
            'final_recall': final_recall,
            'confusion_matrix': conf_matrix,
            'training_time': training_time,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracies': val_accuracies,
            'val_f1_scores': val_f1_scores,
            'val_precisions': val_precisions,
            'val_recalls': val_recalls,
            'trainable_params': count_parameters(model),
            'epochs_trained': len(train_losses),
            'classification_report': classification_report(
                                         final_targets, final_preds,
                                         target_names=[f"Class {i}" for i in sorted(set(final_targets))],
                                         labels=sorted(set(final_targets)),
                                         zero_division=0),
            'mlflow_run_id': mlflow.active_run().info.run_id,
        }

def main():
    print(f"Using device: {device}")
    
    # Initialize MLFlow
    mlflow.set_tracking_uri(_MLFLOW_URI)
    mlflow.set_experiment("AgriBlast_Transfer_Learning")

    # Define transforms
    img_size = 224
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading data...")
    full_dataset = ImageClassificationDataset(IMAGE_FOLDER, CSV_FILE, transform=transform)

    # Train/Val Split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Now that we are in a .py file, we can use num_workers > 0 on Mac without it crashing!
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=4, collate_fn=collate_fn_skip_none)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=4, collate_fn=collate_fn_skip_none)

    # Example: Train a single model
    model_name = "efficientnet_v2_s"
    print(f"Initializing {model_name}...")
    model = get_classification_model(model_name, NUM_CLASSES, fine_tune_mode="full")
    
    results = train_and_evaluate_model(
        model_name=model_name, 
        model=model, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        device=device,
        max_epochs=1, 
        patience=1, 
        lr=0.001
    )

    print(f"Training complete! Final F1: {results['final_f1']:.4f}")

if __name__ == "__main__":
    main()
