#!/usr/bin/env python3
"""
train.py - KneeWise ML Model Training Pipeline

Implements training & validation loops, learning rate scheduling,
early stopping, mixed precision training (AMP), class-weighted CrossEntropyLoss,
checkpoint management, and performance visualization for ResNet18.
"""

import os
import sys
import argparse
import time
import json
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

# Ensure matplotlib writes cache to temp
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    DEFAULT_ARCHITECTURE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_EPOCHS,
    DEFAULT_LR,
    DEFAULT_WEIGHT_DECAY,
    EARLY_STOPPING_PATIENCE,
    RANDOM_SEED,
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    BEST_MODEL_PATH,
    LATEST_CHECKPOINT_PATH,
    TRAINING_OUTPUT_DIR,
    KL_CLASSES,
    get_device,
    use_mixed_precision,
)
from dataset import create_dataloaders
from model import build_model, save_checkpoint


def set_seed(seed: int = RANDOM_SEED):
    """Sets random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def calculate_class_weights(manifest_csv: Path, device: torch.device) -> Optional[torch.Tensor]:
    """
    Computes balanced class weights for CrossEntropyLoss to address class imbalance.
    """
    df = pd.read_csv(manifest_csv)
    labels = df["kl_grade"].values
    classes = np.array(KL_CLASSES)

    # Compute balanced weights
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print("\n[*] Computed Balanced Class Weights:")
    for grade, w in zip(KL_CLASSES, weights):
        print(f"  Class {grade}: {w:.4f}")
    return weight_tensor


def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Runs a single training epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Train]", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


@torch.no_grad()
def evaluate_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Tuple[float, float, float]:
    """Runs validation evaluation for one epoch."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Epoch {epoch:02d} [Val  ]", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_targets, all_preds)
    epoch_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return epoch_loss, epoch_acc, epoch_f1


def plot_training_curves(history: pd.DataFrame, save_path: Path):
    """Plots and saves loss, accuracy, and macro F1 curves."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    axes[0].plot(history["epoch"], history["train_loss"], label="Train Loss", marker="o", color="#1f77b4")
    axes[0].plot(history["epoch"], history["val_loss"], label="Val Loss", marker="s", color="#ff7f0e")
    axes[0].set_title("CrossEntropy Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["epoch"], history["train_acc"], label="Train Acc", marker="o", color="#2ca02c")
    axes[1].plot(history["epoch"], history["val_acc"], label="Val Acc", marker="s", color="#d62728")
    axes[1].set_title("5-Class Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Macro F1
    axes[2].plot(history["epoch"], history["train_f1"], label="Train Macro F1", marker="o", color="#9467bd")
    axes[2].plot(history["epoch"], history["val_f1"], label="Val Macro F1", marker="s", color="#8c564b")
    axes[2].set_title("Macro F1-Score")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("F1 Score")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved training curves to: {save_path}")


def train_pipeline(
    train_manifest: Path = TRAIN_MANIFEST_CSV,
    val_manifest: Path = VAL_MANIFEST_CSV,
    architecture: str = DEFAULT_ARCHITECTURE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    epochs: int = DEFAULT_EPOCHS,
    lr: float = DEFAULT_LR,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    use_class_weights: bool = True,
    freeze_backbone: bool = False,
    patience: int = EARLY_STOPPING_PATIENCE,
    output_dir: Path = TRAINING_OUTPUT_DIR,
):
    """Executes the complete end-to-end model training workflow."""
    set_seed(RANDOM_SEED)
    device = get_device()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — Model Training Execution           ")
    print("=" * 65)
    print(f"[*] Target Architecture:   {architecture}")
    print(f"[*] Device:                {device}")
    print(f"[*] Batch Size:            {batch_size}")
    print(f"[*] Max Epochs:            {epochs}")
    print(f"[*] Initial Learning Rate: {lr}")
    print(f"[*] Freeze Backbone:       {freeze_backbone}")

    # Check manifests
    if not Path(train_manifest).exists() or not Path(val_manifest).exists():
        print("\n[!] Error: Training or Validation manifest not found!")
        print("Please run 'python src/preprocess.py' first to build manifests.")
        return

    # DataLoaders
    loaders = create_dataloaders(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        batch_size=batch_size,
    )
    print(f"[+] DataLoaders ready: {len(loaders['train'].dataset):,} train, {len(loaders['val'].dataset):,} val")

    # Build model
    model = build_model(
        architecture=architecture,
        num_classes=len(KL_CLASSES),
        pretrained=True,
        freeze_backbone=freeze_backbone,
    )
    model.to(device)

    trainable_p, total_p = model.count_trainable_parameters()
    print(f"[+] Model Parameters: {trainable_p:,} trainable / {total_p:,} total")

    # Loss Criterion
    if use_class_weights:
        weights = calculate_class_weights(train_manifest, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    # Mixed Precision Scaler
    enable_amp = use_mixed_precision() and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if enable_amp else None
    print(f"[*] Mixed Precision (FP16 AMP): {'ENABLED' if enable_amp else 'DISABLED'}")

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    epochs_no_improve = 0

    training_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_loss, train_acc, train_f1 = train_one_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            epoch=epoch,
        )

        val_loss, val_acc, val_f1 = evaluate_epoch(
            model=model,
            loader=loaders["val"],
            criterion=criterion,
            device=device,
            epoch=epoch,
        )

        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]["lr"]
        epoch_dur = time.time() - epoch_start

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "train_f1": train_f1,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_f1,
            "lr": current_lr,
            "time_sec": epoch_dur,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{epochs:02d} [{epoch_dur:.1f}s] | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.3f} F1: {train_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | LR: {current_lr:.6f}"
        )

        # Checkpoint Best Model (based on Val Macro F1)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            epochs_no_improve = 0

            config_meta = {
                "architecture": architecture,
                "input_size": [224, 224],
                "batch_size": batch_size,
                "epochs": epochs,
                "best_epoch": best_epoch,
                "lr": lr,
                "num_classes": 5,
            }
            save_checkpoint(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_f1,
                config=config_meta,
                save_path=BEST_MODEL_PATH,
            )
            print(f"  [>>>] New Best Checkpoint Saved! (Val Macro F1: {val_f1:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered: No validation improvement for {patience} consecutive epochs.")
                break

    total_time = time.time() - training_start
    print("\n" + "=" * 65)
    print(f"[+] Training Completed in {total_time / 60:.2f} minutes!")
    print(f"[+] Best Validation Macro F1: {best_val_f1:.4f} (at Epoch {best_epoch})")
    print(f"[+] Best Model Checkpoint:     {BEST_MODEL_PATH}")
    print("=" * 65)

    # Save History CSV
    history_df = pd.DataFrame(history)
    history_csv_path = output_dir / "training_history.csv"
    history_df.to_csv(history_csv_path, index=False)
    print(f"[+] Training history saved to: {history_csv_path}")

    # Plot Curves
    plot_training_curves(history_df, output_dir / "training_curves.png")


def main():
    parser = argparse.ArgumentParser(description="Train KneeWise Osteoarthritis Severity Model")
    parser.add_argument("--architecture", type=str, default=DEFAULT_ARCHITECTURE)
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight_decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--no_class_weights", action="store_true")
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    args = parser.parse_args()

    train_pipeline(
        architecture=args.architecture,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_class_weights=not args.no_class_weights,
        freeze_backbone=args.freeze_backbone,
        patience=args.patience,
    )


if __name__ == "__main__":
    main()
