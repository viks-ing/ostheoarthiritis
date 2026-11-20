"""
phase4_experiments.py - KneeWise Phase 4 Targeted Model Improvement Suite
Experiments:
- Exp E: EfficientNet-B0 with Targeted Radiographic Augmentation (sharpness + micro-scaling)
- Exp F: EfficientNet-B0 with Focal Loss (gamma=1.5, class-weighted alpha)
- Exp G: EfficientNet-B0 with KL1/KL2-Focused Sampling
"""

import os
import sys
import time
import argparse
import random
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import matplotlib.pyplot as plt

# Matplotlib temp directory
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    BINARY_DEFINITE_GRADES,
    BINARY_UNLIKELY_GRADES,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_BATCH_SIZE,
    RANDOM_SEED,
    get_device,
    use_mixed_precision,
    use_pinned_memory,
)
from dataset import KneeXRayDataset, get_val_test_transforms
from model import KneeWiseClassifier, save_checkpoint, load_checkpoint


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# -------------------------------------------------------------
# Targeted Augmentations for Experiment E
# -------------------------------------------------------------
def get_targeted_train_transforms():
    """
    Experiment E: Targeted radiographic augmentation.
    Features:
    - Micro-rotation (+/- 5 deg) and micro-translation (+/- 3%)
    - Micro-scaling (0.96 - 1.04) simulating focal distance differences
    - Subtle sharpness adjustment (1.4x) highlighting cortical margins & osteophytes
    - Mild brightness & contrast shifts (12%)
    - Bilateral horizontal flip (50%)
    Strictly avoids severe distortions that alter apparent joint space width.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.03, 0.03),
            scale=(0.96, 1.04),
        ),
        transforms.RandomAdjustSharpness(sharpness_factor=1.4, p=0.4),
        transforms.ColorJitter(brightness=0.12, contrast=0.12),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# -------------------------------------------------------------
# Focal Loss Implementation for Experiment F
# -------------------------------------------------------------
class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss with class weighting:
    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    Down-weights well-classified easy samples (KL0) and focuses gradients on hard cases (KL1, KL2).
    """
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 1.5, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# -------------------------------------------------------------
# Targeted Sampler for Experiment G
# -------------------------------------------------------------
def create_kl_focused_sampler(manifest_csv: Path) -> WeightedRandomSampler:
    """
    Derives training sampling weights boosting KL1 and KL2 representations.
    Derived strictly from training manifest labels.
    """
    df = pd.read_csv(manifest_csv)
    labels = df["kl_grade"].values
    counts = np.bincount(labels, minlength=5)

    # Base inverse frequency weights
    weights = 1.0 / np.maximum(counts, 1).astype(float)
    # Give a controlled 1.4x boost to KL1 and KL2
    weights[1] *= 1.4
    weights[2] *= 1.4
    # Normalize
    weights = weights / weights.sum()

    sample_weights = np.array([weights[y] for y in labels], dtype=np.float32)
    return WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Ep {epoch:02d} [Trn]", leave=False)
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

    ep_loss = running_loss / len(loader.dataset)
    ep_acc = accuracy_score(all_targets, all_preds)
    ep_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return ep_loss, ep_acc, ep_f1


@torch.no_grad()
def evaluate_validation(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    pbar = tqdm(loader, desc=f"Ep {epoch:02d} [Val]", leave=False)
    for images, targets, _ in pbar:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        running_loss += loss.item() * images.size(0)
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.detach().cpu().numpy())

    ep_loss = running_loss / len(loader.dataset)
    ep_acc = accuracy_score(all_targets, all_preds)
    ep_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    per_f1 = f1_score(all_targets, all_preds, average=None, labels=KL_CLASSES, zero_division=0)
    return ep_loss, ep_acc, ep_f1, per_f1


@torch.no_grad()
def full_evaluation(model, loader, device, threshold: float = 0.30):
    model.eval()
    all_targets = []
    all_probs = []

    for images, targets, _ in loader:
        images = images.to(device)
        probs = model.predict_probabilities(images)
        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    all_preds = np.argmax(all_probs, axis=1)

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_targets, all_preds, average="weighted", zero_division=0)
    prec_per = precision_score(all_targets, all_preds, average=None, labels=KL_CLASSES, zero_division=0)
    rec_per = recall_score(all_targets, all_preds, average=None, labels=KL_CLASSES, zero_division=0)
    f1_per = f1_score(all_targets, all_preds, average=None, labels=KL_CLASSES, zero_division=0)
    cm_5c = confusion_matrix(all_targets, all_preds, labels=KL_CLASSES)

    # Binary metrics
    true_bin = np.isin(all_targets, BINARY_DEFINITE_GRADES).astype(int)
    prob_oa = all_probs[:, BINARY_DEFINITE_GRADES].sum(axis=1)
    pred_bin = (prob_oa >= threshold).astype(int)
    cm_bin = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm_bin.ravel()

    roc_auc = roc_auc_score(true_bin, prob_oa)
    pr_auc = average_precision_score(true_bin, prob_oa)
    brier = brier_score_loss(true_bin, prob_oa)

    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    prec_bin = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    bin_f1 = 2 * (prec_bin * sens) / (prec_bin + sens) if (prec_bin + sens) > 0 else 0.0

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "precision_per_class": prec_per.tolist(),
        "recall_per_class": rec_per.tolist(),
        "f1_per_class": f1_per.tolist(),
        "cm_5class": cm_5c.tolist(),
        "binary_roc_auc": roc_auc,
        "binary_pr_auc": pr_auc,
        "brier_score": brier,
        "threshold": threshold,
        "binary_sensitivity": sens,
        "binary_specificity": spec,
        "binary_precision": prec_bin,
        "binary_f1": bin_f1,
        "false_negatives": int(fn),
        "false_positives": int(fp),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    }


def run_phase4_experiment(
    exp_id: str,
    augmentation_type: str = "standard",  # "standard" or "targeted"
    loss_type: str = "weighted_ce",       # "weighted_ce" or "focal"
    use_focused_sampler: bool = False,
    epochs: int = 6,
    patience: int = 2,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lr: float = 1e-4,
) -> Dict[str, Any]:
    set_seed(RANDOM_SEED)
    device = get_device()
    output_dir = Path("KneeWise/ml/outputs/training/phase_4") / exp_id
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = Path("KneeWise/ml/models/phase_4") / f"{exp_id}.pth"

    print("\n" + "=" * 75)
    print(f"      STARTING PHASE 4 EXPERIMENT: {exp_id}")
    print("=" * 75)
    print(f" Architecture:        EfficientNet-B0 (Compound Scaled, 5.3M params)")
    print(f" Augmentation:        {augmentation_type}")
    print(f" Loss Function:       {loss_type}")
    print(f" KL-Focused Sampler:  {use_focused_sampler}")
    print(f" Learning Rate:       {lr}")
    print(f" Batch Size / Epochs: {batch_size} / {epochs} (Patience: {patience})")
    print(f" Target Checkpoint:   {ckpt_path}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Transforms
    if augmentation_type == "targeted":
        train_transform = get_targeted_train_transforms()
    else:
        from dataset import get_train_transforms
        train_transform = get_train_transforms()

    val_transform = get_val_test_transforms()

    # Dataloaders
    val_ds = KneeXRayDataset(VAL_MANIFEST_CSV, transform=val_transform)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pinned_memory())

    train_ds = KneeXRayDataset(TRAIN_MANIFEST_CSV, transform=train_transform)
    if use_focused_sampler:
        sampler = create_kl_focused_sampler(TRAIN_MANIFEST_CSV)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=use_pinned_memory())
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pinned_memory())

    # Build Model
    model = KneeWiseClassifier(backbone_name="efficientnet_b0", num_classes=len(KL_CLASSES), pretrained=True)
    model.to(device)

    # Class weights from train manifest
    df_trn = pd.read_csv(TRAIN_MANIFEST_CSV)
    labels = df_trn["kl_grade"].values
    weights = compute_class_weight("balanced", classes=np.array(KL_CLASSES), y=labels)
    weight_tensor = torch.tensor(weights, dtype=torch.float32, device=device)

    # Loss criterion
    if loss_type == "focal":
        criterion = FocalLoss(alpha=weight_tensor, gamma=1.5)
        val_criterion = FocalLoss(alpha=weight_tensor, gamma=1.5)
    else:
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
        val_criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    enable_amp = use_mixed_precision() and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if enable_amp else None

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_one_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_f1, val_per_f1 = evaluate_validation(model, val_loader, val_criterion, device, epoch)

        scheduler.step(val_f1)
        cur_lr = optimizer.param_groups[0]["lr"]
        dur = time.time() - ep_t0

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "val_f1_kl0": val_per_f1[0], "val_f1_kl1": val_per_f1[1],
            "val_f1_kl2": val_per_f1[2], "val_f1_kl3": val_per_f1[3],
            "val_f1_kl4": val_per_f1[4],
            "lr": cur_lr, "time_sec": dur,
        })

        print(
            f"Ep {epoch:02d}/{epochs:02d} [{dur:.1f}s] | "
            f"Trn Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} F1: {tr_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} (KL1: {val_per_f1[1]:.3f}, KL2: {val_per_f1[2]:.3f}) | LR: {cur_lr:.6f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            no_improve = 0
            save_checkpoint(
                model=model,
                epoch=epoch,
                optimizer=optimizer,
                scheduler=scheduler,
                val_loss=val_loss,
                val_acc=val_acc,
                val_macro_f1=val_f1,
                config={"exp_id": exp_id, "arch": "efficientnet_b0", "batch_size": batch_size, "lr": lr},
                save_path=ckpt_path,
            )
            print(f"  [>>>] New Best Checkpoint Saved! (Val Macro F1: {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(f"\n[+] Experiment {exp_id} completed in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"[+] Peak VRAM: {peak_vram:.1f} MB | Best Val Macro F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # Save history CSV
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "history.csv", index=False)

    # Plot Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title(f"{exp_id} Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist_df["epoch"], hist_df["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(hist_df["epoch"], hist_df["val_acc"], label="Val Acc", marker="s")
    axes[1].set_title(f"{exp_id} Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hist_df["epoch"], hist_df["val_f1"], label="Val Macro F1", marker="o", color="purple")
    axes[2].plot(hist_df["epoch"], hist_df["val_f1_kl1"], label="Val KL1 F1", marker="^", color="orange")
    axes[2].plot(hist_df["epoch"], hist_df["val_f1_kl2"], label="Val KL2 F1", marker="v", color="green")
    axes[2].set_title(f"{exp_id} Validation F1 Metrics")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "curves.png", dpi=200)
    plt.close()

    # Load best checkpoint for validation summary
    best_model = KneeWiseClassifier(backbone_name="efficientnet_b0", num_classes=5, pretrained=False)
    load_checkpoint(ckpt_path, best_model, device=device)
    best_model.to(device)

    # Validation Set Full Evaluation (Strict protocol: DO NOT touch test set yet!)
    val_metrics = full_evaluation(best_model, val_loader, device, threshold=0.30)

    summary = {
        "exp_id": exp_id,
        "architecture": "efficientnet_b0",
        "checkpoint_path": str(ckpt_path),
        "runtime_seconds": total_time,
        "peak_vram_mb": peak_vram,
        "best_epoch": best_epoch,
        "validation_metrics": val_metrics,
    }

    with open(output_dir / "validation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["E", "F", "G"])
    args = parser.parse_args()

    if args.exp == "E":
        run_phase4_experiment(
            exp_id="exp_e_targeted_aug",
            augmentation_type="targeted",
            loss_type="weighted_ce",
            use_focused_sampler=False,
            epochs=6,
            patience=2,
            batch_size=8,
            lr=1e-4,
        )
    elif args.exp == "F":
        run_phase4_experiment(
            exp_id="exp_f_focal_loss",
            augmentation_type="standard",
            loss_type="focal",
            use_focused_sampler=False,
            epochs=6,
            patience=2,
            batch_size=8,
            lr=1e-4,
        )
    elif args.exp == "G":
        run_phase4_experiment(
            exp_id="exp_g_focused_sampling",
            augmentation_type="standard",
            loss_type="weighted_ce",
            use_focused_sampler=True,
            epochs=6,
            patience=2,
            batch_size=8,
            lr=1e-4,
        )


if __name__ == "__main__":
    main()
