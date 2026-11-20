"""
experiments.py - Modular ML Experimentation Suite for KneeWise
Runs structured experiments:
- Exp A1: Unweighted CrossEntropyLoss
- Exp A3: WeightedRandomSampler with Unweighted CrossEntropyLoss
- Exp B: Fine-tuning with Layer-wise Discriminative Learning Rates
- Exp D: EfficientNet-B0 Architecture Comparison
"""

import os
import sys
import time
import argparse
import random
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    KL_CLASSES,
    BINARY_DEFINITE_GRADES,
    BINARY_UNLIKELY_GRADES,
    DEFAULT_BATCH_SIZE,
    RANDOM_SEED,
    get_device,
    use_mixed_precision,
    use_pinned_memory,
)
from dataset import (
    KneeXRayDataset,
    get_train_transforms,
    get_val_test_transforms,
)
from model import KneeWiseClassifier, save_checkpoint, load_checkpoint


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_sampler_dataloader(
    manifest_csv: Path,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Tuple[DataLoader, torch.utils.data.Dataset]:
    """Creates a DataLoader with WeightedRandomSampler for inverse frequency sampling."""
    df = pd.read_csv(manifest_csv)
    labels = df["kl_grade"].values
    class_counts = np.bincount(labels, minlength=len(KL_CLASSES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = np.array([class_weights[y] for y in labels], dtype=np.float32)

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.float),
        num_samples=len(sample_weights),
        replacement=True,
    )

    dataset = KneeXRayDataset(
        manifest=manifest_csv,
        transform=get_train_transforms(),
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=use_pinned_memory(),
    )
    return loader, dataset


def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
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
def evaluate_epoch(model, loader, criterion, device, epoch):
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
    return ep_loss, ep_acc, ep_f1


@torch.no_grad()
def full_test_evaluation(model, loader, device, threshold: float = 0.5):
    model.eval()
    all_targets = []
    all_preds = []
    all_probs = []

    for images, targets, _ in loader:
        images = images.to(device)
        probs = model.predict_probabilities(images)
        preds = torch.argmax(probs, dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_targets.extend(targets.numpy())
        all_preds.extend(preds.cpu().numpy())

    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    # 5-class metrics
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


def run_experiment(
    exp_name: str,
    arch: str = "resnet18",
    use_sampler: bool = False,
    use_class_weights: bool = False,
    discriminative_lr: bool = False,
    base_lr: float = 1e-4,
    backbone_lr: float = 1e-5,
    epochs: int = 8,
    patience: int = 3,
    batch_size: int = DEFAULT_BATCH_SIZE,
):
    set_seed(RANDOM_SEED)
    device = get_device()
    output_dir = Path("KneeWise/ml/outputs/experiments") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = Path("KneeWise/ml/models") / f"{exp_name}.pth"

    print("\n" + "=" * 70)
    print(f"      STARTING EXPERIMENT: {exp_name}")
    print("=" * 70)
    print(f" Architecture:        {arch}")
    print(f" Use Sampler:         {use_sampler}")
    print(f" Use Class Weights:   {use_class_weights}")
    print(f" Discriminative LR:   {discriminative_lr} (Backbone: {backbone_lr}, Head: {base_lr})")
    print(f" Max Epochs:          {epochs} (Patience: {patience})")
    print(f" Checkpoint Target:   {checkpoint_path}")

    # Reset peak memory stats
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Dataloaders
    val_dataset = KneeXRayDataset(VAL_MANIFEST_CSV, transform=get_val_test_transforms())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pinned_memory())

    if use_sampler:
        train_loader, _ = create_sampler_dataloader(TRAIN_MANIFEST_CSV, batch_size=batch_size)
    else:
        train_dataset = KneeXRayDataset(TRAIN_MANIFEST_CSV, transform=get_train_transforms())
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=use_pinned_memory())

    # Model
    model = KneeWiseClassifier(backbone_name=arch, num_classes=len(KL_CLASSES), pretrained=True)
    model.to(device)

    # Criterion
    if use_class_weights:
        df = pd.read_csv(TRAIN_MANIFEST_CSV)
        labels = df["kl_grade"].values
        weights = compute_class_weight("balanced", classes=np.array(KL_CLASSES), y=labels)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    if discriminative_lr:
        if arch == "resnet18":
            backbone_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("fc")]
            head_params = list(model.backbone.fc.parameters())
        elif arch == "efficientnet_b0":
            backbone_params = [p for n, p in model.backbone.named_parameters() if not n.startswith("classifier")]
            head_params = list(model.backbone.classifier.parameters())
        else:
            raise ValueError(f"Unknown arch: {arch}")

        param_groups = [
            {"params": backbone_params, "lr": backbone_lr},
            {"params": head_params, "lr": base_lr},
        ]
        optimizer = Adam(param_groups, weight_decay=1e-4)
    else:
        optimizer = Adam(model.parameters(), lr=base_lr, weight_decay=1e-4)

    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    enable_amp = use_mixed_precision() and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if enable_amp else None

    history = []
    best_val_f1 = -1.0
    best_epoch = 0
    no_improve = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        ep_t0 = time.time()
        tr_loss, tr_acc, tr_f1 = train_epoch(model, train_loader, criterion, optimizer, scaler, device, epoch)
        val_loss, val_acc, val_f1 = evaluate_epoch(model, val_loader, criterion, device, epoch)

        scheduler.step(val_f1)
        cur_lr = optimizer.param_groups[0]["lr"]
        dur = time.time() - ep_t0

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc, "train_f1": tr_f1,
            "val_loss": val_loss, "val_acc": val_acc, "val_f1": val_f1,
            "lr": cur_lr, "time_sec": dur,
        })

        print(
            f"Ep {epoch:02d}/{epochs:02d} [{dur:.1f}s] | "
            f"Trn Loss: {tr_loss:.4f} Acc: {tr_acc:.3f} F1: {tr_f1:.3f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.3f} F1: {val_f1:.3f} | LR: {cur_lr:.6f}"
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
                config={"exp_name": exp_name, "arch": arch, "batch_size": batch_size, "lr": base_lr},
                save_path=checkpoint_path,
            )
            print(f"  [>>>] New Best Checkpoint Saved! (Val Macro F1: {val_f1:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[!] Early stopping triggered at epoch {epoch}.")
                break

    total_time = time.time() - t0
    peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    print(f"\n[+] Experiment {exp_name} completed in {total_time:.1f}s ({total_time/60:.2f} min).")
    print(f"[+] Peak VRAM: {peak_vram:.1f} MB | Best Val Macro F1: {best_val_f1:.4f} (Epoch {best_epoch})")

    # Save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(output_dir / "history.csv", index=False)

    # Plot curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].plot(hist_df["epoch"], hist_df["train_loss"], label="Train Loss", marker="o")
    axes[0].plot(hist_df["epoch"], hist_df["val_loss"], label="Val Loss", marker="s")
    axes[0].set_title(f"{exp_name} Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(hist_df["epoch"], hist_df["train_acc"], label="Train Acc", marker="o")
    axes[1].plot(hist_df["epoch"], hist_df["val_acc"], label="Val Acc", marker="s")
    axes[1].set_title(f"{exp_name} Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(hist_df["epoch"], hist_df["train_f1"], label="Train F1", marker="o")
    axes[2].plot(hist_df["epoch"], hist_df["val_f1"], label="Val F1", marker="s")
    axes[2].set_title(f"{exp_name} Macro F1")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "curves.png", dpi=200)
    plt.close()

    # Now evaluate on untouched test set using the best saved checkpoint
    test_ds = KneeXRayDataset(TEST_MANIFEST_CSV, transform=get_val_test_transforms())
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=use_pinned_memory())

    best_model = KneeWiseClassifier(backbone_name=arch, num_classes=len(KL_CLASSES), pretrained=False)
    load_checkpoint(checkpoint_path, best_model, device=device)
    best_model.to(device)

    # Evaluate at default threshold 0.5 AND operating screening threshold 0.30
    test_results_05 = full_test_evaluation(best_model, test_loader, device, threshold=0.50)
    test_results_03 = full_test_evaluation(best_model, test_loader, device, threshold=0.30)

    summary = {
        "exp_name": exp_name,
        "arch": arch,
        "runtime_sec": total_time,
        "peak_vram_mb": peak_vram,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_metrics_default_th_0.50": test_results_05,
        "test_metrics_screening_th_0.30": test_results_03,
    }

    import json
    with open(output_dir / "test_evaluation.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[+] Test Evaluation Completed for {exp_name}:")
    print(f"    5-Class Accuracy: {test_results_05['accuracy']*100:.2f}% | Macro F1: {test_results_05['macro_f1']:.4f}")
    print(f"    Binary Sens (th=0.50): {test_results_05['binary_sensitivity']*100:.2f}% | Spec: {test_results_05['binary_specificity']*100:.2f}%")
    print(f"    Binary Sens (th=0.30): {test_results_03['binary_sensitivity']*100:.2f}% | Spec: {test_results_03['binary_specificity']*100:.2f}%")

    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, required=True, choices=["A1", "A3", "B", "D", "all"])
    args = parser.parse_args()

    results = {}

    if args.exp in ["A1", "all"]:
        # Exp A1: Unweighted CrossEntropyLoss
        results["A1"] = run_experiment(
            exp_name="exp_a1_unweighted_resnet18",
            arch="resnet18",
            use_sampler=False,
            use_class_weights=False,
            discriminative_lr=False,
            base_lr=1e-4,
            epochs=6,
            patience=2,
        )

    if args.exp in ["A3", "all"]:
        # Exp A3: WeightedRandomSampler (with unweighted CE)
        results["A3"] = run_experiment(
            exp_name="exp_a3_sampler_resnet18",
            arch="resnet18",
            use_sampler=True,
            use_class_weights=False,
            discriminative_lr=False,
            base_lr=1e-4,
            epochs=6,
            patience=2,
        )

    if args.exp in ["B", "all"]:
        # Exp B: Fine-tuning with layer-wise discriminative learning rate
        results["B"] = run_experiment(
            exp_name="exp_b_finetune_resnet18",
            arch="resnet18",
            use_sampler=False,
            use_class_weights=True,
            discriminative_lr=True,
            base_lr=1e-4,
            backbone_lr=1e-5,
            epochs=6,
            patience=2,
        )

    if args.exp in ["D", "all"]:
        # Exp D: EfficientNet-B0 Architecture Comparison
        results["D"] = run_experiment(
            exp_name="exp_d_efficientnet_b0",
            arch="efficientnet_b0",
            use_sampler=False,
            use_class_weights=True,
            discriminative_lr=False,
            base_lr=1e-4,
            epochs=6,
            patience=2,
        )


if __name__ == "__main__":
    main()
