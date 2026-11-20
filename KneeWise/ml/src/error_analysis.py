"""
error_analysis.py - In-depth Error Analysis & Validation Diagnostic Engine
Analyzes baseline ResNet18 performance, adjacent-grade misclassifications,
class distribution across splits, and validation set threshold tuning.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, f1_score, precision_score, recall_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    BEST_MODEL_PATH,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    BINARY_DEFINITE_GRADES,
    BINARY_UNLIKELY_GRADES,
    get_device,
)
from dataset import create_dataloaders
from model import build_model, load_checkpoint


def analyze_split_distributions():
    """Inspects class counts and proportions across train, val, and test manifests."""
    train_df = pd.read_csv(TRAIN_MANIFEST_CSV)
    val_df = pd.read_csv(VAL_MANIFEST_CSV)
    test_df = pd.read_csv(TEST_MANIFEST_CSV)

    summary = []
    for g in KL_CLASSES:
        tr_cnt = int((train_df["kl_grade"] == g).sum())
        va_cnt = int((val_df["kl_grade"] == g).sum())
        te_cnt = int((test_df["kl_grade"] == g).sum())
        tot = tr_cnt + va_cnt + te_cnt

        summary.append({
            "grade": f"KL {g}",
            "train_count": tr_cnt,
            "train_pct": f"{tr_cnt / len(train_df) * 100:.2f}%",
            "val_count": va_cnt,
            "val_pct": f"{va_cnt / len(val_df) * 100:.2f}%",
            "test_count": te_cnt,
            "test_pct": f"{te_cnt / len(test_df) * 100:.2f}%",
            "total_count": tot,
            "total_pct": f"{tot / (len(train_df) + len(val_df) + len(test_df)) * 100:.2f}%",
        })

    return pd.DataFrame(summary), len(train_df), len(val_df), len(test_df)


@torch.no_grad()
def get_predictions_and_probabilities(model, loader, device):
    """Gathers all targets, predicted classes, and full probability distributions."""
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

    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def plot_annotated_matrix(cm, classes, title, save_path, fmt="d", normalize=False):
    """Plots and saves confusion matrix with formatting."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Radiographic Reference Label",
        xlabel="AI Model Predicted Grade",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.1%}" if normalize else f"{val:d}"
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if val > thresh else "black",
                fontweight="bold" if i == j else "normal",
                fontsize=9,
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def compute_adjacent_grade_metrics(true_labels, pred_labels):
    """
    Computes off-by-zero, off-by-one, off-by-two, and off-by-three+ distances.
    In Kellgren-Lawrence grading, off-by-one is widely considered standard clinical observer variability.
    """
    diffs = np.abs(true_labels - pred_labels)
    total = len(true_labels)

    exact = int((diffs == 0).sum())
    off_by_1 = int((diffs == 1).sum())
    off_by_2 = int((diffs == 2).sum())
    off_by_3_plus = int((diffs >= 3).sum())

    within_one = exact + off_by_1

    return {
        "total": total,
        "exact_match_pct": (exact / total) * 100,
        "exact_match_count": exact,
        "off_by_1_pct": (off_by_1 / total) * 100,
        "off_by_1_count": off_by_1,
        "off_by_2_pct": (off_by_2 / total) * 100,
        "off_by_2_count": off_by_2,
        "off_by_3_plus_pct": (off_by_3_plus / total) * 100,
        "off_by_3_plus_count": off_by_3_plus,
        "within_1_grade_accuracy_pct": (within_one / total) * 100,
    }


def analyze_thresholds_on_validation(val_targets, val_probs):
    """
    Evaluates P(OA) = P(KL2) + P(KL3) + P(KL4) across thresholds on VALIDATION set.
    Never uses test labels!
    """
    true_binary = np.isin(val_targets, BINARY_DEFINITE_GRADES).astype(int)
    p_oa = val_probs[:, BINARY_DEFINITE_GRADES].sum(axis=1)

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    records = []

    for th in thresholds:
        pred_binary = (p_oa >= th).astype(int)
        cm = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(true_binary)

        records.append({
            "threshold": th,
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1": f1,
            "accuracy": acc,
            "false_negatives": fn,
            "false_positives": fp,
            "true_positives": tp,
            "true_negatives": tn,
        })

    return pd.DataFrame(records)


def main():
    device = get_device()
    output_dir = Path("KneeWise/ml/outputs/error_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("      KneeWise — Detailed Baseline Error Analysis & Inspection")
    print("=" * 70)

    # 1. Dataset Class Distribution
    dist_df, n_train, n_val, n_test = analyze_split_distributions()
    print("\n[*] Dataset Class Distribution Across Splits:")
    print(dist_df.to_string(index=False))

    # 2. Load Checkpoint and Test Dataloaders
    loaders = create_dataloaders(
        train_manifest=TRAIN_MANIFEST_CSV,
        val_manifest=VAL_MANIFEST_CSV,
        test_manifest=TEST_MANIFEST_CSV,
        batch_size=8,
    )

    model = build_model(pretrained=False)
    checkpoint = load_checkpoint(BEST_MODEL_PATH, model, device=device)
    model.to(device)

    print(f"\n[+] Loaded Best Baseline Checkpoint from Epoch {checkpoint['epoch']}")
    print(f"    Validation Loss: {checkpoint['val_loss']:.4f} | Validation Macro F1: {checkpoint['val_macro_f1']:.4f}")

    # Gather test predictions
    test_targets, test_preds, test_probs = get_predictions_and_probabilities(model, loaders["test"], device)
    val_targets, val_preds, val_probs = get_predictions_and_probabilities(model, loaders["val"], device)

    # 3. Five-Class Confusion Matrices
    cm_raw = confusion_matrix(test_targets, test_preds, labels=KL_CLASSES)
    cm_norm = cm_raw.astype(float) / cm_raw.sum(axis=1)[:, np.newaxis]

    plot_annotated_matrix(
        cm_raw,
        classes=[f"KL{c}" for c in KL_CLASSES],
        title="KneeWise Baseline Test Confusion Matrix (Counts)",
        save_path=output_dir / "cm_test_raw.png",
        fmt="d",
        normalize=False,
    )

    plot_annotated_matrix(
        cm_norm,
        classes=[f"KL{c}" for c in KL_CLASSES],
        title="KneeWise Baseline Test Confusion Matrix (Normalized Recall %)",
        save_path=output_dir / "cm_test_normalized.png",
        fmt=".1%",
        normalize=True,
    )

    # 4. Adjacent-Grade Distance Metrics
    adj_metrics = compute_adjacent_grade_metrics(test_targets, test_preds)
    print("\n[*] Adjacent-Grade Misclassification Analysis (Test Set):")
    print(f"  Exact Match (KL Grade agreement):       {adj_metrics['exact_match_pct']:.2f}% ({adj_metrics['exact_match_count']}/{adj_metrics['total']})")
    print(f"  Off-by-1 Grade (Adjacent agreement):     {adj_metrics['off_by_1_pct']:.2f}% ({adj_metrics['off_by_1_count']}/{adj_metrics['total']})")
    print(f"  Within +/- 1 KL Grade Clinical Window:   {adj_metrics['within_1_grade_accuracy_pct']:.2f}%")
    print(f"  Off-by-2 Grades:                         {adj_metrics['off_by_2_pct']:.2f}% ({adj_metrics['off_by_2_count']}/{adj_metrics['total']})")
    print(f"  Off-by-3 or more (Severe Error):         {adj_metrics['off_by_3_plus_pct']:.2f}% ({adj_metrics['off_by_3_plus_count']}/{adj_metrics['total']})")

    # 5. KL 1 and KL 2 Deep Dive
    print("\n[*] KL 1 and KL 2 Error Breakdown:")
    kl1_mask = (test_targets == 1)
    kl1_preds = test_preds[kl1_mask]
    print(f"  KL Grade 1 (Total {len(kl1_preds)} images):")
    for g in KL_CLASSES:
        cnt = (kl1_preds == g).sum()
        pct = cnt / len(kl1_preds) * 100
        print(f"    Predicted as KL {g}: {cnt:3d} ({pct:5.1f}%)")

    kl2_mask = (test_targets == 2)
    kl2_preds = test_preds[kl2_mask]
    print(f"\n  KL Grade 2 (Total {len(kl2_preds)} images):")
    for g in KL_CLASSES:
        cnt = (kl2_preds == g).sum()
        pct = cnt / len(kl2_preds) * 100
        print(f"    Predicted as KL {g}: {cnt:3d} ({pct:5.1f}%)")

    # 6. Validation Set Binary Threshold Tuning (Experiment C Prep)
    print("\n[*] Evaluating Binary Screening Thresholds on VALIDATION Set (P(OA) = P(KL2)+P(KL3)+P(KL4)):")
    val_th_df = analyze_thresholds_on_validation(val_targets, val_probs)
    print(val_th_df[["threshold", "sensitivity", "specificity", "precision", "f1", "false_negatives", "false_positives"]].to_string(index=False))

    val_th_df.to_csv(output_dir / "validation_threshold_sweep.csv", index=False)

    # Save summary report
    with open(output_dir / "error_analysis_report.json", "w") as f:
        json.dump({
            "distribution": dist_df.to_dict(orient="records"),
            "adjacent_metrics": adj_metrics,
            "kl_1_confusion": {f"pred_{g}": int((kl1_preds == g).sum()) for g in KL_CLASSES},
            "kl_2_confusion": {f"pred_{g}": int((kl2_preds == g).sum()) for g in KL_CLASSES},
            "validation_thresholds": val_th_df.to_dict(orient="records"),
        }, f, indent=2)

    print(f"\n[+] Error analysis report and plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
