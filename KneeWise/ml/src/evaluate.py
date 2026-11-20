#!/usr/bin/env python3
"""
evaluate.py - KneeWise ML Comprehensive Evaluation Engine

Evaluates trained KneeWise models on the untouched test set across:
1. Five-Class Kellgren-Lawrence severity grading (0, 1, 2, 3, 4)
2. Derived Binary Radiographic OA screening (OA unlikely vs Definite OA)

Generates confusion matrices, per-class metrics, clinical sensitivity/specificity,
and saves publication-ready evaluation reports to outputs/evaluation/.
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# Ensure matplotlib cache is in temp
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    TEST_MANIFEST_CSV,
    BEST_MODEL_PATH,
    EVALUATION_OUTPUT_DIR,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    BINARY_CLASSES,
    BINARY_UNLIKELY_GRADES,
    BINARY_DEFINITE_GRADES,
    DEFAULT_BATCH_SIZE,
    get_device,
)
from dataset import create_dataloaders
from model import build_model, load_checkpoint


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: list,
    title: str,
    save_path: Path,
    cmap: str = "Blues",
    fmt: str = "d",
):
    """Generates and saves an annotated confusion matrix plot without external seaborn dependency."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=classes,
        yticklabels=classes,
        title=title,
        ylabel="True Reference Label",
        xlabel="AI Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val_str = f"{cm[i, j]:{fmt}}"
            ax.text(
                j, i, val_str,
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontweight="bold" if i == j else "normal",
            )

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[+] Saved confusion matrix to: {save_path}")


def evaluate_test_set(
    model_path: Path = BEST_MODEL_PATH,
    test_manifest: Path = TEST_MANIFEST_CSV,
    output_dir: Path = EVALUATION_OUTPUT_DIR,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> Dict[str, Any]:
    """Evaluates the model on test set for both 5-class and derived binary tasks."""
    device = get_device()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — Test Set Evaluation Engine        ")
    print("=" * 65)

    if not Path(model_path).exists():
        print(f"\n[!] Model checkpoint not found at: {model_path}")
        print("Please train the model first using 'python src/train.py'.")
        return {}

    if not Path(test_manifest).exists():
        print(f"\n[!] Test manifest not found at: {test_manifest}")
        print("Please generate manifests first using 'python src/preprocess.py'.")
        return {}

    # Load test data
    loaders = create_dataloaders(
        train_manifest=test_manifest,  # dummy for signature
        val_manifest=test_manifest,    # dummy
        test_manifest=test_manifest,
        batch_size=batch_size,
    )
    test_loader = loaders["test"]

    # Load Model
    model = build_model(pretrained=False)
    checkpoint = load_checkpoint(model_path, model, device=device)
    model.to(device)
    model.eval()

    print(f"[+] Loaded weights from checkpoint (Epoch {checkpoint.get('epoch', 'N/A')})")
    print(f"[+] Testing on {len(test_loader.dataset):,} test images...\n")

    all_true_kl = []
    all_pred_kl = []
    all_probs = []

    with torch.no_grad():
        for images, targets, _ in test_loader:
            images = images.to(device)
            probs = model.predict_probabilities(images)
            preds = torch.argmax(probs, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_true_kl.extend(targets.numpy())
            all_pred_kl.extend(preds.cpu().numpy())

    all_true_kl = np.array(all_true_kl)
    all_pred_kl = np.array(all_pred_kl)
    all_probs = np.array(all_probs)

    # ---------------------------------------------------------
    # 1. Five-Class Severity Metrics
    # ---------------------------------------------------------
    acc_5c = accuracy_score(all_true_kl, all_pred_kl)
    macro_f1_5c = f1_score(all_true_kl, all_pred_kl, average="macro", zero_division=0)
    weighted_f1_5c = f1_score(all_true_kl, all_pred_kl, average="weighted", zero_division=0)
    macro_prec_5c = precision_score(all_true_kl, all_pred_kl, average="macro", zero_division=0)
    macro_rec_5c = recall_score(all_true_kl, all_pred_kl, average="macro", zero_division=0)

    per_class_prec = precision_score(all_true_kl, all_pred_kl, average=None, labels=KL_CLASSES, zero_division=0)
    per_class_rec = recall_score(all_true_kl, all_pred_kl, average=None, labels=KL_CLASSES, zero_division=0)
    per_class_f1 = f1_score(all_true_kl, all_pred_kl, average=None, labels=KL_CLASSES, zero_division=0)
    cm_5c = confusion_matrix(all_true_kl, all_pred_kl, labels=KL_CLASSES)

    # ---------------------------------------------------------
    # 2. Derived Binary OA Screening Metrics
    # ---------------------------------------------------------
    # True binary: 0 = OA unlikely (KL 0-1), 1 = Definite OA (KL 2-4)
    true_binary = np.isin(all_true_kl, BINARY_DEFINITE_GRADES).astype(int)

    # AI derived probability: P(Definite OA) = sum(P(KL 2, 3, 4))
    pred_prob_definite = all_probs[:, BINARY_DEFINITE_GRADES].sum(axis=1)
    pred_binary = (pred_prob_definite >= 0.5).astype(int)

    cm_binary = confusion_matrix(true_binary, pred_binary, labels=[0, 1])
    tn, fp, fn, tp = cm_binary.ravel()

    binary_acc = accuracy_score(true_binary, pred_binary)
    binary_prec = precision_score(true_binary, pred_binary, zero_division=0)
    binary_rec = recall_score(true_binary, pred_binary, zero_division=0)  # Sensitivity
    binary_f1 = f1_score(true_binary, pred_binary, zero_division=0)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # ---------------------------------------------------------
    # Output to Console
    # ---------------------------------------------------------
    print("=" * 65)
    print("          5-CLASS KELLGREN-LAWRENCE EVALUATION RESULTS          ")
    print("=" * 65)
    print(f"Overall Accuracy:  {acc_5c:.4f} ({acc_5c * 100:.2f}%)")
    print(f"Macro Precision:   {macro_prec_5c:.4f}")
    print(f"Macro Recall:      {macro_rec_5c:.4f}")
    print(f"Macro F1-Score:    {macro_f1_5c:.4f}")
    print(f"Weighted F1-Score: {weighted_f1_5c:.4f}\n")

    print("Per-Class Performance:")
    print("Grade | Precision | Recall | F1-Score | Support")
    print("-" * 50)
    for g in KL_CLASSES:
        sup = np.sum(all_true_kl == g)
        print(f"  {g}   |   {per_class_prec[g]:.3f}   |  {per_class_rec[g]:.3f} |  {per_class_f1[g]:.3f}   |  {sup:4d}")

    print("\n" + "=" * 65)
    print("       DERIVED BINARY SCREENING RESULTS (OA Unlikely vs Definite) ")
    print("=" * 65)
    print(f"Binary Accuracy:     {binary_acc:.4f} ({binary_acc * 100:.2f}%)")
    print(f"Clinical Sensitivity (Recall of OA): {sensitivity:.4f} ({sensitivity * 100:.2f}%)")
    print(f"Clinical Specificity:                {specificity:.4f} ({specificity * 100:.2f}%)")
    print(f"Binary Precision:                    {binary_prec:.4f}")
    print(f"Binary F1-Score:                     {binary_f1:.4f}")
    print(f"False Negatives (Missed OA):         {fn:d} patients")
    print(f"False Positives (Over-referred):     {fp:d} patients")
    print("=" * 65)

    # ---------------------------------------------------------
    # Save Visualizations
    # ---------------------------------------------------------
    plot_confusion_matrix(
        cm_5c,
        classes=[f"KL{c}" for c in KL_CLASSES],
        title="KneeWise 5-Class KL Confusion Matrix",
        save_path=output_dir / "confusion_matrix_5class.png",
        cmap="Blues",
    )

    plot_confusion_matrix(
        cm_binary,
        classes=["OA Unlikely", "Definite OA"],
        title="KneeWise Derived Binary OA Confusion Matrix",
        save_path=output_dir / "confusion_matrix_binary.png",
        cmap="Greens",
    )

    # Save Markdown Evaluation Report
    report_md = [
        "# KneeWise — Model Evaluation Report\n",
        "## 1. Five-Class Kellgren-Lawrence Severity Grading\n",
        f"- **Accuracy:** {acc_5c:.4f}",
        f"- **Macro F1:** {macro_f1_5c:.4f}",
        f"- **Weighted F1:** {weighted_f1_5c:.4f}\n",
        "### Per-Class Metrics Table\n",
        "| Grade | Description | Precision | Recall | F1 | Support |",
        "|---|---|---|---|---|---|",
    ]
    for g in KL_CLASSES:
        sup = int(np.sum(all_true_kl == g))
        report_md.append(
            f"| Grade {g} | {KL_GRADE_DESCRIPTIONS[g]} | {per_class_prec[g]:.3f} | {per_class_rec[g]:.3f} | {per_class_f1[g]:.3f} | {sup:,} |"
        )

    report_md.extend([
        "\n## 2. Derived Binary Screening Performance\n",
        f"- **Sensitivity (Recall of Definite OA):** {sensitivity:.4f} ({sensitivity * 100:.1f}%)",
        f"- **Specificity (Correct identification of Unlikely):** {specificity:.4f} ({specificity * 100:.1f}%)",
        f"- **Precision:** {binary_prec:.4f}",
        f"- **Binary F1:** {binary_f1:.4f}",
        f"- **False Negatives:** {fn} (Crucial for clinical screening)",
        f"- **False Positives:** {fp}\n",
        "## 3. Important Medical AI Considerations\n",
        "> [!IMPORTANT]\n",
        "> This system is an AI-assisted **screening support tool**, not an autonomous diagnostic system.\n",
        "> Predictions must be confirmed by certified radiologists or clinicians in conjunction with clinical symptoms.",
    ])

    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_md))
    print(f"\n[+] Comprehensive Evaluation Report saved to: {report_path}")

    return {
        "accuracy_5class": float(acc_5c),
        "macro_f1_5class": float(macro_f1_5c),
        "binary_sensitivity": float(sensitivity),
        "binary_specificity": float(specificity),
        "binary_f1": float(binary_f1),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate KneeWise Model on Test Set")
    parser.add_argument("--model_path", type=str, default=str(BEST_MODEL_PATH))
    parser.add_argument("--test_manifest", type=str, default=str(TEST_MANIFEST_CSV))
    parser.add_argument("--output_dir", type=str, default=str(EVALUATION_OUTPUT_DIR))
    parser.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()

    evaluate_test_set(
        model_path=Path(args.model_path),
        test_manifest=Path(args.test_manifest),
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
