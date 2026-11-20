# KneeWise — Freeze Champion & Start Phase 4: Targeted Model Improvement

You are working on the KneeWise machine learning project.

We have completed the initial architecture comparison and selected **Experiment D: EfficientNet-B0 with balanced class weights** as the current champion.

Your task is to:

1. Formally freeze and preserve the existing champion and all its evaluation artifacts.
2. Audit the existing training pipeline and experiment configurations.
3. Start a new, controlled training phase focused on improving classification performance, especially KL1 and KL2.
4. Compare new experiments against the frozen champion using identical evaluation conditions.

Do not modify the Django backend or frontend.

---

## PHASE 1 — FREEZE THE CURRENT CHAMPION

Current champion:

- Architecture: EfficientNet-B0
- Experiment ID: `exp_d_efficientnet_b0`
- Checkpoint: `models/exp_d_efficientnet_b0.pth`
- Test Accuracy: 66.61%
- Test Macro F1: 0.6724
- KL0 F1: 0.778
- KL1 F1: 0.347
- KL2 F1: 0.598
- KL3 F1: 0.798
- KL4 F1: 0.841
- Binary ROC-AUC: 0.9289
- Binary PR-AUC: 0.9228
- Test Sensitivity at threshold 0.30: 80.86%
- Test Specificity at threshold 0.30: 88.66%
- Brier Score: 0.1114
- ECE: 0.0731
- Patient-level split audit: 4,130 unique patients, zero overlap across train, validation and test.

### Required actions

1. Verify that the champion checkpoint exists and is readable.
2. Calculate and record its SHA-256 checksum.
3. Create a frozen experiment manifest containing:
   - Architecture
   - Experiment ID
   - Checkpoint path
   - Checkpoint checksum
   - Training configuration
   - Dataset identifier and split information
   - Preprocessing configuration
   - Class mapping
   - Validation metrics
   - Test metrics
   - Threshold-selection methodology
4. Preserve the existing technical dossier, JSON dossier, threshold CSV and plots.
5. Copy the champion checkpoint into:

`models/frozen_champion/efficientnet_b0_champion.pth`

6. Do not overwrite or retrain this checkpoint.
7. Keep the ResNet18 checkpoint as a reference baseline.

If the original experiment configuration or dataset provenance cannot be established, explicitly report the missing information. Never fabricate it.

---

## PHASE 2 — AUDIT THE TRAINING PIPELINE

Before launching new training, inspect:

- `src/config.py`
- `src/model.py`
- `src/dataset.py`
- `src/preprocess.py`
- `src/train.py`
- `src/evaluate.py`
- Existing experiment configurations
- Existing checkpoints and training logs

Determine the exact settings used in Experiment D:

- Input resolution
- Image normalization
- Augmentation
- Optimizer
- Learning rate
- Scheduler
- Batch size
- Number of epochs
- Early stopping
- Class weights
- Loss function
- Train/validation/test split

Do not assume settings. Read the actual implementation and saved experiment configuration.

Verify that all new experiments use the same patient-level split.

IMPORTANT:
- Train only on training data.
- Use validation data for hyperparameter selection and model selection.
- Keep the test set untouched until the final evaluation of a selected candidate.
- Never use test results to tune hyperparameters or thresholds.

---

## PHASE 3 — TARGETED TRAINING EXPERIMENTS

The primary weakness is KL1:

- KL1 F1: 0.347
- KL1 recall: 0.345

KL2 F1 is also an improvement target:

- KL2 F1: 0.598
- KL2 recall: 0.494

The objective is to improve the five-class macro F1 without severely damaging the other classes.

Create a new experiment directory:

`outputs/training/phase_4/`

Use unique experiment IDs and separate checkpoints.

### Experiment E — EfficientNet-B0 with targeted augmentation

Start with the exact EfficientNet-B0 architecture used in Experiment D.

Investigate training-only augmentation that may improve robustness:

- Small rotations
- Slight translations
- Mild brightness and contrast changes
- Small scale variations

Avoid transformations that distort meaningful radiographic anatomy.

Use the same validation preprocessing as Experiment D.

Do not apply augmentation to validation or test images.

### Experiment F — EfficientNet-B0 with alternative loss

Investigate an alternative to the existing class-weighted cross-entropy loss.

Evaluate focal loss with a conservative gamma value and compare it against the original loss.

Use class weighting only if justified by the training distribution.

Avoid excessive weighting that could destabilize training.

### Experiment G — EfficientNet-B0 with KL1/KL2-focused sampling

Investigate a training-only weighted sampling strategy that gives underrepresented KL1 and KL2 samples more opportunities during training.

Requirements:

- Derive sampling weights from training labels only.
- Do not modify validation or test distributions.
- Avoid extreme oversampling.
- Monitor overfitting and class-specific validation metrics.
- Record the effective class distribution seen during training.

Run these experiments independently, not simultaneously.

Do not automatically launch all experiments if GPU memory or disk capacity is insufficient.

---

## PHASE 4 — HARDWARE CONSTRAINTS

Target hardware:

- NVIDIA RTX 3050 Laptop GPU
- 4 GB VRAM
- 8 GB system RAM
- Windows environment

Maintain resource safeguards:

- Mixed precision FP16 where supported.
- Conservative batch size.
- `num_workers=0` on Windows unless verified safe.
- Clear GPU cache between experiments.
- Save checkpoints safely.
- Log peak GPU memory.
- Avoid loading multiple large models simultaneously.
- Do not silently fall back to CPU when CUDA was explicitly requested.

If an out-of-memory error occurs, reduce batch size and document the change.

---

## PHASE 5 — EVALUATION

For each new experiment, record:

### Five-class classification
- Test accuracy
- Macro precision
- Macro recall
- Macro F1
- Per-class precision, recall and F1
- 5×5 confusion matrix
- Adjacent-grade concordance

### Binary screening
- ROC-AUC
- PR-AUC
- Sensitivity
- Specificity
- Precision
- F1
- Confusion matrix

### Training efficiency
- Training time
- Peak VRAM
- Best validation macro F1
- Best epoch
- Model checkpoint size

Do not evaluate candidates repeatedly against the test set during experimentation.

First select the best candidate using validation results. Then evaluate that selected candidate once on the untouched test set.

Use the same evaluation methodology as the frozen champion.

---

## PHASE 6 — MODEL SELECTION

The current champion remains:

`EfficientNet-B0 — exp_d_efficientnet_b0`

A new model can replace it only if it demonstrates meaningful improvement.

Prioritize:

1. Higher validation macro F1.
2. Better KL1 and KL2 performance.
3. No severe degradation in KL0, KL3 or KL4.
4. Stable training and reproducible results.
5. Reasonable inference cost for the target laptop.

Do not select a model solely because its accuracy is higher.

If none of the new experiments improves the champion meaningfully, retain Experiment D.

Never automatically replace the frozen champion.

---

## PHASE 7 — FINAL REPORT

Create:

- `phase_4_experiment_report.md`
- `phase_4_experiment_report.json`
- Per-experiment training logs
- Validation comparison plots
- Per-class F1 comparison chart
- Confusion matrices for the final selected candidate
- Checkpoint manifest with SHA-256 checksums

Include a comparison table:

| Experiment | Accuracy | Macro F1 | KL1 F1 | KL2 F1 | KL3 F1 | KL4 F1 | Peak VRAM | Runtime |
|---|---|---|---|---|---|---|---|---|

Clearly distinguish validation metrics from final test metrics.

---

## STRICT RULES

- Never overwrite the frozen champion.
- Never change the original dataset splits.
- Never train on validation or test images.
- Never tune thresholds using test results.
- Never fabricate metrics.
- Never claim clinical validation.
- Never change the original experiment results.
- Do not modify the Django backend or frontend.
- Preserve reproducibility and log every configuration change.

### STARTING INSTRUCTION

First complete Phase 1 and Phase 2.

Show me:
1. Champion checkpoint verification and checksum.
2. Frozen manifest location.
3. Exact Experiment D training configuration.
4. Current dataset and patient-split verification.
5. Proposed configurations for Experiments E, F and G.

Then start **Experiment E only** after confirming that the frozen champion and all existing artifacts are preserved.

Do not wait for all three experiment configurations to run together. Complete them sequentially and report each result before moving to the next."""
generate_candidate_artifact.py - KneeWise Final Model Candidate Verification Engine
Preserves:
1. Full 5x5 confusion matrix (raw & normalized)
2. Precision, Recall, and F1 for all five KL grades
3. ROC-AUC and PR-AUC for binary OA screening
4. Sensitivity/Specificity sweep across validation thresholds with Youden's J
5. Patient-level split verification (zero overlap proof)
6. Calibration assessment (Expected Calibration Error, Brier score, Reliability curve)
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    brier_score_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

# Ensure temp directory for matplotlib
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
    get_device,
    use_pinned_memory,
)
from dataset import KneeXRayDataset, get_val_test_transforms
from model import KneeWiseClassifier, load_checkpoint


def verify_patient_splits() -> Dict[str, Any]:
    """Demonstrates complete patient-level separation across train, val, and test splits."""
    train_df = pd.read_csv(TRAIN_MANIFEST_CSV)
    val_df = pd.read_csv(VAL_MANIFEST_CSV)
    test_df = pd.read_csv(TEST_MANIFEST_CSV)

    train_pts = set(train_df["patient_id"].astype(str))
    val_pts = set(val_df["patient_id"].astype(str))
    test_pts = set(test_df["patient_id"].astype(str))

    train_val_overlap = train_pts.intersection(val_pts)
    train_test_overlap = train_pts.intersection(test_pts)
    val_test_overlap = val_pts.intersection(test_pts)

    all_pts = train_pts.union(val_pts).union(test_pts)

    return {
        "train_patients": len(train_pts),
        "val_patients": len(val_pts),
        "test_patients": len(test_pts),
        "total_unique_patients": len(all_pts),
        "train_images": len(train_df),
        "val_images": len(val_df),
        "test_images": len(test_df),
        "total_images": len(train_df) + len(val_df) + len(test_df),
        "train_val_overlap_count": len(train_val_overlap),
        "train_test_overlap_count": len(train_test_overlap),
        "val_test_overlap_count": len(val_test_overlap),
        "is_leakage_free": (
            len(train_val_overlap) == 0
            and len(train_test_overlap) == 0
            and len(val_test_overlap) == 0
        ),
    }


@torch.no_grad()
def collect_model_predictions(model, loader, device):
    """Collects targets, predictions, and full class probability distributions."""
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

    return all_targets, all_preds, all_probs


def compute_expected_calibration_error(y_true_binary: np.ndarray, y_prob_binary: np.ndarray, n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
    """Computes Expected Calibration Error (ECE) for binary classification."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_accs = []
    bin_confs = []

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (y_prob_binary >= bin_lower) & (y_prob_binary < bin_upper) if i < n_bins - 1 else (y_prob_binary >= bin_lower) & (y_prob_binary <= bin_upper)
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true_binary[in_bin])
            avg_confidence_in_bin = np.mean(y_prob_binary[in_bin])
            ece += np.abs(accuracy_in_bin - avg_confidence_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((bin_lower + bin_upper) / 2.0)

    return ece, np.array(bin_accs), np.array(bin_confs)


def evaluate_validation_thresholds(val_targets, val_probs) -> pd.DataFrame:
    """Evaluates fine-grained screening thresholds on validation set."""
    true_bin = np.isin(val_targets, BINARY_DEFINITE_GRADES).astype(int)
    prob_oa = val_probs[:, BINARY_DEFINITE_GRADES].sum(axis=1)

    thresholds = np.linspace(0.10, 0.90, 17)
    records = []

    for th in thresholds:
        pred_bin = (prob_oa >= th).astype(int)
        cm = confusion_matrix(true_bin, pred_bin, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        f1 = 2 * (prec * sens) / (prec + sens) if (prec + sens) > 0 else 0.0
        acc = (tp + tn) / len(true_bin)
        youden_j = sens + spec - 1.0

        records.append({
            "threshold": round(float(th), 2),
            "sensitivity": sens,
            "specificity": spec,
            "precision": prec,
            "f1_score": f1,
            "accuracy": acc,
            "youden_j": youden_j,
            "false_negatives": int(fn),
            "false_positives": int(fp),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

    return pd.DataFrame(records)


def main():
    device = get_device()
    output_dir = Path("KneeWise/ml/outputs/evaluation/candidate")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("      KNEEWISE — FINAL MODEL CANDIDATE STABILITY & PRESERVATION ENGINE")
    print("=" * 80)

    # 1. Patient-Level Split Verification
    print("\n[1/6] Verifying Patient-Level Split Separation...")
    split_audit = verify_patient_splits()
    print(f"  Total Patients: {split_audit['total_unique_patients']:,} across {split_audit['total_images']:,} images")
    print(f"  Train: {split_audit['train_patients']} patients ({split_audit['train_images']} images)")
    print(f"  Val:   {split_audit['val_patients']} patients ({split_audit['val_images']} images)")
    print(f"  Test:  {split_audit['test_patients']} patients ({split_audit['test_images']} images)")
    print(f"  Train-Val Overlap:  {split_audit['train_val_overlap_count']} patients")
    print(f"  Train-Test Overlap: {split_audit['train_test_overlap_count']} patients")
    print(f"  Val-Test Overlap:   {split_audit['val_test_overlap_count']} patients")
    print(f"  Data Leakage Audit: {'PASSED (Zero Overlap)' if split_audit['is_leakage_free'] else 'FAILED'}")

    # 2. Load Champion Model (EfficientNet-B0)
    champion_path = Path("KneeWise/ml/models/exp_d_efficientnet_b0.pth")
    if not champion_path.exists():
        champion_path = Path("KneeWise/ml/models/best_kneewise_resnet18.pth")
        model_arch = "resnet18"
    else:
        model_arch = "efficientnet_b0"

    print(f"\n[2/6] Loading Candidate Champion Model ({model_arch}) from {champion_path}...")
    model = KneeWiseClassifier(backbone_name=model_arch, num_classes=5, pretrained=False)
    load_checkpoint(champion_path, model, device=device)
    model.to(device)

    # Dataloaders
    val_ds = KneeXRayDataset(VAL_MANIFEST_CSV, transform=get_val_test_transforms())
    test_ds = KneeXRayDataset(TEST_MANIFEST_CSV, transform=get_val_test_transforms())
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=8, shuffle=False, num_workers=0)

    # Collect predictions
    val_targets, val_preds, val_probs = collect_model_predictions(model, val_loader, device)
    test_targets, test_preds, test_probs = collect_model_predictions(model, test_loader, device)

    # 3. Full 5x5 Confusion Matrix & Per-Class Metrics
    print("\n[3/6] Generating 5x5 Confusion Matrix and Per-Class Breakdown...")
    cm_5x5 = confusion_matrix(test_targets, test_preds, labels=KL_CLASSES)
    cm_5x5_norm = cm_5x5.astype(float) / cm_5x5.sum(axis=1)[:, np.newaxis]

    per_class_metrics = []
    prec_list = precision_score(test_targets, test_preds, average=None, labels=KL_CLASSES, zero_division=0)
    rec_list = recall_score(test_targets, test_preds, average=None, labels=KL_CLASSES, zero_division=0)
    f1_list = f1_score(test_targets, test_preds, average=None, labels=KL_CLASSES, zero_division=0)

    for g in KL_CLASSES:
        sup = int((test_targets == g).sum())
        per_class_metrics.append({
            "grade": f"KL {g}",
            "description": KL_GRADE_DESCRIPTIONS[g],
            "precision": float(prec_list[g]),
            "recall": float(rec_list[g]),
            "f1_score": float(f1_list[g]),
            "support": sup,
        })

    # Plot 5x5 Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mat, title, fmt in zip(
        axes,
        [cm_5x5, cm_5x5_norm],
        ["5x5 Confusion Matrix (Counts)", "5x5 Confusion Matrix (Normalized Recall %)"],
        ["d", ".1%"],
    ):
        im = ax.imshow(mat, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set(
            xticks=np.arange(5), yticks=np.arange(5),
            xticklabels=[f"KL{c}" for c in KL_CLASSES],
            yticklabels=[f"KL{c}" for c in KL_CLASSES],
            title=title,
            ylabel="True Reference Label",
            xlabel="AI Model Predicted Grade",
        )
        thresh = mat.max() / 2.0
        for i in range(5):
            for j in range(5):
                val = mat[i, j]
                txt = f"{val:{fmt}}"
                ax.text(j, i, txt, ha="center", va="center",
                        color="white" if val > thresh else "black",
                        fontweight="bold" if i == j else "normal")
    plt.tight_layout()
    cm_plot_path = output_dir / "candidate_confusion_matrices.png"
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"  [+] Saved 5x5 confusion matrices to: {cm_plot_path}")

    # 4. Binary Screening ROC-AUC and PR-AUC
    print("\n[4/6] Computing ROC-AUC, PR-AUC, and Curve Generation...")
    true_bin_test = np.isin(test_targets, BINARY_DEFINITE_GRADES).astype(int)
    prob_oa_test = test_probs[:, BINARY_DEFINITE_GRADES].sum(axis=1)

    roc_auc = roc_auc_score(true_bin_test, prob_oa_test)
    pr_auc = average_precision_score(true_bin_test, prob_oa_test)
    fpr, tpr, roc_thresholds = roc_curve(true_bin_test, prob_oa_test)
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(true_bin_test, prob_oa_test)

    print(f"  Binary Screening ROC-AUC : {roc_auc:.4f} ({roc_auc*100:.2f}%)")
    print(f"  Binary Screening PR-AUC  : {pr_auc:.4f} ({pr_auc*100:.2f}%)")

    # Plot ROC & PR Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    # ROC
    axes[0].plot(fpr, tpr, color="#1f77b4", lw=2.5, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="gray", linestyle="--", lw=1.5, label="Chance Baseline (AUC = 0.50)")
    axes[0].set_title("Receiver Operating Characteristic (ROC)", fontsize=13)
    axes[0].set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11)
    axes[0].set_ylabel("True Positive Rate (Sensitivity)", fontsize=11)
    axes[0].legend(loc="lower right")
    axes[0].grid(True, alpha=0.3)

    # PR
    baseline_pr = np.mean(true_bin_test)
    axes[1].plot(rec_curve, prec_curve, color="#2ca02c", lw=2.5, label=f"PR Curve (PR-AUC = {pr_auc:.4f})")
    axes[1].axhline(baseline_pr, color="gray", linestyle="--", lw=1.5, label=f"Prevalence Baseline ({baseline_pr:.2f})")
    axes[1].set_title("Precision-Recall Curve (PR)", fontsize=13)
    axes[1].set_xlabel("Recall (Sensitivity)", fontsize=11)
    axes[1].set_ylabel("Precision (Positive Predictive Value)", fontsize=11)
    axes[1].legend(loc="lower left")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    roc_pr_path = output_dir / "candidate_roc_pr_curves.png"
    plt.savefig(roc_pr_path, dpi=300)
    plt.close()
    print(f"  [+] Saved ROC & PR curves to: {roc_pr_path}")

    # 5. Validation Threshold Sweep (Sensitivity / Specificity Trade-Off)
    print("\n[5/6] Performing Validation Threshold Sweep & Youden's J Analysis...")
    val_th_df = evaluate_validation_thresholds(val_targets, val_probs)
    val_th_path = output_dir / "validation_threshold_curve.csv"
    val_th_df.to_csv(val_th_path, index=False)

    best_f1_row = val_th_df.loc[val_th_df["f1_score"].idxmax()]
    best_j_row = val_th_df.loc[val_th_df["youden_j"].idxmax()]
    print(f"  Peak Validation F1 Operating Point: Threshold = {best_f1_row['threshold']:.2f}")
    print(f"    Sensitivity: {best_f1_row['sensitivity']*100:.2f}% | Specificity: {best_f1_row['specificity']*100:.2f}% | F1: {best_f1_row['f1_score']:.4f}")
    print(f"  Peak Youden's J Index Operating Point: Threshold = {best_j_row['threshold']:.2f} (J = {best_j_row['youden_j']:.4f})")
    print(f"    Sensitivity: {best_j_row['sensitivity']*100:.2f}% | Specificity: {best_j_row['specificity']*100:.2f}%")

    # 6. Model Calibration Assessment
    print("\n[6/6] Computing Calibration Assessment (ECE & Reliability Diagram)...")
    brier_score = brier_score_loss(true_bin_test, prob_oa_test)
    ece_10, bin_accs, bin_confs = compute_expected_calibration_error(true_bin_test, prob_oa_test, n_bins=10)
    prob_true, prob_pred = calibration_curve(true_bin_test, prob_oa_test, n_bins=10)

    print(f"  Brier Score Loss: {brier_score:.4f} (Lower is better; 0 = perfect)")
    print(f"  Expected Calibration Error (ECE - 10 Bins): {ece_10:.4f} ({ece_10*100:.2f}%)")

    # Plot Reliability Curve
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect Calibration")
    ax.plot(prob_pred, prob_true, marker="s", color="#d62728", lw=2, label=f"Model Calibration (ECE = {ece_10:.3f})")
    ax.set_title("Reliability Diagram (Model Risk Calibration)", fontsize=13)
    ax.set_xlabel("Mean Predicted OA Probability", fontsize=11)
    ax.set_ylabel("Empirical Fraction of Positives", fontsize=11)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    calib_path = output_dir / "candidate_calibration_curve.png"
    plt.savefig(calib_path, dpi=300)
    plt.close()
    print(f"  [+] Saved Calibration Reliability curve to: {calib_path}")

    # 7. Assemble Comprehensive Verification Report
    report_dict = {
        "candidate_model": model_arch,
        "checkpoint_file": str(champion_path.name),
        "split_verification": split_audit,
        "five_class_metrics": {
            "overall_accuracy": float(accuracy_score(test_targets, test_preds)),
            "macro_f1": float(f1_score(test_targets, test_preds, average="macro")),
            "weighted_f1": float(f1_score(test_targets, test_preds, average="weighted")),
            "per_class": per_class_metrics,
            "confusion_matrix_5x5": cm_5x5.tolist(),
        },
        "binary_screening_metrics": {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "brier_score": float(brier_score),
            "expected_calibration_error_10bin": float(ece_10),
            "default_operating_point_th_0.50": {
                "sensitivity": float(recall_score(true_bin_test, (prob_oa_test >= 0.50).astype(int))),
                "specificity": float(recall_score(1 - true_bin_test, 1 - (prob_oa_test >= 0.50).astype(int))),
                "precision": float(precision_score(true_bin_test, (prob_oa_test >= 0.50).astype(int))),
                "f1_score": float(f1_score(true_bin_test, (prob_oa_test >= 0.50).astype(int))),
            },
            "clinical_screening_operating_point_th_0.30": {
                "sensitivity": float(recall_score(true_bin_test, (prob_oa_test >= 0.30).astype(int))),
                "specificity": float(recall_score(1 - true_bin_test, 1 - (prob_oa_test >= 0.30).astype(int))),
                "precision": float(precision_score(true_bin_test, (prob_oa_test >= 0.30).astype(int))),
                "f1_score": float(f1_score(true_bin_test, (prob_oa_test >= 0.30).astype(int))),
            },
        },
        "validation_threshold_sweep": val_th_df.to_dict(orient="records"),
    }

    json_path = output_dir / "candidate_verification_dossier.json"
    with open(json_path, "w") as f:
        json.dump(report_dict, f, indent=2)
    print(f"\n[+] Saved complete verification dossier to: {json_path}")


if __name__ == "__main__":
    main()
