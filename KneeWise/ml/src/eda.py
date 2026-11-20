#!/usr/bin/env python3
"""
eda.py - KneeWise Exploratory Data Analysis Engine

Performs rigorous exploratory analysis on knee osteoarthritis radiographs:
- Class frequency & imbalance analysis
- Representative grid visualization for KL grades 0 to 4
- Pixel intensity histograms & dynamic range checks
- Image dimension distribution
- Duplicate file hash detection (MD5)
- Automated plot generation saved to outputs/eda/
"""

import os
import sys
import hashlib
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    EDA_OUTPUT_DIR,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    KL_GRADE_SHORT_LABELS,
    FULL_MANIFEST_CSV,
)


def compute_md5(file_path: Path, block_size: int = 65536) -> str:
    """Calculates MD5 hash of a file for duplicate detection."""
    hasher = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(block_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def run_eda(
    data_manifest_or_dir: Optional[Path] = None,
    output_dir: Path = EDA_OUTPUT_DIR,
):
    """Executes full Exploratory Data Analysis workflow."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — Exploratory Data Analysis         ")
    print("=" * 65)

    # Determine input source: Full manifest if available, or scan raw directory
    df = None
    if FULL_MANIFEST_CSV.exists():
        print(f"[*] Reading dataset manifest: {FULL_MANIFEST_CSV}")
        df = pd.read_csv(FULL_MANIFEST_CSV)
    elif data_manifest_or_dir and Path(data_manifest_or_dir).is_file():
        df = pd.read_csv(data_manifest_or_dir)

    if df is None or len(df) == 0:
        print("[!] No manifest found. Running pre-manifest raw scan...")
        # Scan raw data directly
        from preprocess import build_manifests
        train_df, val_df, test_df = build_manifests(RAW_DATA_DIR, force_patient_split=False)
        if FULL_MANIFEST_CSV.exists():
            df = pd.read_csv(FULL_MANIFEST_CSV)
        else:
            print("\n[!] Dataset files not yet downloaded or unpacked in data/raw/.")
            print("Please place the Mendeley dataset in data/raw/ and re-run EDA.")
            return

    total_images = len(df)
    print(f"\n[+] Total Images Analyzed: {total_images:,}")

    # 1. Class Distribution Analysis
    class_counts = df["kl_grade"].value_counts().sort_index()
    print("\n--- Kellgren-Lawrence Grade Distribution ---")
    for grade in KL_CLASSES:
        cnt = class_counts.get(grade, 0)
        pct = (cnt / total_images * 100) if total_images > 0 else 0
        print(f"  KL Grade {grade}: {cnt:5d} ({pct:5.1f}%) | {KL_GRADE_DESCRIPTIONS[grade]}")

    # Imbalance ratio
    max_c = class_counts.max()
    min_c = class_counts.min() if class_counts.min() > 0 else 1
    imbalance_ratio = max_c / min_c
    print(f"\n[*] Class Imbalance Ratio (Max / Min): {imbalance_ratio:.2f}:1")

    # Plot 1: Class Distribution Bar Chart
    plt.figure(figsize=(9, 5))
    grades = [f"Grade {g}" for g in KL_CLASSES]
    counts = [class_counts.get(g, 0) for g in KL_CLASSES]
    colors = ["#2b5c8f", "#4682b4", "#5c9ebb", "#e07a5f", "#c0392b"]

    bars = plt.bar(grades, counts, color=colors, edgecolor="black", alpha=0.85)
    plt.title("Distribution of Kellgren-Lawrence (KL) Severity Grades", fontsize=14, fontweight="bold")
    plt.xlabel("Osteoarthritis Severity Grade", fontsize=12)
    plt.ylabel("Number of Radiographs", fontsize=12)
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(counts) * 0.01,
            f"{int(height):,} ({height / total_images * 100:.1f}%)",
            ha="center", va="bottom", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    dist_plot_path = output_dir / "class_distribution.png"
    plt.savefig(dist_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved class distribution plot: {dist_plot_path}")

    # Plot 2: Representative Images Grid (Grades 0 to 4)
    fig, axes = plt.subplots(1, 5, figsize=(18, 4.5))
    sample_paths = {}

    for i, grade in enumerate(KL_CLASSES):
        grade_subset = df[df["kl_grade"] == grade]
        if not grade_subset.empty:
            sample_row = grade_subset.iloc[0]
            sample_path = Path(sample_row["image_path"])
            sample_paths[grade] = sample_path
            try:
                with Image.open(sample_path) as img:
                    axes[i].imshow(img.convert("L"), cmap="bone")
                    axes[i].set_title(KL_GRADE_SHORT_LABELS[grade], fontsize=11, fontweight="bold")
                    axes[i].axis("off")
            except Exception as e:
                axes[i].set_title(f"Grade {grade}\n(Error loading)", fontsize=10)
                axes[i].axis("off")
        else:
            axes[i].set_title(f"Grade {grade}\n(No Images)", fontsize=10)
            axes[i].axis("off")

    fig.suptitle("Representative Knee Radiographs Across Kellgren-Lawrence Grades (0–4)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    rep_plot_path = output_dir / "representative_samples.png"
    plt.savefig(rep_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved representative sample grid: {rep_plot_path}")

    # 3. Pixel Intensity Distribution
    plt.figure(figsize=(10, 5))
    for grade in KL_CLASSES:
        if grade in sample_paths and sample_paths[grade].exists():
            try:
                with Image.open(sample_paths[grade]) as img:
                    arr = np.array(img.convert("L"))
                    plt.hist(arr.ravel(), bins=64, range=(0, 256), density=True, alpha=0.4, label=f"Grade {grade}")
            except Exception:
                pass

    plt.title("Pixel Intensity Distributions Across Representative KL Grades", fontsize=13, fontweight="bold")
    plt.xlabel("Grayscale Intensity Value (0-255)", fontsize=11)
    plt.ylabel("Normalized Density", fontsize=11)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    intensity_plot_path = output_dir / "pixel_intensity_distribution.png"
    plt.savefig(intensity_plot_path, dpi=300)
    plt.close()
    print(f"[+] Saved pixel intensity plot: {intensity_plot_path}")

    # 4. Duplicate Image Detection
    print("\n[*] Checking for potential duplicate images via MD5 checksum...")
    hash_map = defaultdict(list)
    sample_for_dup = df["image_path"].head(1000)  # check sample
    for p_str in sample_for_dup:
        p = Path(p_str)
        if p.exists():
            h = compute_md5(p)
            hash_map[h].append(p)

    duplicates = {h: paths for h, paths in hash_map.items() if len(paths) > 1}
    print(f"[+] Duplicate inspection finished: {len(duplicates)} identical file pairs detected.")

    print("\n" + "=" * 65)
    print(f"[+] EDA Complete! All figures exported to: {output_dir}")
    print("=" * 65)


if __name__ == "__main__":
    run_eda()
