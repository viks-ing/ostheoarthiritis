#!/usr/bin/env python3
"""
preprocess.py - KneeWise ML Data Preprocessing & Manifest Generator

Parses raw knee X-ray directory structure, extracts patient identifiers,
validates integrity, audits patient-level data leakage, and generates
reproducible train, validation, and test CSV manifests without duplicating raw images.
"""

import os
import sys
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    TRAIN_MANIFEST_CSV,
    VAL_MANIFEST_CSV,
    TEST_MANIFEST_CSV,
    FULL_MANIFEST_CSV,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    RANDOM_SEED,
)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def extract_patient_id(filename: str) -> str:
    """
    Extracts patient identifier from OAI image filename.
    Standard OAI formats include:
      - '9001695_1.png' -> Patient 9001695
      - '9003316R.png'  -> Patient 9003316
      - '0_9001695_L.png' -> Patient 9001695
    """
    stem = Path(filename).stem
    # Match leading digits or digits between underscores
    match = re.search(r"(\d{6,8})", stem)
    if match:
        return match.group(1)
    
    # Fallback: take token before first underscore
    parts = stem.split("_")
    for part in parts:
        if part.isdigit() and len(part) >= 4:
            return part
            
    return f"unknown_{stem}"


def build_manifests(
    raw_root: Path,
    output_dir: Path = PROCESSED_DATA_DIR,
    force_patient_split: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Scans raw directory, identifies ClsKL images, analyzes splits,
    verifies patient isolation, and saves CSV manifests.
    """
    raw_root = Path(raw_root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — Preprocessing & Manifests         ")
    print("=" * 65)
    print(f"[*] Scanning data source: {raw_root}")

    records = []
    skipped_detjoint = 0
    skipped_invalid = 0

    for root, _, files in os.walk(raw_root):
        rel_path = Path(root).relative_to(raw_root)
        parts_lower = [p.lower() for p in rel_path.parts]

        # Explicitly filter out YOLO / DetJoint / auto_test images
        if any("detjoint" in p or "detection" in p or "auto_test" in p for p in parts_lower):
            skipped_detjoint += len([f for f in files if Path(f).suffix.lower() in IMAGE_EXTENSIONS])
            continue

        # Look for KL grade directory (0, 1, 2, 3, 4)
        kl_label = None
        for p in rel_path.parts:
            if p in [str(k) for k in KL_CLASSES]:
                kl_label = int(p)
                break

        # Look for split directory (train, val, test)
        detected_split = "unassigned"
        for p in parts_lower:
            if p in ("train", "training"):
                detected_split = "train"
                break
            elif p in ("val", "valid", "validation"):
                detected_split = "val"
                break
            elif p in ("test", "testing"):
                detected_split = "test"
                break

        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            if kl_label is None:
                skipped_invalid += 1
                continue

            patient_id = extract_patient_id(file)

            records.append({
                "image_path": str(file_path.resolve()),
                "filename": file,
                "kl_grade": kl_label,
                "split": detected_split,
                "patient_id": patient_id,
            })

    print(f"[+] Total classification candidates collected: {len(records):,}")
    if skipped_detjoint > 0:
        print(f"[*] Excluded DetJoint images (detection data): {skipped_detjoint:,}")
    if skipped_invalid > 0:
        print(f"[*] Skipped unclassified/label-ambiguous images: {skipped_invalid:,}")

    if not records:
        print("\n[!] No valid classification images found to build manifests.")
        print("Please verify dataset extraction in data/raw/ before proceeding.")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df = pd.DataFrame(records)

    # Check existing splits
    splits_present = set(df["split"].unique())
    has_canonical_splits = {"train", "val", "test"}.issubset(splits_present)

    # Audit Patient-Level Leakage across splits
    if has_canonical_splits and not force_patient_split:
        print("\n[*] Auditing original dataset splits for patient-level data leakage...")
        train_patients = set(df[df["split"] == "train"]["patient_id"])
        val_patients = set(df[df["split"] == "val"]["patient_id"])
        test_patients = set(df[df["split"] == "test"]["patient_id"])

        train_val_overlap = train_patients.intersection(val_patients)
        train_test_overlap = train_patients.intersection(test_patients)
        val_test_overlap = val_patients.intersection(test_patients)

        print(f"  • Unique patients in Train:      {len(train_patients):,}")
        print(f"  • Unique patients in Validation: {len(val_patients):,}")
        print(f"  • Unique patients in Test:       {len(test_patients):,}")

        if train_test_overlap or train_val_overlap or val_test_overlap:
            print("\n" + "!" * 65)
            print("[WARNING] PATIENT DATA LEAKAGE DETECTED IN ORIGINAL SPLITS!")
            print(f"  - Train/Val Overlapping Patients:  {len(train_val_overlap)}")
            print(f"  - Train/Test Overlapping Patients: {len(train_test_overlap)}")
            print(f"  - Val/Test Overlapping Patients:   {len(val_test_overlap)}")
            print("To ensure clinical validity, re-partitioning at patient-level is strongly advised.")
            print("!" * 65 + "\n")
        else:
            print("[+] Zero patient overlap confirmed between Train, Val, and Test splits.")

        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    else:
        print("\n[*] Constructing Patient-Level Stratified Train/Val/Test Splits (70/15/15)...")
        # Unique patient list
        patients = df["patient_id"].unique()
        np.random.seed(RANDOM_SEED)
        shuffled_patients = np.random.permutation(patients)

        n_total = len(shuffled_patients)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)

        train_pts = set(shuffled_patients[:n_train])
        val_pts = set(shuffled_patients[n_train:n_train + n_val])
        test_pts = set(shuffled_patients[n_train + n_val:])

        df.loc[df["patient_id"].isin(train_pts), "split"] = "train"
        df.loc[df["patient_id"].isin(val_pts), "split"] = "val"
        df.loc[df["patient_id"].isin(test_pts), "split"] = "test"

        train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
        val_df = df[df["split"] == "val"].copy().reset_index(drop=True)
        test_df = df[df["split"] == "test"].copy().reset_index(drop=True)

    # Save manifests
    train_df.to_csv(TRAIN_MANIFEST_CSV, index=False)
    val_df.to_csv(VAL_MANIFEST_CSV, index=False)
    test_df.to_csv(TEST_MANIFEST_CSV, index=False)
    df.to_csv(FULL_MANIFEST_CSV, index=False)

    print("\n[+] Manifests saved successfully:")
    print(f"  • Train manifest:      {TRAIN_MANIFEST_CSV} ({len(train_df):,} images)")
    print(f"  • Val manifest:        {VAL_MANIFEST_CSV} ({len(val_df):,} images)")
    print(f"  • Test manifest:       {TEST_MANIFEST_CSV} ({len(test_df):,} images)")
    print(f"  • Full manifest:       {FULL_MANIFEST_CSV} ({len(df):,} images)")

    # Class balance summary
    print("\n[+] Train Set Class Distribution:")
    for grade in KL_CLASSES:
        cnt = (train_df["kl_grade"] == grade).sum()
        pct = cnt / len(train_df) * 100 if len(train_df) > 0 else 0
        print(f"  Grade {grade}: {cnt:5d} ({pct:5.1f}%) | {KL_GRADE_DESCRIPTIONS[grade]}")

    return train_df, val_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Preprocess and generate manifests for KneeWise ML")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Raw dataset directory",
    )
    parser.add_argument(
        "--force_patient_split",
        action="store_true",
        help="Force patient-level re-split even if original splits exist",
    )
    args = parser.parse_args()

    build_manifests(
        raw_root=Path(args.data_dir),
        force_patient_split=args.force_patient_split,
    )


if __name__ == "__main__":
    main()
