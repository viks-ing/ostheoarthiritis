#!/usr/bin/env python3
"""
inspect_data.py - KneeWise ML Dataset Inspection Script (Kaggle & Multi-Source Support)

Recursively inspects knee osteoarthritis radiographs for the Kaggle dataset structure:
  train/0-4
  val/0-4 (or valid/0-4)
  test/0-4

Key Capabilities:
- Identifies splits (train, val, test) and 5 Kellgren-Lawrence severity grades (0, 1, 2, 3, 4).
- Does not assume Mendeley folder structures (ClsKLData).
- Extracts patient identifiers from filenames (e.g., OAI standard IDs like 9001695_...).
- Rigorously checks for patient overlap / data leakage across train, val, and test partitions.
- Computes class distributions and class imbalance ratios.
- Calculates derived binary OA screening balance (KL 0-1 vs KL 2-4).
- Tests image integrity, color modes, and resolution consistency.
- Generates a comprehensive markdown report in outputs/inspection/.
"""

import os
import sys
import re
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from config import (
        RAW_DATA_DIR,
        INSPECTION_OUTPUT_DIR,
        KL_CLASSES,
        KL_GRADE_DESCRIPTIONS,
        KL_GRADE_SHORT_LABELS,
        BINARY_UNLIKELY_GRADES,
        BINARY_DEFINITE_GRADES,
    )
except ImportError:
    RAW_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
    INSPECTION_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs" / "inspection"
    KL_CLASSES = [0, 1, 2, 3, 4]
    KL_GRADE_DESCRIPTIONS = {
        0: "Grade 0: No radiographic OA (Normal)",
        1: "Grade 1: Doubtful OA",
        2: "Grade 2: Mild radiographic OA",
        3: "Grade 3: Moderate radiographic OA",
        4: "Grade 4: Severe radiographic OA",
    }
    KL_GRADE_SHORT_LABELS = {0: "KL0", 1: "KL1", 2: "KL2", 3: "KL3", 4: "KL4"}
    BINARY_UNLIKELY_GRADES = [0, 1]
    BINARY_DEFINITE_GRADES = [2, 3, 4]

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
METADATA_EXTENSIONS = {".csv", ".txt", ".json", ".md", ".xml", ".tsv"}


def extract_patient_id(filename: str) -> str:
    """
    Extracts OAI patient identifier from filename.
    Standard patterns:
      - '9001695_1.png'   -> 9001695
      - '9003316R.png'    -> 9003316
      - '0_9001695_L.png' -> 9001695
      - '9000099.png'     -> 9000099
    """
    stem = Path(filename).stem
    # Match 6-8 digit OAI patient IDs
    match = re.search(r"(\d{6,8})", stem)
    if match:
        return match.group(1)

    # Fallback: take token before first underscore if alphanumeric/digits
    parts = stem.split("_")
    for p in parts:
        if p.isdigit() and len(p) >= 4:
            return p

    return f"unknown_{stem}"


class DatasetInspector:
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root).resolve()
        self.all_files: List[Path] = []
        self.image_files: List[Path] = []
        self.metadata_files: List[Path] = []
        self.other_files: List[Path] = []
        self.corrupted_files: List[Tuple[Path, str]] = []

        # Directory and partition analysis
        self.image_extensions: Counter = Counter()
        self.image_dimensions: Counter = Counter()
        self.image_modes: Counter = Counter()

        # Classification structures (Kaggle train/val/test/0-4)
        self.labeled_images: List[Dict] = []
        self.unlabeled_images: List[Path] = []
        self.label_distribution: Counter = Counter()
        self.split_distribution: Dict[str, Counter] = defaultdict(Counter)

        # Patient isolation tracking
        self.patients_all: Set[str] = set()
        self.patients_by_split: Dict[str, Set[str]] = defaultdict(set)
        self.patient_to_images: Dict[str, List[Path]] = defaultdict(list)

    def scan_files(self):
        """Recursively scans data_root for all images, metadata, and files."""
        if not self.data_root.exists():
            return

        for root, _, files in os.walk(self.data_root):
            for file in files:
                file_path = Path(root) / file
                self.all_files.append(file_path)
                ext = file_path.suffix.lower()

                if ext in IMAGE_EXTENSIONS:
                    self.image_files.append(file_path)
                    self.image_extensions[ext] += 1
                elif ext in METADATA_EXTENSIONS:
                    self.metadata_files.append(file_path)
                else:
                    self.other_files.append(file_path)

    def analyze_dataset_structure(self):
        """
        Parses folder structure to identify splits (train/val/test) and KL grades (0-4).
        Directly supports Kaggle layout:
          train/0, train/1, train/2, train/3, train/4
          val/0,   val/1,   val/2,   val/3,   val/4  (or valid/0..)
          test/0,  test/1,  test/2,  test/3,  test/4
        Also supports arbitrary parent folder paths or subfolders.
        """
        for img in self.image_files:
            rel_parts = img.relative_to(self.data_root).parts
            parts_lower = [p.lower() for p in rel_parts]

            # 1. Detect partition (train, val/valid, test, auto_test)
            split_name = "unassigned"
            for p in parts_lower:
                if p in ("train", "training"):
                    split_name = "train"
                    break
                elif p in ("val", "valid", "validation"):
                    split_name = "val"
                    break
                elif p == "auto_test":
                    split_name = "auto_test"
                    break
                elif p in ("test", "testing"):
                    split_name = "test"
                    break

            # 2. Detect KL grade (0, 1, 2, 3, 4) from folder name
            kl_grade = None
            for p in rel_parts:
                if p in [str(k) for k in KL_CLASSES]:
                    kl_grade = int(p)
                    break

            # 3. Patient ID extraction
            patient_id = extract_patient_id(img.name)

            if kl_grade is not None:
                self.label_distribution[kl_grade] += 1
                self.split_distribution[split_name][kl_grade] += 1

                record = {
                    "path": img,
                    "filename": img.name,
                    "split": split_name,
                    "kl_grade": kl_grade,
                    "patient_id": patient_id,
                }
                self.labeled_images.append(record)

                self.patients_all.add(patient_id)
                self.patients_by_split[split_name].add(patient_id)
                self.patient_to_images[patient_id].append(img)
            else:
                self.unlabeled_images.append(img)

    def audit_patient_overlap(self) -> Dict[str, Set[str]]:
        """
        Computes intersection of patient sets across train, val, and test splits
        to verify clinical data isolation.
        """
        train_pts = self.patients_by_split.get("train", set())
        val_pts = self.patients_by_split.get("val", set())
        test_pts = self.patients_by_split.get("test", set())

        return {
            "train_val": train_pts.intersection(val_pts),
            "train_test": train_pts.intersection(test_pts),
            "val_test": val_pts.intersection(test_pts),
        }

    def verify_image_integrity(self, sample_limit: int = 500):
        """
        Verifies image decodability, detects corruptions, checks dimensions and color modes.
        Checks a representative sample across all partitions and classes.
        """
        import random
        candidates = self.image_files
        if len(candidates) > sample_limit:
            random.seed(42)
            candidates = random.sample(candidates, sample_limit)

        for img_path in candidates:
            try:
                if img_path.stat().st_size == 0:
                    self.corrupted_files.append((img_path, "Zero-byte file"))
                    continue

                with Image.open(img_path) as img:
                    img.verify()

                with Image.open(img_path) as img:
                    self.image_dimensions[img.size] += 1
                    self.image_modes[img.mode] += 1

            except Exception as e:
                self.corrupted_files.append((img_path, str(e)))

    def generate_report(self) -> str:
        """Constructs an exhaustive Markdown inspection report."""
        report = []
        report.append("# KneeWise — Dataset Inspection Report (Kaggle OAI Cohort)\n")
        report.append(f"**Inspection Root:** `{self.data_root}`\n")

        if not self.data_root.exists() or len(self.image_files) == 0:
            report.append("## Status: DATASET NOT FOUND IN RAW DIRECTORY\n")
            report.append("> [!WARNING]\n")
            report.append("> No image files were detected in `data/raw/`.\n")
            report.append("### To obtain and configure the Kaggle dataset:\n")
            report.append("1. **Kaggle Dataset Identifier:** `shashwatwork/knee-osteoarthritis-dataset-with-severity`\n")
            report.append("2. **Option A (Kaggle CLI):**\n")
            report.append("   ```powershell\n")
            report.append("   kaggle datasets download -d shashwatwork/knee-osteoarthritis-dataset-with-severity -p data/raw --unzip\n")
            report.append("   ```\n")
            report.append("3. **Option B (Browser Download):**\n")
            report.append("   - Visit: https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity\n")
            report.append("   - Download the archive and extract it directly into `data/raw/`.\n")
            report.append("4. **Expected Structure:**\n")
            report.append("   ```text\n")
            report.append("   data/raw/\n")
            report.append("   ├── train/  (folders: 0, 1, 2, 3, 4)\n")
            report.append("   ├── val/    (folders: 0, 1, 2, 3, 4)\n")
            report.append("   └── test/   (folders: 0, 1, 2, 3, 4)\n")
            report.append("   ```\n")
            report.append("5. Re-run `python src/inspect_data.py` once extracted.\n")
            return "\n".join(report)

        report.append("## 1. Summary of Discovered Files\n")
        report.append(f"- **Total Files Found:** {len(self.all_files):,}")
        report.append(f"- **Total Radiographs:** {len(self.image_files):,}")
        report.append(f"- **Successfully Mapped (Split + Grade 0–4):** {len(self.labeled_images):,}")
        report.append(f"- **Uncategorized / Missing Labels:** {len(self.unlabeled_images):,}")
        report.append(f"- **Metadata Files:** {len(self.metadata_files):,}\n")

        # Partition & Class Matrix
        report.append("## 2. Partition & Class Distribution Matrix\n")
        report.append("| Partition | Grade 0 | Grade 1 | Grade 2 | Grade 3 | Grade 4 | Total Images |")
        report.append("|---|---|---|---|---|---|---|")
        for split_name in sorted(self.split_distribution.keys()):
            dist = self.split_distribution[split_name]
            total_s = sum(dist.values())
            report.append(
                f"| **`{split_name}`** | {dist[0]:,} | {dist[1]:,} | {dist[2]:,} | {dist[3]:,} | {dist[4]:,} | **{total_s:,}** |"
            )

        total_labeled = len(self.labeled_images)
        if total_labeled > 0:
            report.append(
                f"| **TOTAL** | **{self.label_distribution[0]:,}** | **{self.label_distribution[1]:,}** | "
                f"**{self.label_distribution[2]:,}** | **{self.label_distribution[3]:,}** | **{self.label_distribution[4]:,}** | "
                f"**{total_labeled:,}** |"
            )

        # Class Imbalance Analysis
        report.append("\n## 3. Class Imbalance & Clinical Proportions\n")
        report.append("| KL Grade | Clinical Severity | Image Count | Percentage | Weight Tier |")
        report.append("|---|---|---|---|---|")
        max_cnt = max(self.label_distribution.values()) if self.label_distribution else 1
        min_cnt = min(c for c in self.label_distribution.values() if c > 0) if self.label_distribution else 1
        imbalance_ratio = max_cnt / min_cnt

        for grade in KL_CLASSES:
            cnt = self.label_distribution[grade]
            pct = (cnt / total_labeled * 100) if total_labeled > 0 else 0
            tier = "Majority" if cnt == max_cnt else ("Minority" if cnt == min_cnt else "Intermediate")
            report.append(f"| Grade {grade} | {KL_GRADE_DESCRIPTIONS[grade]} | {cnt:,} | {pct:.1f}% | {tier} |")

        report.append(f"\n- **Overall Imbalance Ratio (Max / Min Class):** `{imbalance_ratio:.2f} : 1`")

        # Derived Binary OA Screening Breakdown
        unlikely_cnt = sum(self.label_distribution[g] for g in BINARY_UNLIKELY_GRADES)
        definite_cnt = sum(self.label_distribution[g] for g in BINARY_DEFINITE_GRADES)
        report.append("\n### Derived Binary Screening Proportions\n")
        report.append(f"- **OA Unlikely (KL 0–1):** {unlikely_cnt:,} ({unlikely_cnt / total_labeled * 100:.1f}%)" if total_labeled else "-")
        report.append(f"- **Definite Radiographic OA (KL 2–4):** {definite_cnt:,} ({definite_cnt / total_labeled * 100:.1f}%)" if total_labeled else "-")

        # Patient Overlap & Data Leakage Audit
        report.append("\n## 4. Patient Isolation & Data Leakage Audit\n")
        overlaps = self.audit_patient_overlap()
        train_pts = len(self.patients_by_split.get("train", set()))
        val_pts = len(self.patients_by_split.get("val", set()))
        test_pts = len(self.patients_by_split.get("test", set()))

        report.append(f"- **Total Unique Patient Identifiers:** {len(self.patients_all):,}")
        report.append(f"- **Unique Patients in Train:** {train_pts:,}")
        report.append(f"- **Unique Patients in Validation:** {val_pts:,}")
        report.append(f"- **Unique Patients in Test:** {test_pts:,}\n")

        leakage_detected = any(len(pts) > 0 for pts in overlaps.values())
        if leakage_detected:
            report.append("> [!WARNING]\n")
            report.append("> **Patient-level data leakage detected across original splits!**\n")
            report.append(f"> - Train ∩ Val Overlap:  {len(overlaps['train_val']):,} patients\n")
            report.append(f"> - Train ∩ Test Overlap: {len(overlaps['train_test']):,} patients\n")
            report.append(f"> - Val ∩ Test Overlap:   {len(overlaps['val_test']):,} patients\n")
            report.append(">\n> *Action:* To ensure clinically rigorous benchmark evaluation, run `python src/preprocess.py --force_patient_split` to re-partition by patient.\n")
        else:
            report.append("> [!NOTE]\n")
            report.append("> **Zero patient overlap confirmed.** Patient isolation is preserved across Train, Val, and Test splits.\n")

        # Image Characteristics
        report.append("## 5. Image Characteristics & Integrity\n")
        report.append(f"- **Corrupted / Damaged Files:** {len(self.corrupted_files)}")
        if self.corrupted_files:
            report.append("\n> [!CAUTION]\n> Unreadable files detected:")
            for p, err in self.corrupted_files[:5]:
                report.append(f"- `{p.name}`: {err}")

        report.append("\n### Image Dimensions (Sampled):\n")
        for dim, count in self.image_dimensions.most_common(5):
            report.append(f"- `{dim[0]} x {dim[1]}` pixels: {count:,} images")

        report.append("\n### Color Modes:\n")
        for mode, count in self.image_modes.items():
            report.append(f"- `{mode}`: {count:,} images")

        report.append("\n## 6. Sample Image Paths\n")
        for record in self.labeled_images[:5]:
            report.append(f"- `{record['split']}/{record['kl_grade']}/{record['filename']}` (Patient: {record['patient_id']})")

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Inspect Knee Osteoarthritis Dataset (Kaggle Layout)")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=str(RAW_DATA_DIR),
        help="Root directory of raw dataset",
    )
    parser.add_argument(
        "--report_out",
        type=str,
        default=str(INSPECTION_OUTPUT_DIR / "dataset_inspection_report.md"),
        help="Destination path for inspection report",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    report_out = Path(args.report_out).resolve()
    report_out.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("      KneeWise ML Pipeline — Dataset Inspection (Kaggle)       ")
    print("=" * 65)
    print(f"[*] Scanning data directory: {data_dir}")

    inspector = DatasetInspector(data_dir)
    inspector.scan_files()
    inspector.analyze_dataset_structure()
    inspector.verify_image_integrity()

    report_md = inspector.generate_report()

    with open(report_out, "w", encoding="utf-8") as f:
        f.write(report_md)

    print(f"[+] Inspection complete!")
    print(f"[+] Full Report generated at: {report_out}")
    print("-" * 65)
    print(report_md[:1400])
    if len(report_md) > 1400:
        print("\n... [View full inspection report at destination] ...")


if __name__ == "__main__":
    main()
