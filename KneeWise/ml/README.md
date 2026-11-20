# KneeWise — Machine Learning Pipeline

An AI-powered Knee Osteoarthritis (KOA) screening and severity grading system developed for clinical decision support in underserved communities.

> [!IMPORTANT]
> **Clinical AI Classification Notice:**
> KneeWise is designed as an **AI-assisted screening support tool**, not an autonomous diagnostic system or a replacement for board-certified radiologists. All AI predictions must be clinically corroborated by medical professionals alongside patient history, physical examination, and symptoms.

---

## 1. Clinical & Technical Objectives

KneeWise processes digital knee radiographs to deliver two clinically-aligned outputs from a single calibrated model:

1. **Five-Class Kellgren–Lawrence (KL) Severity Grading:**
   - **Grade 0:** No radiographic OA (Normal joint space, absence of osteophytes).
   - **Grade 1:** Doubtful OA (Possible osteophytic lipping, doubtful joint space narrowing).
   - **Grade 2:** Mild OA (Definite osteophytes, potential joint space narrowing).
   - **Grade 3:** Moderate OA (Multiple osteophytes, definite joint space narrowing, sclerosis).
   - **Grade 4:** Severe OA (Large osteophytes, marked joint space narrowing, severe bone sclerosis).

2. **Derived Binary Radiographic OA Screening:**
   Instead of training an independent, potentially discordant secondary binary model, binary probabilities are derived directly from the five-class probability distribution:
   $$\mathcal{P}(\text{OA unlikely}) = \mathcal{P}(\text{KL 0}) + \mathcal{P}(\text{KL 1})$$
   $$\mathcal{P}(\text{Definite radiographic OA}) = \mathcal{P}(\text{KL 2}) + \mathcal{P}(\text{KL 3}) + \mathcal{P}(\text{KL 4})$$

---

## 2. Hardware Constraints & Optimizations

This pipeline is engineered specifically for development on:
- **GPU:** NVIDIA GeForce RTX 3050 Laptop GPU (4 GB VRAM, Ampere Architecture, Compute Capability 8.6)
- **RAM:** 8 GB Host RAM
- **OS:** Windows 11 / Python 3.11

### Applied Optimizations:
| Hardware Constraint | Pipeline Solution |
|---|---|
| **4 GB VRAM Ceiling** | Default batch size of `8` (fallback to `4` if needed); ResNet-18 lightweight backbone (~11.2M params). |
| **Windows DataLoader IPC** | `num_workers=0` (conservative default) to prevent worker deadlocks and RAM overhead. |
| **Host-to-Device Bandwidth** | Pinned memory enabled (`pin_memory=True`) when CUDA is active. |
| **Tensor Core Acceleration** | PyTorch Native Automatic Mixed Precision (`torch.amp.autocast`) with `GradScaler` for FP16 training. |
| **Memory Leak Prevention** | On-the-fly preprocessing using CSV manifests—raw images are never loaded monolithically into RAM. |
| **Fail-Safe Warnings** | `src/config.py` raises loud runtime warnings if CUDA falls back to CPU. |

---

## 3. Directory Structure

```text
KneeWise/
└── ml/
    ├── data/
    │   ├── raw/                 # Downloaded Mendeley dataset archives & extracted folders
    │   └── processed/           # Train/Val/Test CSV manifests (no raw image duplicates)
    ├── notebooks/
    │   └── exploration.ipynb    # Interactive Exploratory Data Analysis
    ├── src/
    │   ├── config.py            # Central configuration, clinical labels, hyperparameters
    │   ├── verify_gpu.py        # CUDA hardware diagnostics & VRAM monitoring
    │   ├── inspect_data.py      # Recursive dataset inspector & integrity checker
    │   ├── eda.py               # Automated EDA script & plot generator
    │   ├── preprocess.py        # Manifest generator & patient leakage auditor
    │   ├── dataset.py           # PyTorch Dataset, transforms & DataLoader factory
    │   ├── model.py             # ResNet18 & EfficientNet-B0 architectures
    │   ├── train.py             # Mixed-precision training pipeline with early stopping
    │   ├── evaluate.py          # 5-class & binary evaluation with confusion matrices
    │   ├── predict.py           # Single-image inference returning structured JSON
    │   └── export_onnx.py       # ONNX exporter & numerical tolerance validator
    ├── models/                  # Checkpoints (.pth) and ONNX exports (.onnx)
    ├── outputs/
    │   ├── eda/                 # Class distributions, sample grids, intensity plots
    │   ├── inspection/          # Markdown dataset inspection reports
    │   ├── evaluation/          # Confusion matrices and evaluation reports
    │   └── training/            # Training curves and history CSVs
    ├── requirements.txt         # Compatible Python dependencies
    └── README.md                # Project documentation
```

---

## 4. Setup & Execution Instructions

### Step 1: Environment Setup & GPU Verification

1. Install PyTorch with CUDA 12.1 acceleration:
   ```powershell
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```
2. Install remaining dependencies:
   ```powershell
   cd KneeWise/ml
   pip install -r requirements.txt
   ```
3. Run the hardware verification utility:
   ```powershell
   python src/verify_gpu.py
   ```
   *Expected output: Displays CUDA availability, RTX 3050 name, 4.00 GB VRAM, and runs a GPU matrix multiplication test.*

---

### Step 2: Dataset Acquisition & Verification

You can retrieve the dataset automatically or manually:

#### Option A: Using the Automated Downloader (`src/download_dataset.py`)
1. Inspect public dataset metadata without downloading:
   ```powershell
   python src/download_dataset.py --metadata-only
   ```
2. Automated download:
   - If downloading directly with an OAuth token:
     ```powershell
     python src/download_dataset.py --token "<YOUR_MENDELEY_TOKEN>"
     ```
   - If downloading from a direct S3 or mirror link:
     ```powershell
     python src/download_dataset.py --url "<DIRECT_DOWNLOAD_URL>"
     ```
3. Verify integrity of an existing downloaded archive (`KneeXrayData.zip`):
   ```powershell
   python src/download_dataset.py --verify-only
   ```

#### Option B: Direct 1-Click Browser Download (Simplest & Most Reliable)
1. Open the Mendeley Data page in your browser:
   [https://data.mendeley.com/datasets/56rmx5bjcr/1](https://data.mendeley.com/datasets/56rmx5bjcr/1)
2. Click the blue **"Download all (6.68 GB)"** button (free, no API key or token required).
3. Move the downloaded `KneeXrayData.zip` (or extracted contents) into:
   ```text
   KneeWise/ml/data/raw/
   ```

#### Understanding Dataset Components:
- `ClsKLData` (or `kneeKL224/`): Contains pre-cropped knee joints organized by KL grade (`0`, `1`, `2`, `3`, `4`). **This is our target for classification.**
- `DetJointData`: Contains full radiograph images with bounding boxes for YOLO joint localization. **Do NOT mix with classification data.**

---

### Step 3: Run Dataset Inspection

Verify raw images, detect file corruptions, and separate `ClsKL` from `DetJoint`:
```powershell
python src/inspect_data.py
```
*Outputs a detailed inspection report to `outputs/inspection/dataset_inspection_report.md`.*

---

### Step 4: Exploratory Data Analysis (EDA)

Run automated EDA and generate publication-ready plots:
```powershell
python src/eda.py
```
Or open the interactive notebook:
```powershell
jupyter notebook notebooks/exploration.ipynb
```
*Outputs generated in `outputs/eda/`:*
- `class_distribution.png`: Bar chart of KL grades 0–4 with class percentages.
- `representative_samples.png`: Visual grid comparing radiographs across all five grades.
- `pixel_intensity_distribution.png`: Grayscale intensity histograms.

---

### Step 5: Data Preprocessing & Patient-Level Manifest Generation

Build reproducible manifests and audit against patient data leakage:
```powershell
python src/preprocess.py
```
- Audits patient identifiers between splits.
- If patient overlap is detected in original partitions, use:
  ```powershell
  python src/preprocess.py --force_patient_split
  ```
- Creates `data/processed/train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`, and `full_dataset_manifest.csv`.

---

### Step 6: Model Training

Train the ResNet18 baseline with ImageNet transfer learning and mixed precision (AMP):
```powershell
python src/train.py --architecture resnet18 --batch_size 8 --epochs 10 --lr 0.0001
```
Key features:
- Automatically computes balanced class weights to manage class imbalance.
- Saves best checkpoint to `models/best_kneewise_resnet18.pth` based on Validation Macro F1.
- Generates `outputs/training/training_curves.png` and `outputs/training/training_history.csv`.

---

### Step 7: Comprehensive Evaluation

Evaluate the best model checkpoint on the untouched test set:
```powershell
python src/evaluate.py
```
Generates:
- 5-class metrics: Accuracy, Precision, Recall, Macro F1, Weighted F1.
- Derived binary metrics: Clinical Sensitivity, Specificity, Precision, Recall, F1.
- False negative & false positive counts for clinical screening assessment.
- Plots saved in `outputs/evaluation/`:
  - `confusion_matrix_5class.png`
  - `confusion_matrix_binary.png`
  - `evaluation_report.md`

---

### Step 8: Single-Image Clinical Prediction (Inference)

Run prediction on any individual knee radiograph:
```powershell
python src/predict.py --image "data/raw/kneeKL224/test/2/sample.png"
```
**Example Structured Output:**
```json
{
  "severity_prediction": {
    "grade": 2,
    "clinical_description": "Grade 2: Mild radiographic OA (Definite osteophytes, possible joint space narrowing)",
    "probabilities": {
      "0": 0.0421,
      "1": 0.1154,
      "2": 0.6328,
      "3": 0.1582,
      "4": 0.0515
    }
  },
  "binary_prediction": {
    "label": "Definite radiographic OA",
    "oa_probability": 0.8425,
    "oa_unlikely_probability": 0.1575
  },
  "metadata": {
    "input_image": "sample.png",
    "model_version": "KneeWise-ResNet18-v1.0",
    "device": "cuda:0",
    "inference_time_ms": 14.82,
    "clinical_disclaimer": "AI-generated screening support requiring clinician interpretation. Not an autonomous diagnostic system."
  }
}
```

---

### Step 9: ONNX Model Export

Export the PyTorch model into ONNX format for future backend integration (Django/FastAPI):
```powershell
python src/export_onnx.py
```
- Validates the ONNX computational graph.
- Confirms numerical equivalence between PyTorch and ONNX Runtime ($< 10^{-4}$ tolerance).
- Saves exported graph to `models/kneewise_resnet18.onnx`.

---

## 5. Medical AI Safety & Governance Considerations

1. **Dataset Provenance & Demographic Limits:**
   - The Mendeley dataset is derived from the Osteoarthritis Initiative (OAI) cohort in the United States.
   - Algorithms trained on this dataset risk performance degradation (domain shift) when deployed in populations with differing demographics, imaging equipment, beam calibration, or positioning protocols.
2. **Patient-Level Data Leakage:**
   - X-rays from the same patient (e.g. left and right knees, or longitudinal follow-up visits) must never appear simultaneously in training and test sets. The `preprocess.py` module strictly verifies patient separation.
3. **Clinical Intended Use:**
   - KneeWise is intended to assist general practitioners, physiotherapists, and community health workers in triage and rehabilitation prioritization.
   - It is explicitly **not** an autonomous diagnostic device.
