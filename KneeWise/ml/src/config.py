"""
config.py - Central Configuration for the KneeWise ML Pipeline

Maintains all file paths, dataset schema, clinical label definitions,
hyperparameters, and hardware-specific configurations.
"""

from pathlib import Path
import warnings
import torch

import tempfile
import os

# Ensure matplotlib uses a writable temp directory across all platforms
mpl_cache = Path(tempfile.gettempdir()) / "kneewise_matplotlib"
mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(mpl_cache)

# ==============================================================================
# 1. Directory Structure and Relative Paths
# ==============================================================================
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent  # Resolves to ml/

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_OUTPUT_DIR = OUTPUTS_DIR / "eda"
EVALUATION_OUTPUT_DIR = OUTPUTS_DIR / "evaluation"
TRAINING_OUTPUT_DIR = OUTPUTS_DIR / "training"
INSPECTION_OUTPUT_DIR = OUTPUTS_DIR / "inspection"

# Manifest Files
TRAIN_MANIFEST_CSV = PROCESSED_DATA_DIR / "train_manifest.csv"
VAL_MANIFEST_CSV = PROCESSED_DATA_DIR / "val_manifest.csv"
TEST_MANIFEST_CSV = PROCESSED_DATA_DIR / "test_manifest.csv"
FULL_MANIFEST_CSV = PROCESSED_DATA_DIR / "full_dataset_manifest.csv"

# Model Checkpoint Paths
BEST_MODEL_PATH = MODELS_DIR / "best_kneewise_resnet18.pth"
LATEST_CHECKPOINT_PATH = MODELS_DIR / "latest_checkpoint.pth"
ONNX_EXPORT_PATH = MODELS_DIR / "kneewise_resnet18.onnx"

# ==============================================================================
# 2. Clinical & Severity Class Definitions (Kellgren-Lawrence Grading System)
# ==============================================================================
NUM_CLASSES = 5

KL_CLASSES = [0, 1, 2, 3, 4]

KL_GRADE_DESCRIPTIONS = {
    0: "Grade 0: No radiographic OA (Normal joint space, no osteophytes)",
    1: "Grade 1: Doubtful OA (Possible osteophytic lipping, doubtful joint space narrowing)",
    2: "Grade 2: Mild radiographic OA (Definite osteophytes, possible joint space narrowing)",
    3: "Grade 3: Moderate radiographic OA (Multiple osteophytes, definite joint space narrowing, some sclerosis)",
    4: "Grade 4: Severe radiographic OA (Large osteophytes, marked joint space narrowing, severe sclerosis)",
}

KL_GRADE_SHORT_LABELS = {
    0: "KL0 - Normal",
    1: "KL1 - Doubtful",
    2: "KL2 - Mild",
    3: "KL3 - Moderate",
    4: "KL4 - Severe",
}

# Binary OA Mapping (Screening-Support Rule)
# KL 0-1: OA unlikely
# KL 2-4: Definite radiographic OA
BINARY_CLASSES = ["OA unlikely", "Definite radiographic OA"]
BINARY_UNLIKELY_GRADES = [0, 1]
BINARY_DEFINITE_GRADES = [2, 3, 4]

# ==============================================================================
# 3. Model Architecture & Training Hyperparameters
# ==============================================================================
DEFAULT_ARCHITECTURE = "resnet18"
INPUT_SIZE = (224, 224)
IN_CHANNELS = 3  # Standard for ImageNet pretrained backbones

# Pretrained ImageNet Normalization Constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Optimized for NVIDIA RTX 3050 (4 GB VRAM) & 8 GB RAM on Windows
DEFAULT_BATCH_SIZE = 8
FALLBACK_BATCH_SIZE = 4
DEFAULT_EPOCHS = 10
DEFAULT_LR = 0.0001
DEFAULT_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 3

# Dataloader optimization for Windows: Avoid multiprocessing lockups
NUM_WORKERS = 0  # Conservative for Windows; set to 2 if verified stable

# Reproducibility Seed
RANDOM_SEED = 42

# ==============================================================================
# 4. Device Management and Hardware Acceleration
# ==============================================================================
def get_device() -> torch.device:
    """
    Selects CUDA device if available, otherwise raises explicit warning and falls back to CPU.
    Never silently falls back to CPU.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_name = torch.cuda.get_device_name(0)
        # Verify device properties
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return device
    else:
        warnings.warn(
            "\n" + "!" * 70 + "\n"
            "[HARDWARE WARNING] CUDA is NOT available! Falling back to CPU.\n"
            "Training on CPU will be substantially slower and will not utilize your RTX 3050 GPU.\n"
            "Please ensure CUDA-enabled PyTorch is installed if an NVIDIA GPU is present.\n"
            + "!" * 70,
            RuntimeWarning,
            stacklevel=2,
        )
        return torch.device("cpu")

def use_mixed_precision() -> bool:
    """Returns True if GPU is available and Ampere/modern architecture supports FP16 autocast."""
    return torch.cuda.is_available()

def use_pinned_memory() -> bool:
    """Returns True if GPU is available to enable faster host-to-device transfers."""
    return torch.cuda.is_available()
