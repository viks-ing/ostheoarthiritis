"""
inference.py - KneeWise Standalone Inference Engine
Production-ready, deterministic inference pipeline utilizing the frozen champion
checkpoint (EfficientNet-B0) with cryptographic SHA-256 integrity verification.
"""

import os
import sys
import time
import io
import hashlib
from pathlib import Path
from typing import Dict, Any, Union, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
# Path constants
ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CHAMPION_PATH = ROOT_DIR / "models" / "frozen_champion" / "efficientnet_b0_champion.pth"
EXPECTED_CHAMPION_SHA256 = "d8b65306864362f2d83bcf94ee3f1940ea87feb7ce19ff08736c5dc881e2dc1b"

# Import KneeWiseClassifier to ensure exact architecture parity
sys.path.insert(0, str(ROOT_DIR / "src"))
from model import KneeWiseClassifier

KL_CLASSES = [0, 1, 2, 3, 4]
KL_GRADE_DESCRIPTIONS = {
    0: "Grade 0: No radiographic OA (Normal joint space, no osteophytes)",
    1: "Grade 1: Doubtful OA (Possible osteophytic lipping, doubtful joint space narrowing)",
    2: "Grade 2: Mild radiographic OA (Definite osteophytes, possible joint space narrowing)",
    3: "Grade 3: Moderate radiographic OA (Multiple osteophytes, marked joint space narrowing, some sclerosis)",
    4: "Grade 4: Severe radiographic OA (Large osteophytes, severe joint space narrowing, severe subchondral sclerosis)",
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
PROVISIONAL_SCREENING_THRESHOLD = 0.30


def calculate_sha256(file_path: Union[str, Path]) -> str:
    """Computes SHA-256 hash for a given file."""
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192 * 1024):
            h.update(chunk)
    return h.hexdigest()


class KneeWiseInferenceEngine:
    """
    KneeWise Standalone Inference Engine.
    Loads frozen EfficientNet-B0 champion checkpoint, validates cryptographic integrity,
    applies clinical preprocessing, and outputs 5-class severity & binary screening risk.
    """
    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CHAMPION_PATH,
        device: Optional[str] = None,
        verify_checksum: bool = True,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Champion checkpoint not found at: {self.checkpoint_path}")

        # Cryptographic Verification
        if verify_checksum:
            actual_sha = calculate_sha256(self.checkpoint_path)
            if actual_sha != EXPECTED_CHAMPION_SHA256:
                raise RuntimeError(
                    f"Integrity check failed for {self.checkpoint_path}!\n"
                    f"Expected SHA-256: {EXPECTED_CHAMPION_SHA256}\n"
                    f"Actual SHA-256:   {actual_sha}\n"
                    "Model weights appear tampered with or corrupted."
                )
            self.sha256 = actual_sha
        else:
            self.sha256 = "verification_skipped"

        # Device determination
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Reconstruct exact architecture
        self.model = self._build_model()
        self._load_checkpoint()
        self.model.to(self.device)
        self.model.eval()

        # Deterministic preprocessing transforms
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ])

    def _build_model(self) -> nn.Module:
        """Constructs KneeWiseClassifier with identical 5-class classification head."""
        return KneeWiseClassifier(
            backbone_name="efficientnet_b0",
            num_classes=len(KL_CLASSES),
            pretrained=False,
            dropout_p=0.3,
        )

    def _load_checkpoint(self):
        """Loads state dictionary strictly."""
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in ckpt:
            self.model.load_state_dict(ckpt["model_state_dict"])
            self.best_epoch = ckpt.get("epoch", 5)
            self.val_macro_f1 = ckpt.get("val_macro_f1", 0.6505)
        else:
            self.model.load_state_dict(ckpt)
            self.best_epoch = "N/A"
            self.val_macro_f1 = "N/A"

    def preprocess_image(self, image_input: Union[str, Path, bytes, io.BytesIO, Image.Image]) -> torch.Tensor:
        """
        Validates input and applies deterministic RGB ImageNet preprocessing.
        Supports file path, raw bytes, or PIL Image.
        """
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Image file does not exist: {img_path}")
            try:
                with Image.open(img_path) as raw_img:
                    pil_img = raw_img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to decode image from path {img_path}: {e}")
        elif isinstance(image_input, bytes):
            try:
                with Image.open(io.BytesIO(image_input)) as raw_img:
                    pil_img = raw_img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to decode image from raw bytes: {e}")
        elif isinstance(image_input, io.BytesIO):
            try:
                with Image.open(image_input) as raw_img:
                    pil_img = raw_img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to decode image from BytesIO: {e}")
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")

        # Transform to tensor
        tensor = self.transform(pil_img)  # Shape: (3, 224, 224)
        return tensor.unsqueeze(0)        # Shape: (1, 3, 224, 224)

    @torch.no_grad()
    def predict(
        self,
        image_input: Union[str, Path, bytes, io.BytesIO, Image.Image],
        screening_threshold: float = PROVISIONAL_SCREENING_THRESHOLD,
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference on a single knee radiograph.
        Returns 5-class severity distribution and derived binary screening status.
        """
        t0 = time.perf_counter()

        # 1. Preprocess
        input_tensor = self.preprocess_image(image_input).to(self.device)

        # 2. Forward pass & Softmax
        logits = self.model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # 3. Five-class predictions
        pred_grade = int(np.argmax(probs))
        prob_dict = {f"KL_{g}": float(np.round(probs[g], 4)) for g in KL_CLASSES}

        # 4. Binary screening risk: P(Definite OA) = P(KL2) + P(KL3) + P(KL4)
        prob_definite_oa = float(np.round(probs[2] + probs[3] + probs[4], 4))
        is_oa_likely = prob_definite_oa >= screening_threshold

        category = "Definite Radiographic OA Likely (Refer for Clinical Evaluation)" if is_oa_likely else "OA Radiographically Unlikely / Doubtful"

        return {
            "prediction": {
                "kellgren_lawrence_grade": pred_grade,
                "grade_description": KL_GRADE_DESCRIPTIONS[pred_grade],
                "confidence": float(round(probs[pred_grade], 4)),
                "probabilities": prob_dict,
            },
            "binary_screening": {
                "screening_category": category,
                "is_oa_likely": bool(is_oa_likely),
                "binary_oa_probability": float(round(prob_definite_oa, 4)),
                "unlikely_oa_probability": float(round(1.0 - prob_definite_oa, 4)),
                "screening_threshold_used": float(round(screening_threshold, 2)),
                "threshold_notice": "Threshold 0.30 is a provisional research operating point optimizing sensitivity; not an autonomous clinical diagnostic threshold.",
            },
            "model_metadata": {
                "model_name": "KneeWise-EfficientNet-B0-Champion",
                "experiment_id": "exp_d_efficientnet_b0",
                "checkpoint_sha256": self.sha256,
                "device": str(self.device),
                "inference_time_ms": float(round(latency_ms, 2)),
            },
            "disclaimer": (
                "KneeWise is an AI screening-support tool, not an autonomous diagnostic system. "
                "All predictions must be clinically correlated with patient symptoms and verified by a licensed radiologist or healthcare provider."
            ),
        }


def get_inference_engine(device: Optional[str] = None) -> KneeWiseInferenceEngine:
    """Factory helper returning singleton or fresh instance."""
    return KneeWiseInferenceEngine(device=device)


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KneeWise Standalone Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to knee X-ray image (PNG/JPEG)")
    parser.add_argument("--threshold", type=float, default=PROVISIONAL_SCREENING_THRESHOLD, help="Binary screening threshold (default: 0.30)")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    args = parser.parse_args()

    engine = KneeWiseInferenceEngine(device=args.device)
    result = engine.predict(args.image, screening_threshold=args.threshold)
    print(json.dumps(result, indent=2))
