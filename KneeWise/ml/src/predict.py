#!/usr/bin/env python3
"""
predict.py - KneeWise Single Image Inference Pipeline

Accepts an individual knee X-ray radiograph, validates input format,
runs model inference with PyTorch/CUDA, and returns structured JSON predictions
including 5-class Kellgren-Lawrence probabilities and derived binary OA status.
"""

import sys
import argparse
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional

import torch
from torchvision import transforms
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    BEST_MODEL_PATH,
    DEFAULT_ARCHITECTURE,
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    KL_CLASSES,
    KL_GRADE_DESCRIPTIONS,
    BINARY_CLASSES,
    BINARY_UNLIKELY_GRADES,
    BINARY_DEFINITE_GRADES,
    get_device,
)
from model import build_model, load_checkpoint


def get_inference_transform(input_size=INPUT_SIZE) -> transforms.Compose:
    """Deterministic preprocessing for inference matching training validation."""
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class KneeWisePredictor:
    """Production inference wrapper for KneeWise models."""
    def __init__(self, model_path: Path = BEST_MODEL_PATH, device: Optional[torch.device] = None):
        self.device = device if device is not None else get_device()
        self.model_path = Path(model_path)
        self.transform = get_inference_transform()
        self.model_version = "KneeWise-ResNet18-v1.0"

        self.model = build_model(architecture=DEFAULT_ARCHITECTURE, pretrained=False)
        if self.model_path.exists():
            load_checkpoint(self.model_path, self.model, device=self.device)
            self.has_trained_weights = True
        else:
            self.has_trained_weights = False

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image_input: Any) -> Dict[str, Any]:
        """Runs inference on a single image path or PIL Image object."""
        if not self.has_trained_weights:
            raise RuntimeError(
                f"Trained model checkpoint not found at: {self.model_path}. "
                "Please train the model using 'python src/train.py' before running predictions."
            )

        # Validate and open image
        if isinstance(image_input, (str, Path)):
            img_path = Path(image_input)
            if not img_path.exists():
                raise FileNotFoundError(f"Input image not found: {img_path}")
            try:
                with Image.open(img_path) as raw_img:
                    image = raw_img.convert("RGB")
            except Exception as e:
                raise ValueError(f"Unable to decode input image file {img_path}: {e}")
        elif isinstance(image_input, Image.Image):
            image = image_input.convert("RGB")
            img_path = Path("in_memory_image")
        else:
            raise TypeError("image_input must be a file path (str, Path) or PIL.Image")

        # Preprocess
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Run inference and measure latency
        start_time = time.perf_counter()
        with torch.no_grad():
            probs, p_unlikely_tensor, p_definite_tensor = self.model.predict_with_binary(tensor)
        inference_time_ms = (time.perf_counter() - start_time) * 1000.0

        # Extract values
        class_probs = probs[0].cpu().numpy()
        pred_grade = int(class_probs.argmax())

        p_unlikely = float(p_unlikely_tensor[0].cpu().numpy())
        p_definite = float(p_definite_tensor[0].cpu().numpy())

        binary_label = (
            "Definite radiographic OA" if p_definite >= 0.5 else "OA unlikely"
        )

        probabilities_dict = {
            str(grade): round(float(class_probs[grade]), 4)
            for grade in KL_CLASSES
        }

        response = {
            "severity_prediction": {
                "grade": pred_grade,
                "clinical_description": KL_GRADE_DESCRIPTIONS.get(pred_grade, "Unknown"),
                "probabilities": probabilities_dict,
            },
            "binary_prediction": {
                "label": binary_label,
                "oa_probability": round(p_definite, 4),
                "oa_unlikely_probability": round(p_unlikely, 4),
            },
            "metadata": {
                "input_image": str(img_path),
                "model_version": self.model_version,
                "device": str(self.device),
                "inference_time_ms": round(inference_time_ms, 2),
                "clinical_disclaimer": (
                    "AI-generated screening support requiring clinician interpretation. "
                    "Not an autonomous diagnostic system."
                ),
            },
        }

        return response


def main():
    parser = argparse.ArgumentParser(description="KneeWise Single Image Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to knee X-ray image")
    parser.add_argument("--model_path", type=str, default=str(BEST_MODEL_PATH), help="Path to checkpoint")
    args = parser.parse_args()

    predictor = KneeWisePredictor(model_path=Path(args.model_path))
    result = predictor.predict(args.image)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
