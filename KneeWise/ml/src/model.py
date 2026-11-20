"""
model.py - KneeWise Deep Learning Model Architectures

Implements pretrained ResNet18 (and modular EfficientNet-B0) with custom
5-class Kellgren-Lawrence classification head, frozen backbone transfer learning,
progressive fine-tuning utilities, and combined 5-class + binary probability output.
"""

from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn
from torchvision import models

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    NUM_CLASSES,
    DEFAULT_ARCHITECTURE,
    BINARY_UNLIKELY_GRADES,
    BINARY_DEFINITE_GRADES,
)


class KneeWiseClassifier(nn.Module):
    """
    KneeWise Osteoarthritis Severity Classifier.
    Supports ResNet18 and EfficientNet-B0 backbones.
    """
    def __init__(
        self,
        backbone_name: str = DEFAULT_ARCHITECTURE,
        num_classes: int = NUM_CLASSES,
        pretrained: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.num_classes = num_classes

        if self.backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            in_features = self.backbone.fc.in_features
            # Replace classifier head with dropout + linear
            self.backbone.fc = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
            self.classifier_layer = self.backbone.fc

        elif self.backbone_name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            self.backbone = models.efficientnet_b0(weights=weights)
            in_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(p=dropout_p),
                nn.Linear(in_features, num_classes),
            )
            self.classifier_layer = self.backbone.classifier

        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}. Choose 'resnet18' or 'efficientnet_b0'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw classification logits of shape (batch_size, num_classes)."""
        return self.backbone(x)

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Applies softmax over raw logits to return calibrated probabilities."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict_with_binary(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          - class_probs: shape (batch_size, 5)
          - p_unlikely: shape (batch_size,) = P(KL 0) + P(KL 1)
          - p_definite: shape (batch_size,) = P(KL 2) + P(KL 3) + P(KL 4)
        """
        probs = self.predict_probabilities(x)
        p_unlikely = probs[:, BINARY_UNLIKELY_GRADES].sum(dim=1)
        p_definite = probs[:, BINARY_DEFINITE_GRADES].sum(dim=1)
        return probs, p_unlikely, p_definite

    def freeze_backbone(self):
        """Freezes all layers except the newly initialized classification head."""
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze head
        for param in self.classifier_layer.parameters():
            param.requires_grad = True

    def unfreeze_backbone(self, unfreeze_all: bool = True):
        """Unfreezes backbone parameters for progressive fine-tuning."""
        if unfreeze_all:
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze top feature layers only
            if self.backbone_name == "resnet18":
                for layer in [self.backbone.layer4, self.backbone.fc]:
                    for param in layer.parameters():
                        param.requires_grad = True
            elif self.backbone_name == "efficientnet_b0":
                for layer in [self.backbone.features[-2:], self.backbone.classifier]:
                    for param in layer.parameters():
                        param.requires_grad = True

    def count_trainable_parameters(self) -> Tuple[int, int]:
        """Returns (trainable_params, total_params)."""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        return trainable, total


def build_model(
    architecture: str = DEFAULT_ARCHITECTURE,
    num_classes: int = NUM_CLASSES,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout_p: float = 0.3,
) -> KneeWiseClassifier:
    """Factory function for initializing and configuring a KneeWise model."""
    model = KneeWiseClassifier(
        backbone_name=architecture,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
    )
    if freeze_backbone:
        model.freeze_backbone()
    return model


def save_checkpoint(
    model: nn.Module,
    epoch: int,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    val_loss: float,
    val_acc: float,
    val_macro_f1: float,
    config: Dict[str, Any],
    save_path: Path,
):
    """Saves model checkpoint with training state and hyperparameter metadata."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "val_loss": val_loss,
        "val_acc": val_acc,
        "val_macro_f1": val_macro_f1,
        "config": config,
    }
    torch.save(state, save_path)


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Loads checkpoint weights and resumes training state if requested."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint
