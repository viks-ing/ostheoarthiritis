"""
dataset.py - KneeWise PyTorch Dataset & Transform Pipeline

Handles on-the-fly medical image loading, RGB conversion, conservative clinical
data augmentation, ImageNet normalization, and DataLoader instantiation.
"""

from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import (
    INPUT_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    DEFAULT_BATCH_SIZE,
    NUM_WORKERS,
    use_pinned_memory,
)


def get_train_transforms(input_size: Tuple[int, int] = INPUT_SIZE) -> transforms.Compose:
    """
    Returns training transformations with conservative, clinically-safe augmentations.
    Preserves anatomical joint space geometry without severe distortion.
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        # Mild rotation: knees can be slightly tilted during positioning (+/- 7 degrees)
        transforms.RandomRotation(degrees=7),
        # Subtle translation (+/- 4% shift) simulating slight patient re-centering
        transforms.RandomAffine(degrees=0, translate=(0.04, 0.04)),
        # Mild brightness & contrast shifts simulating varied exposure levels
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        # Mild horizontal flip: left/right knee symmetry is valid for general feature learning
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def get_val_test_transforms(input_size: Tuple[int, int] = INPUT_SIZE) -> transforms.Compose:
    """
    Returns deterministic validation and test transformations without random augmentation.
    """
    return transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class KneeXRayDataset(Dataset):
    """
    PyTorch Dataset for Kellgren-Lawrence Knee Osteoarthritis classification.
    Reads image paths and KL grades from a manifest DataFrame or CSV.
    """
    def __init__(
        self,
        manifest: Union[pd.DataFrame, str, Path],
        transform: Optional[transforms.Compose] = None,
        is_test: bool = False,
    ):
        if isinstance(manifest, (str, Path)):
            manifest_path = Path(manifest)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Manifest file not found: {manifest_path}")
            self.df = pd.read_csv(manifest_path)
        elif isinstance(manifest, pd.DataFrame):
            self.df = manifest.copy()
        else:
            raise TypeError("manifest must be a DataFrame or path to CSV.")

        # Ensure required columns exist
        required_cols = {"image_path"}
        if not is_test:
            required_cols.add("kl_grade")

        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Manifest missing required columns: {missing}")

        self.transform = transform
        self.is_test = is_test

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, Dict[str, Any]]:
        row = self.df.iloc[idx]
        img_path = Path(row["image_path"])

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        try:
            # Load image and ensure 3-channel RGB for ImageNet architectures
            with Image.open(img_path) as raw_img:
                img = raw_img.convert("RGB")
        except Exception as e:
            raise RuntimeError(f"Failed to read image {img_path}: {e}")

        if self.transform is not None:
            img_tensor = self.transform(img)
        else:
            to_tensor = transforms.ToTensor()
            img_tensor = to_tensor(img)

        label = int(row["kl_grade"]) if not self.is_test and "kl_grade" in row else -1

        metadata = {
            "image_path": str(img_path),
            "patient_id": str(row.get("patient_id", "unknown")),
            "split": str(row.get("split", "unknown")),
        }

        return img_tensor, label, metadata


def create_dataloaders(
    train_manifest: Union[pd.DataFrame, str, Path],
    val_manifest: Union[pd.DataFrame, str, Path],
    test_manifest: Optional[Union[pd.DataFrame, str, Path]] = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    input_size: Tuple[int, int] = INPUT_SIZE,
) -> Dict[str, DataLoader]:
    """
    Constructs train, validation, and optional test DataLoaders.
    Optimized for 4 GB VRAM GPU environments with pinned memory.
    """
    pin_mem = use_pinned_memory()

    train_ds = KneeXRayDataset(
        manifest=train_manifest,
        transform=get_train_transforms(input_size),
    )
    val_ds = KneeXRayDataset(
        manifest=val_manifest,
        transform=get_val_test_transforms(input_size),
    )

    loaders = {
        "train": DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        ),
        "val": DataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        ),
    }

    if test_manifest is not None:
        test_ds = KneeXRayDataset(
            manifest=test_manifest,
            transform=get_val_test_transforms(input_size),
        )
        loaders["test"] = DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_mem,
            drop_last=False,
        )

    return loaders
