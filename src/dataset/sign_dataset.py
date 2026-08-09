"""
PyTorch Dataset and DataLoader for INCLUDE-50 Sign Language Landmarks.
Loads .npz feature matrices on demand and applies optional training landmark augmentations.
"""

import json
import random
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SignLandmarkDataset(Dataset):
    """
    PyTorch Dataset reading landmark features (.npz) and class labels.
    """

    def __init__(
        self,
        manifest_csv: Path = Path("data/metadata/features_manifest.csv"),
        label_map_json: Path = Path("data/metadata/label_map.json"),
        split: str = "train",
        augment: bool = False,
    ):
        self.split = split
        self.augment = augment

        with open(label_map_json, "r", encoding="utf-8") as f:
            self.label_map = json.load(f)

        df = pd.read_csv(manifest_csv)
        self.samples = df[(df["split"] == split) & (df["status"] == "processed")].to_dict("records")

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_features(self, features: np.ndarray) -> np.ndarray:
        """
        Applies lightweight landmark coordinate augmentation:
        - Small Gaussian noise
        - Small spatial scale jitter
        - Small translation jitter
        """
        augmented = features.copy()

        # Scale jitter (95% - 105%)
        scale = random.uniform(0.95, 1.05)
        augmented = augmented * scale

        # Translation jitter (-0.02 to 0.02)
        shift = random.uniform(-0.02, 0.02)
        augmented = augmented + shift

        # Gaussian noise
        noise = np.random.normal(0.0, 0.005, size=augmented.shape)
        augmented = augmented + noise

        return augmented.astype(np.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        feat_path = Path(sample["feature_path"])
        label_name = str(sample["label"])

        class_id = self.label_map[label_name]

        with np.load(feat_path) as data:
            features = data["features"].astype(np.float32)  # Shape (32, 225)

        if self.augment and self.split == "train":
            features = self._augment_features(features)

        features_tensor = torch.from_numpy(features)  # (32, 225)
        label_tensor = torch.tensor(class_id, dtype=torch.long)

        return features_tensor, label_tensor


def create_dataloaders(
    manifest_csv: Path = Path("data/metadata/features_manifest.csv"),
    label_map_json: Path = Path("data/metadata/label_map.json"),
    batch_size: int = 16,
    num_workers: int = 0,
    augment_train: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    """
    Creates PyTorch DataLoaders for train and validation splits.
    """
    train_dataset = SignLandmarkDataset(
        manifest_csv=manifest_csv,
        label_map_json=label_map_json,
        split="train",
        augment=augment_train,
    )

    val_dataset = SignLandmarkDataset(
        manifest_csv=manifest_csv,
        label_map_json=label_map_json,
        split="val",
        augment=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return train_loader, val_loader
