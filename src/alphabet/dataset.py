"""
ISL Alphabet + Digit Dataset Module.

Loads images from the structured directory:
    data/isl_alphabet_digits/{split}/{label}/

Returns (image_tensor, class_id) pairs for PyTorch DataLoader.
"""

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ISLAlphabetDataset(Dataset):
    """
    PyTorch Dataset for ISL Alphabet (A-Z) and Digit (0-9) image classification.

    Expects directory structure:
        root_dir/{split}/{label}/image.jpg

    Args:
        root_dir:    Path to `data/isl_alphabet_digits/`
        split:       One of 'train', 'val', or 'test'
        label_map:   Dict mapping label string -> class_id (e.g. {'A': 0, ..., '9': 35})
        transform:   Torchvision transforms to apply
    """

    def __init__(
        self,
        root_dir: Path,
        split: str,
        label_map: Dict[str, int],
        transform: Optional[Callable] = None,
    ):
        self.split_dir = Path(root_dir) / split
        self.label_map = label_map
        self.id_to_label: Dict[int, str] = {v: k for k, v in label_map.items()}
        self.transform = transform

        self.samples: List[Tuple[Path, int]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Scans split directory and populates self.samples with (path, class_id) tuples."""
        if not self.split_dir.exists():
            logger.warning(f"Split directory does not exist: {self.split_dir}")
            return

        skipped = 0
        for label_dir in sorted(self.split_dir.iterdir()):
            if not label_dir.is_dir():
                continue

            label = label_dir.name
            if label not in self.label_map:
                logger.warning(f"Unknown label directory ignored: {label_dir}")
                skipped += 1
                continue

            class_id = self.label_map[label]
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                    continue
                self.samples.append((img_path, class_id))

        logger.info(f"[{self.split_dir.name}] Loaded {len(self.samples)} samples. Skipped {skipped} unknown dirs.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, class_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Failed to open image {img_path}: {e}. Returning black image.")
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))

        if self.transform:
            image = self.transform(image)

        return image, class_id

    def get_class_counts(self) -> Dict[str, int]:
        """Returns dict of {label: sample_count} for all target classes."""
        counts: Dict[str, int] = {label: 0 for label in self.label_map}
        for _, class_id in self.samples:
            label = self.id_to_label[class_id]
            counts[label] += 1
        return counts

    def get_class_weights(self) -> torch.Tensor:
        """
        Returns normalized inverse-frequency class weights for CrossEntropyLoss.
        Classes with zero samples are assigned weight = 1.0.
        """
        counts = self.get_class_counts()
        ordered = [counts[self.id_to_label[i]] for i in range(len(self.label_map))]
        totals = np.array(ordered, dtype=np.float32)
        weights = np.where(totals > 0, 1.0 / totals, 1.0)
        weights = weights / weights.sum() * len(weights)  # normalize
        return torch.from_numpy(weights)


def build_dataloaders(
    root_dir: Path,
    label_map: Dict[str, int],
    train_transform: Callable,
    val_transform: Callable,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Builds train/val/test DataLoaders from the structured dataset directory.

    Returns:
        train_loader, val_loader, test_loader
    """
    train_ds = ISLAlphabetDataset(root_dir, "train", label_map, transform=train_transform)
    val_ds = ISLAlphabetDataset(root_dir, "val", label_map, transform=val_transform)
    test_ds = ISLAlphabetDataset(root_dir, "test", label_map, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    logger.info(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")
    return train_loader, val_loader, test_loader
