
"""
Training Script for ISL Alphabet (A-Z) + Digit (0-9) CNN Classifier.

Usage:
    python scripts/train_alphabet.py
    python scripts/train_alphabet.py --resume
    python scripts/train_alphabet.py --phase2
    python scripts/train_alphabet.py --config configs/alphabet_config.yaml
"""

import argparse
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alphabet.dataset import build_dataloaders
from src.alphabet.model import build_model
from src.alphabet.preprocessing import build_eval_transform, build_train_transform
from src.alphabet.train import AlphabetTrainer
from src.utils.config import load_config
from src.utils.labels import (
    create_deterministic_alphabet_label_map,
    load_label_map,
    save_label_map,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def detect_hardware() -> torch.device:
    """Detects and reports available hardware."""
    print("=" * 60)
    print("HARDWARE DETECTION")
    print("=" * 60)
    import platform
    import psutil
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    print(f"OS           : {platform.system()} {platform.release()}")
    print(f"RAM          : {ram_gb:.1f} GB")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA         : {'Available' if cuda_avail else 'Not available (using CPU)'}")
    if cuda_avail:
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"GPU          : {gpu_name}")
        print(f"VRAM         : {vram:.1f} GB")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Training on  : {device}")
    print()
    return device


def set_seed(seed: int) -> None:
    """Sets random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_alphabet_config(config_path: str) -> dict:
    """Loads alphabet_config.yaml."""
    import yaml
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Train ISL Alphabet + Digit CNN Classifier (36 classes)"
    )
    parser.add_argument(
        "--config", type=str, default="configs/alphabet_config.yaml",
        help="Path to alphabet config YAML"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume training from latest checkpoint"
    )
    parser.add_argument(
        "--phase2", action="store_true",
        help="Run Phase 2 backbone fine-tuning after Phase 1"
    )
    args = parser.parse_args()

    # Load config
    config = load_alphabet_config(args.config)
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    preproc_cfg = config.get("preprocessing", {})
    dataset_cfg = config.get("dataset", {})

    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    device = detect_hardware()

    print("=" * 60)
    print("ISL ALPHABET + DIGIT RECOGNITION — TRAINING")
    print("=" * 60)

    # Dataset & label map
    data_dir = Path(dataset_cfg.get("target_dir", "data/isl_alphabet_digits"))
    label_map_path = Path(dataset_cfg.get("label_map", "models/alphabet/label_map.json"))

    if not data_dir.exists():
        print(f"ERROR: Dataset not found at {data_dir}")
        print("Run: python scripts/prepare_alphabet_dataset.py --source <raw_dataset>")
        sys.exit(1)

    # Load or create label map
    if label_map_path.exists():
        label_map = load_label_map(label_map_path)
        logger.info(f"Loaded label map from {label_map_path} ({len(label_map)} classes)")
    else:
        label_map = create_deterministic_alphabet_label_map()
        save_label_map(label_map, label_map_path)
        logger.info(f"Created and saved label map to {label_map_path}")

    # Transforms
    image_size = preproc_cfg.get("image_size", 224)
    h_flip = preproc_cfg.get("horizontal_flip", False)
    rotation = preproc_cfg.get("rotation_degrees", 10)
    translate = preproc_cfg.get("translate", 0.05)
    bc = preproc_cfg.get("brightness_contrast", 0.1)

    train_transform = build_train_transform(
        image_size=image_size,
        horizontal_flip=h_flip,
        rotation_degrees=rotation,
        translate=translate,
        brightness_contrast=bc,
    )
    eval_transform = build_eval_transform(image_size=image_size)

    # DataLoaders
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    train_loader, val_loader, _ = build_dataloaders(
        root_dir=data_dir,
        label_map=label_map,
        train_transform=train_transform,
        val_transform=eval_transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if len(train_loader.dataset) == 0:
        print("ERROR: No training samples found! Check dataset structure and labels.")
        sys.exit(1)

    print(f"Train samples : {len(train_loader.dataset)}")
    print(f"Val samples   : {len(val_loader.dataset)}")
    print(f"Batch size    : {batch_size}")
    print(f"Device        : {device}")
    print()

    # Model
    model = build_model(config, num_classes=len(label_map))
    model.to(device)
    model.freeze_backbone()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model         : {model_cfg.get('name', 'mobilenet_v3_small')}")
    print(f"Parameters    : {total:,} total | {trainable:,} trainable")
    print()

    # Checkpoint & best model paths
    ckpt_dir = Path("models/alphabet/checkpoints")
    best_dir = Path("models/alphabet/best_model")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)

    # Copy label map next to best model dir
    save_label_map(label_map, best_dir / "label_map.json")

    # Optional class weights
    class_weights = None
    if train_cfg.get("use_class_weights", False):
        class_weights = train_loader.dataset.get_class_weights()
        logger.info(f"Class weights: {class_weights}")

    # Trainer
    trainer = AlphabetTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        label_map=label_map,
        device=device,
        checkpoint_dir=ckpt_dir,
        best_model_dir=best_dir,
        class_weights=class_weights,
    )

    start_epoch = 0
    if args.resume:
        start_epoch = trainer.resume()

    # Phase 1: Train head
    t0 = time.time()
    history = trainer.train(start_epoch=start_epoch)

    # Phase 2: Fine-tuning (optional)
    if args.phase2 and model_cfg.get("fine_tune_backbone", False):
        unfreeze_n = model_cfg.get("unfreeze_layers", 2)
        fine_tune_history = trainer.fine_tune(unfreeze_layers=unfreeze_n)
        history.extend(fine_tune_history)

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"Training complete in {elapsed/60:.1f} min.")
    print(f"Best val accuracy : {trainer.best_val_accuracy:.2f}%")
    print(f"Best model saved  : {best_dir / 'best.pt'}")
    print("=" * 60)
    print()
    print("Next: python scripts/evaluate_alphabet.py")


if __name__ == "__main__":
    main()
