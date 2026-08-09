"""
Model Training Script for INCLUDE-50 Sign Language Recognition.
Builds BiLSTM model, sets reproducible random seeds, supports CUDA/CPU, handles training/validation loops,
checkpointing, early stopping, and training history reporting.
"""

import argparse
import json
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from src.dataset.sign_dataset import create_dataloaders
from src.models.bilstm import SignBiLSTMModel
from src.training.trainer import ModelTrainer
from src.utils.config import load_config


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train INCLUDE-50 Sign Recognition Model")
    parser.add_argument("--resume", action="store_true", help="Resume training from latest.pt checkpoint")
    parser.add_argument("--config", type=str, default="configs/config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    set_seed(config["training"].get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("INCLUDE-50 MODEL TRAINING (BiLSTM)")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Epochs: {config['training']['epochs']}")
    print(f"Batch size: {config['training']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")

    manifest_csv = Path(config["dataset"]["features_manifest"])
    label_map_json = Path(config["dataset"]["label_map"])

    if not manifest_csv.exists() or not label_map_json.exists():
        print("Error: Preprocessing features or label map not found! Run preprocess_dataset.py first.")
        sys.exit(1)

    with open(label_map_json, "r", encoding="utf-8") as f:
        label_map = json.load(f)

    train_loader, val_loader = create_dataloaders(
        manifest_csv=manifest_csv,
        label_map_json=label_map_json,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        augment_train=True,
    )

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples  : {len(val_loader.dataset)}")
    print(f"Num classes  : {len(label_map)}")

    model = SignBiLSTMModel(
        feature_dim=225,
        proj_dim=128,
        hidden_size=config["training"]["hidden_size"],
        num_layers=config["training"]["num_layers"],
        dropout=config["training"]["dropout"],
        num_classes=len(label_map),
    )

    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        label_map=label_map,
        config=config,
        device=device,
    )

    results = trainer.run(resume=args.resume)

    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best Validation Accuracy: {results['best_val_accuracy']*100:.2f}%")
    print(f"Best Model Saved To     : {results['best_model_path']}")
    print("Training history saved  : reports/training_history.csv")
    print("Training curves saved   : reports/training_curves.png")


if __name__ == "__main__":
    main()
