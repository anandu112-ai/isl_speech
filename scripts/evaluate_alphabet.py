"""
Evaluation Script for ISL Alphabet (A-Z) + Digit (0-9) CNN Classifier.

Evaluates the best trained model on the TEST split ONLY.
NEVER uses test data for model selection — only final reporting.

Usage:
    python scripts/evaluate_alphabet.py
    python scripts/evaluate_alphabet.py --config configs/alphabet_config.yaml

Outputs:
    reports/alphabet/classification_report.csv
    reports/alphabet/confusion_matrix.png
    reports/alphabet/evaluation_summary.txt
"""

import argparse
import logging
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.alphabet.dataset import ISLAlphabetDataset, build_dataloaders
from src.alphabet.evaluate import AlphabetEvaluator
from src.alphabet.model import load_checkpoint
from src.alphabet.preprocessing import build_eval_transform
from src.utils.labels import create_deterministic_alphabet_label_map, load_label_map

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_alphabet_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate ISL Alphabet + Digit classifier on test set"
    )
    parser.add_argument(
        "--config", type=str, default="configs/alphabet_config.yaml",
        help="Path to alphabet config YAML"
    )
    parser.add_argument(
        "--model", type=str, default="models/alphabet/best_model/best.pt",
        help="Path to best.pt checkpoint"
    )
    args = parser.parse_args()

    config = load_alphabet_config(args.config)
    train_cfg = config.get("training", {})
    preproc_cfg = config.get("preprocessing", {})
    dataset_cfg = config.get("dataset", {})

    # Device
    device_str = config.get("inference", {}).get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    best_model_path = Path(args.model)
    if not best_model_path.exists():
        print(f"ERROR: Model checkpoint not found: {best_model_path}")
        print("Train first: python scripts/train_alphabet.py")
        sys.exit(1)

    # Label map
    label_map_path = Path(dataset_cfg.get("label_map", "models/alphabet/label_map.json"))
    if label_map_path.exists():
        label_map = load_label_map(label_map_path)
    else:
        label_map = create_deterministic_alphabet_label_map()

    # Load model
    model, ckpt = load_checkpoint(best_model_path, device, num_classes=len(label_map))

    print("=" * 60)
    print("ISL ALPHABET + DIGIT — EVALUATION (TEST SET)")
    print("=" * 60)
    print(f"Model         : {best_model_path}")
    print(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    print(f"Best val acc  : {ckpt.get('best_val_accuracy', '?')}%")
    print(f"Device        : {device}")
    print(f"Classes       : {len(label_map)}")
    print()

    # Test DataLoader
    data_dir = Path(dataset_cfg.get("target_dir", "data/isl_alphabet_digits"))
    image_size = preproc_cfg.get("image_size", 224)
    eval_transform = build_eval_transform(image_size=image_size)
    batch_size = train_cfg.get("batch_size", 32)
    num_workers = train_cfg.get("num_workers", 0)

    test_ds = ISLAlphabetDataset(data_dir, "test", label_map, transform=eval_transform)
    if len(test_ds) == 0:
        print(f"ERROR: No test samples found in {data_dir}/test/")
        sys.exit(1)

    from torch.utils.data import DataLoader
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    print(f"Test samples  : {len(test_ds)}")

    # Evaluate
    evaluator = AlphabetEvaluator(
        model=model,
        device=device,
        label_map=label_map,
        reports_dir=Path("reports/alphabet"),
    )

    print("Running evaluation...")
    results = evaluator.run(test_loader)
    evaluator.save_reports(results)


if __name__ == "__main__":
    main()
