"""
Single-Image Inference Script for ISL Alphabet (A-Z) + Digit (0-9) Recognition.

Usage:
    python scripts/predict_alphabet.py --image "path/to/image.jpg"
    python scripts/predict_alphabet.py --image "path/to/image.png" --threshold 0.60
    python scripts/predict_alphabet.py --image "img.jpg" --hand-crop

Output:
    Prediction : A
    Confidence : 96.32%

    Top 5 Predictions:
      1. A  — 96.32%
      2. H  —  1.21%
      3. E  —  0.95%
      4. N  —  0.43%
      5. M  —  0.31%
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

BEST_MODEL_DIR = Path("models/alphabet/best_model")
LABEL_MAP_PATH = Path("models/alphabet/label_map.json")
CONFIG_PATH = Path("configs/alphabet_config.yaml")


def load_alphabet_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ISL Alphabet + Digit Single-Image Inference"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to input image file (.jpg, .png, etc.)"
    )
    parser.add_argument(
        "--threshold", type=float, default=None,
        help="Confidence threshold 0.0-1.0 (overrides config)"
    )
    parser.add_argument(
        "--model-dir", type=str, default=str(BEST_MODEL_DIR),
        help="Path to directory containing best.pt and label_map.json"
    )
    parser.add_argument(
        "--hand-crop", action="store_true",
        help="Enable MediaPipe hand detection and bounding-box crop before classification"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on"
    )
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)

    config = load_alphabet_config(CONFIG_PATH)
    inf_cfg = config.get("inference", {})
    preproc_cfg = config.get("preprocessing", {})

    threshold = args.threshold if args.threshold is not None else inf_cfg.get("confidence_threshold", 0.70)
    image_size = preproc_cfg.get("image_size", 224)
    device_str = args.device if args.device != "auto" else inf_cfg.get("device", "auto")

    from src.alphabet.inference import AlphabetInferenceEngine

    try:
        engine = AlphabetInferenceEngine(
            best_model_dir=Path(args.model_dir),
            label_map_path=LABEL_MAP_PATH if LABEL_MAP_PATH.exists() else None,
            confidence_threshold=threshold,
            device_str=device_str,
            hand_crop=args.hand_crop,
            image_size=image_size,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Train the model first: python scripts/train_alphabet.py")
        sys.exit(1)

    try:
        result = engine.predict_from_path(image_path)
    except Exception as e:
        print(f"ERROR during inference: {e}")
        sys.exit(1)
    finally:
        engine.close()

    # Format output
    print()
    print("=" * 40)
    print("  ISL SIGN RECOGNITION — RESULT")
    print("=" * 40)
    print(f"  Image      : {image_path.name}")
    print(f"  Prediction : {result['prediction']}")
    print(f"  Confidence : {result['confidence']:.2f}%")
    if not result["is_certain"]:
        print(f"  ⚠  Below confidence threshold ({threshold*100:.0f}%) — prediction uncertain")
    print()
    print("  Top 5 Predictions:")
    for rank, (label, pct) in enumerate(result["top5"], 1):
        marker = "★" if rank == 1 else " "
        print(f"  {marker} {rank}. {label:<4} — {pct:6.2f}%")
    print()


if __name__ == "__main__":
    main()
