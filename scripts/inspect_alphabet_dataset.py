"""
Dataset Inspection Script for ISL Alphabet + Digit Dataset.

Inspects data/isl_alphabet_digits/ and reports:
    - Total, train, val, test sample counts
    - Per-class counts (A-Z, 0-9)
    - Corrupt/unreadable files
    - Unsupported format files
    - Class imbalance warnings
    - Generates reports/alphabet/dataset_report.txt

Usage:
    python scripts/inspect_alphabet_dataset.py
    python scripts/inspect_alphabet_dataset.py --data-dir custom/path
"""

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image, UnidentifiedImageError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.labels import get_alphabet_digit_classes, load_label_map

logging.basicConfig(level=logging.WARNING)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DATA_DIR = Path("data/isl_alphabet_digits")
LABEL_MAP_JSON = Path("models/alphabet/label_map.json")
REPORT_FILE = Path("reports/alphabet/dataset_report.txt")
SPLITS = ("train", "val", "test")


def check_image(path: Path):
    """Checks if image is readable. Returns (ok, width, height)."""
    try:
        with Image.open(path) as img:
            w, h = img.width, img.height
            return True, w, h
    except (UnidentifiedImageError, OSError):
        return False, 0, 0


def main():
    parser = argparse.ArgumentParser(
        description="Inspect ISL Alphabet + Digit Dataset"
    )
    parser.add_argument(
        "--data-dir", type=str, default=str(DATA_DIR),
        help=f"Dataset directory (default: {DATA_DIR})"
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"ERROR: Dataset directory not found: {data_dir}")
        print("Run: python scripts/prepare_alphabet_dataset.py --source <your_raw_dataset>")
        sys.exit(1)

    target_classes = get_alphabet_digit_classes()

    stats = {
        "total": 0,
        "by_split": defaultdict(int),
        "by_class": defaultdict(int),
        "by_split_class": defaultdict(lambda: defaultdict(int)),
        "corrupt": [],
        "unsupported": [],
        "tiny": [],  # images smaller than 32x32
    }

    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
        for label_dir in sorted(split_dir.iterdir()):
            if not label_dir.is_dir():
                continue
            label = label_dir.name
            for img_path in sorted(label_dir.iterdir()):
                if img_path.is_dir():
                    continue
                ext = img_path.suffix.lower()
                if ext not in SUPPORTED_EXTS:
                    stats["unsupported"].append(str(img_path))
                    continue
                ok, w, h = check_image(img_path)
                if not ok:
                    stats["corrupt"].append(str(img_path))
                    continue
                if w < 32 or h < 32:
                    stats["tiny"].append(str(img_path))
                stats["total"] += 1
                stats["by_split"][split] += 1
                stats["by_class"][label] += 1
                stats["by_split_class"][split][label] += 1

    lines = []
    lines.append("=" * 60)
    lines.append("ISL ALPHABET + DIGIT DATASET INSPECTION REPORT")
    lines.append("=" * 60)
    lines.append(f"Dataset directory : {data_dir}")
    lines.append(f"Total samples     : {stats['total']}")
    lines.append(f"  Train           : {stats['by_split']['train']}")
    lines.append(f"  Val             : {stats['by_split']['val']}")
    lines.append(f"  Test            : {stats['by_split']['test']}")
    lines.append(f"Corrupt files     : {len(stats['corrupt'])}")
    lines.append(f"Unsupported files : {len(stats['unsupported'])}")
    lines.append(f"Tiny images (<32px): {len(stats['tiny'])}")
    lines.append("")

    lines.append("Per-Class Sample Counts:")
    lines.append(f"{'Class':<8} {'Train':>8} {'Val':>6} {'Test':>6} {'Total':>8}")
    lines.append("-" * 42)

    all_totals = []
    missing_classes = []
    for cls in target_classes:
        tr = stats["by_split_class"]["train"].get(cls, 0)
        va = stats["by_split_class"]["val"].get(cls, 0)
        te = stats["by_split_class"]["test"].get(cls, 0)
        total = tr + va + te
        all_totals.append(total)
        if total == 0:
            missing_classes.append(cls)
        lines.append(f"  {cls:<6} {tr:>8} {va:>6} {te:>6} {total:>8}")

    lines.append("")

    # Imbalance detection
    if all_totals:
        max_count = max(all_totals)
        min_count = min(all_totals)
        imbalance_ratio = max_count / max(min_count, 1)
        lines.append(f"Class balance: max={max_count}, min={min_count}, ratio={imbalance_ratio:.1f}x")

        if imbalance_ratio > 5:
            lines.append("  [WARNING] Severe class imbalance detected (>5x ratio).")
            lines.append("   Consider using use_class_weights: true in alphabet_config.yaml.")
        elif imbalance_ratio > 2:
            lines.append("  [NOTICE] Moderate class imbalance detected (>2x ratio).")

    if missing_classes:
        lines.append("")
        lines.append(f"  [MISSING CLASSES] (0 samples): {missing_classes}")
        lines.append("   These classes currently have 0 samples!")

    if stats["corrupt"]:
        lines.append("")
        lines.append("Corrupt files:")
        for f in stats["corrupt"][:10]:
            lines.append(f"  {f}")
        if len(stats["corrupt"]) > 10:
            lines.append(f"  ... and {len(stats['corrupt'])-10} more.")

    lines.append("")
    lines.append("RESULT: " + ("OK" if not missing_classes and not stats["corrupt"] else "WARNINGS FOUND"))

    report_text = "\n".join(lines)
    print(report_text)

    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    REPORT_FILE.write_text(report_text, encoding="utf-8")
    print(f"\nReport saved to {REPORT_FILE}")


if __name__ == "__main__":
    main()
