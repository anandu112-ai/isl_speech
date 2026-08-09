"""
Dataset Preparation Script for ISL Alphabet + Digit Recognition.

Converts any raw directory of images/videos organised by class label into the
canonical `data/isl_alphabet_digits/{train,val,test}/{label}/` structure.

Usage:
    python scripts/prepare_alphabet_dataset.py --source /path/to/raw_dataset
    python scripts/prepare_alphabet_dataset.py --source /path/to/raw --split 70 15 15
    python scripts/prepare_alphabet_dataset.py --source /path/to/raw --keep-split

Source can be:
    a) A flat directory where each sub-folder is a class label:
       raw_dataset/A/img1.jpg  raw_dataset/B/img2.jpg  ...

    b) A directory with pre-split structure:
       raw_dataset/train/A/img.jpg  raw_dataset/val/A/img.jpg  ...

The script will:
    - Normalize class names (a→A, 'zero'→0, etc.)
    - Reject unknown classes (not in A-Z or 0-9)
    - Detect and skip corrupt images
    - Remove exact-hash duplicates within each class
    - Apply 70/15/15 split (or preserve existing split)
    - Save data/metadata/alphabet_dataset.csv
    - Save models/alphabet/label_map.json
"""

import argparse
import hashlib
import logging
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError

# Allow imports from project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.labels import (
    create_deterministic_alphabet_label_map,
    get_alphabet_digit_classes,
    normalize_alphabet_label,
    save_label_map,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
TARGET_DIR = Path("data/isl_alphabet_digits")
METADATA_CSV = Path("data/metadata/alphabet_dataset.csv")
LABEL_MAP_JSON = Path("models/alphabet/label_map.json")
SPLITS = ("train", "val", "test")


def _file_hash(path: Path) -> str:
    """SHA-256 hash of file content for duplicate detection."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _image_info(path: Path):
    """Returns (width, height) or (0, 0) for corrupt images."""
    try:
        with Image.open(path) as img:
            return img.width, img.height
    except (UnidentifiedImageError, OSError):
        return 0, 0


def discover_source_files(source_dir: Path, has_split: bool):
    """
    Walks source directory, returns list of dicts:
        {path, raw_label, source_split}
    """
    files = []
    if has_split:
        for split_dir in source_dir.iterdir():
            if split_dir.name not in SPLITS or not split_dir.is_dir():
                continue
            for label_dir in split_dir.iterdir():
                if not label_dir.is_dir():
                    continue
                for img in label_dir.rglob("*"):
                    if img.suffix.lower() in SUPPORTED_EXTENSIONS:
                        files.append({
                            "path": img,
                            "raw_label": label_dir.name,
                            "source_split": split_dir.name,
                        })
    else:
        for label_dir in source_dir.iterdir():
            if not label_dir.is_dir():
                continue
            for img in label_dir.rglob("*"):
                if img.suffix.lower() in SUPPORTED_EXTENSIONS:
                    files.append({
                        "path": img,
                        "raw_label": label_dir.name,
                        "source_split": None,
                    })
    return files


def apply_split(class_files: list, ratios=(0.70, 0.15, 0.15)):
    """Assigns train/val/test split indices without leakage."""
    import random
    random.shuffle(class_files)
    n = len(class_files)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])

    for i, f in enumerate(class_files):
        if i < n_train:
            f["split"] = "train"
        elif i < n_train + n_val:
            f["split"] = "val"
        else:
            f["split"] = "test"
    return class_files


def main():
    parser = argparse.ArgumentParser(
        description="Prepare ISL Alphabet + Digit dataset (36 classes)"
    )
    parser.add_argument(
        "--source", type=str, required=True,
        help="Path to source dataset directory"
    )
    parser.add_argument(
        "--split", nargs=3, type=float, default=[70, 15, 15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Split percentages (default: 70 15 15)"
    )
    parser.add_argument(
        "--keep-split", action="store_true",
        help="Preserve existing train/val/test structure in source directory"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducible split (default: 42)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report what would happen without copying any files"
    )
    args = parser.parse_args()

    import random
    random.seed(args.seed)

    source_dir = Path(args.source)
    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Validate split ratios
    total = sum(args.split)
    ratios = tuple(r / total for r in args.split)

    print("=" * 60)
    print("ISL ALPHABET + DIGIT DATASET PREPARATION")
    print("=" * 60)
    print(f"Source       : {source_dir}")
    print(f"Target       : {TARGET_DIR}")
    print(f"Keep split   : {args.keep_split}")
    print(f"Split ratios : {ratios[0]*100:.0f}/{ratios[1]*100:.0f}/{ratios[2]*100:.0f}")
    print(f"Seed         : {args.seed}")
    print()

    label_map = create_deterministic_alphabet_label_map()
    valid_labels = set(get_alphabet_digit_classes())

    # Discover files
    files = discover_source_files(source_dir, has_split=args.keep_split)
    print(f"Discovered {len(files)} image files in source directory.")

    # Normalize labels and filter
    accepted = []
    rejected_unknown = []
    rejected_corrupt = []

    seen_hashes: dict = defaultdict(set)  # label -> set of hashes

    for f in files:
        norm = normalize_alphabet_label(f["raw_label"])
        if not norm or norm not in valid_labels:
            rejected_unknown.append(f)
            continue

        w, h = _image_info(f["path"])
        if w == 0 or h == 0:
            rejected_corrupt.append(f)
            continue

        # Duplicate detection
        fhash = _file_hash(f["path"])
        if fhash in seen_hashes[norm]:
            continue  # Skip duplicate
        seen_hashes[norm].add(fhash)

        accepted.append({
            **f,
            "label": norm,
            "class_id": label_map[norm],
            "width": w,
            "height": h,
            "status": "accepted",
        })

    print(f"Accepted      : {len(accepted)}")
    print(f"Rejected (unknown label) : {len(rejected_unknown)}")
    print(f"Rejected (corrupt/unreadable): {len(rejected_corrupt)}")
    print()

    # Apply split if not keeping original
    if not args.keep_split or accepted[0].get("source_split") is None:
        by_class: dict = defaultdict(list)
        for item in accepted:
            by_class[item["label"]].append(item)
        accepted_split = []
        for label_files in by_class.values():
            accepted_split.extend(apply_split(label_files, ratios=ratios))
        accepted = accepted_split
    else:
        for item in accepted:
            item["split"] = item["source_split"]

    # Report class counts
    print("Class sample counts:")
    by_cls_split: dict = defaultdict(lambda: defaultdict(int))
    for item in accepted:
        by_cls_split[item["label"]][item["split"]] += 1

    for label in get_alphabet_digit_classes():
        counts = by_cls_split.get(label, {})
        tr = counts.get("train", 0)
        va = counts.get("val", 0)
        te = counts.get("test", 0)
        total = tr + va + te
        print(f"  {label}: total={total}  train={tr}  val={va}  test={te}")
    print()

    if args.dry_run:
        print("DRY RUN — no files were copied.")
        return

    # Copy files to target structure
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        for label in valid_labels:
            (TARGET_DIR / split / label).mkdir(parents=True, exist_ok=True)

    copied = 0
    for item in accepted:
        src = item["path"]
        dest_dir = TARGET_DIR / item["split"] / item["label"]
        dest = dest_dir / src.name
        if dest.exists():
            # Avoid collision: append index suffix
            dest = dest_dir / f"{src.stem}_{_file_hash(src)[:6]}{src.suffix}"
        shutil.copy2(src, dest)
        item["dest_path"] = str(dest)
        copied += 1

    print(f"Copied {copied} images to {TARGET_DIR}")

    # Save metadata CSV
    METADATA_CSV.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([{
        "path": item.get("dest_path", ""),
        "split": item["split"],
        "label": item["label"],
        "class_id": item["class_id"],
        "width": item["width"],
        "height": item["height"],
        "status": item["status"],
    } for item in accepted])
    df.to_csv(METADATA_CSV, index=False)
    print(f"Metadata saved to {METADATA_CSV}")

    # Save label map
    LABEL_MAP_JSON.parent.mkdir(parents=True, exist_ok=True)
    save_label_map(label_map, LABEL_MAP_JSON)
    print(f"Label map saved to {LABEL_MAP_JSON}")
    print()
    print("Dataset preparation COMPLETE.")
    print(f"Next: python scripts/inspect_alphabet_dataset.py")


if __name__ == "__main__":
    main()
