"""
Validation utilities for INCLUDE-50 Remote ZIP pipeline.
Includes CRC32 calculation, ZIP path traversal safety, extracted file validation,
and complete dataset integrity verification.
"""

import zlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def calculate_crc32(file_path: Path) -> int:
    """
    Computes unsigned 32-bit CRC32 checksum for a local file by streaming in 64KB chunks.
    """
    crc = 0
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def is_safe_path(base_dir: Path, target_path: Path) -> bool:
    """
    Prevents ZIP path traversal attacks by ensuring the target path remains inside base_dir.
    """
    try:
        resolved_base = base_dir.resolve()
        resolved_target = target_path.resolve()
        return resolved_target.is_relative_to(resolved_base)
    except Exception:
        return False


def validate_extracted_file(
    file_path: Path, expected_crc32: int, expected_size: Optional[int] = None
) -> Tuple[bool, str]:
    """
    Validates an extracted video file: checks existence, size > 0, optional expected size, and CRC32.
    Returns (is_valid, error_message).
    """
    if not file_path.exists():
        return False, f"File does not exist: {file_path}"

    actual_size = file_path.stat().st_size
    if actual_size == 0:
        return False, f"Extracted file is 0 bytes: {file_path}"

    if expected_size is not None and actual_size != expected_size:
        return False, f"Size mismatch: expected {expected_size} bytes, got {actual_size} bytes"

    actual_crc = calculate_crc32(file_path)
    if actual_crc != expected_crc32:
        return (
            False,
            f"CRC32 mismatch for {file_path.name}: expected 0x{expected_crc32:08X}, got 0x{actual_crc:08X}",
        )

    return True, "Valid"


def verify_dataset_integrity(
    selected_csv: Path = Path("data/metadata/selected_videos.csv"),
    video_base_dir: Path = Path("data/videos"),
) -> Dict[str, Any]:
    """
    Comprehensive dataset verification against authoritative selected_videos.csv metadata.
    Checks:
    1. Expected videos (650)
    2. Found videos
    3. Missing videos
    4. Corrupt / 0-byte videos
    5. Duplicate video paths
    6. Train count (500)
    7. Validation count (150)
    8. Classes count (50)
    9. Train/validation data leakage
    10. Class balance (10 train, 3 val per class)
    """
    if not selected_csv.exists():
        return {
            "passed": False,
            "error": f"Metadata file missing: {selected_csv}",
        }

    df = pd.read_csv(selected_csv)
    expected_total = len(df)
    expected_train = len(df[df["split"] == "train"])
    expected_val = len(df[df["split"] == "val"])

    # Normalization: Path label authoritative, format slashes
    df["clean_video_path"] = df["video_path"].astype(str).str.replace("\\", "/", regex=False)

    # Check duplicate video_path entries in CSV
    csv_duplicates = df["clean_video_path"].duplicated().sum()

    missing_files: List[str] = []
    corrupt_files: List[str] = []
    found_train = 0
    found_val = 0
    class_split_counts: Dict[str, Dict[str, int]] = {}

    for _, row in df.iterrows():
        label = str(row["label"])
        split = str(row["split"])
        vpath = str(row["clean_video_path"])
        filename = Path(vpath).name

        target_file = video_base_dir / split / label / filename

        if label not in class_split_counts:
            class_split_counts[label] = {"train": 0, "val": 0}

        if not target_file.exists():
            missing_files.append(vpath)
            continue

        if target_file.stat().st_size == 0:
            corrupt_files.append(f"0-byte file: {vpath}")
            continue

        # File exists and non-empty
        if split == "train":
            found_train += 1
            class_split_counts[label]["train"] += 1
        elif split == "val":
            found_val += 1
            class_split_counts[label]["val"] += 1

    total_found = found_train + found_val

    # Data leakage check: train paths vs val paths
    train_paths = set(df[df["split"] == "train"]["clean_video_path"])
    val_paths = set(df[df["split"] == "val"]["clean_video_path"])
    leakage = train_paths.intersection(val_paths)

    # Class balance check
    unbalanced_classes = []
    for cls, counts in class_split_counts.items():
        if counts["train"] != 10 or counts["val"] != 3:
            unbalanced_classes.append((cls, counts["train"], counts["val"]))

    passed = (
        expected_total == 650
        and total_found == 650
        and len(missing_files) == 0
        and len(corrupt_files) == 0
        and csv_duplicates == 0
        and found_train == 500
        and found_val == 150
        and len(class_split_counts) == 50
        and len(leakage) == 0
        and len(unbalanced_classes) == 0
    )

    return {
        "passed": passed,
        "expected_total": expected_total,
        "found_total": total_found,
        "missing_count": len(missing_files),
        "corrupt_count": len(corrupt_files),
        "duplicate_count": csv_duplicates,
        "found_train": found_train,
        "found_val": found_val,
        "num_classes": len(class_split_counts),
        "leakage_count": len(leakage),
        "unbalanced_classes": unbalanced_classes,
        "missing_files": missing_files,
        "corrupt_files": corrupt_files,
    }
