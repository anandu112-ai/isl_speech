"""
Dataset Verification Script for INCLUDE-50 Remote ZIP Pipeline.
Validates extracted dataset structure, video existence, non-zero file sizes, class balance (10 train, 3 val per class),
train/val split totals (500 train, 150 val), and absence of data leakage or duplicates.
"""

import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.validation import verify_dataset_integrity


def main():
    selected_csv = Path("data/metadata/selected_videos.csv")
    video_base_dir = Path("data/videos")

    res = verify_dataset_integrity(selected_csv, video_base_dir)

    print("Dataset verification")
    print("====================")
    print()
    print(f"Expected videos: {res.get('expected_total', 0)}")
    print(f"Found videos:    {res.get('found_total', 0)}")
    print(f"Missing:         {res.get('missing_count', 0)}")
    print(f"Corrupt:         {res.get('corrupt_count', 0)}")
    print(f"Duplicate:       {res.get('duplicate_count', 0)}")
    print()
    print(f"Train:           {res.get('found_train', 0)}")
    print(f"Validation:      {res.get('found_val', 0)}")
    print()
    print(f"Classes:         {res.get('num_classes', 0)}")
    if res.get("leakage_count", 0) > 0:
        print(f"Data Leakage:    {res['leakage_count']} files in both train & val!")

    if res.get("unbalanced_classes"):
        print("\nUnbalanced classes (Expected 10 train, 3 val):")
        for cls, tr, val in res["unbalanced_classes"]:
            print(f"  Class '{cls}': train={tr}, val={val}")

    print()
    if res.get("passed"):
        print("RESULT: PASS")
        sys.exit(0)
    else:
        print("RESULT: FAIL")
        sys.exit(1)


if __name__ == "__main__":
    main()
