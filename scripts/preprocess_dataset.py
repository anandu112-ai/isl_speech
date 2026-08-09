"""
Preprocessing and Feature Extraction CLI Script for INCLUDE-50.
Samples 32 frames per video, extracts MediaPipe Holistic 225-dim landmarks,
saves compressed .npz feature files, generates label_map.json, and maintains features_manifest.csv.
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from src.features.extractor import LandmarkExtractor
from src.preprocessing.processor import process_video_file
from src.utils.config import load_config


def generate_label_map(selected_df: pd.DataFrame, label_map_path: Path) -> dict:
    """
    Creates a deterministic, sorted label-to-class_id dictionary and saves label_map.json.
    """
    unique_labels = sorted(selected_df["label"].astype(str).unique())
    label_map = {label: idx for idx, label in enumerate(unique_labels)}

    label_map_path.parent.mkdir(parents=True, exist_ok=True)
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=4)

    return label_map


def main():
    parser = argparse.ArgumentParser(description="INCLUDE-50 Preprocessing & Landmark Feature Extraction")
    parser.add_argument("--force", action="store_true", help="Re-extract features even if .npz file already exists")
    parser.add_argument("--num-frames", type=int, default=32, help="Number of frames to sample per video (default: 32)")
    args = parser.parse_args()

    config = load_config()

    selected_csv = Path(config["dataset"]["selected_csv"])
    video_base_dir = Path(config["dataset"]["video_dir"])
    features_base_dir = Path(config["dataset"]["features_dir"])
    features_manifest_csv = Path(config["dataset"]["features_manifest"])
    label_map_json = Path(config["dataset"]["label_map"])

    if not selected_csv.exists():
        print(f"Error: Selected videos CSV not found at {selected_csv}")
        sys.exit(1)

    df = pd.read_csv(selected_csv)

    print("=" * 70)
    print("INCLUDE-50 VIDEO PREPROCESSING & FEATURE EXTRACTION")
    print("=" * 70)
    print(f"Selected videos  : {len(df)}")
    print(f"Target frames    : {args.num_frames}")
    print(f"Features directory: {features_base_dir}")

    # Generate label map
    label_map = generate_label_map(df, label_map_json)
    print(f"Classes generated: {len(label_map)} (Saved to {label_map_json})")

    # Load or initialize features manifest
    manifest_rows = []
    if features_manifest_csv.exists() and not args.force:
        manifest_df = pd.read_csv(features_manifest_csv)
        manifest_dict = {row["video_path"]: row for row in manifest_df.to_dict("records")}
    else:
        manifest_dict = {}

    extractor = LandmarkExtractor(
        min_detection_confidence=config["preprocessing"]["min_detection_confidence"],
        min_tracking_confidence=config["preprocessing"]["min_tracking_confidence"],
    )

    processed_count = 0
    skipped_count = 0
    failed_count = 0
    failed_items = []

    start_time = time.time()

    try:
        for idx, row in df.iterrows():
            vpath_norm = str(row["video_path"]).replace("\\", "/")
            label = str(row["label"])
            split = str(row["split"])
            filename = Path(vpath_norm).name

            local_video = video_base_dir / split / label / filename
            output_npz = features_base_dir / split / label / f"{local_video.stem}.npz"

            if not args.force and output_npz.exists() and output_npz.stat().st_size > 0:
                skipped_count += 1
                manifest_dict[vpath_norm] = {
                    "video_path": vpath_norm,
                    "split": split,
                    "label": label,
                    "feature_path": output_npz.as_posix(),
                    "num_frames": args.num_frames,
                    "feature_dimension": 225,
                    "status": "processed",
                    "error": "",
                }
                continue

            print(f"[{idx+1}/{len(df)}] Processing: {vpath_norm}")
            success, msg, shape = process_video_file(
                video_path=local_video,
                output_npz_path=output_npz,
                extractor=extractor,
                num_frames=args.num_frames,
            )

            if success:
                processed_count += 1
                manifest_dict[vpath_norm] = {
                    "video_path": vpath_norm,
                    "split": split,
                    "label": label,
                    "feature_path": output_npz.as_posix(),
                    "num_frames": shape[0],
                    "feature_dimension": shape[1],
                    "status": "processed",
                    "error": "",
                }
            else:
                failed_count += 1
                failed_items.append((vpath_norm, msg))
                manifest_dict[vpath_norm] = {
                    "video_path": vpath_norm,
                    "split": split,
                    "label": label,
                    "feature_path": output_npz.as_posix(),
                    "num_frames": 0,
                    "feature_dimension": 0,
                    "status": "failed",
                    "error": msg,
                }
                print(f"  FAILED: {msg}")

            # Save manifest periodically
            if (processed_count + failed_count) % 10 == 0:
                pd.DataFrame(list(manifest_dict.values())).to_csv(features_manifest_csv, index=False)

    finally:
        extractor.close()
        # Save final manifest
        pd.DataFrame(list(manifest_dict.values())).to_csv(features_manifest_csv, index=False)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)
    print(f"Total processed: {processed_count}")
    print(f"Total skipped  : {skipped_count}")
    print(f"Total failed   : {failed_count}")
    print(f"Elapsed time   : {elapsed:.2f} s")
    print(f"Features manifest saved to: {features_manifest_csv}")

    if failed_items:
        print("\nFAILED VIDEOS:")
        for vp, err in failed_items:
            print(f"  {vp} -> {err}")


if __name__ == "__main__":
    main()
