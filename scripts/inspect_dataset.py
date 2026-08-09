"""
Dataset Inspection and Video Quality Check Script for INCLUDE-50.
Reports video counts, FPS, duration, resolution distributions, corrupt/empty files,
and generates data/metadata/video_quality.csv.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import pandas as pd


def main():
    selected_csv = Path("data/metadata/selected_videos.csv")
    video_base_dir = Path("data/videos")
    quality_csv = Path("data/metadata/video_quality.csv")

    if not selected_csv.exists():
        print(f"Error: Selection CSV not found: {selected_csv}")
        sys.exit(1)

    df = pd.read_csv(selected_csv)
    print("=" * 70)
    print("INCLUDE-50 DATASET & VIDEO QUALITY INSPECTION")
    print("=" * 70)

    quality_rows = []
    durations = []
    fps_list = []
    resolutions = {}

    unreadable_count = 0
    empty_count = 0
    valid_count = 0

    for _, row in df.iterrows():
        vpath_norm = str(row["video_path"]).replace("\\", "/")
        label = str(row["label"])
        split = str(row["split"])
        filename = Path(vpath_norm).name

        local_file = video_base_dir / split / label / filename

        status = "valid"
        error_msg = ""
        fps = 0.0
        frame_count = 0
        duration = 0.0
        width = 0
        height = 0

        if not local_file.exists():
            status = "invalid"
            error_msg = "File missing"
            unreadable_count += 1
        elif local_file.stat().st_size == 0:
            status = "invalid"
            error_msg = "0-byte file"
            empty_count += 1
        else:
            cap = cv2.VideoCapture(str(local_file))
            if not cap.isOpened():
                status = "invalid"
                error_msg = "OpenCV cannot open file"
                unreadable_count += 1
            else:
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()

                if frame_count == 0:
                    status = "invalid"
                    error_msg = "0 frames in video"
                    empty_count += 1
                else:
                    duration = frame_count / fps if fps > 0 else 0.0
                    durations.append(duration)
                    if fps > 0:
                        fps_list.append(fps)
                    res_key = f"{width}x{height}"
                    resolutions[res_key] = resolutions.get(res_key, 0) + 1
                    valid_count += 1

        quality_rows.append(
            {
                "video_path": vpath_norm,
                "split": split,
                "label": label,
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "width": width,
                "height": height,
                "status": status,
                "error": error_msg,
            }
        )

    quality_df = pd.DataFrame(quality_rows)
    quality_csv.parent.mkdir(parents=True, exist_ok=True)
    quality_df.to_csv(quality_csv, index=False)

    print(f"\nTotal videos selected   : {len(df)}")
    print(f"Train videos           : {len(df[df['split'] == 'train'])}")
    print(f"Validation videos      : {len(df[df['split'] == 'val'])}")
    print(f"Number of classes      : {df['label'].nunique()}")
    print(f"Valid videos found     : {valid_count}")
    print(f"Unreadable videos      : {unreadable_count}")
    print(f"Empty/0-frame videos   : {empty_count}")

    if durations:
        print(f"\nVideo Duration (sec)   : Min = {min(durations):.2f}s, Max = {max(durations):.2f}s, Avg = {sum(durations)/len(durations):.2f}s")
    if fps_list:
        print(f"FPS Range              : Min = {min(fps_list):.1f}, Max = {max(fps_list):.1f}, Avg = {sum(fps_list)/len(fps_list):.1f}")
    if resolutions:
        print(f"Resolutions            : {resolutions}")

    print(f"\nSaved quality report to: {quality_csv}")
    print("Done.")


if __name__ == "__main__":
    main()
