"""
Estimate Download Size Script for INCLUDE-50 Remote ZIP Pipeline.
Inspects central directories of all required archives remotely, calculates exact
compressed byte-range download size and extracted dataset size without downloading video payloads.
"""

import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests

from tools.remote_zip import get_central_directory, parse_central_directory


def main():
    selected_csv = Path("data/metadata/selected_videos.csv")
    if not selected_csv.exists():
        print(f"Error: Selection file {selected_csv} not found.")
        sys.exit(1)

    selected = pd.read_csv(selected_csv)
    selected["clean_vpath"] = selected["video_path"].astype(str).str.replace("\\", "/", regex=False)

    archives = (
        selected[["archive", "archive_url"]]
        .drop_duplicates()
        .sort_values("archive")
    )

    print("=" * 70)
    print("INCLUDE-50 REMOTE VIDEO DOWNLOAD SIZE ESTIMATOR")
    print("=" * 70)

    print(f"\nSelected videos : {len(selected)}")
    print(f"Archives needed : {len(archives)}")

    session = requests.Session()

    total_compressed = 0
    total_original = 0
    found_videos = 0
    missing_videos = []
    failed_archives = []
    unsupported_compression = []

    for index, row in archives.iterrows():
        archive_name = row["archive"]
        url = row["archive_url"]

        group = selected[selected["archive"] == archive_name]

        print("\n" + "-" * 70)
        print(f"[{found_videos + len(failed_archives)}/{len(selected)}]")
        print(f"Archive: {archive_name}")
        print(f"Videos needed: {len(group)}")

        try:
            cd_bytes, cd_offset, entries_count = get_central_directory(url, session=session)
            print(f"Central directory: {len(cd_bytes) / 1024:.1f} KB")
            print(f"ZIP entries: {entries_count}")

            entries = parse_central_directory(cd_bytes)

            archive_compressed = 0
            archive_original = 0
            archive_found = 0

            for _, video_row in group.iterrows():
                vpath = video_row["clean_vpath"]
                info = entries.get(vpath)

                if info is None:
                    # Fallback match on filename
                    matches = [e for name, e in entries.items() if name.endswith(Path(vpath).name)]
                    if matches:
                        info = matches[0]

                if info is None:
                    missing_videos.append((archive_name, vpath))
                    continue

                if info["compression_method"] not in (0, 8):
                    unsupported_compression.append((archive_name, vpath, info["compression_method"]))

                archive_found += 1
                archive_compressed += info["compressed_size"]
                archive_original += info["original_size"]

            found_videos += archive_found
            total_compressed += archive_compressed
            total_original += archive_original

            print(f"Found: {archive_found}/{len(group)}")
            print(f"Selected compressed: {archive_compressed / (1024**2):.1f} MB")
            print(f"Selected original  : {archive_original / (1024**2):.1f} MB")

        except Exception as e:
            print(f"\nFAILED to inspect archive {archive_name}: {e}")
            failed_archives.append(archive_name)

    print("\n" + "=" * 70)
    print("FINAL RESULT")
    print("=" * 70)

    print(f"\nSelected videos: {len(selected)}")
    print(f"Archives: {len(archives)}")
    print(f"Found: {found_videos}/{len(selected)}")
    print(f"Missing: {len(missing_videos)}")
    print(f"Failed archives: {len(failed_archives)}")
    if unsupported_compression:
        print(f"Unsupported compression: {len(unsupported_compression)}")

    print(f"\nCompressed download size: {total_compressed / (1024**3):.2f} GB")
    print(f"Extracted dataset size  : {total_original / (1024**3):.2f} GB")
    print("\nFull archive download size: ~42.38 GB")

    if missing_videos:
        print("\nMISSING VIDEOS:")
        for arch, vp in missing_videos:
            print(f"  {arch} -> {vp}")

    print("\nDone.")


if __name__ == "__main__":
    main()
