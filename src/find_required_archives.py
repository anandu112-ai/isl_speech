import pandas as pd
import requests
import zipfile
import io
import os


ZENODO_API = "https://zenodo.org/api/records/4010759"

METADATA_FILE = "data/metadata/include50.csv"


def main():

    print("=" * 70)
    print("       FINDING INCLUDE-50 ARCHIVES")
    print("=" * 70)

    # -----------------------------------------
    # Load our 881 required videos
    # -----------------------------------------

    data = pd.read_csv(METADATA_FILE)

    required_paths = set(
        data["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    print(f"\nRequired videos: {len(required_paths)}")

    # -----------------------------------------
    # Get Zenodo file list
    # -----------------------------------------

    print("\nReading Zenodo file list...")

    response = requests.get(
        ZENODO_API,
        timeout=30
    )

    response.raise_for_status()

    record = response.json()

    files = record["files"]

    print(
        f"Files found on Zenodo: {len(files)}"
    )

    # -----------------------------------------
    # Show ZIP files
    # -----------------------------------------

    zip_files = []

    for file in files:

        name = file["key"]

        if name.lower().endswith(".zip"):

            zip_files.append(file)

    print(
        f"ZIP archives found: {len(zip_files)}"
    )

    print("\nArchives:")

    for file in zip_files:

        print(
            f" - {file['key']} "
            f"({file['size'] / (1024**3):.2f} GB)"
        )

    print("\n" + "=" * 70)
    print("NOTE")
    print("=" * 70)

    print(
        """
Zenodo archives are large, so we are NOT downloading them yet.

We first need to determine which archive contains
each INCLUDE-50 video.

The next stage will use the archive metadata/index
where available.
"""
    )


if __name__ == "__main__":
    main()