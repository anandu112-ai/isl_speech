import os
import pandas as pd
import requests

from remotezip import RemoteZip


ZENODO_API = "https://zenodo.org/api/records/4010759"
METADATA_FILE = "data/metadata/include50.csv"


def main():

    print("=" * 70)
    print("       MAPPING INCLUDE-50 VIDEOS TO ZENODO ARCHIVES")
    print("=" * 70)

    # -----------------------------------------
    # Load metadata
    # -----------------------------------------

    data = pd.read_csv(METADATA_FILE)

    # Normalize paths
    data["video_path"] = (
        data["video_path"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
    )

    # Remove duplicate video paths
    data = data.drop_duplicates(
        subset=["video_path"]
    )

    required_paths = set(
        data["video_path"]
    )

    print(
        f"\nUnique videos required: "
        f"{len(required_paths)}"
    )

    # -----------------------------------------
    # Get Zenodo files
    # -----------------------------------------

    print("\nGetting Zenodo archive list...")

    response = requests.get(
        ZENODO_API,
        timeout=30
    )

    response.raise_for_status()

    record = response.json()

    zip_files = [
        f for f in record["files"]
        if f["key"].lower().endswith(".zip")
    ]

    print(
        f"Found {len(zip_files)} ZIP archives."
    )

    # -----------------------------------------
    # Find matching archives
    # -----------------------------------------

    results = []

    remaining = set(required_paths)

    os.makedirs(
        "data/metadata",
        exist_ok=True
    )

    for index, archive in enumerate(zip_files, 1):

        archive_name = archive["key"]

        # We only need categories represented
        # in our INCLUDE-50 metadata.
        category = archive_name.split("_")[0]

        print(
            f"\n[{index}/{len(zip_files)}] "
            f"Checking {archive_name}"
        )

        print(
            "This reads the ZIP index remotely; "
            "the video data is NOT downloaded."
        )

        try:

            with RemoteZip(
                archive["links"]["self"]
            ) as remote_zip:

                names = set(
                    remote_zip.namelist()
                )

                matches = (
                    required_paths
                    & names
                )

                if matches:

                    print(
                        f"FOUND {len(matches)} "
                        f"required videos!"
                    )

                    for path in sorted(matches):

                        results.append({
                            "video_path": path,
                            "archive": archive_name,
                            "archive_url":
                                archive["links"]["self"]
                        })

                    remaining -= matches

                else:

                    print(
                        "No INCLUDE-50 videos "
                        "found in this archive."
                    )

        except Exception as e:

            print(
                f"Could not inspect archive: {e}"
            )

    # -----------------------------------------
    # Save results
    # -----------------------------------------

    result_df = pd.DataFrame(results)

    result_df.to_csv(
        "data/metadata/include50_archive_map.csv",
        index=False
    )

    # -----------------------------------------
    # Summary
    # -----------------------------------------

    print("\n" + "=" * 70)
    print("                     SUMMARY")
    print("=" * 70)

    print(
        f"\nVideos mapped : {len(result_df)}"
    )

    print(
        f"Videos missing: {len(remaining)}"
    )

    if remaining:

        print("\nMissing videos:")

        for path in sorted(remaining):

            print(
                " -", path
            )

    print("\nArchive distribution:")

    if len(result_df) > 0:

        counts = (
            result_df["archive"]
            .value_counts()
        )

        for archive, count in counts.items():

            print(
                f"{archive:35} {count}"
            )

    print("\nSaved mapping to:")

    print(
        "data/metadata/include50_archive_map.csv"
    )

    print("\n" + "=" * 70)
    print("                       DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()