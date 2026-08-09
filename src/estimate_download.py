import pandas as pd
import requests
import time
from remotezip import RemoteZip

MAP_FILE = "data/metadata/include50_archive_map.csv"
OUTPUT_FILE = "data/metadata/download_sizes.csv"


def check_archive(archive, group):

    url = group["archive_url"].iloc[0]
    required = set(group["video_path"])

    for attempt in range(1, 4):

        try:

            print(
                f"\nAttempt {attempt}/3: {archive}"
            )

            with RemoteZip(
                url,
                timeout=30
            ) as z:

                infos = {
                    info.filename: info
                    for info in z.infolist()
                }

                total = 0

                for path in required:

                    if path in infos:
                        total += infos[path].file_size

                return total

        except Exception as e:

            print(
                f"Connection failed: {e}"
            )

            if attempt < 3:

                print(
                    "Retrying in 5 seconds..."
                )

                time.sleep(5)

    return None


def main():

    print("=" * 70)
    print("       INCLUDE-50 DOWNLOAD SIZE ESTIMATE")
    print("=" * 70)

    data = pd.read_csv(MAP_FILE)

    # Remove previous incomplete results if needed
    if False:
        pass

    results = []

    total = 0
    failed = []

    groups = data.groupby("archive")

    print(
        f"\nArchives to inspect: {len(groups)}"
    )

    for index, (archive, group) in enumerate(
        groups,
        start=1
    ):

        print("\n" + "-" * 70)

        print(
            f"[{index}/{len(groups)}] {archive}"
        )

        size = check_archive(
            archive,
            group
        )

        if size is None:

            print(
                "FAILED — will retry later."
            )

            failed.append(archive)

            continue

        print(
            f"Required videos: {len(group)}"
        )

        print(
            f"Required size: "
            f"{size / (1024 ** 2):.2f} MB"
        )

        results.append({
            "archive": archive,
            "video_count": len(group),
            "required_bytes": size,
            "required_mb":
                size / (1024 ** 2),
            "required_gb":
                size / (1024 ** 3)
        })

        total += size

        # Save progress immediately
        pd.DataFrame(results).to_csv(
            OUTPUT_FILE,
            index=False
        )

    print("\n" + "=" * 70)
    print("                    SUMMARY")
    print("=" * 70)

    print(
        f"\nSuccessfully checked: "
        f"{len(results)}"
    )

    print(
        f"Failed: {len(failed)}"
    )

    print(
        f"\nEstimated download size "
        f"from successful archives:"
    )

    print(
        f"{total / (1024 ** 3):.2f} GB"
    )

    if failed:

        print("\nArchives that need retry:")

        for archive in failed:
            print(" -", archive)

    print(
        f"\nProgress saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()