import io
import struct
import time

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

SELECTED_FILE = "data/metadata/selected_videos.csv"

TAIL_SIZE = 64 * 1024

TIMEOUT = (15, 120)

MAX_RETRIES = 5


# ============================================================
# LOAD SELECTION
# ============================================================

selected = pd.read_csv(SELECTED_FILE)

archives = (
    selected[
        ["archive", "archive_url"]
    ]
    .drop_duplicates()
    .sort_values("archive")
)


print("=" * 70)
print("REMOTE VIDEO SIZE ESTIMATOR")
print("=" * 70)

print(f"\nSelected videos : {len(selected)}")
print(f"Archives needed : {len(archives)}")


# ============================================================
# HTTP RANGE REQUEST WITH RETRIES
# ============================================================

session = requests.Session()


def range_request(url, start, end):

    for attempt in range(1, MAX_RETRIES + 1):

        try:

            response = session.get(
                url,
                headers={
                    "Range": f"bytes={start}-{end}"
                },
                timeout=TIMEOUT
            )

            response.raise_for_status()

            if response.status_code != 206:

                raise RuntimeError(
                    f"Expected HTTP 206, "
                    f"got {response.status_code}"
                )

            return response

        except Exception as e:

            print(
                f"    Attempt {attempt}/{MAX_RETRIES} "
                f"failed: {type(e).__name__}"
            )

            if attempt == MAX_RETRIES:

                raise

            wait = attempt * 3

            print(
                f"    Retrying in {wait} seconds..."
            )

            time.sleep(wait)


# ============================================================
# FIND ZIP CENTRAL DIRECTORY
# ============================================================

def get_zip_directory(url):

    # --------------------------------------------------------
    # Request only the final 64 KB
    # --------------------------------------------------------

    response = session.get(
        url,
        headers={
            "Range": f"bytes=-{TAIL_SIZE}"
        },
        timeout=TIMEOUT
    )

    response.raise_for_status()

    if response.status_code != 206:

        raise RuntimeError(
            f"Expected 206, got "
            f"{response.status_code}"
        )

    tail = response.content

    # --------------------------------------------------------
    # Find End Of Central Directory
    # --------------------------------------------------------

    signature = b"PK\x05\x06"

    position = tail.rfind(signature)

    if position == -1:

        raise RuntimeError(
            "ZIP End Of Central Directory "
            "record not found."
        )

    # EOCD layout:
    #
    # signature       4 bytes
    # disk            2
    # cd start disk   2
    # disk entries    2
    # total entries   2
    # cd size         4
    # cd offset       4
    # comment length  2

    fields = struct.unpack_from(
        "<4s4H2LH",
        tail,
        position
    )

    (
        signature,
        disk,
        cd_start_disk,
        entries_disk,
        total_entries,
        cd_size,
        cd_offset,
        comment_length
    ) = fields

    # --------------------------------------------------------
    # ZIP64 check
    # --------------------------------------------------------

    if (
        total_entries == 0xFFFF
        or cd_size == 0xFFFFFFFF
        or cd_offset == 0xFFFFFFFF
    ):

        raise RuntimeError(
            "ZIP64 archive detected. "
            "This script needs ZIP64 handling."
        )

    # --------------------------------------------------------
    # Request exact central directory
    # --------------------------------------------------------

    cd_end = cd_offset + cd_size - 1

    response = range_request(
        url,
        cd_offset,
        cd_end
    )

    central_directory = response.content

    # --------------------------------------------------------
    # Build a synthetic ZIP containing:
    #
    # local headers aren't required for ZipInfo parsing,
    # but Python's zipfile needs the central directory.
    #
    # We therefore use the central directory directly
    # with manual parsing below.
    # --------------------------------------------------------

    return (
        central_directory,
        cd_offset,
        total_entries
    )


# ============================================================
# PARSE CENTRAL DIRECTORY
# ============================================================

def parse_central_directory(data):

    files = {}

    pos = 0

    while pos < len(data):

        signature = data[pos:pos + 4]

        if signature != b"PK\x01\x02":

            break

        # Central directory fixed header = 46 bytes
        #
        # Filename length at offset 28
        # Extra length at offset 30
        # Comment length at offset 32
        # Compressed size at offset 20
        # Original size at offset 24

        compressed_size = struct.unpack_from(
            "<I",
            data,
            pos + 20
        )[0]

        original_size = struct.unpack_from(
            "<I",
            data,
            pos + 24
        )[0]

        filename_length = struct.unpack_from(
            "<H",
            data,
            pos + 28
        )[0]

        extra_length = struct.unpack_from(
            "<H",
            data,
            pos + 30
        )[0]

        comment_length = struct.unpack_from(
            "<H",
            data,
            pos + 32
        )[0]

        filename_start = pos + 46

        filename_end = (
            filename_start +
            filename_length
        )

        filename = data[
            filename_start:filename_end
        ].decode(
            "utf-8",
            errors="replace"
        )

        files[filename] = {
            "compressed_size": compressed_size,
            "original_size": original_size
        }

        pos = (
            filename_end +
            extra_length +
            comment_length
        )

    return files


# ============================================================
# PROCESS ARCHIVES
# ============================================================

total_compressed = 0
total_original = 0

found_videos = 0
missing_videos = []

failed_archives = []


for index, row in archives.iterrows():

    archive = row["archive"]
    url = row["archive_url"]

    group = selected[
        selected["archive"] == archive
    ]

    print("\n" + "-" * 70)

    print(
        f"[{len(failed_archives) + found_videos}/{len(selected)}]"
    )

    print(f"Archive: {archive}")
    print(
        f"Videos needed: {len(group)}"
    )

    try:

        central_directory, offset, entries = (
            get_zip_directory(url)
        )

        print(
            f"Central directory: "
            f"{len(central_directory) / 1024:.1f} KB"
        )

        print(
            f"ZIP entries: {entries}"
        )

        files = parse_central_directory(
            central_directory
        )

        archive_compressed = 0
        archive_original = 0
        archive_found = 0

        for _, video in group.iterrows():

            path = video["video_path"]

            info = files.get(path)

            if info is None:

                missing_videos.append(
                    (archive, path)
                )

                continue

            archive_found += 1

            archive_compressed += (
                info["compressed_size"]
            )

            archive_original += (
                info["original_size"]
            )

        found_videos += archive_found

        total_compressed += archive_compressed
        total_original += archive_original

        print(
            f"Found: "
            f"{archive_found}/{len(group)}"
        )

        print(
            f"Selected compressed: "
            f"{archive_compressed / 1024**2:.1f} MB"
        )

        print(
            f"Selected original: "
            f"{archive_original / 1024**2:.1f} MB"
        )

    except Exception as e:

        print(
            f"\nFAILED: {type(e).__name__}: {e}"
        )

        failed_archives.append(
            archive
        )


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 70)
print("FINAL RESULT")
print("=" * 70)

print(
    f"\nVideos found: "
    f"{found_videos}/{len(selected)}"
)

print(
    f"Missing videos: "
    f"{len(missing_videos)}"
)

print(
    f"Failed archives: "
    f"{len(failed_archives)}"
)

print(
    f"\nSelected compressed size:"
    f"\n{total_compressed / 1024**3:.2f} GB"
)

print(
    f"\nSelected extracted video size:"
    f"\n{total_original / 1024**3:.2f} GB"
)

print(
    "\nFull archive download would be:"
    "\n42.38 GB"
)


# ============================================================
# DETAILS
# ============================================================

if missing_videos:

    print("\nMISSING VIDEOS:")

    for archive, video in missing_videos:

        print(
            f"{archive} -> {video}"
        )


if failed_archives:

    print("\nFAILED ARCHIVES:")

    for archive in failed_archives:

        print(archive)


print("\nDone.")