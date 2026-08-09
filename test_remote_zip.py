import io
import zipfile
import requests
import pandas as pd


# --------------------------------------------------
# Load selected videos
# --------------------------------------------------

selected = pd.read_csv(
    "data/metadata/selected_videos.csv"
)

# Pick one selected video
row = selected.iloc[0]

archive = row["archive"]
video_path = row["video_path"]
url = row["archive_url"]

print("Archive:")
print(archive)

print("\nVideo:")
print(video_path)

print("\nTesting remote ZIP access...")


# --------------------------------------------------
# First request: get the last 65 KB
# --------------------------------------------------

# ZIP central directory is normally near the end
TAIL_SIZE = 65536

head = requests.get(
    url,
    headers={
        "Range": f"bytes=-{TAIL_SIZE}"
    },
    timeout=60
)

print("\nRange status:", head.status_code)
print("Bytes received:", len(head.content))
print("Content-Range:", head.headers.get("Content-Range"))

if head.status_code != 206:
    raise RuntimeError(
        "Server did not return a partial response."
    )


# --------------------------------------------------
# Find ZIP end-of-central-directory
# --------------------------------------------------

data = head.content

signature = b"PK\x05\x06"

position = data.rfind(signature)

if position == -1:

    raise RuntimeError(
        "Could not find ZIP central directory."
    )

print("\nZIP central directory found.")


# --------------------------------------------------
# Parse central directory
# --------------------------------------------------

# Unfortunately, the central directory may be larger
# than our first 64 KB, so first try Python's zipfile
# using the downloaded tail if possible.

try:

    z = zipfile.ZipFile(
        io.BytesIO(data)
    )

    names = z.namelist()

    print(
        f"Files discovered in tail: {len(names)}"
    )

except Exception:

    print(
        "Central directory is larger than 64 KB."
    )

    print(
        "We'll implement a larger-range reader "
        "in the next step."
    )

    raise


# --------------------------------------------------
# Find our selected video
# --------------------------------------------------

matches = [
    name
    for name in names
    if name == video_path
]

print("\nExact matches:")

for name in matches:
    print(name)

if not matches:

    print(
        "\nVideo was not found in this ZIP."
    )

else:

    info = z.getinfo(matches[0])

    print("\nZIP information:")
    print("Filename:", info.filename)
    print("Compressed size:", info.compress_size)
    print("Original size:", info.file_size)
    print("Compression:", info.compress_type)