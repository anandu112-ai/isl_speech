import pandas as pd
import requests


# Load selected videos
df = pd.read_csv(
    "data/metadata/selected_videos.csv"
)

# One row per archive
archives = (
    df[["archive", "archive_url"]]
    .drop_duplicates()
    .sort_values("archive")
)

print(f"Archives required: {len(archives)}")
print()
print("Checking Zenodo archive sizes...")
print("=" * 70)

total_size = 0

for _, row in archives.iterrows():

    archive = row["archive"]
    url = row["archive_url"]

    try:
        # Convert content URL to Zenodo file API URL
        api_url = url.replace(
            "/content",
            ""
        )

        response = requests.get(
            api_url,
            timeout=30
        )

        response.raise_for_status()

        info = response.json()

        size = info.get("size", 0)

        total_size += size

        size_mb = size / (1024 ** 2)

        print(
            f"{archive:40} "
            f"{size_mb:8.1f} MB"
        )

    except Exception as e:

        print(
            f"{archive:40} ERROR: {e}"
        )


print("=" * 70)

print(
    f"TOTAL DOWNLOAD SIZE: "
    f"{total_size / (1024 ** 3):.2f} GB"
)