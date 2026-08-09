import pandas as pd
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PER_CLASS = 10
VAL_PER_CLASS = 3
RANDOM_SEED = 42

METADATA_FILE = Path("data/metadata/include50.csv")
ARCHIVE_FILE = Path("data/metadata/include50_archive_map.csv")
OUTPUT_FILE = Path("data/metadata/selected_videos.csv")


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("INCLUDE-50 VIDEO SELECTION")
print("=" * 60)

print("\nLoading metadata...")

metadata = pd.read_csv(METADATA_FILE)
archives = pd.read_csv(ARCHIVE_FILE)

metadata = metadata[
    metadata["include_50"] == True
].copy()

print(f"Original INCLUDE-50 rows: {len(metadata)}")


# ============================================================
# EXTRACT AUTHORITATIVE LABEL FROM VIDEO PATH
# ============================================================

print("\nChecking labels using video folders...")

# Example:
#
# Adjectives/1. loud/MVI_9290.MOV
#            ^^^^^^^
#
# We take the folder immediately before the filename.

metadata["path_label"] = (
    metadata["video_path"]
    .str.replace("\\", "/", regex=False)
    .str.split("/")
    .str[-2]
)


# ============================================================
# FIND LABEL CONFLICTS
# ============================================================

conflicts = metadata[
    metadata["label"] != metadata["path_label"]
].copy()

print(
    f"Metadata/path label conflicts: {len(conflicts)}"
)

if len(conflicts) > 0:

    print("\nConflicts found:")
    print(
        conflicts[
            ["label", "path_label", "video_path"]
        ].to_string(index=False)
    )

    print(
        "\nUsing folder label as the authoritative label."
    )


# Replace metadata label with folder label
metadata["label"] = metadata["path_label"]

metadata.drop(
    columns=["path_label"],
    inplace=True
)


# ============================================================
# REMOVE EXACT DUPLICATE VIDEO PATHS
# ============================================================

duplicates = metadata[
    metadata["video_path"].duplicated(keep=False)
]

print(
    f"\nRows involved in duplicate paths: "
    f"{len(duplicates)}"
)

metadata = metadata.drop_duplicates(
    subset="video_path",
    keep="first"
).copy()

print(
    f"Unique videos after duplicate removal: "
    f"{len(metadata)}"
)


# ============================================================
# CONNECT VIDEOS TO ARCHIVES
# ============================================================

print("\nMapping videos to archives...")

df = metadata.merge(
    archives,
    on="video_path",
    how="inner"
)

print(
    f"Videos successfully mapped: {len(df)}"
)


# ============================================================
# CHECK FOR MISSING ARCHIVE MAPPINGS
# ============================================================

missing = metadata[
    ~metadata["video_path"].isin(
        archives["video_path"]
    )
]

if len(missing) > 0:

    print(
        f"\nWARNING: {len(missing)} videos "
        "have no archive mapping."
    )

else:

    print("All videos have archive mappings.")


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df.groupby("label")
    .size()
    .sort_index()
)

print("\nVideos available per sign:")
print(class_counts.to_string())


# ============================================================
# CHECK WHETHER EVERY CLASS HAS ENOUGH VIDEOS
# ============================================================

required = TRAIN_PER_CLASS + VAL_PER_CLASS

too_small = class_counts[
    class_counts < required
]

if len(too_small) > 0:

    print(
        "\nERROR: Some classes do not have enough videos:"
    )

    print(too_small.to_string())

    raise SystemExit(1)


# ============================================================
# BALANCED SELECTION
# ============================================================

print("\nSelecting videos...")

selected_parts = []

for label, group in df.groupby(
    "label",
    sort=True
):

    # Reproducible shuffle
    group = group.sample(
        frac=1,
        random_state=RANDOM_SEED
    ).reset_index(drop=True)

    # -------------------------
    # Training
    # -------------------------

    train = group.iloc[
        :TRAIN_PER_CLASS
    ].copy()

    train["split"] = "train"

    # -------------------------
    # Validation
    # -------------------------

    val = group.iloc[
        TRAIN_PER_CLASS:
        TRAIN_PER_CLASS + VAL_PER_CLASS
    ].copy()

    val["split"] = "val"

    selected_parts.append(train)
    selected_parts.append(val)


selected = pd.concat(
    selected_parts,
    ignore_index=True
)


# ============================================================
# FINAL SAFETY CHECKS
# ============================================================

print("\nRunning safety checks...")


# Duplicate videos
duplicate_count = (
    selected["video_path"]
    .duplicated()
    .sum()
)

if duplicate_count > 0:

    print(
        f"ERROR: {duplicate_count} duplicate "
        "videos remain!"
    )

    raise SystemExit(1)


# Check exact class balance
balance = (
    selected
    .groupby(["label", "split"])
    .size()
    .unstack(fill_value=0)
)

print("\nFinal class balance:")

print(balance.to_string())


# Check total
expected = 50 * required

if len(selected) != expected:

    print(
        f"\nWARNING: Expected {expected} videos "
        f"but selected {len(selected)}."
    )

else:

    print(
        f"\nCorrect total: {expected} videos"
    )


# ============================================================
# SAVE MANIFEST
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

selected.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("SELECTION COMPLETE")
print("=" * 60)

print(
    f"\nTraining videos: "
    f"{(selected['split'] == 'train').sum()}"
)

print(
    f"Validation videos: "
    f"{(selected['split'] == 'val').sum()}"
)

print(
    f"Total videos: "
    f"{len(selected)}"
)

print(
    f"\nSaved manifest:\n"
    f"{OUTPUT_FILE}"
)

print("\nNo videos were downloaded.")