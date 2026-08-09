import pandas as pd
from collections import Counter

FILE = "data/metadata/include50.csv"

data = pd.read_csv(FILE)

print("=" * 60)
print("           INCLUDE-50 DOWNLOAD ANALYSIS")
print("=" * 60)

# Extract top-level folder
data["category"] = data["video_path"].apply(
    lambda x: x.split("/")[0]
)

print("\nVideos by category:")
print("-" * 40)

counts = data["category"].value_counts()

for category, count in counts.items():
    print(f"{category:35} {count}")

print("\n" + "=" * 60)

print(f"Total videos: {len(data)}")
print(f"Categories  : {data['category'].nunique()}")

print("\nCategories we need:")

for category in sorted(data["category"].unique()):
    print(" -", category)

# Save category summary
summary = (
    data.groupby("category")
    .size()
    .reset_index(name="video_count")
)

summary.to_csv(
    "data/metadata/include50_categories.csv",
    index=False
)

print("\nSaved:")
print("data/metadata/include50_categories.csv")

print("\n" + "=" * 60)
print("                    DONE")
print("=" * 60)