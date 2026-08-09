from datasets import load_dataset
import pandas as pd
import os

print("=" * 60)
print("       INCLUDE-50 METADATA")
print("=" * 60)

print("\nDownloading dataset metadata...")

dataset = load_dataset("ai4bharat/INCLUDE")

print("\nMetadata loaded!")

# Convert splits to pandas
train = dataset["train"].to_pandas()
test = dataset["test"].to_pandas()

print(f"Train records: {len(train)}")
print(f"Test records : {len(test)}")

# Select INCLUDE-50
train50 = train[train["include_50"] == True]
test50 = test[test["include_50"] == True]

print("\n" + "=" * 60)
print("             INCLUDE-50")
print("=" * 60)

print(f"\nTrain videos: {len(train50)}")
print(f"Test videos : {len(test50)}")
print(f"Total       : {len(train50) + len(test50)}")

# Combine
include50 = pd.concat(
    [train50, test50],
    ignore_index=True
)

print(
    f"\nNumber of unique signs: "
    f"{include50['label'].nunique()}"
)

print("\nSigns:")

for sign in sorted(include50["label"].unique()):
    print(" -", sign)

# Save metadata
os.makedirs("data/metadata", exist_ok=True)

include50.to_csv(
    "data/metadata/include50.csv",
    index=False
)

print("\nMetadata saved to:")
print("data/metadata/include50.csv")

print("\nExample records:")
print(
    include50[
        ["label", "video_path", "include_50"]
    ].head(10).to_string(index=False)
)

print("\n" + "=" * 60)
print("                    DONE")
print("=" * 60)