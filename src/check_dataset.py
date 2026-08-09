import pandas as pd
import os


DATASET_FILE = "dataset/processed/landmarks.csv"


def main():

    print("=" * 50)
    print("          ISL DATASET CHECK")
    print("=" * 50)

    # Check file exists
    if not os.path.exists(DATASET_FILE):
        print("ERROR: Dataset file not found.")
        return

    # Load dataset
    data = pd.read_csv(DATASET_FILE)

    print("\nDataset shape:")
    print(data.shape)

    print("\nExpected:")
    print("Rows    : 1000")
    print("Columns : 127")

    # ---------------------------------------
    # Check columns
    # ---------------------------------------

    print("\nColumn count:")

    print(len(data.columns))

    # ---------------------------------------
    # Check labels
    # ---------------------------------------

    print("\nSign distribution:")

    print(data["label"].value_counts())

    # ---------------------------------------
    # Check missing values
    # ---------------------------------------

    print("\nMissing values:")

    missing = data.isnull().sum().sum()

    print(missing)

    # ---------------------------------------
    # Check duplicate rows
    # ---------------------------------------

    print("\nDuplicate rows:")

    print(data.duplicated().sum())

    # ---------------------------------------
    # Show first few rows
    # ---------------------------------------

    print("\nFirst 5 rows:")

    print(data.head())

    # ---------------------------------------
    # Dataset information
    # ---------------------------------------

    print("\nDataset information:")

    data.info()

    print("\n" + "=" * 50)
    print("             CHECK COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()