import os

import numpy as np
import pandas as pd


INPUT_FILE = "dataset/processed/landmarks.csv"
OUTPUT_FILE = "dataset/processed/normalized_landmarks.csv"


def normalize_hand(hand):

    """
    Normalize one hand.

    Input:
        63 values
        21 landmarks × 3 coordinates

    Output:
        63 normalized values
    """

    landmarks = np.array(hand).reshape(21, 3)

    # Wrist = landmark 0
    wrist = landmarks[0].copy()

    # Move wrist to origin
    landmarks = landmarks - wrist

    # Calculate hand size
    distances = np.linalg.norm(
        landmarks,
        axis=1
    )

    scale = np.max(distances)

    # Prevent division by zero
    if scale > 0:
        landmarks = landmarks / scale

    return landmarks.flatten().tolist()


def main():

    print("=" * 60)
    print("        ISL LANDMARK NORMALIZATION")
    print("=" * 60)

    if not os.path.exists(INPUT_FILE):

        print("ERROR: Dataset not found.")

        return

    data = pd.read_csv(INPUT_FILE)

    labels = data["label"]

    features = data.drop(
        "label",
        axis=1
    )

    normalized_rows = []

    for _, row in features.iterrows():

        values = row.values.astype(float)

        hand1 = values[:63]

        hand2 = values[63:126]

        normalized_hand1 = normalize_hand(
            hand1
        )

        normalized_hand2 = normalize_hand(
            hand2
        )

        normalized_rows.append(
            normalized_hand1 +
            normalized_hand2
        )

    normalized_data = pd.DataFrame(
        normalized_rows
    )

    normalized_data.insert(
        0,
        "label",
        labels.values
    )

    normalized_data.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("Normalization complete.")

    print(
        f"Original shape: "
        f"{data.shape}"
    )

    print(
        f"New shape: "
        f"{normalized_data.shape}"
    )

    print()
    print(
        f"Saved to: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()