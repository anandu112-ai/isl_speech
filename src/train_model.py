import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_FILE = "dataset/processed/landmarks.csv"
MODEL_FILE = "models/isl_random_forest.pkl"

RANDOM_STATE = 42


# ==========================================
# MAIN
# ==========================================

def main():

    print("=" * 60)
    print("             ISL MODEL TRAINING")
    print("=" * 60)

    # --------------------------------------
    # Load dataset
    # --------------------------------------

    if not os.path.exists(DATASET_FILE):

        print("\nERROR: Dataset not found.")

        return

    data = pd.read_csv(DATASET_FILE)

    print("\nDataset loaded.")
    print(f"Samples : {len(data)}")
    print(f"Columns : {len(data.columns)}")

    # --------------------------------------
    # Separate features and labels
    # --------------------------------------

    X = data.drop("label", axis=1)

    y = data["label"]

    print("\nFeatures:")
    print(X.shape)

    print("\nClasses:")
    print(y.unique())

    # --------------------------------------
    # Train / Test split
    # --------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=RANDOM_STATE,

        stratify=y
    )

    print("\nTrain/Test split:")

    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")

    # --------------------------------------
    # Create Random Forest
    # --------------------------------------

    print("\nCreating Random Forest...")

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=RANDOM_STATE,

        n_jobs=-1,

        class_weight="balanced"
    )

    # --------------------------------------
    # Train
    # --------------------------------------

    print("Training model...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # --------------------------------------
    # Prediction
    # --------------------------------------

    print("\nMaking predictions...")

    y_pred = model.predict(X_test)

    # --------------------------------------
    # Accuracy
    # --------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("\n" + "=" * 60)

    print(
        f"Accuracy: {accuracy * 100:.2f}%"
    )

    print("=" * 60)

    # --------------------------------------
    # Classification report
    # --------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            y_pred
        )
    )

    # --------------------------------------
    # Confusion matrix
    # --------------------------------------

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(cm)

    # --------------------------------------
    # Save model
    # --------------------------------------

    os.makedirs(
        "models",
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\nModel saved successfully:")

    print(MODEL_FILE)

    # --------------------------------------
    # Feature importance
    # --------------------------------------

    print("\nTop 10 important features:")

    importances = model.feature_importances_

    feature_names = X.columns

    importance_data = sorted(
        zip(
            feature_names,
            importances
        ),
        key=lambda x: x[1],
        reverse=True
    )

    for feature, importance in importance_data[:10]:

        print(
            f"{feature:25s} "
            f"{importance:.6f}"
        )

    print("\n" + "=" * 60)
    print("             TRAINING COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()