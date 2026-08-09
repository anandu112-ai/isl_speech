"""
Model Evaluation Script for INCLUDE-50 Sign Language Recognition.
Evaluates best.pt model on validation features ONLY.
Computes Top-1 & Top-5 Accuracy, Macro Precision, Macro Recall, Macro F1-Score,
generates reports/confusion_matrix.png and reports/classification_report.csv, and reports top class confusions.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from src.dataset.sign_dataset import SignLandmarkDataset
from src.models.bilstm import SignBiLSTMModel
from torch.utils.data import DataLoader


def plot_confusion_matrix(cm: np.ndarray, class_names: list, save_path: Path):
    """
    Plots and saves a 50x50 confusion matrix chart.
    """
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="INCLUDE-50 Validation Confusion Matrix (50 Classes)",
        ylabel="True Label",
        xlabel="Predicted Label",
    )

    plt.setp(ax.get_xticklabels(), rotation=90, ha="right", rotation_mode="anchor", fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    best_model_path = Path("models/best_model/best.pt")
    if not best_model_path.exists():
        print(f"Error: Best model checkpoint not found at {best_model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print("INCLUDE-50 MODEL EVALUATION")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Loading checkpoint: {best_model_path}")

    checkpoint = torch.load(best_model_path, map_location=device)
    label_map = checkpoint["label_map"]
    id_to_label = {v: k for k, v in label_map.items()}
    class_names = [id_to_label[i] for i in range(len(label_map))]

    manifest_csv = Path("data/metadata/features_manifest.csv")
    label_map_json = Path("data/metadata/label_map.json")

    val_dataset = SignLandmarkDataset(
        manifest_csv=manifest_csv,
        label_map_json=label_map_json,
        split="val",
        augment=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    model = SignBiLSTMModel(
        feature_dim=225,
        proj_dim=128,
        hidden_size=checkpoint["config"]["training"]["hidden_size"],
        num_layers=checkpoint["config"]["training"]["num_layers"],
        dropout=checkpoint["config"]["training"]["dropout"],
        num_classes=len(label_map),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in val_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)

            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Top-1 & Top-5 Accuracy
    top1_correct = (all_preds == all_labels).sum()
    top1_acc = top1_correct / len(all_labels)

    top5_correct = 0
    for idx, true_lbl in enumerate(all_labels):
        top5_preds = np.argsort(all_probs[idx])[-5:]
        if true_lbl in top5_preds:
            top5_correct += 1
    top5_acc = top5_correct / len(all_labels)

    # Macro Precision, Recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average="macro", zero_division=0)

    print("\nVAL EVALUATION RESULTS:")
    print("-" * 50)
    print(f"Validation Samples: {len(all_labels)}")
    print(f"Top-1 Accuracy    : {top1_acc * 100:.2f}%")
    print(f"Top-5 Accuracy    : {top5_acc * 100:.2f}%")
    print(f"Macro Precision   : {precision * 100:.2f}%")
    print(f"Macro Recall      : {recall * 100:.2f}%")
    print(f"Macro F1-Score    : {f1 * 100:.2f}%")

    # Save Classification Report CSV
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_dict = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose().reset_index().rename(columns={"index": "class"})
    report_csv = reports_dir / "classification_report.csv"
    report_df.to_csv(report_csv, index=False)

    # Save Confusion Matrix PNG
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    cm_png = reports_dir / "confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_png)

    # Top confused pairs
    confused_pairs = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm[i, j] > 0:
                confused_pairs.append((class_names[i], class_names[j], cm[i, j]))

    confused_pairs.sort(key=lambda x: x[2], reverse=True)

    print("\nTOP CLASS CONFUSIONS (True -> Predicted):")
    print("-" * 50)
    if confused_pairs:
        for true_cls, pred_cls, count in confused_pairs[:10]:
            print(f"  {true_cls:25} -> {pred_cls:25} ({count} instances)")
    else:
        print("  None! Perfect prediction alignment.")

    print(f"\nSaved classification report to: {report_csv}")
    print(f"Saved confusion matrix plot to : {cm_png}")


if __name__ == "__main__":
    main()
