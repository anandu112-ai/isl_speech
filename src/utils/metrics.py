"""
Metrics evaluation and reporting tools for ISL Alphabet + Digit Classifier.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_topk_accuracy(logits: np.ndarray, targets: np.ndarray, ks: Tuple[int, ...] = (1, 3, 5)) -> Dict[str, float]:
    """
    Computes Top-k accuracy for k in ks.
    logits: shape (N, num_classes)
    targets: shape (N,) integer class indices
    """
    results = {}
    N = len(targets)
    if N == 0:
        return {f"top{k}_acc": 0.0 for k in ks}

    for k in ks:
        # Get indices of top k probabilities/logits
        topk_preds = np.argsort(logits, axis=1)[:, -k:]
        correct = np.any(topk_preds == targets[:, None], axis=1)
        acc = float(np.sum(correct)) / N * 100.0
        results[f"top{k}_acc"] = round(acc, 2)

    return results


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_logits: np.ndarray,
    class_names: List[str],
) -> Dict:
    """
    Calculates overall and per-class evaluation metrics.
    """
    acc = accuracy_score(y_true, y_pred) * 100.0
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    topk = compute_topk_accuracy(y_logits, y_true, ks=(1, 3, 5))

    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    return {
        "accuracy": round(acc, 2),
        "macro_precision": round(macro_p * 100.0, 2),
        "macro_recall": round(macro_r * 100.0, 2),
        "macro_f1": round(macro_f1 * 100.0, 2),
        "top1_acc": topk["top1_acc"],
        "top3_acc": topk["top3_acc"],
        "top5_acc": topk["top5_acc"],
        "report_dict": report_dict,
        "confusion_matrix": cm,
    }


def save_confusion_matrix_plot(
    cm: np.ndarray,
    class_names: List[str],
    output_path: Path,
    title: str = "36-Class ISL Alphabet & Digit Confusion Matrix",
) -> None:
    """
    Generates and saves a publication-quality 36x36 confusion matrix image.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        square=True,
    )
    plt.title(title, fontsize=14, pad=15)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def save_classification_report_csv(
    report_dict: Dict,
    output_csv_path: Path,
) -> pd.DataFrame:
    """
    Converts sklearn classification report dict into a pandas DataFrame and saves CSV.
    """
    output_csv_path = Path(output_csv_path)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(output_csv_path, index_label="class")
    return df
