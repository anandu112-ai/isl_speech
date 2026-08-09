"""
Evaluation module for ISL Alphabet + Digit CNN classifier.

Evaluates on TEST split only.
Produces:
    - Accuracy, Macro P/R/F1, Top-1/3/5 Accuracy
    - reports/alphabet/classification_report.csv
    - reports/alphabet/confusion_matrix.png
    - reports/alphabet/evaluation_summary.txt
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.utils.metrics import (
    evaluate_predictions,
    save_classification_report_csv,
    save_confusion_matrix_plot,
)

logger = logging.getLogger(__name__)


class AlphabetEvaluator:
    """
    Runs inference on a DataLoader and computes full evaluation metrics.

    Args:
        model:       Trained AlphabetCNNModel in eval mode
        device:      torch.device
        label_map:   {label: class_id}
        reports_dir: Where to save evaluation reports
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        label_map: Dict[str, int],
        reports_dir: Path = Path("reports/alphabet"),
    ):
        self.model = model
        self.device = device
        self.label_map = label_map
        self.id_to_label: Dict[int, str] = {v: k for k, v in label_map.items()}
        # Ordered list of class names by class_id
        self.class_names: List[str] = [
            self.id_to_label[i] for i in range(len(label_map))
        ]
        self.reports_dir = Path(reports_dir)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def run(self, test_loader: DataLoader) -> Dict:
        """
        Runs evaluation on `test_loader`.

        Returns:
            Dict with keys: accuracy, macro_precision, macro_recall, macro_f1,
            top1_acc, top3_acc, top5_acc, report_dict, confusion_matrix
        """
        self.model.eval()
        all_labels: List[int] = []
        all_preds: List[int] = []
        all_logits: List[np.ndarray] = []

        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(self.device, non_blocking=True)
                logits = self.model(images)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_labels.extend(labels.numpy().tolist())
                all_preds.extend(preds.tolist())
                all_logits.append(probs)

        y_true = np.array(all_labels, dtype=np.int64)
        y_pred = np.array(all_preds, dtype=np.int64)
        y_logits = np.concatenate(all_logits, axis=0)

        results = evaluate_predictions(y_true, y_pred, y_logits, self.class_names)
        return results

    def save_reports(self, results: Dict) -> None:
        """Saves confusion matrix PNG and classification report CSV."""
        # Confusion matrix
        cm_path = self.reports_dir / "confusion_matrix.png"
        save_confusion_matrix_plot(
            results["confusion_matrix"],
            self.class_names,
            cm_path,
        )
        logger.info(f"Confusion matrix saved to {cm_path}")

        # Classification report CSV
        csv_path = self.reports_dir / "classification_report.csv"
        save_classification_report_csv(results["report_dict"], csv_path)
        logger.info(f"Classification report saved to {csv_path}")

        # Summary text
        summary_path = self.reports_dir / "evaluation_summary.txt"
        lines = [
            "=" * 60,
            "ISL ALPHABET + DIGIT EVALUATION SUMMARY",
            "=" * 60,
            f"Accuracy         : {results['accuracy']:.2f}%",
            f"Top-1 Accuracy   : {results['top1_acc']:.2f}%",
            f"Top-3 Accuracy   : {results['top3_acc']:.2f}%",
            f"Top-5 Accuracy   : {results['top5_acc']:.2f}%",
            f"Macro Precision  : {results['macro_precision']:.2f}%",
            f"Macro Recall     : {results['macro_recall']:.2f}%",
            f"Macro F1         : {results['macro_f1']:.2f}%",
            "",
        ]

        # Per-class performance
        rd = results["report_dict"]
        lines.append("Per-Class Performance:")
        lines.append(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
        lines.append("-" * 52)
        for cls in self.class_names:
            if cls in rd:
                r = rd[cls]
                lines.append(
                    f"{cls:<8} {r['precision']*100:>9.1f}% {r['recall']*100:>9.1f}% "
                    f"{r['f1-score']*100:>9.1f}% {int(r['support']):>10}"
                )

        # Best and worst performers
        per_class_f1 = {
            cls: rd[cls]["f1-score"]
            for cls in self.class_names
            if cls in rd and "f1-score" in rd[cls]
        }
        if per_class_f1:
            sorted_by_f1 = sorted(per_class_f1.items(), key=lambda x: x[1])
            lines.append("")
            lines.append("Top 5 Worst Performing Classes:")
            for cls, f1 in sorted_by_f1[:5]:
                lines.append(f"  {cls}: F1={f1*100:.1f}%")
            lines.append("Top 5 Best Performing Classes:")
            for cls, f1 in reversed(sorted_by_f1[-5:]):
                lines.append(f"  {cls}: F1={f1*100:.1f}%")

        summary_text = "\n".join(lines)
        summary_path.write_text(summary_text, encoding="utf-8")
        logger.info(f"Evaluation summary saved to {summary_path}")

        print(summary_text)
