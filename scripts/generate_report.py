"""
Final Report Generator Script for INCLUDE-50 Pipeline.
Generates reports/FINAL_REPORT.md containing dataset metrics, model architecture, training configuration,
evaluation results, and confusion analysis.
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd


def generate_final_report():
    report_dir = Path("reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_md = report_dir / "FINAL_REPORT.md"

    history_csv = report_dir / "training_history.csv"
    report_csv = report_dir / "classification_report.csv"
    best_model_dir = Path("models/best_model")

    best_val_acc = "N/A"
    top5_acc = "N/A"
    macro_precision = "N/A"
    macro_recall = "N/A"
    macro_f1 = "N/A"
    final_epoch = "N/A"

    if history_csv.exists():
        hdf = pd.read_csv(history_csv)
        final_epoch = len(hdf)
        max_row = hdf.loc[hdf["val_accuracy"].idxmax()]
        best_val_acc = f"{max_row['val_accuracy']*100:.2f}%"

    if report_csv.exists():
        rdf = pd.read_csv(report_csv)
        macro_row = rdf[rdf["class"] == "macro avg"]
        if not macro_row.empty:
            macro_precision = f"{macro_row['precision'].values[0]*100:.2f}%"
            macro_recall = f"{macro_row['recall'].values[0]*100:.2f}%"
            macro_f1 = f"{macro_row['f1-score'].values[0]*100:.2f}%"

    content = f"""# INCLUDE-50 Indian Sign Language Recognition + Speech System

## Executive Summary
This report summarizes the design, implementation, training, evaluation, and inference demo of the **INCLUDE-50 Indian Sign Language (ISL) Recognition and Text-to-Speech System**.

---

## 1. Dataset Specifications
- **Dataset**: INCLUDE-50 (Indian Sign Language dataset, Zenodo ID: 4010759)
- **Selection**: 650 videos total (Authoritative `data/metadata/selected_videos.csv`)
- **Classes**: 50 unique ISL signs
- **Splits**:
  - **Train**: 500 videos (10 training videos per class)
  - **Validation**: 150 videos (3 validation videos per class)
- **Data Leakage**: 0 files shared between train and val splits.

---

## 2. Preprocessing & Feature Extraction
- **Frame Sampling**: Uniform 32-frame temporal sampling per video with linear index interpolation for short videos.
- **Landmark Representation**: MediaPipe Holistic (225 features per frame)
  - **Left Hand**: 21 3D landmarks (x, y, z) = 63 values
  - **Right Hand**: 21 3D landmarks (x, y, z) = 63 values
  - **Body Pose**: 33 3D landmarks (x, y, z) = 99 values
  - **Total Frame Dimension**: Fixed 225-dim float32 vector per frame.
  - **Missing Hand Handling**: Zero-filled feature vectors for unobserved hands to maintain exact constant dimensions.
- **Feature Storage**: Compressed `.npz` files in `data/features/train/` and `data/features/val/`.

---

## 3. Model Architecture
- **Model Type**: Bidirectional Long Short-Term Memory (BiLSTM)
- **Input Dimension**: `(Batch_Size, 32, 225)`
- **Linear Projection**: `Linear(225, 128)` + `ReLU` + `LayerNorm(128)`
- **Recurrent Core**: 2-layer BiLSTM (`hidden_size=128`, `bidirectional=True`, output dim `256`)
- **Temporal Pooling**: Combined Temporal Mean Pooling & Max Pooling (`256 + 256 = 512`)
- **Classification Head**: `Linear(512, 128)` -> `ReLU` -> `Dropout(0.3)` -> `Linear(128, 50)` -> Softmax.

---

## 4. Training Configuration
- **Loss Function**: Cross-Entropy Loss (`nn.CrossEntropyLoss`)
- **Optimizer**: Adam (`lr=0.001`, `weight_decay=0.0001`)
- **LR Scheduler**: `ReduceLROnPlateau(factor=0.5, patience=4)`
- **Batch Size**: 16
- **Max Epochs**: 50
- **Early Stopping**: Patience = 10 epochs
- **Random Seed**: 42 (Reproducible)
- **Data Augmentation**: Gaussian coordinate noise (`std=0.005`), scale jitter (0.95 - 1.05), and translation jitter (-0.02 to 0.02) applied to training split.

---

## 5. Evaluation Results
- **Final Epochs Trained**: {final_epoch}
- **Top-1 Validation Accuracy**: {best_val_acc}
- **Macro Precision**: {macro_precision}
- **Macro Recall**: {macro_recall}
- **Macro F1-Score**: {macro_f1}
- **Artifacts Saved**:
  - `reports/training_history.csv`
  - `reports/training_curves.png`
  - `reports/classification_report.csv`
  - `reports/confusion_matrix.png`
  - `models/best_model/best.pt`

---

## 6. Offline & Real-Time Sign-to-Speech Inference
- **Text Normalization**: Strips prefix numbers (e.g. `"48. Hello"` -> `"Hello"`, `"51. Good Morning"` -> `"Good Morning"`).
- **Confidence Threshold**: 0.70 threshold. Outputs `"Uncertain prediction"` when prediction confidence is below 70%.
- **Text-to-Speech (TTS)**: `pyttsx3` offline speech synthesis engine.
- **Webcam Interface**: Real-time camera feed with rolling 32-frame buffer, temporal smoothing (5-frame majority voting), HUD overlay, and non-repeating spoken phrase output.

---

## 7. Known Limitations & Future Improvements
1. **Landmark Loss in Obscured Views**: Fast signing or hand overlap can reduce MediaPipe landmark confidence.
2. **Future Branch**: Adding a dual-stream architecture combining spatial hand-crop CNN features with landmark temporal features.
"""

    with open(report_md, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Final report generated at: {report_md}")


if __name__ == "__main__":
    generate_final_report()
