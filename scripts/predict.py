"""
Inference Script for INCLUDE-50 Sign Language Recognition.
Takes a single video file path, extracts 32-frame MediaPipe landmarks, runs model prediction,
applies confidence thresholding (0.70), and outputs predicted sign & top-5 probabilities.
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.features.extractor import LandmarkExtractor
from src.models.bilstm import SignBiLSTMModel
from src.preprocessing.processor import sample_frame_indices
from src.utils.labels import normalize_label


def predict_video(
    video_path: Path,
    model: torch.nn.Module,
    id_to_label: dict,
    device: torch.device,
    confidence_threshold: float = 0.70,
) -> dict:
    """
    Runs end-to-end inference on a single video file.
    """
    vpath = Path(video_path)
    if not vpath.exists():
        raise FileNotFoundError(f"Video file not found: {vpath}")

    extractor = LandmarkExtractor()
    try:
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            raise RuntimeError(f"OpenCV cannot open video file: {vpath}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            frameCount = 0
            while True:
                ret, _ = cap.read()
                if not ret:
                    break
                frameCount += 1
            total_frames = frameCount
            cap.release()
            cap = cv2.VideoCapture(str(vpath))

        target_indices = set(sample_frame_indices(total_frames, num_frames=32))
        ordered_indices = sample_frame_indices(total_frames, num_frames=32)

        frame_map = {}
        curr = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if curr in target_indices:
                frame_map[curr] = extractor.extract_frame_features(frame)
            curr += 1
        cap.release()

        fallback = next(iter(frame_map.values())) if frame_map else np.zeros(225, dtype=np.float32)
        features_list = [frame_map.get(idx, fallback) for idx in ordered_indices]
        feature_matrix = np.array(features_list, dtype=np.float32)  # (32, 225)
    finally:
        extractor.close()

    input_tensor = torch.from_numpy(feature_matrix).unsqueeze(0).to(device)  # (1, 32, 225)

    model.eval()
    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    top5_indices = np.argsort(probs)[-5:][::-1]
    top_class_id = top5_indices[0]
    top_confidence = float(probs[top_class_id])
    raw_label = id_to_label[top_class_id]
    clean_label = normalize_label(raw_label)

    is_certain = top_confidence >= confidence_threshold
    display_sign = clean_label if is_certain else "Uncertain prediction"

    top5_results = []
    for idx in top5_indices:
        raw_name = id_to_label[idx]
        norm_name = normalize_label(raw_name)
        top5_results.append((norm_name, float(probs[idx]) * 100))

    return {
        "raw_label": raw_label,
        "clean_label": clean_label,
        "display_sign": display_sign,
        "confidence": top_confidence * 100,
        "is_certain": is_certain,
        "top5": top5_results,
    }


def main():
    parser = argparse.ArgumentParser(description="Predict Sign Language from Video File")
    parser.add_argument("--video", type=str, required=True, help="Path to video file (.MOV)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    args = parser.parse_args()

    model_path = Path("models/best_model/best.pt")
    if not model_path.exists():
        print(f"Error: Best model checkpoint not found at {model_path}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint["label_map"]
    id_to_label = {v: k for k, v in label_map.items()}

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

    res = predict_video(
        video_path=Path(args.video),
        model=model,
        id_to_label=id_to_label,
        device=device,
        confidence_threshold=args.threshold,
    )

    print("=" * 60)
    print("INCLUDE-50 SIGN RECOGNITION INFERENCE")
    print("=" * 60)
    print(f"Video file      : {args.video}")
    print(f"Predicted sign  : {res['display_sign']}")
    print(f"Confidence      : {res['confidence']:.2f}%")
    print(f"Certainty Check : {'PASS' if res['is_certain'] else 'UNCERTAIN (< ' + str(args.threshold*100) + '%)'}")

    print("\nTop 5 Predictions:")
    print("-" * 40)
    for idx, (sign, prob) in enumerate(res["top5"], 1):
        print(f"  {idx}. {sign:25} — {prob:.2f}%")


if __name__ == "__main__":
    main()
