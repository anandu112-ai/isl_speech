"""
Complete Offline Sign-to-Speech Demo Script for INCLUDE-50.
Executes end-to-end pipeline:
Video File -> 32-Frame Sampling -> MediaPipe Landmarks -> BiLSTM Model -> Normalized Text -> Speech Engine Output.
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from scripts.predict import predict_video
from src.models.bilstm import SignBiLSTMModel
from src.speech.tts import SpeechEngine


def main():
    parser = argparse.ArgumentParser(description="INCLUDE-50 Complete Sign-to-Speech Demo")
    parser.add_argument("--video", type=str, required=True, help="Path to video file (.MOV)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--no-speak", action="store_true", help="Disable audio speech output")
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

    print("=" * 60)
    print("INCLUDE-50 OFFLINE SIGN-TO-SPEECH DEMO")
    print("=" * 60)

    res = predict_video(
        video_path=Path(args.video),
        model=model,
        id_to_label=id_to_label,
        device=device,
        confidence_threshold=args.threshold,
    )

    print(f"\n[INPUT VIDEO]    : {args.video}")
    print(f"[RECOGNIZED SIGN]: {res['display_sign']}")
    print(f"[CONFIDENCE]     : {res['confidence']:.2f}%")

    if not args.no_speak and res["is_certain"]:
        engine = SpeechEngine()
        engine.speak(res["display_sign"])


if __name__ == "__main__":
    main()
