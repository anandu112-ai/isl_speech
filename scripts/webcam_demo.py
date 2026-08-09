"""
Real-Time Webcam Sign-to-Speech Demo Script for INCLUDE-50.
Features:
- Live OpenCV camera feed
- Rolling 32-frame buffer of MediaPipe 225-dim landmarks
- Real-time BiLSTM inference
- Temporal prediction smoothing (rolling window majority voting)
- On-screen HUD displaying predicted sign and confidence score
- Non-repeating TTS speech engine output for high-confidence predictions (>= 0.70)
"""

import argparse
from collections import Counter, deque
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch

from src.features.extractor import LandmarkExtractor
from src.models.bilstm import SignBiLSTMModel
from src.speech.tts import SpeechEngine
from src.utils.labels import normalize_label


def main():
    parser = argparse.ArgumentParser(description="INCLUDE-50 Real-Time Webcam Sign-to-Speech Demo")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument("--threshold", type=float, default=0.70, help="Confidence threshold (default: 0.70)")
    parser.add_argument("--no-speak", action="store_true", help="Disable audio TTS output")
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
    model.eval()

    tts = SpeechEngine() if not args.no_speak else None

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Error: Could not open camera {args.camera}")
        sys.exit(1)

    extractor = LandmarkExtractor()

    frame_buffer = deque(maxlen=32)
    history_predictions = deque(maxlen=5)

    last_spoken_sign = ""
    last_spoken_time = 0

    print("=" * 60)
    print("REAL-TIME WEBCAM SIGN RECOGNITION DEMO")
    print("=" * 60)
    print("Press 'q' to quit.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Mirror frame for intuitive webcam view
            frame = cv2.flip(frame, 1)

            # Extract frame landmarks
            feat = extractor.extract_frame_features(frame)
            frame_buffer.append(feat)

            display_text = "Buffering frames..."
            confidence_str = ""

            if len(frame_buffer) == 32:
                matrix = np.array(list(frame_buffer), dtype=np.float32)  # (32, 225)
                input_tensor = torch.from_numpy(matrix).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(input_tensor)
                    probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

                top_idx = int(np.argmax(probs))
                conf = float(probs[top_idx])
                raw_sign = id_to_label[top_idx]
                clean_sign = normalize_label(raw_sign)

                history_predictions.append((clean_sign, conf))

                # Temporal smoothing via majority voting
                recent_signs = [s for s, c in history_predictions if c >= args.threshold]
                if recent_signs:
                    smoothed_sign, count = Counter(recent_signs).most_common(1)[0]
                    avg_conf = np.mean([c for s, c in history_predictions if s == smoothed_sign])
                    display_text = smoothed_sign
                    confidence_str = f"{avg_conf * 100:.1f}%"

                    # Speak stable sign if new and enough time elapsed
                    if (
                        tts is not None
                        and smoothed_sign != last_spoken_sign
                        and (time.time() - last_spoken_time) > 2.0
                    ):
                        tts.speak(smoothed_sign)
                        last_spoken_sign = smoothed_sign
                        last_spoken_time = time.time()
                else:
                    display_text = "Uncertain sign"
                    confidence_str = f"{conf * 100:.1f}%"

            # Draw HUD overlay
            cv2.rectangle(frame, (10, 10), (450, 90), (0, 0, 0), -1)
            cv2.putText(frame, f"Sign: {display_text}", (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            if confidence_str:
                cv2.putText(frame, f"Conf: {confidence_str}", (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            cv2.imshow("INCLUDE-50 Sign-to-Speech", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        extractor.close()


if __name__ == "__main__":
    main()
