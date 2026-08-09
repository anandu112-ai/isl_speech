"""
Real-Time Webcam Script for ISL Alphabet (A-Z) + Digit (0-9) Recognition.

Features:
    - Live webcam or video file input
    - Per-frame CNN inference
    - Temporal smoothing (rolling majority vote over N frames)
    - On-screen HUD with predicted character and confidence bar
    - Optional TTS speech output on stable high-confidence predictions
    - Press Q to quit

Usage:
    python scripts/webcam_alphabet.py
    python scripts/webcam_alphabet.py --source 1
    python scripts/webcam_alphabet.py --source "video.mp4"
    python scripts/webcam_alphabet.py --no-speak
    python scripts/webcam_alphabet.py --smooth 10
"""

import argparse
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml

BEST_MODEL_DIR = Path("models/alphabet/best_model")
LABEL_MAP_PATH = Path("models/alphabet/label_map.json")
CONFIG_PATH = Path("configs/alphabet_config.yaml")

# HUD color palette
COLOR_BG = (20, 20, 20)
COLOR_ACCENT = (0, 200, 120)
COLOR_UNCERTAIN = (0, 130, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 220, 220)
COLOR_GRAY = (140, 140, 140)


def draw_hud(frame: np.ndarray, prediction: str, confidence: float, is_certain: bool, top3: list) -> np.ndarray:
    """Draws the prediction HUD overlay on the frame."""
    h, w = frame.shape[:2]

    # Semi-transparent black background banner
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 120), COLOR_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

    # Main prediction character
    color = COLOR_ACCENT if is_certain else COLOR_UNCERTAIN
    char_to_show = prediction if is_certain else "?"
    cv2.putText(frame, char_to_show, (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 3.2, color, 5, cv2.LINE_AA)

    # Confidence bar
    bar_x, bar_y, bar_w, bar_h = 140, 20, 300, 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), COLOR_GRAY, -1)
    fill = int(bar_w * min(confidence / 100.0, 1.0))
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    cv2.putText(frame, f"{confidence:.1f}%", (bar_x + bar_w + 8, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Top-3 labels
    for i, (label, pct) in enumerate(top3[:3]):
        txt = f"{'★' if i == 0 else ' '} {label}: {pct:.1f}%"
        cv2.putText(frame, txt, (140, 60 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_YELLOW if i == 0 else COLOR_GRAY, 1, cv2.LINE_AA)

    # Press Q hint
    cv2.putText(frame, "Press Q to quit", (w - 200, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_GRAY, 1, cv2.LINE_AA)

    return frame


def load_alphabet_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ISL Alphabet + Digit Real-Time Webcam Recognition"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (0, 1) or path to video file")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Confidence threshold 0.0-1.0")
    parser.add_argument("--smooth", type=int, default=8,
                        help="Temporal smoothing window size (frames, default 8)")
    parser.add_argument("--no-speak", action="store_true",
                        help="Disable TTS speech output")
    parser.add_argument("--hand-crop", action="store_true",
                        help="Enable MediaPipe hand crop before inference")
    parser.add_argument("--model-dir", type=str, default=str(BEST_MODEL_DIR),
                        help="Path to model directory")
    args = parser.parse_args()

    config = load_alphabet_config(CONFIG_PATH)
    inf_cfg = config.get("inference", {})
    preproc_cfg = config.get("preprocessing", {})

    threshold = args.threshold if args.threshold is not None else inf_cfg.get("confidence_threshold", 0.70)
    image_size = preproc_cfg.get("image_size", 224)

    from src.alphabet.inference import AlphabetInferenceEngine

    try:
        engine = AlphabetInferenceEngine(
            best_model_dir=Path(args.model_dir),
            label_map_path=LABEL_MAP_PATH if LABEL_MAP_PATH.exists() else None,
            confidence_threshold=threshold,
            device_str="auto",
            hand_crop=args.hand_crop,
            image_size=image_size,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Train the model first: python scripts/train_alphabet.py")
        sys.exit(1)

    # TTS engine
    tts = None
    if not args.no_speak:
        try:
            from src.speech.tts import SpeechEngine
            tts = SpeechEngine()
        except Exception as e:
            print(f"[Warning] TTS unavailable: {e}")

    # Open video source
    source = args.source
    if source.isdigit():
        cap = cv2.VideoCapture(int(source), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"ERROR: Cannot open video source: {source}")
        engine.close()
        sys.exit(1)

    # Smoothing buffer and TTS debounce
    pred_buffer: deque = deque(maxlen=args.smooth)
    last_spoken = ""
    last_spoken_time = 0.0
    SPEAK_COOLDOWN = 2.5  # seconds between TTS utterances

    print("=" * 50)
    print("  ISL ALPHABET + DIGIT — REAL-TIME RECOGNITION")
    print("=" * 50)
    print(f"  Source     : {source}")
    print(f"  Threshold  : {threshold*100:.0f}%")
    print(f"  Smoothing  : {args.smooth} frames")
    print(f"  Hand crop  : {args.hand_crop}")
    print(f"  TTS        : {'disabled' if args.no_speak else 'enabled'}")
    print("  Press Q to quit")
    print()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video / cannot read frame.")
                break

            # Mirror for intuitive webcam view
            if source.isdigit():
                frame = cv2.flip(frame, 1)

            # Run inference
            result = engine.predict_from_numpy(frame)
            pred_buffer.append(result["prediction"] if result["is_certain"] else None)

            # Temporal smoothing: majority vote over buffer
            valid = [p for p in pred_buffer if p is not None]
            if valid:
                smoothed = Counter(valid).most_common(1)[0][0]
                smooth_conf = result["confidence"] if result["prediction"] == smoothed else 50.0
            else:
                smoothed = "?"
                smooth_conf = 0.0

            is_certain = result["is_certain"] and bool(valid)

            # Draw HUD
            frame = draw_hud(
                frame,
                prediction=smoothed,
                confidence=result["confidence"],
                is_certain=is_certain,
                top3=result["top5"][:3],
            )

            cv2.imshow("ISL Alphabet + Digit Recognition — Press Q to quit", frame)

            # TTS: speak stable prediction
            if tts and is_certain and smoothed != "?":
                now = time.time()
                if smoothed != last_spoken or (now - last_spoken_time) > SPEAK_COOLDOWN:
                    tts.speak(smoothed)
                    last_spoken = smoothed
                    last_spoken_time = now

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        engine.close()
        print("Recognition stopped.")


if __name__ == "__main__":
    main()
