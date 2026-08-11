"""
Real-Time Webcam Script for ISL Alphabet (A-Z) Recognition.

Features:
    - Automatic MediaPipe hand detection & crop bounding box
    - Real-time hand tracking bounding box on video feed
    - MASSIVE hero letter display when a gesture match is detected
    - Temporal smoothing (rolling majority vote over N frames)
    - HUD overlay with confidence bar & Top-3 predictions
    - Optional TTS speech output on stable high-confidence predictions
    - Press Q to quit

Usage:
    python scripts/webcam_alphabet.py
    python scripts/webcam_alphabet.py --no-speak
    python scripts/webcam_alphabet.py --source 1
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

# Color palette
COLOR_BG_DARK = (15, 15, 15)
COLOR_NEON_GREEN = (0, 240, 120)
COLOR_NEON_CYAN = (240, 240, 0)
COLOR_UNCERTAIN = (0, 140, 255)
COLOR_WHITE = (255, 255, 255)
COLOR_YELLOW = (0, 220, 255)
COLOR_GRAY = (120, 120, 120)
COLOR_DARK_GREEN = (0, 100, 50)


def draw_massive_hud(
    frame: np.ndarray,
    prediction: str,
    confidence: float,
    is_certain: bool,
    top3: list,
    hand_box: tuple = None,
) -> np.ndarray:
    """
    Draws a prominent, high-visibility HUD on the webcam frame.
    When matched (is_certain = True), shows the matched letter in MASSIVE bold font.
    """
    h, w = frame.shape[:2]

    # 1. Top HUD Banner (semi-transparent dark gradient)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 140), COLOR_BG_DARK, -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # 2. Draw Hand Bounding Box on Frame (if detected)
    if hand_box is not None:
        hx1, hy1, hx2, hy2 = hand_box
        box_color = COLOR_NEON_GREEN if is_certain else COLOR_UNCERTAIN
        thickness = 3 if is_certain else 2
        cv2.rectangle(frame, (hx1, hy1), (hx2, hy2), box_color, thickness)
        
        # Label tag above hand box
        tag_txt = f"{prediction} ({confidence:.0f}%)" if is_certain else "Hand Detected"
        cv2.rectangle(frame, (hx1, max(0, hy1 - 32)), (hx1 + 180, hy1), box_color, -1)
        cv2.putText(frame, tag_txt, (hx1 + 8, hy1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2, cv2.LINE_AA)

    # 3. MASSIVE MATCHED LETTER CARD (Top-Right Hero Display)
    card_w, card_h = 240, 240
    card_x = w - card_w - 20
    card_y = 20

    if is_certain and prediction not in ("?", "nothing", "del", "space"):
        # Match Glow Border on whole screen
        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), COLOR_NEON_GREEN, 6)

        # Hero Card Box Background
        card_bg = frame.copy()
        cv2.rectangle(card_bg, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_DARK_GREEN, -1)
        cv2.addWeighted(card_bg, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_NEON_GREEN, 4)

        # Card Title
        cv2.putText(frame, "MATCHED SIGN", (card_x + 35, card_y + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_NEON_CYAN, 2, cv2.LINE_AA)

        # MASSIVE LETTER TEXT (Centered in card, scale 5.5, thickness 12)
        font_scale = 5.5
        font_thick = 12
        (tw, th), _ = cv2.getTextSize(prediction, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        text_x = card_x + (card_w - tw) // 2
        text_y = card_y + (card_h + th) // 2 + 10

        # Drop shadow + Glow text
        cv2.putText(frame, prediction, (text_x + 4, text_y + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thick + 4, cv2.LINE_AA)
        cv2.putText(frame, prediction, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_WHITE, font_thick, cv2.LINE_AA)

        # Confidence Badge bottom of card
        conf_str = f"{confidence:.1f}% ACCURACY"
        cv2.putText(frame, conf_str, (card_x + 30, card_y + card_h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_NEON_GREEN, 2, cv2.LINE_AA)
    else:
        # Searching / Uncertain State Card
        card_bg = frame.copy()
        cv2.rectangle(card_bg, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_BG_DARK, -1)
        cv2.addWeighted(card_bg, 0.70, frame, 0.30, 0, frame)
        cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), COLOR_GRAY, 2)

        cv2.putText(frame, "SHOW HAND SIGN", (card_x + 30, card_y + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, COLOR_YELLOW, 1, cv2.LINE_AA)

        cv2.putText(frame, "?", (card_x + 85, card_y + 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 4.5, COLOR_GRAY, 8, cv2.LINE_AA)

    # 4. Top-Left HUD Info (Status, Confidence Bar, Top-3)
    cv2.putText(frame, "ISL ALPHABET DETECTOR", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_NEON_CYAN, 2, cv2.LINE_AA)

    # Confidence Bar
    bar_x, bar_y, bar_w, bar_h = 20, 50, 320, 22
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    fill = int(bar_w * min(confidence / 100.0, 1.0))
    bar_color = COLOR_NEON_GREEN if is_certain else COLOR_UNCERTAIN
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), bar_color, -1)
    cv2.putText(frame, f"Conf: {confidence:.1f}%", (bar_x + bar_w + 10, bar_y + 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    # Top-3 Probabilities
    for i, (label, pct) in enumerate(top3[:3]):
        txt = f"{'★' if i == 0 else ' '} {label}: {pct:.1f}%"
        cv2.putText(frame, txt, (20, 95 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    COLOR_YELLOW if i == 0 else COLOR_GRAY, 1, cv2.LINE_AA)

    # Bottom Quit Hint
    cv2.putText(frame, "Press Q to Quit", (20, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WHITE, 1, cv2.LINE_AA)

    return frame


def load_alphabet_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser(
        description="ISL Alphabet Real-Time Webcam Recognition (High-Visibility HUD)"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Camera index (0, 1) or path to video file")
    parser.add_argument("--threshold", type=float, default=0.65,
                        help="Confidence threshold 0.0-1.0 (default 0.65)")
    parser.add_argument("--smooth", type=int, default=6,
                        help="Temporal smoothing window size (frames, default 6)")
    parser.add_argument("--no-speak", action="store_true",
                        help="Disable TTS speech output")
    parser.add_argument("--no-hand-crop", action="store_true",
                        help="Disable automatic MediaPipe hand crop")
    parser.add_argument("--model-dir", type=str, default=str(BEST_MODEL_DIR),
                        help="Path to model directory")
    args = parser.parse_args()

    config = load_alphabet_config(CONFIG_PATH)
    preproc_cfg = config.get("preprocessing", {})
    image_size = preproc_cfg.get("image_size", 224)

    # Hand crop enabled by default for maximum accuracy
    hand_crop = not args.no_hand_crop

    from src.alphabet.inference import AlphabetInferenceEngine

    try:
        engine = AlphabetInferenceEngine(
            best_model_dir=Path(args.model_dir),
            label_map_path=LABEL_MAP_PATH if LABEL_MAP_PATH.exists() else None,
            confidence_threshold=args.threshold,
            device_str="auto",
            hand_crop=hand_crop,
            image_size=image_size,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
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

    # Set camera resolution to 1280x720 if possible
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Smoothing buffer and TTS debounce
    pred_buffer: deque = deque(maxlen=args.smooth)
    last_spoken = ""
    last_spoken_time = 0.0
    SPEAK_COOLDOWN = 2.5  # seconds

    print("=" * 60)
    print("  ISL ALPHABET — REAL-TIME WEBCAM RECOGNITION")
    print("=" * 60)
    print(f"  Source     : {source}")
    print(f"  Threshold  : {args.threshold*100:.0f}%")
    print(f"  Smoothing  : {args.smooth} frames")
    print(f"  Hand Crop  : {'ENABLED' if hand_crop else 'DISABLED'}")
    print(f"  TTS        : {'DISABLED' if args.no_speak else 'ENABLED'}")
    print("  Press Q to quit")
    print("=" * 60 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video / cannot read frame.")
                break

            # Mirror frame for intuitive webcam view
            if source.isdigit():
                frame = cv2.flip(frame, 1)

            # Get image dimensions
            h, w = frame.shape[:2]

            # Run inference (applies MediaPipe hand crop internally if enabled)
            result = engine.predict_from_numpy(frame)
            pred_buffer.append(result["prediction"] if result["is_certain"] else None)

            # Temporal smoothing (majority vote)
            valid = [p for p in pred_buffer if p is not None]
            if valid:
                smoothed = Counter(valid).most_common(1)[0][0]
            else:
                smoothed = "?"

            is_certain = result["is_certain"] and (smoothed != "?")

            # Extract hand bounding box for drawing (if detected)
            hand_box = None
            if engine.hand_preprocessor is not None and engine.hand_preprocessor._detector is not None:
                try:
                    import mediapipe as mp
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                    res = engine.hand_preprocessor._detector.detect(mp_img)
                    if res.hand_landmarks:
                        hand = res.hand_landmarks[0]
                        xs = [lm.x for lm in hand]
                        ys = [lm.y for lm in hand]
                        pad = 0.1
                        hx1 = max(0, int((min(xs) - pad) * w))
                        hy1 = max(0, int((min(ys) - pad) * h))
                        hx2 = min(w, int((max(xs) + pad) * w))
                        hy2 = min(h, int((max(ys) + pad) * h))
                        if hx2 > hx1 and hy2 > hy1:
                            hand_box = (hx1, hy1, hx2, hy2)
                except Exception:
                    pass

            # Draw Massive HUD Overlay
            frame = draw_massive_hud(
                frame,
                prediction=smoothed,
                confidence=result["confidence"],
                is_certain=is_certain,
                top3=result["top5"][:3],
                hand_box=hand_box,
            )

            cv2.imshow("ISL Alphabet Recognition — Press Q to Quit", frame)

            # TTS: speak stable prediction
            if tts and is_certain and smoothed not in ("?", "nothing", "del", "space"):
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
        print("Webcam recognition stopped.")


if __name__ == "__main__":
    main()
