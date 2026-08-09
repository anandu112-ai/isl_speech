import os

import cv2
import joblib
import mediapipe as mp


MODEL_PATH = "models/isl_random_forest.pkl"
HAND_MODEL_PATH = "models/hand_landmarker.task"


def extract_landmarks(result):

    features = []

    detected_hands = result.hand_landmarks

    # Always create exactly 126 features
    for hand_index in range(2):

        if hand_index < len(detected_hands):

            hand = detected_hands[hand_index]

            for landmark in hand:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

        else:

            features.extend([0.0] * 63)

    return features


def draw_landmarks(frame, result):

    height, width = frame.shape[:2]

    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (5, 9), (9, 10), (10, 11), (11, 12),
        (9, 13), (13, 14), (14, 15), (15, 16),
        (13, 17), (17, 18), (18, 19), (19, 20),
        (0, 17)
    ]

    for hand in result.hand_landmarks:

        # Draw points
        for landmark in hand:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )

        # Draw connections
        for start, end in connections:

            x1 = int(hand[start].x * width)
            y1 = int(hand[start].y * height)

            x2 = int(hand[end].x * width)
            y2 = int(hand[end].y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


def main():

    import argparse
    parser = argparse.ArgumentParser(description="Real-Time ISL Recognition with Random Forest")
    parser.add_argument("--source", type=str, default="0", help="Camera index (0, 1) or path to video file (.MOV, .mp4)")
    args = parser.parse_args()

    # =========================================
    # CHECK FILES
    # =========================================

    if not os.path.exists(MODEL_PATH):

        print("ERROR: Trained model not found.")
        print(MODEL_PATH)

        return

    if not os.path.exists(HAND_MODEL_PATH):

        print("ERROR: MediaPipe model not found.")
        print(HAND_MODEL_PATH)

        return

    # =========================================
    # LOAD RANDOM FOREST
    # =========================================

    print("Loading ISL classifier...")

    model = joblib.load(MODEL_PATH)

    print("Model loaded.")

    # =========================================
    # MEDIAPIPE
    # =========================================

    BaseOptions = mp.tasks.BaseOptions
    Vision = mp.tasks.vision

    options = Vision.HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=HAND_MODEL_PATH
        ),

        running_mode=Vision.RunningMode.IMAGE,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

    )

    detector = Vision.HandLandmarker.create_from_options(
        options
    )

    # =========================================
    # VIDEO / CAMERA CAPTURE
    # =========================================

    source = args.source
    if source.isdigit():
        cam_idx = int(source)
        # Try DirectShow backend on Windows first, fallback to default
        camera = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not camera.isOpened():
            camera = cv2.VideoCapture(cam_idx)
    else:
        camera = cv2.VideoCapture(source)

    if not camera.isOpened():

        print(f"ERROR: Could not open video source: {source}")

        detector.close()

        return

    print()
    print("======================================")
    print("       REAL-TIME ISL RECOGNITION")
    print("======================================")
    print()
    print(f"Source: {source}")
    print("Show one of the trained signs.")
    print("Press Q to quit.")
    print()

    # =========================================
    # CAMERA LOOP
    # =========================================

    while True:

        success, frame = camera.read()

        if not success:

            print("ERROR: Could not read webcam.")

            break

        # -------------------------------------
        # Convert BGR → RGB
        # -------------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------------------
        # MediaPipe image
        # -------------------------------------

        mp_image = mp.Image(

            image_format=mp.ImageFormat.SRGB,

            data=rgb_frame
        )

        # -------------------------------------
        # Detect hands
        # -------------------------------------

        result = detector.detect(mp_image)

        prediction = "No hand"

        confidence = 0.0

        # -------------------------------------
        # Predict
        # -------------------------------------

        if result.hand_landmarks:

            features = extract_landmarks(result)

            if len(features) == 126:

                prediction = model.predict(
                    [features]
                )[0]

                probabilities = model.predict_proba(
                    [features]
                )[0]

                confidence = max(
                    probabilities
                ) * 100

        # -------------------------------------
        # Draw landmarks
        # -------------------------------------

        draw_landmarks(
            frame,
            result
        )

        # -------------------------------------
        # Display prediction
        # -------------------------------------

        cv2.rectangle(
            frame,
            (10, 10),
            (500, 100),
            (0, 0, 0),
            -1
        )

        cv2.putText(

            frame,

            f"Sign: {prediction}",

            (25, 50),

            cv2.FONT_HERSHEY_SIMPLEX,

            1,

            (0, 255, 0),

            2
        )

        cv2.putText(

            frame,

            f"Confidence: {confidence:.1f}%",

            (25, 85),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.7,

            (255, 255, 255),

            2
        )

        # -------------------------------------
        # Show
        # -------------------------------------

        cv2.imshow(
            "ISL Recognition",
            frame
        )

        # -------------------------------------
        # Quit
        # -------------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break

    # =========================================
    # CLEANUP
    # =========================================

    camera.release()

    cv2.destroyAllWindows()

    detector.close()

    print("ISL recognition stopped.")


if __name__ == "__main__":
    main()