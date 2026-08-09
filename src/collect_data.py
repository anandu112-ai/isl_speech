import csv
import os
import time

import cv2
import mediapipe as mp


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "models/hand_landmarker.task"
DATASET_FILE = "dataset/processed/landmarks.csv"

SIGNS = [
    "HELLO",
    "YES",
    "NO",
    "THANK_YOU",
    "I_LOVE_YOU",
]

SAMPLES_PER_SIGN = 200


# ==========================================
# CREATE DATASET FILE
# ==========================================

def create_dataset_file():
    """Create the CSV file and header if it doesn't exist."""

    os.makedirs("dataset/processed", exist_ok=True)

    if os.path.exists(DATASET_FILE):
        return

    header = ["label"]

    # 2 hands
    # 21 landmarks per hand
    # x, y, z for every landmark
    #
    # 2 × 21 × 3 = 126 features

    for hand_number in range(1, 3):

        for landmark_number in range(21):

            header.extend([
                f"hand{hand_number}_x{landmark_number}",
                f"hand{hand_number}_y{landmark_number}",
                f"hand{hand_number}_z{landmark_number}",
            ])

    with open(
        DATASET_FILE,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)
        writer.writerow(header)

    print(f"Created dataset file: {DATASET_FILE}")


# ==========================================
# EXTRACT LANDMARK FEATURES
# ==========================================

def extract_landmarks(result):
    """
    Extract hand landmarks.

    Always returns exactly 126 values:

        2 hands
        × 21 landmarks
        × 3 coordinates

        = 126 features

    If only one hand is detected,
    the second hand is filled with zeros.
    """

    features = []

    detected_hands = result.hand_landmarks

    for hand_index in range(2):

        # A hand exists
        if hand_index < len(detected_hands):

            hand = detected_hands[hand_index]

            for landmark in hand:

                features.extend([
                    landmark.x,
                    landmark.y,
                    landmark.z
                ])

        # No hand detected for this slot
        else:

            features.extend([0.0] * 63)

    return features


# ==========================================
# DRAW LANDMARKS
# ==========================================

def draw_landmarks(frame, result):
    """Draw detected hand landmarks on the webcam frame."""

    height, width = frame.shape[:2]

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
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

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


# ==========================================
# MAIN PROGRAM
# ==========================================

def main():

    print()
    print("========================================")
    print("       ISL DATASET COLLECTOR")
    print("========================================")

    # --------------------------------------
    # Check model
    # --------------------------------------

    if not os.path.exists(MODEL_PATH):

        print()
        print("ERROR: Hand Landmarker model not found.")
        print()
        print(f"Expected location:")
        print(MODEL_PATH)
        print()

        return

    # --------------------------------------
    # Create dataset
    # --------------------------------------

    create_dataset_file()

    # --------------------------------------
    # MediaPipe configuration
    # --------------------------------------

    BaseOptions = mp.tasks.BaseOptions
    Vision = mp.tasks.vision

    options = Vision.HandLandmarkerOptions(

        base_options=BaseOptions(
            model_asset_path=MODEL_PATH
        ),

        running_mode=Vision.RunningMode.IMAGE,

        num_hands=2,

        min_hand_detection_confidence=0.5,

        min_hand_presence_confidence=0.5,

    )

    detector = Vision.HandLandmarker.create_from_options(
        options
    )

    # --------------------------------------
    # Open webcam
    # --------------------------------------

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():

        print()
        print("ERROR: Could not open webcam.")
        print()

        detector.close()

        return

    # --------------------------------------
    # Project information
    # --------------------------------------

    print()

    print("Signs:")
    
    for number, sign in enumerate(SIGNS, start=1):

        print(f"{number}. {sign}")

    print()

    print(f"Samples per sign : {SAMPLES_PER_SIGN}")

    print(
        f"Total samples    : "
        f"{len(SIGNS) * SAMPLES_PER_SIGN}"
    )

    print()

    print("Press Q at any time to stop.")
    print()

    # ======================================
    # COLLECT EACH SIGN
    # ======================================

    for sign in SIGNS:

        print()
        print("----------------------------------------")
        print(f"NEXT SIGN: {sign}")
        print("----------------------------------------")
        print()

        input(
            "Position yourself and press ENTER "
            "when ready..."
        )

        # ----------------------------------
        # Countdown
        # ----------------------------------

        print()

        for number in [3, 2, 1]:

            print(number)

            time.sleep(1)

        print()
        print("COLLECTING...")
        print()

        samples = 0

        # ==================================
        # COLLECT SAMPLES
        # ==================================

        while samples < SAMPLES_PER_SIGN:

            success, frame = camera.read()

            if not success:

                print(
                    "ERROR: Could not read "
                    "from webcam."
                )

                break

            # ----------------------------------
            # Convert BGR → RGB
            # ----------------------------------

            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            # ----------------------------------
            # Create MediaPipe image
            # ----------------------------------

            mp_image = mp.Image(

                image_format=mp.ImageFormat.SRGB,

                data=rgb_frame
            )

            # ----------------------------------
            # Detect hands
            # ----------------------------------

            result = detector.detect(mp_image)

            # ----------------------------------
            # Draw landmarks
            # ----------------------------------

            draw_landmarks(
                frame,
                result
            )

            # ----------------------------------
            # Extract features
            # ----------------------------------

            if result.hand_landmarks:

                features = extract_landmarks(
                    result
                )

                # Safety check
                if len(features) == 126:

                    with open(
                        DATASET_FILE,
                        "a",
                        newline=""
                    ) as file:

                        writer = csv.writer(file)

                        writer.writerow(
                            [sign] + features
                        )

                    samples += 1

            # ==================================
            # DISPLAY INFORMATION
            # ==================================

            cv2.putText(

                frame,

                f"Sign: {sign}",

                (20, 40),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.9,

                (0, 255, 0),

                2
            )

            cv2.putText(

                frame,

                f"Samples: "
                f"{samples}/{SAMPLES_PER_SIGN}",

                (20, 80),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.8,

                (0, 255, 0),

                2
            )

            cv2.putText(

                frame,

                "Press Q to stop",

                (20, 120),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.7,

                (0, 255, 255),

                2
            )

            # ----------------------------------
            # Show webcam
            # ----------------------------------

            cv2.imshow(
                "ISL Dataset Collector",
                frame
            )

            # ----------------------------------
            # Quit
            # ----------------------------------

            if cv2.waitKey(1) & 0xFF == ord("q"):

                print()
                print("Collection stopped by user.")

                camera.release()

                cv2.destroyAllWindows()

                detector.close()

                return

        # ----------------------------------
        # Sign completed
        # ----------------------------------

        print()
        print(
            f"{sign}: "
            f"{samples} samples collected."
        )

    # ======================================
    # CLEANUP
    # ======================================

    camera.release()

    cv2.destroyAllWindows()

    detector.close()

    # ======================================
    # COMPLETE
    # ======================================

    print()
    print("========================================")
    print("          DATASET COMPLETE")
    print("========================================")
    print()

    print(f"Dataset saved to:")

    print(DATASET_FILE)

    print()

    print(
        f"Total expected samples: "
        f"{len(SIGNS) * SAMPLES_PER_SIGN}"
    )

    print()


# ==========================================
# PROGRAM ENTRY POINT
# ==========================================

if __name__ == "__main__":
    main()