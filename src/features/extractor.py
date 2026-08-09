"""
Landmark Feature Extractor using MediaPipe Holistic.
Extracts 3D coordinates for Left Hand (21), Right Hand (21), and Body Pose (33).
Produces a constant 225-dimensional feature vector per video frame.
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple


class LandmarkExtractor:
    """
    Extracts fixed-length (225-dim) landmark feature vectors from video frames using MediaPipe Holistic.
    """

    def __init__(self, min_detection_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def extract_frame_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Extracts 225-dim feature vector from a single BGR OpenCV frame.
        Layout:
        - Left Hand  : 21 landmarks * 3 (x, y, z) = 63 floats
        - Right Hand : 21 landmarks * 3 (x, y, z) = 63 floats
        - Pose       : 33 landmarks * 3 (x, y, z) = 99 floats
        Total = 225 floats
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return np.zeros(225, dtype=np.float32)

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False

        try:
            results = self.holistic.process(image_rgb)
        except Exception:
            return np.zeros(225, dtype=np.float32)

        # Left Hand (21 x 3 = 63)
        if results.left_hand_landmarks:
            lh = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            lh = np.zeros(63, dtype=np.float32)

        # Right Hand (21 x 3 = 63)
        if results.right_hand_landmarks:
            rh = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark], dtype=np.float32).flatten()
        else:
            rh = np.zeros(63, dtype=np.float32)

        # Pose (33 x 3 = 99)
        if results.pose_landmarks:
            pose = np.array([[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark], dtype=np.float32).flatten()
        else:
            pose = np.zeros(99, dtype=np.float32)

        features = np.concatenate([lh, rh, pose])
        return features.astype(np.float32)

    def close(self):
        """Releases MediaPipe resources."""
        if hasattr(self, "holistic") and self.holistic:
            self.holistic.close()
