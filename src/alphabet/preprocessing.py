"""
Image preprocessing and augmentation pipelines for ISL Alphabet + Digit recognition.

Provides:
    build_train_transform()   — Training pipeline with configurable augmentations
    build_eval_transform()    — Validation/Test deterministic pipeline (no random ops)
    HandCropPreprocessor      — Optional MediaPipe-based hand bounding box crop
"""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

logger = logging.getLogger(__name__)

# ImageNet statistics (used since backbone is ImageNet-pretrained)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_train_transform(
    image_size: int = 224,
    horizontal_flip: bool = False,
    rotation_degrees: float = 10,
    translate: float = 0.05,
    brightness_contrast: float = 0.1,
) -> Callable:
    """
    Builds the training augmentation pipeline.

    IMPORTANT: `horizontal_flip` defaults to False for ISL, where handedness matters.

    Args:
        image_size:         Target image size (square).
        horizontal_flip:    Whether to randomly flip horizontally (False = safe for ISL).
        rotation_degrees:   Max degrees of random rotation.
        translate:          Max translation fraction (both axes).
        brightness_contrast: Brightness/contrast jitter factor.
    """
    aug_list = [
        transforms.RandomResizedCrop(
            image_size,
            scale=(0.75, 1.0),
            ratio=(0.85, 1.15),
        ),
        transforms.RandomAffine(
            degrees=rotation_degrees,
            translate=(translate, translate),
        ),
        transforms.ColorJitter(
            brightness=brightness_contrast,
            contrast=brightness_contrast,
            saturation=0.05,
        ),
    ]

    if horizontal_flip:
        logger.warning(
            "horizontal_flip=True is enabled. Ensure this is appropriate for your ISL dataset."
        )
        aug_list.append(transforms.RandomHorizontalFlip(p=0.5))

    aug_list += [
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]

    return transforms.Compose(aug_list)


def build_eval_transform(image_size: int = 224) -> Callable:
    """
    Builds a deterministic evaluation transform — resizes directly to square target size
    without CenterCrop to avoid clipping fingertips from cropped hand images.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HandCropPreprocessor:
    """
    Optional MediaPipe hand detection and bounding-box crop preprocessor.
    Crops a padded square bounding box around the detected hand to preserve aspect ratio.
    """

    def __init__(self, padding: float = 0.20, min_hand_detection_confidence: float = 0.4):
        self.padding = padding
        self.min_detection_confidence = min_hand_detection_confidence
        self._detector = None
        self._init_detector()

    def _init_detector(self) -> None:
        """Initialize MediaPipe hand detection (Tasks API)."""
        if self._detector is not None:
            return
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            Vision = mp.tasks.vision

            possible_paths = [
                Path("models/hand_landmarker.task"),
                Path("models/alphabet/hand_landmarker.task"),
            ]
            model_path = next((p for p in possible_paths if p.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    "hand_landmarker.task not found. Place it in models/ directory."
                )

            options = Vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=Vision.RunningMode.IMAGE,
                num_hands=2,  # Support BOTH Left and Right Hands for ISL
                min_hand_detection_confidence=self.min_detection_confidence,
            )
            self._detector = Vision.HandLandmarker.create_from_options(options)
            logger.info("HandCropPreprocessor: Dual-Hand MediaPipe detector initialized successfully.")
        except Exception as e:
            logger.warning(f"HandCropPreprocessor: Could not initialize MediaPipe: {e}. Hand crop disabled.")
            self._detector = None

    def detect_hand_bbox(self, frame_bgr: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[np.ndarray], Optional[object]]:
        """
        Detects 1 or 2 hands in a BGR numpy frame.
        Returns: ((x1, y1, x2, y2), cropped_hand_bgr, mp_result) or (None, None, None).
        """
        if self._detector is None:
            return None, None, None

        try:
            import mediapipe as mp
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = self._detector.detect(mp_image)

            if not result.hand_landmarks:
                return None, None, None

            h, w = frame_bgr.shape[:2]
            
            # Aggregate all landmark coordinates across all detected hands (1 or 2)
            all_xs = []
            all_ys = []
            for hand in result.hand_landmarks:
                all_xs.extend([lm.x for lm in hand])
                all_ys.extend([lm.y for lm in hand])

            x_min_raw, x_max_raw = min(all_xs) * w, max(all_xs) * w
            y_min_raw, y_max_raw = min(all_ys) * h, max(all_ys) * h

            box_w = x_max_raw - x_min_raw
            box_h = y_max_raw - y_min_raw
            cx = (x_min_raw + x_max_raw) / 2.0
            cy = (y_min_raw + y_max_raw) / 2.0

            side = max(box_w, box_h) * (1.0 + self.padding * 2.0)

            x1 = max(0, int(cx - side / 2.0))
            y1 = max(0, int(cy - side / 2.0))
            x2 = min(w, int(cx + side / 2.0))
            y2 = min(h, int(cy + side / 2.0))

            if x2 <= x1 or y2 <= y1:
                return None, None, result

            crop_bgr = frame_bgr[y1:y2, x1:x2]
        except Exception as e:
            logger.warning(f"detect_hand_bbox failed: {e}")
            return None, None, None

    def crop(self, image: Image.Image) -> Image.Image:
        """
        Detects the first hand in `image` and returns a padded, square bounding-box crop.
        Falls back to the original image if no hand is detected or detector unavailable.
        """
        if self._detector is None:
            return image

        try:
            img_np = np.array(image.convert("RGB"))
            bbox, crop_rgb = self.detect_hand_bbox(cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
            if crop_rgb is not None and crop_rgb.size > 0:
                return Image.fromarray(cv2.cvtColor(crop_rgb, cv2.COLOR_BGR2RGB))
            return image
        except Exception as e:
            logger.warning(f"Hand crop failed: {e}. Returning original image.")
            return image

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
