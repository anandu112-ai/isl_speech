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
    Builds a deterministic evaluation/test pipeline — no random operations.

    Args:
        image_size: Target image size (square).
    """
    # Resize slightly larger then center-crop for stable evaluation
    resize_to = int(image_size * 1.143)  # ~256 for 224
    return transforms.Compose([
        transforms.Resize(resize_to),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


class HandCropPreprocessor:
    """
    Optional MediaPipe hand detection and bounding-box crop preprocessor.

    Use when `hand_crop: true` is set in config. Falls back to full image
    when no hand is detected.

    Usage:
        preprocessor = HandCropPreprocessor(padding=0.1)
        cropped_pil = preprocessor.crop(pil_image)
    """

    def __init__(self, padding: float = 0.1, min_hand_detection_confidence: float = 0.5):
        self.padding = padding
        self.min_detection_confidence = min_hand_detection_confidence
        self._detector = None

    def _init_detector(self) -> None:
        """Lazy-initialize MediaPipe hand detection (Tasks API)."""
        try:
            import mediapipe as mp
            BaseOptions = mp.tasks.BaseOptions
            Vision = mp.tasks.vision

            # Search for the task model file
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
                num_hands=1,
                min_hand_detection_confidence=self.min_detection_confidence,
            )
            self._detector = Vision.HandLandmarker.create_from_options(options)
            logger.info("HandCropPreprocessor: MediaPipe hand detector initialized.")
        except Exception as e:
            logger.warning(f"HandCropPreprocessor: Could not initialize MediaPipe: {e}. Hand crop disabled.")
            self._detector = None

    def crop(self, image: Image.Image) -> Image.Image:
        """
        Detects the first hand in `image` and returns the padded bounding-box crop.
        Falls back to the original image if no hand is detected or detector unavailable.
        """
        if self._detector is None:
            self._init_detector()

        if self._detector is None:
            return image

        try:
            import mediapipe as mp
            img_np = np.array(image.convert("RGB"))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_np)
            result = self._detector.detect(mp_image)

            if not result.hand_landmarks:
                return image

            h, w = img_np.shape[:2]
            hand = result.hand_landmarks[0]
            xs = [lm.x for lm in hand]
            ys = [lm.y for lm in hand]

            pad = self.padding
            x_min = max(0, int((min(xs) - pad) * w))
            y_min = max(0, int((min(ys) - pad) * h))
            x_max = min(w, int((max(xs) + pad) * w))
            y_max = min(h, int((max(ys) + pad) * h))

            if x_max <= x_min or y_max <= y_min:
                return image

            cropped = img_np[y_min:y_max, x_min:x_max]
            return Image.fromarray(cropped)
        except Exception as e:
            logger.warning(f"Hand crop failed: {e}. Returning original image.")
            return image

    def close(self) -> None:
        if self._detector is not None:
            try:
                self._detector.close()
            except Exception:
                pass
