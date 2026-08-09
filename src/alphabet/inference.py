"""
Inference engine for ISL Alphabet + Digit CNN classifier.

Loads best.pt checkpoint and performs single-image or single-frame inference.
Returns normalized class label (one char: A-Z or 0-9), confidence, and top-5 predictions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError

from src.alphabet.model import load_checkpoint
from src.alphabet.preprocessing import HandCropPreprocessor, build_eval_transform
from src.utils.labels import load_label_map

logger = logging.getLogger(__name__)


class AlphabetInferenceEngine:
    """
    Loads the best trained AlphabetCNNModel and runs single-character ISL inference.

    Args:
        best_model_dir:       Path to directory containing best.pt & label_map.json
        confidence_threshold: Minimum confidence to accept as a valid prediction (0-1)
        device_str:           'auto', 'cuda', or 'cpu'
        hand_crop:            Whether to apply MediaPipe hand detection & crop
    """

    def __init__(
        self,
        best_model_dir: Path = Path("models/alphabet/best_model"),
        label_map_path: Optional[Path] = None,
        confidence_threshold: float = 0.70,
        device_str: str = "auto",
        hand_crop: bool = False,
        image_size: int = 224,
    ):
        self.confidence_threshold = confidence_threshold
        self.image_size = image_size

        # Resolve device
        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)
        logger.info(f"Inference device: {self.device}")

        # Load label map
        lm_path = label_map_path or (Path(best_model_dir) / "label_map.json")
        if not lm_path.exists():
            lm_path = Path("models/alphabet/label_map.json")
        if not lm_path.exists():
            raise FileNotFoundError(f"label_map.json not found at {lm_path}")
        self.label_map: Dict[str, int] = load_label_map(lm_path)
        self.id_to_label: Dict[int, str] = {v: k for k, v in self.label_map.items()}
        logger.info(f"Loaded label map with {len(self.label_map)} classes from {lm_path}")

        # Load model
        best_pt = Path(best_model_dir) / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Best model checkpoint not found: {best_pt}")
        self.model, self.ckpt = load_checkpoint(best_pt, self.device, num_classes=len(self.label_map))
        self.model.eval()

        # Preprocessing transform
        self.transform = build_eval_transform(image_size=image_size)

        # Optional hand crop
        self.hand_preprocessor: Optional[HandCropPreprocessor] = None
        if hand_crop:
            self.hand_preprocessor = HandCropPreprocessor()

    def predict(self, image: Image.Image) -> Dict:
        """
        Runs inference on a PIL image.

        Returns:
            {
                'prediction': 'A',          # Best class (single char)
                'confidence': 96.32,        # As percentage
                'is_certain': True,         # Confidence >= threshold
                'top5': [('A', 96.32), ...] # Top-5 (label, %)
            }
        """
        # Optional hand crop
        if self.hand_preprocessor is not None:
            image = self.hand_preprocessor.crop(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top5_indices = np.argsort(probs)[-5:][::-1]
        top_class_id = int(top5_indices[0])
        top_confidence = float(probs[top_class_id]) * 100.0
        predicted_label = self.id_to_label.get(top_class_id, "?")

        is_certain = top_confidence / 100.0 >= self.confidence_threshold
        display = predicted_label if is_certain else "?"

        top5 = [
            (self.id_to_label.get(int(idx), "?"), float(probs[idx]) * 100.0)
            for idx in top5_indices
        ]

        return {
            "prediction": predicted_label,
            "display": display,
            "confidence": round(top_confidence, 2),
            "is_certain": is_certain,
            "top5": top5,
            "all_probs": probs,
        }

    def predict_from_path(self, image_path: Path) -> Dict:
        """Loads an image from disk and runs inference."""
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        try:
            image = Image.open(image_path).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"Cannot read image {image_path}: {e}")
        return self.predict(image)

    def predict_from_numpy(self, frame_bgr: np.ndarray) -> Dict:
        """Accepts a BGR OpenCV frame and runs inference."""
        import cv2
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.predict(image)

    def close(self) -> None:
        if self.hand_preprocessor is not None:
            self.hand_preprocessor.close()
