"""
Inference engine for ISL Alphabet CNN classifier.

Supports checkpoints saved by both:
  - The original project training script (AlphabetCNNModel / MobileNetV3)
  - External Kaggle-trained checkpoints (ResNet18 or other torchvision models)

Checkpoint auto-detection logic:
  1. If checkpoint contains 'model_name' key  → build that model directly.
  2. Otherwise                                → use config['model']['name'] (legacy).

Returns prediction label, confidence %, is_certain flag, and top-5 list.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torchvision import models

from src.alphabet.preprocessing import HandCropPreprocessor, build_eval_transform
from src.utils.labels import load_label_map

logger = logging.getLogger(__name__)

# ── Supported backbone output feature dimensions ──────────────
BACKBONE_OUT = {
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
    "efficientnet_b0":    1280,
}


# ─────────────────────────────────────────────────────────────
#  Model builders
# ─────────────────────────────────────────────────────────────

def _build_mobilenet_model(backbone_name: str, num_classes: int,
                           dropout: float = 0.3) -> nn.Module:
    """Builds the original project MobileNetV3 / EfficientNet model (no pretrain)."""
    from src.alphabet.model import AlphabetCNNModel
    return AlphabetCNNModel(
        num_classes=num_classes,
        backbone_name=backbone_name,
        pretrained=False,
        dropout=dropout,
    )


def _build_resnet_model(model_name: str, num_classes: int) -> nn.Module:
    """
    Builds a standard torchvision ResNet / other model with the correct
    output head size. Supports resnet18, resnet34, resnet50, resnet101, resnet152.
    """
    builder = getattr(models, model_name, None)
    if builder is None:
        raise ValueError(
            f"Unknown model_name '{model_name}' in checkpoint. "
            f"Add support in inference.py if needed."
        )
    model = builder(weights=None)  # no pretrained — we load from checkpoint

    # Replace the final fully-connected layer to match num_classes
    if hasattr(model, "fc"):
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(model, "classifier"):
        # EfficientNet / MobileNet style
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
            model.classifier[-1] = nn.Linear(in_features, num_classes)
        else:
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise RuntimeError(f"Don't know how to replace head for model '{model_name}'.")

    return model


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    label_map: Dict[str, int],
) -> nn.Module:
    """
    Universal checkpoint loader. Detects whether the checkpoint was created
    by the original project (MobileNetV3) or an external Kaggle model (ResNet18, etc.).

    The loaded model is returned in eval() mode on the target device.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    num_classes = len(label_map)

    # ── Detect model type ──────────────────────────────────
    model_name = ckpt.get("model_name", None)          # e.g. "resnet18" from Kaggle
    config      = ckpt.get("config", {})               # original project format
    legacy_name = config.get("model", {}).get("name")  # e.g. "mobilenet_v3_small"

    if model_name is not None:
        # ── External / Kaggle checkpoint (ResNet18, etc.) ──
        logger.info(f"Loading external checkpoint: model_name='{model_name}', num_classes={num_classes}")
        model = _build_resnet_model(model_name, num_classes)
    elif legacy_name is not None and legacy_name in BACKBONE_OUT:
        # ── Original project checkpoint (MobileNetV3 / EfficientNet) ──
        dropout = config.get("model", {}).get("dropout", 0.3)
        logger.info(f"Loading project checkpoint: backbone='{legacy_name}', num_classes={num_classes}")
        model = _build_mobilenet_model(legacy_name, num_classes, dropout)
    else:
        # ── Fallback: try ResNet18 ──────────────────────────
        logger.warning(
            "Cannot determine model architecture from checkpoint. "
            "Attempting fallback: ResNet18."
        )
        model = _build_resnet_model("resnet18", num_classes)

    # Load weights
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {ckpt.get('epoch', '?')}, "
                f"best_val_acc={ckpt.get('best_val_accuracy', '?')})")
    return model


# ─────────────────────────────────────────────────────────────
#  Inference Engine
# ─────────────────────────────────────────────────────────────

# Labels that should NOT trigger TTS or be treated as actionable signs
SILENT_LABELS = {"del", "nothing", "space"}


class AlphabetInferenceEngine:
    """
    Loads a trained ISL Alphabet model checkpoint and runs single-character inference.

    Supports both original-project checkpoints (MobileNetV3) and
    external Kaggle checkpoints (ResNet18, etc.) — auto-detected.

    Args:
        best_model_dir:       Directory containing best.pt and label_map.json
        label_map_path:       Optional override path for label_map.json
        confidence_threshold: Minimum confidence (0-1) to accept a prediction
        device_str:           'auto', 'cuda', or 'cpu'
        hand_crop:            Whether to use MediaPipe hand crop before inference
        image_size:           Input image size (default 224)
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

        # Load label map — prefer best_model_dir/label_map.json
        lm_path = label_map_path or (Path(best_model_dir) / "label_map.json")
        if not lm_path.exists():
            lm_path = Path("models/alphabet/label_map.json")
        if not lm_path.exists():
            raise FileNotFoundError(f"label_map.json not found at {lm_path}")
        self.label_map: Dict[str, int] = load_label_map(lm_path)
        self.id_to_label: Dict[int, str] = {v: k for k, v in self.label_map.items()}
        logger.info(f"Loaded label map: {len(self.label_map)} classes from {lm_path}")

        # Load model (auto-detect architecture)
        best_pt = Path(best_model_dir) / "best.pt"
        if not best_pt.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {best_pt}")
        self.model = load_model_from_checkpoint(best_pt, self.device, self.label_map)

        # Preprocessing transform
        self.transform = build_eval_transform(image_size=image_size)

        # Optional hand crop
        self.hand_preprocessor: Optional[HandCropPreprocessor] = None
        if hand_crop:
            self.hand_preprocessor = HandCropPreprocessor()

    def predict(self, image: Image.Image) -> Dict:
        """
        Runs inference on a PIL image.

        Returns dict:
            prediction  : predicted class label (e.g. 'A', 'del', 'space')
            confidence  : confidence % (0-100)
            is_certain  : True if confidence >= threshold AND not a silent label
            top5        : list of (label, confidence%) for top 5 predictions
            all_probs   : full probability array (numpy)
        """
        if self.hand_preprocessor is not None:
            image = self.hand_preprocessor.crop(image)

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        top5_indices    = np.argsort(probs)[-5:][::-1]
        top_class_id    = int(top5_indices[0])
        top_confidence  = float(probs[top_class_id]) * 100.0
        predicted_label = self.id_to_label.get(top_class_id, "?")

        # is_certain: confident enough AND not a non-sign label
        is_certain = (
            top_confidence / 100.0 >= self.confidence_threshold
            and predicted_label not in SILENT_LABELS
        )

        top5 = [
            (self.id_to_label.get(int(idx), "?"), float(probs[idx]) * 100.0)
            for idx in top5_indices
        ]

        return {
            "prediction":  predicted_label,
            "confidence":  round(top_confidence, 2),
            "is_certain":  is_certain,
            "top5":        top5,
            "all_probs":   probs,
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
        rgb   = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        return self.predict(image)

    def close(self) -> None:
        if self.hand_preprocessor is not None:
            self.hand_preprocessor.close()
