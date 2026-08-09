"""
CNN Transfer-Learning Model for ISL Alphabet (A-Z) + Digit (0-9) Recognition.

Architecture:
    Pretrained CNN Backbone (MobileNetV3 / EfficientNet-B0)
    → Global Average Pooling (built-in)
    → Linear(backbone_out, 128) → ReLU → Dropout(p)
    → Linear(128, 36)
    → Softmax at inference

Supports two training phases:
    PHASE 1: Frozen backbone — trains classification head only.
    PHASE 2: Partial backbone fine-tuning — unfreezes last N backbone layers.
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torchvision import models

logger = logging.getLogger(__name__)

# Registry of supported backbone names → torchvision factory
BACKBONE_REGISTRY: Dict[str, str] = {
    "mobilenet_v3_small": "mobilenet_v3_small",
    "mobilenet_v3_large": "mobilenet_v3_large",
    "efficientnet_b0": "efficientnet_b0",
}


def _get_backbone_out_features(backbone_name: str) -> int:
    """Returns the feature dimension of the last backbone pooling layer."""
    return {
        "mobilenet_v3_small": 576,
        "mobilenet_v3_large": 960,
        "efficientnet_b0": 1280,
    }.get(backbone_name, 1280)


class AlphabetCNNModel(nn.Module):
    """
    Transfer-learning CNN for 36-class ISL character recognition.

    Args:
        num_classes:    Number of target classes (default 36).
        backbone_name:  Name of pretrained backbone (see BACKBONE_REGISTRY).
        pretrained:     Whether to use ImageNet pretrained weights.
        dropout:        Dropout probability in classification head.
    """

    def __init__(
        self,
        num_classes: int = 36,
        backbone_name: str = "mobilenet_v3_small",
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.num_classes = num_classes

        if backbone_name not in BACKBONE_REGISTRY:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Choose from: {list(BACKBONE_REGISTRY.keys())}"
            )

        # Load pretrained backbone
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)

        # Strip classification head — keep only feature extraction part
        if backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large"):
            # MobileNetV3: features → avgpool → classifier[0]
            self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
            backbone_out = _get_backbone_out_features(backbone_name)
        elif backbone_name == "efficientnet_b0":
            # EfficientNet: features → avgpool
            self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
            backbone_out = _get_backbone_out_features(backbone_name)
        else:
            raise ValueError(f"Backbone '{backbone_name}' not properly configured.")

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        logger.info(
            f"AlphabetCNNModel: backbone={backbone_name} | "
            f"backbone_out={backbone_out} | num_classes={num_classes} | pretrained={pretrained}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns raw logits of shape (N, num_classes)."""
        features = self.backbone(x)
        return self.classifier(features)

    def freeze_backbone(self) -> None:
        """Phase 1: Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info("Backbone FROZEN — training classification head only.")

    def unfreeze_backbone(self, last_n_layers: int = 2) -> None:
        """
        Phase 2: Partially unfreeze backbone for fine-tuning.
        Unfreezes the last `last_n_layers` children of backbone.
        """
        # First freeze everything
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Unfreeze the last N children
        children = list(self.backbone.children())
        for child in children[-last_n_layers:]:
            for param in child.parameters():
                param.requires_grad = True

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            f"Backbone PARTIALLY UNFROZEN (last {last_n_layers} layers). "
            f"Trainable params: {n_trainable:,}"
        )

    def unfreeze_all(self) -> None:
        """Unfreeze all backbone parameters for full fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
        logger.info("Backbone FULLY UNFROZEN.")


def build_model(config: dict, num_classes: int = 36) -> AlphabetCNNModel:
    """Builds AlphabetCNNModel from config dict."""
    model_cfg = config.get("model", {})
    return AlphabetCNNModel(
        num_classes=num_classes,
        backbone_name=model_cfg.get("name", "mobilenet_v3_small"),
        pretrained=model_cfg.get("pretrained", True),
        dropout=model_cfg.get("dropout", 0.3),
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes: int = 36,
) -> Tuple[AlphabetCNNModel, Optional[Dict]]:
    """
    Loads model weights from a checkpoint file.

    Returns:
        (model, checkpoint_dict)
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get("config", {})
    label_map = ckpt.get("label_map", {})

    model = build_model(config, num_classes=len(label_map) or num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {ckpt.get('epoch', '?')}, "
        f"best_val_acc={ckpt.get('best_val_accuracy', '?')})"
    )
    return model, ckpt
