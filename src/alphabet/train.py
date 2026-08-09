"""
Training loop and checkpointing for ISL Alphabet + Digit CNN classifier.

Features:
    - Phase 1: Frozen backbone, trains head only
    - Phase 2: Partial backbone fine-tuning (configurable)
    - AdamW optimizer with cosine LR scheduler
    - Early stopping on validation accuracy
    - Automatic checkpointing (latest.pt, best.pt, epoch_N.pt)
    - Per-epoch metrics logging
    - Class-weighted CrossEntropyLoss (optional)
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Tracks validation accuracy and signals when training should stop."""

    def __init__(self, patience: int = 7, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.should_stop = False

    def step(self, val_acc: float) -> bool:
        """Returns True if training should stop."""
        if self.best_score is None or val_acc > self.best_score + self.min_delta:
            self.best_score = val_acc
            self.counter = 0
        else:
            self.counter += 1
            logger.info(f"EarlyStopping: No improvement for {self.counter}/{self.patience} epochs.")
            if self.counter >= self.patience:
                logger.info("EarlyStopping: Stopping training.")
                self.should_stop = True
        return self.should_stop


class AlphabetTrainer:
    """
    Manages the full training lifecycle for AlphabetCNNModel.

    Args:
        model:           AlphabetCNNModel instance
        train_loader:    DataLoader for training split
        val_loader:      DataLoader for validation split
        config:          Full config dict from alphabet_config.yaml
        label_map:       Dict[str, int] — {label: class_id}
        device:          torch.device
        checkpoint_dir:  Where to save checkpoints
        best_model_dir:  Where to save best.pt
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: dict,
        label_map: Dict[str, int],
        device: torch.device,
        checkpoint_dir: Path,
        best_model_dir: Path,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.label_map = label_map
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_model_dir = Path(best_model_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 30)
        self.lr = train_cfg.get("learning_rate", 0.0001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.patience = train_cfg.get("early_stopping_patience", 7)

        # Loss function
        if train_cfg.get("use_class_weights", False) and class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
            logger.info("Using class-weighted CrossEntropyLoss.")
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Optimizer — only trainable parameters
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )

        self.early_stopping = EarlyStopping(patience=self.patience)
        self.best_val_accuracy = 0.0
        self.history: List[Dict] = []

    def _run_epoch(self, loader: DataLoader, train: bool) -> Tuple[float, float]:
        """Runs a single train or evaluation epoch. Returns (avg_loss, accuracy%)."""
        self.model.train(train)
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.set_grad_enabled(train):
            for batch_idx, (images, labels) in enumerate(loader):
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)

                logits = self.model(images)
                loss = self.criterion(logits, labels)

                if train:
                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                total_loss += loss.item() * images.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += images.size(0)

                if train and (batch_idx + 1) % 20 == 0:
                    running_acc = correct / total * 100
                    logger.debug(
                        f"  Batch {batch_idx+1}/{len(loader)} | "
                        f"loss={loss.item():.4f} | acc={running_acc:.1f}%"
                    )

        avg_loss = total_loss / max(total, 1)
        accuracy = correct / max(total, 1) * 100.0
        return avg_loss, accuracy

    def _save_checkpoint(self, epoch: int, filename: str = "latest.pt") -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "label_map": self.label_map,
            "config": self.config,
        }
        torch.save(state, self.checkpoint_dir / filename)

    def _save_best(self, epoch: int) -> None:
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_accuracy": self.best_val_accuracy,
            "label_map": self.label_map,
            "config": self.config,
        }
        torch.save(state, self.best_model_dir / "best.pt")

    def resume(self, checkpoint_path: Optional[Path] = None) -> int:
        """Loads a checkpoint to resume training. Returns the starting epoch."""
        ckpt_path = checkpoint_path or (self.checkpoint_dir / "latest.pt")
        if not Path(ckpt_path).exists():
            logger.info("No checkpoint found — starting from scratch.")
            return 0

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.best_val_accuracy = ckpt.get("best_val_accuracy", 0.0)
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {ckpt_path} at epoch {start_epoch}.")
        return start_epoch

    def train(self, start_epoch: int = 0) -> List[Dict]:
        """
        Runs training for Phase 1 (frozen backbone).
        Returns history of per-epoch metrics.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Training classification head (backbone frozen)")
        logger.info("=" * 60)

        for epoch in range(start_epoch, self.epochs):
            t0 = time.time()
            train_loss, train_acc = self._run_epoch(self.train_loader, train=True)
            val_loss, val_acc = self._run_epoch(self.val_loader, train=False)
            elapsed = time.time() - t0

            self.scheduler.step()
            lr = self.optimizer.param_groups[0]["lr"]

            row = {
                "epoch": epoch + 1,
                "train_loss": round(train_loss, 4),
                "train_acc": round(train_acc, 2),
                "val_loss": round(val_loss, 4),
                "val_acc": round(val_acc, 2),
                "lr": lr,
            }
            self.history.append(row)

            print(
                f"Epoch [{epoch+1:3d}/{self.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | "
                f"lr={lr:.2e} | {elapsed:.1f}s"
            )

            # Save latest & periodic checkpoints
            self._save_checkpoint(epoch, "latest.pt")
            if (epoch + 1) % 5 == 0:
                self._save_checkpoint(epoch, f"epoch_{epoch+1}.pt")

            # Save best model
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                self._save_best(epoch)
                logger.info(f"  ✓ New best val_acc={val_acc:.2f}% — saved best.pt")

            # Early stopping
            if self.early_stopping.step(val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch+1}.")
                break

        return self.history

    def fine_tune(self, unfreeze_layers: int = 2, start_epoch: int = 0) -> List[Dict]:
        """
        Phase 2: Partially unfreezes backbone and fine-tunes with lower LR.
        """
        logger.info("=" * 60)
        logger.info(f"PHASE 2: Fine-tuning (unfreezing last {unfreeze_layers} backbone layers)")
        logger.info("=" * 60)

        self.model.unfreeze_backbone(last_n_layers=unfreeze_layers)

        # Rebuild optimizer with lower learning rate for Phase 2
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.lr * 0.1,
            weight_decay=self.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.001,
        )
        self.early_stopping = EarlyStopping(patience=self.patience)

        return self.train(start_epoch=start_epoch)
