"""
Trainer Module for INCLUDE-50 Sign Language Recognition Model.
Handles training loop, validation, metrics tracking, model checkpointing,
early stopping, and plotting training/validation curves.
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.bilstm import SignBiLSTMModel


class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        label_map: Dict[str, int],
        config: Dict[str, Any],
        device: torch.device,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.label_map = label_map
        self.config = config
        self.device = device

        train_cfg = config.get("training", {})
        self.epochs = train_cfg.get("epochs", 50)
        self.lr = train_cfg.get("learning_rate", 0.001)
        self.weight_decay = train_cfg.get("weight_decay", 0.0001)
        self.patience = train_cfg.get("early_stopping_patience", 10)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=4
        )

        # Output paths
        self.reports_dir = Path(config.get("reports", {}).get("dir", "reports"))
        self.checkpoint_dir = Path(config.get("model", {}).get("checkpoint_dir", "models/checkpoints"))
        self.best_model_dir = Path(config.get("model", {}).get("best_model_dir", "models/best_model"))

        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_model_dir.mkdir(parents=True, exist_ok=True)

        self.history_csv = self.reports_dir / "training_history.csv"
        self.curves_png = self.reports_dir / "training_curves.png"

    def train_epoch(self) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.train_loader:
            features = features.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = total_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        return epoch_loss, epoch_acc

    def validate(self) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for features, labels in self.val_loader:
                features = features.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item() * features.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_loss = total_loss / max(total, 1)
        val_acc = correct / max(total, 1)
        return val_loss, val_acc

    def save_checkpoint(self, path: Path, epoch: int, best_val_acc: float):
        state = {
            "epoch": epoch,
            "best_val_accuracy": best_val_acc,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "label_map": self.label_map,
            "config": self.config,
        }
        torch.save(state, path)

    def plot_history(self, history: pd.DataFrame):
        plt.figure(figsize=(12, 5))

        # Loss Plot
        plt.subplot(1, 2, 1)
        plt.plot(history["epoch"], history["train_loss"], label="Train Loss", color="blue")
        plt.plot(history["epoch"], history["val_loss"], label="Val Loss", color="red")
        plt.title("Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)

        # Accuracy Plot
        plt.subplot(1, 2, 2)
        plt.plot(history["epoch"], history["train_accuracy"] * 100, label="Train Acc (%)", color="blue")
        plt.plot(history["epoch"], history["val_accuracy"] * 100, label="Val Acc (%)", color="red")
        plt.title("Accuracy Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (%)")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(self.curves_png, dpi=150)
        plt.close()

    def run(self, resume: bool = False) -> Dict[str, Any]:
        start_epoch = 1
        best_val_acc = 0.0
        history_rows = []

        latest_path = self.checkpoint_dir / "latest.pt"
        best_path = self.best_model_dir / "best.pt"

        if resume and latest_path.exists():
            checkpoint = torch.load(latest_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_val_acc = checkpoint["best_val_accuracy"]
            print(f"Resuming training from epoch {start_epoch} (Best Val Acc: {best_val_acc * 100:.2f}%)")

        patience_counter = 0

        for epoch in range(start_epoch, self.epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            self.scheduler.step(val_acc)

            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }
            history_rows.append(row)

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%"
            )

            # Save latest checkpoint
            self.save_checkpoint(latest_path, epoch, best_val_acc)

            # Check if best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(best_path, epoch, best_val_acc)

                # Save label map and config beside best model
                with open(self.best_model_dir / "label_map.json", "w", encoding="utf-8") as f:
                    json.dump(self.label_map, f, indent=4)
                with open(self.best_model_dir / "config.yaml", "w", encoding="utf-8") as f:
                    import yaml
                    yaml.dump(self.config, f)
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                print(f"\nEarly stopping triggered at epoch {epoch} (No improvement for {self.patience} epochs).")
                break

        # Save history CSV and plot
        history_df = pd.DataFrame(history_rows)
        history_df.to_csv(self.history_csv, index=False)
        self.plot_history(history_df)

        return {
            "best_val_accuracy": best_val_acc,
            "final_epoch": epoch,
            "best_model_path": best_path.as_posix(),
        }
