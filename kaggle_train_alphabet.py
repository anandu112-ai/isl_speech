"""
================================================================
  ISL ALPHABET + DIGIT — KAGGLE TRAINING NOTEBOOK
================================================================
Dataset: ASL Alphabet (Kaggle) — https://www.kaggle.com/datasets/grassknoted/asl-alphabet
         (Works with any A-Z folder-structured image dataset)

Instructions:
  1. Upload this file to a new Kaggle Notebook (Code > New Notebook > select .py)
  2. Add the dataset:  Datasets > Add Dataset > search "asl alphabet" by grassknoted
  3. Enable GPU:       Settings > Accelerator > GPU T4 x2
  4. Run All
  5. Download:         /kaggle/working/best.pt  (from Output panel)
  6. Place locally at: models/alphabet/best_model/best.pt

================================================================
"""

# ── 0. Install missing packages ──────────────────────────────
import subprocess, sys

def pip(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

pip("torchvision")
pip("Pillow")

# ── 1. Imports ───────────────────────────────────────────────
import os, json, time, random, logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import models, transforms

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 2. Config ────────────────────────────────────────────────
CONFIG = {
    # ── Paths ──────────────────────────────────────────────
    # Kaggle mounts datasets under /kaggle/input/<dataset-slug>/
    # Check the exact path in the Kaggle data panel (left sidebar).
    # Common options:
    #   /kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train   <- grassknoted dataset
    #   /kaggle/input/asl-alphabet/Train                                   <- some variants
    "DATA_DIR":         "/kaggle/input/asl-alphabet/asl_alphabet_train/asl_alphabet_train",

    # Output directory — Kaggle persists /kaggle/working/
    "OUTPUT_DIR":       "/kaggle/working",

    # ── Model ──────────────────────────────────────────────
    "BACKBONE":         "mobilenet_v3_small",   # or "efficientnet_b0"
    "NUM_CLASSES":      36,
    "PRETRAINED":       True,
    "DROPOUT":          0.3,

    # ── Training ───────────────────────────────────────────
    "EPOCHS":           30,
    "BATCH_SIZE":       64,
    "LR":               1e-4,
    "WEIGHT_DECAY":     1e-4,
    "PATIENCE":         7,
    "SEED":             42,
    "NUM_WORKERS":      2,

    # Phase 2 fine-tuning
    "PHASE2":           True,
    "UNFREEZE_LAYERS":  2,
    "PHASE2_EPOCHS":    10,

    # ── Preprocessing ──────────────────────────────────────
    "IMAGE_SIZE":       224,
    "VAL_SPLIT":        0.15,
    "TEST_SPLIT":       0.05,

    # Set to None to auto-detect all folders.
    # Set to list to filter: e.g. ["A","B",...,"Z"] for letters only
    "TARGET_CLASSES":   None,
}

# ── 3. Seed ──────────────────────────────────────────────────
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG["SEED"])

# ── 4. Device ────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n{'='*60}")
print(f"  Device : {device}")
if torch.cuda.is_available():
    print(f"  GPU    : {torch.cuda.get_device_name(0)}")
    print(f"  VRAM   : {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
print(f"{'='*60}\n")

# ── 5. Build Label Map ───────────────────────────────────────
def build_label_map(data_dir: str, target_classes=None) -> Dict[str, int]:
    data_path = Path(data_dir)
    folders = sorted([d.name for d in data_path.iterdir() if d.is_dir()])
    if target_classes:
        folders = [f for f in folders if f in target_classes]
    label_map = {name: idx for idx, name in enumerate(folders)}
    logger.info(f"Label map built: {len(label_map)} classes -> {list(label_map.keys())}")
    return label_map

label_map = build_label_map(CONFIG["DATA_DIR"], CONFIG["TARGET_CLASSES"])
NUM_CLASSES = len(label_map)
CONFIG["NUM_CLASSES"] = NUM_CLASSES
print(f"Classes detected: {NUM_CLASSES}")
print(f"Labels: {list(label_map.keys())}\n")

# Save label map
label_map_path = Path(CONFIG["OUTPUT_DIR"]) / "label_map.json"
with open(label_map_path, "w") as f:
    json.dump(label_map, f, indent=2, sort_keys=True)
print(f"Label map saved -> {label_map_path}")

# ── 6. Dataset ───────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

class AlphabetDataset(Dataset):
    def __init__(self, data_dir: str, label_map: Dict[str, int], transform=None):
        self.data_dir  = Path(data_dir)
        self.label_map = label_map
        self.transform = transform
        self.samples: List[Tuple[Path, int]] = []
        self._load()

    def _load(self):
        for label, class_id in self.label_map.items():
            label_dir = self.data_dir / label
            if not label_dir.exists():
                logger.warning(f"Label folder not found: {label_dir}")
                continue
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() in SUPPORTED_EXT:
                    self.samples.append((img_path, class_id))
        logger.info(f"Loaded {len(self.samples)} samples from {self.data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, class_id = self.samples[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except (UnidentifiedImageError, OSError):
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, class_id


def build_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(image_size * 1.143)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


train_tf, eval_tf = build_transforms(CONFIG["IMAGE_SIZE"])

full_ds = AlphabetDataset(CONFIG["DATA_DIR"], label_map, transform=None)
n_total = len(full_ds)
n_val   = int(n_total * CONFIG["VAL_SPLIT"])
n_test  = int(n_total * CONFIG["TEST_SPLIT"])
n_train = n_total - n_val - n_test

train_ds_raw, val_ds_raw, test_ds_raw = random_split(
    full_ds, [n_train, n_val, n_test],
    generator=torch.Generator().manual_seed(CONFIG["SEED"])
)

class TransformWrapper(Dataset):
    def __init__(self, subset, transform):
        self.subset    = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img_path, class_id = self.subset.dataset.samples[self.subset.indices[idx]]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8))
        if self.transform:
            image = self.transform(image)
        return image, class_id


train_ds = TransformWrapper(train_ds_raw, train_tf)
val_ds   = TransformWrapper(val_ds_raw,   eval_tf)
test_ds  = TransformWrapper(test_ds_raw,  eval_tf)

train_loader = DataLoader(train_ds, batch_size=CONFIG["BATCH_SIZE"], shuffle=True,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=CONFIG["BATCH_SIZE"], shuffle=False,
                          num_workers=CONFIG["NUM_WORKERS"], pin_memory=True)

print(f"\nDataset split:")
print(f"  Train : {len(train_ds):,}")
print(f"  Val   : {len(val_ds):,}")
print(f"  Test  : {len(test_ds):,}")
print(f"  Total : {n_total:,}\n")

# ── 7. Model ─────────────────────────────────────────────────
BACKBONE_OUT = {
    "mobilenet_v3_small": 576,
    "mobilenet_v3_large": 960,
    "efficientnet_b0":    1280,
}

class AlphabetCNNModel(nn.Module):
    def __init__(self, num_classes=36, backbone_name="mobilenet_v3_small",
                 pretrained=True, dropout=0.3):
        super().__init__()
        self.backbone_name = backbone_name
        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, backbone_name)(weights=weights)

        if backbone_name in ("mobilenet_v3_small", "mobilenet_v3_large", "efficientnet_b0"):
            self.backbone = nn.Sequential(backbone.features, backbone.avgpool)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")

        backbone_out = BACKBONE_OUT[backbone_name]
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self, last_n=2):
        for p in self.backbone.parameters():
            p.requires_grad = False
        for child in list(self.backbone.children())[-last_n:]:
            for p in child.parameters():
                p.requires_grad = True


model = AlphabetCNNModel(
    num_classes=NUM_CLASSES,
    backbone_name=CONFIG["BACKBONE"],
    pretrained=CONFIG["PRETRAINED"],
    dropout=CONFIG["DROPOUT"],
)
model.to(device)
model.freeze_backbone()

total_p     = sum(p.numel() for p in model.parameters())
trainable_p = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Model: {CONFIG['BACKBONE']}")
print(f"  Total params    : {total_p:,}")
print(f"  Trainable params: {trainable_p:,}  (head only, backbone frozen)\n")

# ── 8. Training Utilities ────────────────────────────────────
class EarlyStopping:
    def __init__(self, patience=7):
        self.patience = patience
        self.counter  = 0
        self.best     = None
        self.stop     = False

    def step(self, val_acc):
        if self.best is None or val_acc > self.best:
            self.best    = val_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            logits = model(images)
            loss   = criterion(logits, labels)
            if training:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += (logits.argmax(1) == labels).sum().item()
            total      += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1) * 100.0


def train_phase(model, train_loader, val_loader, epochs, lr, weight_decay,
                patience, phase_name="Phase 1"):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 0.01
    )
    es = EarlyStopping(patience=patience)
    best_val_acc = 0.0
    output_dir   = Path(CONFIG["OUTPUT_DIR"])

    print(f"\n{'='*60}")
    print(f"  {phase_name}")
    print(f"{'='*60}")

    for epoch in range(epochs):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader, criterion)
        scheduler.step()
        elapsed = time.time() - t0

        marker = " <- best" if vl_acc > best_val_acc else ""
        print(f"  [{epoch+1:3d}/{epochs}] "
              f"train={tr_acc:.2f}% val={vl_acc:.2f}% "
              f"lr={optimizer.param_groups[0]['lr']:.2e} "
              f"({elapsed:.1f}s){marker}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            state = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "best_val_accuracy": best_val_acc,
                "label_map": label_map,
                "config": {
                    "model": {
                        "name":       CONFIG["BACKBONE"],
                        "pretrained": CONFIG["PRETRAINED"],
                        "dropout":    CONFIG["DROPOUT"],
                    }
                },
            }
            torch.save(state, output_dir / "best.pt")

        if es.step(vl_acc):
            print(f"\n  Early stopping at epoch {epoch+1}.")
            break

    print(f"\n  Best val accuracy ({phase_name}): {best_val_acc:.2f}%")
    return best_val_acc


# ── 9. Phase 1: Train Head ───────────────────────────────────
best_p1 = train_phase(
    model, train_loader, val_loader,
    epochs       = CONFIG["EPOCHS"],
    lr           = CONFIG["LR"],
    weight_decay = CONFIG["WEIGHT_DECAY"],
    patience     = CONFIG["PATIENCE"],
    phase_name   = "Phase 1 — Head Training (Backbone Frozen)",
)

# ── 10. Phase 2: Fine-tune Backbone (optional) ───────────────
best_p2 = 0.0
if CONFIG["PHASE2"]:
    model.unfreeze_backbone(last_n=CONFIG["UNFREEZE_LAYERS"])
    trainable_p2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nPhase 2: Unfroze last {CONFIG['UNFREEZE_LAYERS']} backbone layers.")
    print(f"  Trainable params: {trainable_p2:,}")
    best_p2 = train_phase(
        model, train_loader, val_loader,
        epochs       = CONFIG["PHASE2_EPOCHS"],
        lr           = CONFIG["LR"] * 0.1,
        weight_decay = CONFIG["WEIGHT_DECAY"],
        patience     = CONFIG["PATIENCE"],
        phase_name   = "Phase 2 — Backbone Fine-tuning",
    )

best_val_acc = max(best_p1, best_p2)

# ── 11. Test Evaluation ──────────────────────────────────────
print(f"\n{'='*60}")
print("  TEST SET EVALUATION")
print(f"{'='*60}")

ckpt = torch.load(Path(CONFIG["OUTPUT_DIR"]) / "best.pt", map_location=device)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
test_loss, test_acc = run_epoch(model, test_loader, nn.CrossEntropyLoss())
print(f"  Test Accuracy : {test_acc:.2f}%")
print(f"  Test Loss     : {test_loss:.4f}")

# ── 12. Save Summary ─────────────────────────────────────────
summary = {
    "best_val_accuracy": best_val_acc,
    "test_accuracy":     test_acc,
    "num_classes":       NUM_CLASSES,
    "backbone":          CONFIG["BACKBONE"],
    "label_map":         label_map,
}
with open(Path(CONFIG["OUTPUT_DIR"]) / "training_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print(f"\n{'='*60}")
print("  TRAINING COMPLETE")
print(f"{'='*60}")
print(f"  Best Val Accuracy : {best_val_acc:.2f}%")
print(f"  Test  Accuracy    : {test_acc:.2f}%")
print(f"\n  Files saved to /kaggle/working/:")
print(f"    best.pt              <- trained model weights")
print(f"    label_map.json       <- class label mapping")
print(f"    training_summary.json")
print(f"\n  Download both files from the Kaggle Output panel!")
print(f"{'='*60}")
