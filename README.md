# ISL Speech — INCLUDE-50 Indian Sign Language Recognition + Speech System

A complete end-to-end pipeline for **Indian Sign Language (ISL) recognition and text-to-speech conversion** using the [INCLUDE-50 dataset](https://zenodo.org/records/4010759).

The system downloads selected ISL videos from Zenodo using **HTTP byte-range requests** (no full 42 GB download required), extracts **MediaPipe Holistic landmarks**, trains a **BiLSTM temporal classifier**, and speaks predictions aloud using **offline TTS**.

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Setup](#setup)
- [Complete Workflow](#complete-workflow)
  - [Phase 1 — Download Dataset](#phase-1--download-dataset)
  - [Phase 2 — Inspect & Verify](#phase-2--inspect--verify)
  - [Phase 3 — Preprocessing & Feature Extraction](#phase-3--preprocessing--feature-extraction)
  - [Phase 4 — Training](#phase-4--training)
  - [Phase 5 — Evaluation](#phase-5--evaluation)
  - [Phase 6 — Inference & Demo](#phase-6--inference--demo)
- [Module Reference](#module-reference)
  - [tools/ — Remote ZIP Download Engine](#tools--remote-zip-download-engine)
  - [src/ — Model & Pipeline Modules](#src--model--pipeline-modules)
  - [scripts/ — CLI Entrypoints](#scripts--cli-entrypoints)
  - [configs/ — Configuration](#configs--configuration)
- [Key Design Decisions](#key-design-decisions)
- [Hardware Requirements](#hardware-requirements)
- [Dependencies](#dependencies)
- [Known Issues & Limitations](#known-issues--limitations)
- [Acceptance Criteria Checklist](#acceptance-criteria-checklist)

---

## Architecture Overview

```
INCLUDE-50 Zenodo Dataset (42.38 GB total ZIP archives)
        │
        │  HTTP Range Requests — download ONLY selected video bytes
        │  (8.10 GB compressed / 8.24 GB extracted)
        ▼
data/videos/train/{label}/*.MOV
data/videos/val/{label}/*.MOV
        │
        │  32-frame uniform temporal sampling
        │  MediaPipe Holistic → 225-dim float32 vector/frame
        ▼
data/features/train/{label}/*.npz   → shape (32, 225)
data/features/val/{label}/*.npz     → shape (32, 225)
        │
        │  BiLSTM Temporal Classifier
        │  Input (N, 32, 225) → Linear(225,128) → LayerNorm
        │  → 2-layer BiLSTM(128, bidirectional) → Mean+Max Pool
        │  → Linear(512,128) → Dropout(0.3) → Linear(128, 50)
        ▼
models/best_model/best.pt   → 50-class softmax
        │
        │  Label normalization ("48. Hello" → "Hello")
        │  Confidence threshold (default ≥ 0.70)
        ▼
Text Output + pyttsx3 Speech Output
```

---

## Project Structure

```
isl_speech/
│
├── configs/
│   └── config.yaml                   ← All hyperparameters & paths
│
├── data/
│   ├── metadata/
│   │   ├── include50.csv             ← Original INCLUDE-50 metadata (881 rows)
│   │   ├── include50_archive_map.csv ← video_path → archive → archive_url
│   │   ├── selected_videos.csv       ← AUTHORITATIVE SELECTION (650 videos)
│   │   ├── download_manifest.csv     ← Live download state (auto-generated)
│   │   ├── features_manifest.csv     ← Preprocessing state (auto-generated)
│   │   ├── label_map.json            ← {label: class_id} deterministic mapping
│   │   ├── video_quality.csv         ← Per-video quality metrics
│   │   └── download_sizes.csv        ← Archive size estimates
│   │
│   ├── videos/
│   │   ├── train/{label}/*.MOV       ← Downloaded training videos
│   │   └── val/{label}/*.MOV         ← Downloaded validation videos
│   │
│   └── features/
│       ├── train/{label}/*.npz       ← (32, 225) float32 landmark features
│       └── val/{label}/*.npz
│
├── models/
│   ├── checkpoints/
│   │   └── latest.pt                 ← Auto-saved after every epoch
│   └── best_model/
│       ├── best.pt                   ← Best validation accuracy checkpoint
│       ├── label_map.json            ← Copied alongside model for inference
│       └── config.yaml               ← Copied alongside model for inference
│
├── reports/
│   ├── training_history.csv          ← Epoch-level loss/accuracy log
│   ├── training_curves.png           ← Loss & accuracy plots
│   ├── confusion_matrix.png          ← 50×50 confusion matrix
│   ├── classification_report.csv     ← Per-class precision/recall/F1
│   └── FINAL_REPORT.md               ← Human-readable project report
│
├── logs/
│   └── download.log                  ← HTTP download event log
│
├── tools/                            ← Remote ZIP download engine
│   ├── remote_zip.py                 ← HTTP Range requests, ZIP parsing, ZIP64
│   ├── downloader.py                 ← Streaming decompress + CRC32 validate
│   ├── manifest.py                   ← Download state persistence
│   └── validation.py                 ← CRC32, path safety, dataset integrity
│
├── src/                              ← Model & pipeline source modules
│   ├── features/
│   │   └── extractor.py              ← MediaPipe Holistic 225-dim extractor
│   ├── preprocessing/
│   │   └── processor.py              ← 32-frame sampling + .npz writer
│   ├── dataset/
│   │   └── sign_dataset.py           ← PyTorch Dataset + DataLoader
│   ├── models/
│   │   └── bilstm.py                 ← BiLSTM classifier architecture
│   ├── training/
│   │   └── trainer.py                ← Training loop, checkpoint, early stop
│   ├── speech/
│   │   └── tts.py                    ← pyttsx3 offline TTS engine
│   └── utils/
│       ├── labels.py                 ← normalize_label(), label_to_text()
│       └── config.py                 ← YAML config loader
│
├── scripts/                          ← CLI entry points
│   ├── verify_dataset.py             ← Dataset integrity checker
│   ├── inspect_dataset.py            ← Video quality analysis
│   ├── estimate_download_size.py     ← Remote ZIP size estimator (no download)
│   ├── download_selected_videos.py   ← Main video downloader
│   ├── preprocess_dataset.py         ← Frame sampling + feature extraction
│   ├── extract_features.py           ← Alias for preprocess_dataset.py
│   ├── train.py                      ← Model training
│   ├── evaluate.py                   ← Model evaluation
│   ├── predict.py                    ← Single video inference
│   ├── speak.py                      ← TTS utility
│   ├── demo.py                       ← Offline sign-to-speech demo
│   ├── webcam_demo.py                ← Real-time webcam demo
│   └── generate_report.py            ← FINAL_REPORT.md generator
│
├── select_videos.py                  ← Video selection script (DO NOT re-run)
├── estimate_video_download.py        ← Legacy size estimator (root-level)
├── test_remote_zip.py                ← Legacy ZIP range-request tester
├── check_archive_sizes.py            ← Legacy archive size checker
└── requirements.txt
```

---

## Dataset

**INCLUDE-50** is an Indian Sign Language video dataset.

| Property | Value |
|---|---|
| Zenodo Record | [4010759](https://zenodo.org/records/4010759) |
| Zenodo API | `https://zenodo.org/api/records/4010759` |
| Total archive size | 42.38 GB (35 ZIP files) |
| **Selected download size** | **8.10 GB compressed / 8.24 GB extracted** |
| Total videos selected | 650 |
| Classes | 50 ISL signs |
| Train videos | 500 (10 per class) |
| Validation videos | 150 (3 per class) |

### Authoritative Selection

`data/metadata/selected_videos.csv` is the **authoritative, fixed selection**.

> ⚠️ **DO NOT re-run `select_videos.py`**. The selection is already generated with conflict resolution (14 metadata/path label conflicts corrected) and duplicate removal (881 original → 866 unique → 650 selected). Re-running will regenerate with a different random seed and break reproducibility.

Columns: `parent_label`, `label`, `video_path`, `include_50`, `archive`, `archive_url`, `split`

### Metadata Files

| File | Description |
|---|---|
| `include50.csv` | Raw INCLUDE-50 metadata |
| `include50_archive_map.csv` | Maps every `video_path` to its ZIP archive URL |
| `selected_videos.csv` | **Fixed 650-video selection with split assignments** |
| `download_manifest.csv` | Per-video download state (auto-managed, do not edit) |
| `features_manifest.csv` | Per-video preprocessing state (auto-managed) |
| `label_map.json` | Deterministic `{label: class_id}` sorted mapping (saved during preprocessing) |

---

## Setup

### Requirements

- Windows 10/11
- Python 3.13
- ~8.5 GB free disk space (for downloaded videos)
- ~500 MB additional for features

### Install Dependencies

```powershell
# Create virtual environment (already exists)
# .venv\Scripts\Activate.ps1

# Install all dependencies
.venv\Scripts\pip.exe install -r requirements.txt
```

Key dependencies: `torch==2.13.0`, `torchvision==0.28.0`, `mediapipe==1.0.0`, `opencv-python==5.0.0.93`, `pandas==3.0.5`, `requests==2.34.2`, `pyttsx3==2.99`, `pyyaml==6.0.3`, `scikit-learn==1.9.0`, `matplotlib==3.11.1`, `tqdm==4.70.0`

---

## Complete Workflow

### Phase 1 — Download Dataset

The downloader uses **HTTP Range requests** to fetch only the compressed bytes of selected videos from Zenodo ZIP archives — no full archive is ever downloaded locally.

#### Estimate download size (no video downloaded)
```powershell
.venv\Scripts\python.exe scripts/estimate_download_size.py
```
Expected output:
```
Found: 650/650 | Compressed: 8.10 GB | Extracted: 8.24 GB
```

#### Dry run (inspect archives, match videos, calculate sizes — no download)
```powershell
.venv\Scripts\python.exe scripts/download_selected_videos.py --dry-run
```

#### Test single video extraction
```powershell
.venv\Scripts\python.exe scripts/download_selected_videos.py --test-one
```
Expected output: `Remote extraction test: PASS`

#### Start full download
```powershell
.venv\Scripts\python.exe scripts/download_selected_videos.py --workers 2
```

- Resumable: re-run the same command to continue from where it stopped
- Ctrl+C safe: manifest is saved on interrupt
- Retry failed: `--retry-failed` flag

#### Check download progress
```powershell
.venv\Scripts\python.exe -c "import sys; sys.path.insert(0,''); from tools.manifest import ManifestManager; m=ManifestManager(); print(m.get_summary())"
```

---

### Phase 2 — Inspect & Verify

#### Verify dataset integrity
```powershell
.venv\Scripts\python.exe scripts/verify_dataset.py
```
Expected output:
```
Dataset verification
====================
Expected videos: 650
Found videos:    650
Missing:         0
Corrupt:         0
Duplicate:       0

Train:           500
Validation:      150

Classes:         50

RESULT: PASS
```

> ⚠️ **Do not proceed to preprocessing if this fails.**

#### Inspect video quality
```powershell
.venv\Scripts\python.exe scripts/inspect_dataset.py
```
Outputs `data/metadata/video_quality.csv` with per-video: FPS, frame count, duration, resolution, and validity status.

---

### Phase 3 — Preprocessing & Feature Extraction

Extracts **MediaPipe Holistic landmarks** from 32 uniformly sampled frames per video. Produces `.npz` files with shape `(32, 225)`.

**Feature layout (225 floats per frame):**
- Left hand: 21 landmarks × 3 (x, y, z) = 63
- Right hand: 21 landmarks × 3 (x, y, z) = 63
- Body pose: 33 landmarks × 3 (x, y, z) = 99

Missing hands → zero-filled (constant dimension guaranteed).

```powershell
.venv\Scripts\python.exe scripts/preprocess_dataset.py
```

- Resumable: skips existing `.npz` files
- Force re-extract: `--force`
- Custom frame count: `--num-frames 32`

After completion, verify:
```
data/metadata/features_manifest.csv   ← Check all 650 rows have status=processed
data/metadata/label_map.json          ← 50-class sorted label→id mapping
```

---

### Phase 4 — Training

```powershell
.venv\Scripts\python.exe scripts/train.py
```

Resume from checkpoint:
```powershell
.venv\Scripts\python.exe scripts/train.py --resume
```

**Model Architecture:**
```
Input (N, 32, 225)
    ↓ Linear(225, 128) + ReLU + LayerNorm
    ↓ 2-layer BiLSTM(128, bidirectional=True)  →  output: (N, 32, 256)
    ↓ Temporal Mean Pooling + Max Pooling       →  (N, 512)
    ↓ Linear(512, 128) + ReLU + Dropout(0.3)
    ↓ Linear(128, 50)
    ↓ Softmax → 50-class probabilities
```

**Training Configuration** (from `configs/config.yaml`):

| Parameter | Default |
|---|---|
| `num_frames` | 32 |
| `batch_size` | 16 |
| `epochs` | 50 |
| `learning_rate` | 0.001 |
| `weight_decay` | 0.0001 |
| `dropout` | 0.3 |
| `hidden_size` | 128 |
| `num_layers` | 2 |
| `early_stopping_patience` | 10 |
| `seed` | 42 |

Outputs:
- `models/checkpoints/latest.pt` — saved after every epoch
- `models/best_model/best.pt` — best validation accuracy
- `reports/training_history.csv`
- `reports/training_curves.png`

---

### Phase 5 — Evaluation

```powershell
.venv\Scripts\python.exe scripts/evaluate.py
```

Reports:
- Top-1 Accuracy
- Top-5 Accuracy
- Macro Precision, Recall, F1-Score
- Top class confusion pairs
- `reports/confusion_matrix.png`
- `reports/classification_report.csv`

---

### Phase 6 — Inference & Demo

#### Predict from video file
```powershell
.venv\Scripts\python.exe scripts/predict.py --video "path/to/video.MOV"
```
Output:
```
Predicted sign  : Dog
Confidence      : 87.42%

Top 5 Predictions:
  1. Dog           — 87.42%
  2. Cow           — 4.31%
  ...
```

#### Complete sign-to-speech demo (video file)
```powershell
.venv\Scripts\python.exe scripts/demo.py --video "path/to/video.MOV"
```

#### Text-to-speech utility
```powershell
.venv\Scripts\python.exe scripts/speak.py --text "48. Hello"
# Speaks: "Hello"
```

#### Real-time webcam demo
```powershell
.venv\Scripts\python.exe scripts/webcam_demo.py
```
Features:
- Rolling 32-frame buffer
- BiLSTM inference after every frame
- 5-frame majority voting smoothing
- Threshold-gated TTS (≥ 0.70 confidence)
- Press `q` to quit

#### Generate final report
```powershell
.venv\Scripts\python.exe scripts/generate_report.py
# Creates: reports/FINAL_REPORT.md
```

---

## Module Reference

### tools/ — Remote ZIP Download Engine

| File | Key Functions / Classes |
|---|---|
| `remote_zip.py` | `range_request()`, `get_zip_tail()`, `parse_eocd()`, `parse_zip64_locator_and_eocd()`, `get_central_directory()`, `parse_central_directory()`, `get_local_file_data_offset()` |
| `downloader.py` | `download_and_extract_video()` — streams compressed bytes, Deflate/Store decompression, CRC32 verification, atomic file move |
| `manifest.py` | `ManifestManager` — persistent CSV state, thread-safe updates, filesystem reconciliation |
| `validation.py` | `calculate_crc32()`, `is_safe_path()`, `validate_extracted_file()`, `verify_dataset_integrity()` |

**How the remote ZIP download works:**
1. `get_zip_tail()` — range request for last 64 KB of remote ZIP
2. `parse_eocd()` — locate End of Central Directory signature `PK\x05\x06`
3. ZIP64 check — if any field is `0xFFFF`/`0xFFFFFFFF`, parse ZIP64 EOCD (`PK\x06\x06`) and locator (`PK\x06\x07`)
4. `get_central_directory()` — range request for exact central directory bytes
5. `parse_central_directory()` — parse all `PK\x01\x02` entries, extract `local_header_offset`, `compressed_size`, `crc32`, `compression_method`
6. `get_local_file_data_offset()` — range request for 30-byte local file header `PK\x03\x04` to get exact data start offset
7. `download_and_extract_video()` — range request `bytes=data_start-(data_start+compressed_size-1)`, stream to `.comp` temp file, decompress with `zlib.decompressobj(-15)` to `.part` temp file, verify CRC32, atomic rename to final path

### src/ — Model & Pipeline Modules

| Module | Purpose |
|---|---|
| `src/features/extractor.py` | `LandmarkExtractor` — MediaPipe Holistic, extracts 225-dim vector per frame |
| `src/preprocessing/processor.py` | `sample_frame_indices()`, `process_video_file()` — uniform sampling + `.npz` save |
| `src/dataset/sign_dataset.py` | `SignLandmarkDataset`, `create_dataloaders()` — PyTorch Dataset with training augmentation |
| `src/models/bilstm.py` | `SignBiLSTMModel` — BiLSTM classifier |
| `src/training/trainer.py` | `ModelTrainer` — training loop, validation, early stopping, plot curves |
| `src/speech/tts.py` | `SpeechEngine` — pyttsx3 wrapper with label normalization |
| `src/utils/labels.py` | `normalize_label()`, `label_to_text()` — strip dataset numbering |
| `src/utils/config.py` | `load_config()` — YAML config loader |

### scripts/ — CLI Entrypoints

| Script | Command | Purpose |
|---|---|---|
| `verify_dataset.py` | `python scripts/verify_dataset.py` | Full dataset integrity check (run after download) |
| `inspect_dataset.py` | `python scripts/inspect_dataset.py` | Video quality CSV + statistics |
| `estimate_download_size.py` | `python scripts/estimate_download_size.py` | Size estimation (no download) |
| `download_selected_videos.py` | `python scripts/download_selected_videos.py` | Main downloader (supports `--dry-run`, `--test-one`, `--retry-failed`, `--workers`) |
| `preprocess_dataset.py` | `python scripts/preprocess_dataset.py` | Extract features (supports `--force`, `--num-frames`) |
| `extract_features.py` | `python scripts/extract_features.py` | Alias for preprocess_dataset.py |
| `train.py` | `python scripts/train.py` | Train BiLSTM model (supports `--resume`) |
| `evaluate.py` | `python scripts/evaluate.py` | Evaluate best model on validation set |
| `predict.py` | `python scripts/predict.py --video PATH` | Single video inference |
| `speak.py` | `python scripts/speak.py --text LABEL` | TTS pronunciation |
| `demo.py` | `python scripts/demo.py --video PATH` | Sign-to-speech demo |
| `webcam_demo.py` | `python scripts/webcam_demo.py` | Real-time webcam demo |
| `generate_report.py` | `python scripts/generate_report.py` | Generate FINAL_REPORT.md |

### configs/ — Configuration

`configs/config.yaml` controls all pipeline parameters. Changes take effect on next run.

```yaml
preprocessing:
  num_frames: 32

training:
  batch_size: 16
  epochs: 50
  learning_rate: 0.001
  early_stopping_patience: 10
  seed: 42

inference:
  confidence_threshold: 0.70
  smoothing_window: 5        # webcam majority voting window
```

---

## Key Design Decisions

### Remote ZIP extraction (no full download)
The Zenodo server supports HTTP Range requests (`HTTP 206 Partial Content`). The pipeline uses this to:
1. Read only the ZIP central directory (~10–14 KB per archive)
2. Read only the 30-byte local file header for offset calculation
3. Download only the compressed bytes of selected videos

This reduces download from **42.38 GB → 8.10 GB**.

### Fixed train/val split
The split in `selected_videos.csv` is **immutable**. Never re-split. The validation set is held out completely until final evaluation.

### Label normalization
Dataset labels include prefix numbers: `"48. Hello"`, `"1. Dog"`, `"46. you (plural)"`.
`normalize_label()` strips these for user-facing output only. Internal label keys are never modified.

### Constant-dimension features
MediaPipe landmarks are zero-filled when a hand is absent. Every sample is always `(32, 225)` regardless of hand visibility. This is essential for batch processing.

### Manifest-driven resume
Both the downloader and preprocessor use CSV manifests. On restart, already-verified/processed entries are skipped automatically. No data is reprocessed unless `--force` is passed.

### CRC32 validation
Every downloaded video is validated against the CRC32 stored in the ZIP central directory. Mismatches trigger retry and deletion of the corrupt output file.

### Path safety
`is_safe_path()` in `tools/validation.py` prevents ZIP path traversal attacks by verifying all output paths resolve inside `data/videos/`.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | 4 cores | 8 cores |
| RAM | 8 GB | 16 GB |
| Disk | 10 GB free | 20 GB free |
| GPU | Not required (CPU mode) | NVIDIA CUDA for faster training |
| Internet | Required for download phase only | Stable broadband |

**CUDA detection is automatic.** Training uses GPU if available, falls back to CPU.

The preprocessing and training pipelines are designed to run on CPU:
- Features are `(32, 225)` float32 arrays — no large image tensors in RAM
- Videos are processed one at a time and released immediately
- `num_workers=0` (safe for Windows multiprocessing)

---

## Dependencies

```
torch==2.13.0+cpu          # Deep learning (CPU; install CUDA version for GPU)
torchvision==0.28.0
mediapipe==1.0.0           # Landmark extraction
opencv-python==5.0.0.93    # Video reading
numpy==2.5.1
pandas==3.0.5
scikit-learn==1.9.0
matplotlib==3.11.1
requests==2.34.2           # HTTP Range requests for remote ZIP
pyttsx3==2.99              # Offline Windows TTS
pyyaml==6.0.3
tqdm==4.70.0
```

---

## Known Issues & Limitations

1. **Slow download speed** — Zenodo free tier is rate-limited (~260 kB/s). Downloading 8.10 GB takes ~8–10 hours at this speed.
2. **Intermittent DNS failures** — The downloader retries with exponential backoff (3s, 6s, 12s, 24s, 48s). If Zenodo is unavailable, re-run with `--retry-failed` after connectivity restores.
3. **MediaPipe hand visibility** — Fast-moving or occluded hands produce zero-filled features. This is handled gracefully but reduces feature quality for those frames.
4. **Windows `num_workers=0`** — PyTorch DataLoader multiprocessing has known issues on Windows. Training uses `num_workers=0` by default.
5. **50-class dataset is small** — With 10 training videos per class, overfitting is a risk. Landmark augmentation (noise, scale jitter, translation) is applied during training to mitigate this.

---

## Acceptance Criteria Checklist

### Download Pipeline
- [x] HTTP Range requests enforcing HTTP 206
- [x] ZIP central directory parsing (including ZIP64)
- [x] Local file header offset calculation (`PK\x03\x04`)
- [x] Streaming Deflate (method 8) and Store (method 0) decompression
- [x] CRC32 validation against ZIP central directory
- [x] Exponential backoff retry (5 attempts)
- [x] Persistent manifest with resume support
- [x] Ctrl+C safe with partial file cleanup
- [x] Path traversal security guard
- [x] `--dry-run`, `--test-one`, `--retry-failed`, `--workers` CLI flags
- [x] Remote extraction test PASS: `Animals/1. Dog/MVI_3060.MOV` (CRC: `0x5023369E`)

### Preprocessing Pipeline
- [x] 32-frame uniform temporal sampling
- [x] Short video interpolation (< 32 frames handled safely)
- [x] 225-dim fixed-length feature per frame (63+63+99)
- [x] Zero-filling for absent hands
- [x] `.npz` compressed feature files
- [x] Resumable feature extraction (skip existing)
- [x] Deterministic label map saved to `label_map.json`
- [x] `features_manifest.csv` tracking

### Model & Training
- [x] BiLSTM architecture with mean+max temporal pooling
- [x] Checkpoint saved after every epoch (`latest.pt`)
- [x] Best model saved separately (`best.pt`) with label_map + config
- [x] Resume training from `latest.pt`
- [x] Early stopping (patience=10)
- [x] Training augmentation (noise, scale, translation jitter)
- [x] No augmentation on validation
- [x] Training history CSV + curve plots

### Evaluation
- [x] Top-1 accuracy
- [x] Top-5 accuracy
- [x] Macro precision, recall, F1
- [x] Confusion matrix PNG (50×50)
- [x] Per-class classification report CSV
- [x] Top confusion pairs printed

### Inference & Speech
- [x] Single video inference with confidence threshold
- [x] Top-5 predicted signs output
- [x] Label normalization (`"48. Hello"` → `"Hello"`)
- [x] "Uncertain prediction" below threshold
- [x] pyttsx3 offline TTS
- [x] Real-time webcam with 5-frame smoothing
- [x] Non-repeating TTS output (2-second cooldown)