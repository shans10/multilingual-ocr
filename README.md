# Multilingual OCR Training (CRNN + Swin)

Production-ready multilingual OCR training using CTC (Connectionist Temporal Classification) loss. Supports any number of languages with automatic dataset balancing and evaluation metrics (CER, WER, Accuracy).

**Two model options:**
- `train_swin_multilingual.py` - Swin Transformer for higher accuracy
- `train_crnn_multilingual.py` - CRNN for faster, lightweight training

**Key features:**
- Two dataset formats: LMDB (recommended, fast) and TSV (legacy)
- Language balancing with weighted sampling and ratio capping
- AMP (Automatic Mixed Precision) for faster training with less VRAM
- Checkpoint management with best/latest saving and resume support
- Automatic corrupt image handling with detailed logging

For **technical details**, architecture decisions, and theory behind the implementation, see [`documentation.md`](documentation.md).

---

## Table of Contents

1. [Installation](#installation)
2. [Dataset Setup](#dataset-setup)
3. [Training Commands](#training-commands)
4. [Complete CLI Arguments](#complete-cli-arguments)
5. [Performance Tuning](#performance-tuning)
6. [Output Artifacts](#output-artifacts)
7. [Troubleshooting](#troubleshooting)
8. [Health Checks](#health-checks)

---

## Environment Setup

### Method 1: Conda (Recommended)

```bash
conda env create -f environment.yml
conda activate multilingual-ocr
```

### Method 2: pip (Legacy)

```bash
pip install -r requirements.txt
```

**Requirements:**
- `torch>=2.2`, `torchvision>=0.17`
- `pandas>=2.0`, `numpy>=1.24`
- `matplotlib>=3.7`, `Pillow>=9.0`
- `tqdm>=4.66`, `pyarrow>=14.0`
- `msgpack>=0.9`, `lmdb>=0.9`

---

## Dataset Setup

The scripts support **two dataset formats** - auto-detected based on folder structure.

### LMDB Format (Recommended)

Optimized for faster training, smaller storage, and easier transfer.

```
dataset_lmdb/
├── train.lmdb              # Training images
├── val.lmdb                # Validation images
├── test.lmdb               # Test images
└── meta/
    ├── dataset_info.json
    ├── train_index.csv
    ├── val_index.csv
    └── test_index.csv
```

**Benefits:**
- Faster image loading (memory-mapped)
- Smaller storage (built-in compression)
- Faster dataset transfer (single files, not millions of images)
- Preserves train/val/test splits from conversion

### TSV Format (Legacy)

Original format with language subfolders containing `.tsv` files:

```
dataset/
├── english/
│   ├── imposs/
│   │   ├── imposs.tsv       # columns: image_path, ground_truth
│   │   └── images/
│   └── modern/
└── konkani/
    └── base/
```

**Auto-detects columns:** `image_path`/`img_path`/`image`/`path`/`filename`/`file` and `ground_truth`/`label`/`text`/`transcription`/`word`

### TSV vs LMDB Performance

| Aspect | TSV | LMDB |
|--------|-----|------|
| First run | O(n) full scan | O(1) direct |
| Later runs | O(n) cache load | O(1) direct |
| Memory | Higher | Lower |
| Disk I/O | Multiple files | Single file |

**Manifest Build Time (TSV):**
| Scenario | Time |
|----------|------|
| First run (1000 images) | ~10 seconds |
| Subsequent runs (cached) | ~1 second |
| With `--rebuild-manifest` | Same as first run |

### Converting TSV to LMDB

One-time conversion process:

```bash
# Basic conversion
python convert_dataset_to_lmdb.py --source-root ./dataset --output-root ./dataset_lmdb

# Faster conversion with parallel workers
python convert_dataset_to_lmdb.py --source-root ./dataset --output-root ./dataset_lmdb --num-workers 8

# Custom batch size for LMDB commits
python convert_dataset_to_lmdb.py --source-root ./dataset --output-root ./dataset_lmdb --batch-size 2000
```

---

## Training Commands

### Quick Start

```bash
# Swin (default LMDB format)
python3 train_swin_multilingual.py

# CRNN (default LMDB format)
python3 train_crnn_multilingual.py
```

### With LMDB (Explicit)

```bash
python3 train_swin_multilingual.py --dataset-root ./dataset_lmdb
```

### With TSV Format

```bash
python3 train_swin_multilingual.py --dataset-root ./dataset
```

### Resume Training

```bash
# Resume from latest checkpoint
python3 train_swin_multilingual.py --resume

# Resume from specific checkpoint
python3 train_swin_multilingual.py --resume-from ./swin_multilingual_outputs/checkpoints/best_checkpoint.pth
```

### Language Balancing

```bash
# Enable weighted sampling for equal language representation
python3 train_swin_multilingual.py --balance-languages

# With hard cap on dominant language ratio
python3 train_swin_multilingual.py --balance-languages --max-dominant-to-second-ratio 1.5
```

### With AMP (Faster, Less VRAM)

```bash
python3 train_swin_multilingual.py --amp
python3 train_crnn_multilingual.py --amp
```

### Custom Parameters

```bash
# Custom image size, batch, epochs
python3 train_swin_multilingual.py --image-size 256x64 --batch-size 64 --epochs 80

# Limit training samples (for testing)
python3 train_swin_multilingual.py --max-samples 10000

# Custom learning rate
python3 train_swin_multilingual.py --learning-rate 5e-4
```

---

## Complete CLI Arguments

### Dataset Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset-root` | `./dataset_lmdb` | Path to dataset (LMDB or TSV format, auto-detected) |
| `--verify-paths` | `false` | Verify all image paths exist at startup (slow, for debugging) |
| `--max-samples` | `null` | Limit number of training samples (null = use all) |
| `--max-rows-per-tsv` | `null` | Limit rows per TSV file (null = use all) |

### Output Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | varies | Output directory (Swin: `./swin_multilingual_outputs`, CRNN: `./crnn_multilingual_outputs`) |
| `--manifest-out` | auto | TSV manifest output path (TSV format only) |
| `--manifest-cache` | auto | Parquet cache path for manifest (TSV format only) |
| `--rebuild-manifest` | `false` | Force rebuild of TSV manifest cache |

### Model Arguments

| Argument | Swin Default | CRNN Default | Description |
|----------|--------------|--------------|-------------|
| `--image-size` | 224x48 | 128x32 | Image size as WxH (e.g., '224x48') |
| `--batch-size` | 32 | 32 | Training batch size |
| `--epochs` | 60 | 60 | Number of training epochs |
| `--learning-rate` | 1e-4 | 1e-4 | Initial learning rate |
| `--patience` | 8 | 8 | Early stopping patience (epochs without improvement) |
| `--num-workers` | 4 | 4 | DataLoader worker processes |

### Training Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--balance-languages` | `false` | Enable weighted sampling for equal language representation |
| `--max-dominant-to-second-ratio` | `null` | Hard cap on dominant language ratio (e.g., 1.5) |
| `--amp` | `false` | Enable Automatic Mixed Precision (GPU only, ~50% less VRAM) |
| `--compile` | `false` | Enable torch.compile for faster training (5-15% speedup, may show SM warning on laptop GPUs) |

### Resume Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--resume` | `false` | Resume from latest checkpoint in output-dir |
| `--resume-from` | `null` | Resume from specific checkpoint path |

### Utility Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--smoke-test` | `false` | Run health check (dataset, dataloader, model forward pass) |

---

## Performance Tuning

### Batch Size Selection

| GPU VRAM | Swin (no AMP) | Swin (AMP) | CRNN (no AMP) | CRNN (AMP) |
|----------|---------------|------------|---------------|------------|
| 8GB | 16-32 | 64-128 | 32-64 | 128-256 |
| 16GB | 32-64 | 128-256 | 64-128 | 256-512 |
| 24GB+ | 64-128 | 256-512 | 128-256 | 512-1024 |

### DataLoader Workers

| Storage Type | Recommended `--num-workers` |
|--------------|-----------------------------|
| HDD | 0 or 1 (parallel seeks hurt HDD) |
| SSD | 4-8 |
| NVMe SSD | 8-16 |

### Recommended Performance Flags

```bash
# For fast training on SSD/NVMe with modern GPU
python3 train_swin_multilingual.py --amp --batch-size 128 --num-workers 8

# For HDD (reduce workers to avoid seek thrashing)
python3 train_swin_multilingual.py --amp --batch-size 128 --num-workers 0
```

---

## Output Artifacts

Each script saves to its output directory:

| Path | Description |
|------|-------------|
| `checkpoints/best_checkpoint.pth` | Best model (lowest validation loss) |
| `checkpoints/latest_checkpoint.pth` | Most recent model (every epoch) |
| `logs/training_log.csv` | Per-epoch metrics (loss, CER, WER, accuracy) |
| `logs/test_metrics.csv` | Final test set metrics |
| `logs/training_report.md` | Markdown summary report |
| `logs/skipped_images/` | Per-worker skipped image logs |
| `logs/skipped_images_summary.csv` | Aggregated skipped image summary |
| `plots/training_metrics.png` | 4-panel plot (loss, CER, WER, accuracy) |
| `plots/loss.png` | Loss curve |
| `plots/cer.png` | CER curve |
| `plots/wer.png` | WER curve |
| `plots/accuracy.png` | Accuracy curve |

---

## Troubleshooting

### 1) Training appears stuck after device print (TSV format)

**Cause:** Manifest building on large TSV datasets takes time.

**Fix:**
```bash
# First run - rebuild cache
python3 train_swin_multilingual.py --rebuild-manifest

# Subsequent runs - reuse cache
python3 train_swin_multilingual.py
```

**Note:** LMDB format is much faster and doesn't need manifest building.

### 2) CUDA out-of-memory (OOM)

**Fix in order:**
1. Reduce `--batch-size`
2. Reduce `--image-size` (e.g., 224x48 → 160x32)
3. Reduce `--num-workers`
4. Enable `--amp` (uses ~50% less VRAM)

**Laptop example (RTX 4050):**
```bash
python3 train_swin_multilingual.py --batch-size 8 --amp
```

### 3) Slow training throughput

| Cause | Solution |
|-------|----------|
| HDD storage | Use `--num-workers 0` |
| No AMP | Add `--amp` |
| Small batch | Increase `--batch-size` |
| No LMDB | Convert dataset to LMDB format |

### 4) Resume fails or metrics reset

- Ensure same dataset/vocabulary when resuming
- Don't delete logs manually before resuming
- Use `--resume` or `--resume-from <path>`

### 5) Corrupt/unreadable images

Scripts automatically skip bad images (don't crash).

**Check skipped images:**
```bash
head -20 ./swin_multilingual_outputs/logs/skipped_images_summary.csv
```

### 6) Test not running every epoch?

Correct - test runs **once** after training completes, using the best checkpoint. Validation runs every epoch.

---

## Health Checks

### Smoke Test (Automated)

```bash
python3 train_swin_multilingual.py --smoke-test
python3 train_crnn_multilingual.py --smoke-test
```

Validates: dataset loading, LMDB connection, dataloader iteration, model forward pass.

### Manual Checks

```bash
# Check GPU
nvidia-smi

# Check PyTorch CUDA
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"

# Check dataset
ls -lh ./dataset_lmdb/

# Check latest checkpoint
ls -lh ./swin_multilingual_outputs/checkpoints/

# Tail training log
tail -5 ./swin_multilingual_outputs/logs/training_log.csv
```