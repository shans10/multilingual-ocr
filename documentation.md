# Theory and Implementation

## 1) Project Overview

This repository provides two multilingual OCR training scripts:

| Script | Model | Use Case |
|--------|-------|----------|
| `train_swin_multilingual.py` | Swin Transformer + CTC | High-accuracy production model |
| `train_crnn_multilingual.py` | CRNN + CTC | Lightweight baseline, fast training |

Both scripts share core components for:
- Dataset handling (LMDB/TSV formats)
- Language balancing and split strategies
- Training optimizations (AMP, gradient clipping, schedulers)
- Checkpoint management and resume
- Metrics evaluation (CER, WER, accuracy)

For **how to run** the scripts, see `README.md`. This file covers **how and why** things work.

---

## 2) Dataset Design Decisions

The scripts support two formats (auto-detected):

| Format | Pros | Cons |
|--------|------|------|
| **LMDB** | Fast loading, small size, preserved splits | One-time conversion needed |
| **TSV** | No conversion, human-readable | Slower loading, larger storage |

**Why LMDB is recommended:**
- Memory-mapped access eliminates filesystem overhead
- Single-file structure enables fast dataset transfer
- Index files (`train_index.csv`, etc.) enable instant training startup

**Why TSV is still supported:**
- Original format, no conversion needed
- Easy to inspect/debug
- Good for small datasets or development

### 2.3 TSV Manifest Building

For TSV format datasets, the scripts build a manifest (index) on first run to enable fast loading.

**What is a Manifest?**

A manifest is a cached index of all training samples:
```
metadata/
├── manifest.parquet    # Cached index
├── train_manifest.csv # Per-split indexes
├── val_manifest.csv
└── test_manifest.csv
```

**Manifest Contents:**

For each sample, the manifest stores:
- `key`: Unique identifier (for LMDB key)
- `image_path`: Full path to image file
- `ground_truth`: OCR label text
- `language`: Language code
- `split`: train/val/test

**Column Auto-Detection:**

The script automatically detects TSV column names:

| Preferred | Alternatives |
|-----------|--------------|
| `image_path` | `img_path`, `image`, `path`, `filename`, `file` |
| `ground_truth` | `label`, `text`, `transcription`, `word` |

**Caching:**

The manifest is cached to parquet for faster subsequent runs:
- First run: Full scan of all TSV files
- Later runs: Load from cache (`manifest.parquet`)

**When to Rebuild:**

| Scenario | Action |
|-----------|-------|
| Adding new TSV files | Rebuild |
| Changing dataset structure | Rebuild |
| Updating ground truth labels | Rebuild |
| Manifest is corrupted | Rebuild |

- For rebuild command, see README
- For performance benchmarks, see README Dataset Setup section

**Recommended Workflow:**

1. **Development**: Use TSV (no conversion needed)
2. **Production**: Convert to LMDB (one-time, then fast)
3. **Large datasets**: Always use LMDB

**Split strategy:**
- **LMDB**: Preserved splits from conversion time (consistent across runs)
- **TSV**: Per-language 80/10/10 split (re-split each run, preserves minority languages)

**Language balancing:**
- Weighted sampling for equal language representation
- Hard cap on dominant language ratio before split

### 2.4 Language Balancing Implementation

For multilingual datasets with imbalanced language distributions, language balancing ensures the model learns from all languages equally.

**Why Language Balancing Matters:**

Without balancing, the model may:
- Overfit to dominant language (e.g., 80% English samples)
- Ignore minority languages entirely
- Produce poor predictions for underrepresented scripts

**Two Balancing Strategies:**

#### Strategy 1: Weighted Sampling (--balance-languages)

Enables weighted sampling where each sample's probability is inversely proportional to language frequency:

```python
# Key concept: Weight = total / (n_languages * language_count)
# This gives equal probability to each language regardless of size
```

**Example:**
```
Dataset: English (8000), Hindi (1000), Arabic (1000)
Without balancing: P(English)=0.8, P(Hindi)=0.1, P(Arabic)=0.1
With balancing: All languages have equal sampling probability (0.333)
```
- Keeps all samples (none removed)
- Equal sampling probability per language
- For CLI commands, see README

#### Strategy 2: Hard Cap (--max-dominant-to-second-ratio)

Limits the ratio between the largest and second-largest language:

```python
# Key concept: max_allowed = second_largest * max_ratio
# Randomly sample down dominant language to this limit
```

**Example:**
```
Dataset: English (8000), Hindi (1000), Arabic (1000)
--max-dominant-to-second-ratio 1.5
Result: English capped at 1500 (1000 * 1.5)
```
- Removes samples from dominant language
- Preserves ratio while reducing size
- For CLI commands, see README

**When to Use Which:**

| Scenario | Recommended Approach |
|-----------|---------------------|
| Equal importance to all languages | `--balance-languages` |
| Limit dominant language but keep most data | `--max-dominant-to-second-ratio` |
| Both strategies desired | Use both flags together |

**Important Notes:**

1. Language balancing is applied **before** train/val/test split
2. Weighted sampling doesn't remove data (keeps all samples)
3. Ratio cap removes samples from dominant language only
4. Check `train_index.csv` to see final language distribution
5. For CLI commands, see README

---

## 3) Design Philosophy and Trade-offs

### 3.1 Design Principles

| Principle | Implementation |
|-----------|----------------|
| **Simplicity** | No complex augmentation, standard components |
| **Reproducibility** | Fixed seeds, complete checkpoints |
| **Robustness** | Skip corrupt images, early stopping, graceful edge cases |

### 3.1.1 Corrupt Image Handling

The training scripts automatically skip corrupt/unreadable images instead of crashing, ensuring training continues even with problematic data.

**Why Skip Instead of Crash?**

Corrupt images in large multilingual datasets can be caused by:
- File system issues during dataset creation
- Image encoding errors in source files
- Missing or truncated image data
- Permission issues

**How It Works:**

```python
def __getitem__(self, idx):
    try:
        # Normal loading
        image = self._load_image(idx)
        return image, label, 0  # 0 = not skipped
    except Exception as e:
        # Log and skip
        self._log_skipped_image(idx, str(e), self.split_name)
        return None, label, 1  # 1 = skipped
```

**What Gets Logged:**

Each skipped image is logged to:
```
logs/skipped_images/skipped_{worker_id}.csv
```

**Log format:**
| key | error_type | error_message | split |
|-----|----------|-------------|-------|
| hindi_001.jpg | IOError | Cannot identify image file | train |
| arabic_002.png | OSError | file truncated | train |

**Aggregated Summary:**

After training, a summary is created at:
```
logs/skipped_images_summary.csv
```

**Summary format:**
| split | error_type | count | examples |
|-------|-----------|-------|----------|
| train | IOError | 5 | hindi_001.jpg, ... |
| val | OSError | 2 | arabic_002.png, ... |

- For how to check skipped images, see README

**Why Not Crash?**

1. Single corrupt image shouldn't stop entire training
2. Small number of corrupt images won't affect model quality
3. Logs enable fix later if needed

**Important Notes:**

- Skipped images don't contribute to training loss
- Count is logged but model continues
- Fix source files to include in future runs

### 3.2 Trade-off Decisions

| Decision | Rationale |
|----------|-----------|
| Swin over ViT | Shifted windows more efficient than full attention |
| CRNN as baseline | Proven, simple, fast - good comparison point |
| CTC over attention | Works without alignment, variable length |
| ReduceLROnPlateau | Adaptive - handles noisy OCR loss curves |
| LMDB format | Dramatically faster than file-based loading |
| No augmentation | Reproducibility, CTC is naturally robust |

---

## 4) Model Architecture Deep Dive

### 4.1 Swin Transformer + CTC (SwinCTC)

**Why Swin Transformer for OCR?**

Swin Transformer was chosen over traditional CNNs for the following reasons:

1. **Global attention mechanism**: Unlike CNNs which have local receptive fields, Swin's shifted window attention can capture long-range dependencies in text images
2. **Hierarchical features**: Swin produces multi-scale features (stage 1-4) which is beneficial for text of varying sizes
3. **State-of-the-art**: Swin-T offers excellent accuracy/compute tradeoff on vision tasks

**Architecture Code:**

```python
class SwinCTC(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        # swin_t = Tiny Swin Transformer (~28M params)
        # weights=None: random init (training from scratch)
        self.backbone = swin_t(weights=None)
        
        # Projection: 768-dim (Swin-T output) -> num_classes
        # 768 comes from Swin-T's hidden dimension
        self.proj = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input: [batch, 3, 48, 224] (RGB images)
        
        # Extract features through Swin backbone
        # Output: [batch, patches, 768] 
        feat = self.backbone.features(x)
        
        # Apply layer normalization
        feat = self.backbone.norm(feat)
        
        # Global average pooling over spatial dimensions
        # [batch, 768] - collapsed sequence
        feat = feat.mean(dim=1)
        
        # Project to CTC classes
        # [batch, num_classes] -> [seq_len, batch, num_classes]
        logits = self.proj(feat)
        return logits.permute(1, 0, 2)
```

**Tensor Shape Flow:**

```
Input Image:     [batch, 3, 48, 224]
    ↓
Swin Features:   [batch, 49, 768]   (7x7 patches from last stage)
    ↓
Norm:            [batch, 49, 768]
    ↓
Mean Pool:       [batch, 768]       (spatial collapse)
    ↓
Linear Proj:     [batch, num_classes]
    ↓
Permute:         [seq_len, batch, num_classes]
                  (seq_len=1 for CTC)
```

**Why output seq_len=1?**

The Swin model collapses spatial information into a single vector via global average pooling. While this loses sequential information (which matters for long text), it's a design choice for simplicity. For longer text, a different architecture (CNN+BiLSTM or Swin with sequence output) would be better.

### 4.1.1 Swin Layer Breakdown

**Stage-wise Architecture:**

| Stage | #Swin Blocks | Channels | Attention Heads | Output Size |
|-------|--------------|----------|-----------------|-------------|
| Patch Embed | 1 | 96 | - | H/4 × W/4 |
| Stage 1 | 2 | 96 | 3 | H/4 × W/4 |
| Stage 2 | 2 | 192 | 6 | H/8 × W/8 |
| Stage 3 | 6 | 384 | 12 | H/16 × W/16 |
| Stage 4 | 2 | 768 | 24 | H/32 × W/32 |

**Feature Map Dimensions:**

| Stage | Spatial (H×W) | Channels |
|-------|---------------|----------|
| Input | 48×224 | 3 (RGB) |
| Stage 1 | 12×56 | 96 |
| Stage 2 | 6×28 | 192 |
| Stage 3 | 3×14 | 384 |
| Stage 4 | 1×7 | 768 |

#### Which Swin Variant Are We Using?

This code uses **Swin-Tiny** from torchvision:
```python
from torchvision.models import swin_t
self.backbone = swin_t(weights=None)
```

**Available Swin Variants:**

| Variant | Parameters | Channel Progression |
|---------|------------|---------------------|
| **Swin-Tiny** (ours) | ~28M | 96 → 192 → 384 → 768 |
| Swin-Small | ~50M | 96 → 192 → 384 → 768 |
| Swin-Base | ~88M | 128 → 256 → 512 → 1024 |

We use Swin-Tiny because it balances accuracy and computational cost for multilingual OCR tasks.

#### About Spatial Reduction

Each stage halves both height and width while doubling channels. This trade-off enables the model to:
- Process larger effective receptive fields in later stages
- Reduce computational cost for attention layers
- Capture hierarchical feature representations

**Why This Matters:**
The spatial reduction from 48×224 to 1×7 compresses the entire image into a single global vector. While this loses exact positional information, the attention mechanism within each window still captures local features effectively.

**Parameter Count (~28M total):**

| Component | Parameters | Percentage |
|-----------|-------------|------------|
| Patch Embed | ~9K | 0.03% |
| Stage 1 (2 blocks) | ~3.4M | 12% |
| Stage 2 (2 blocks) | ~12.5M | 45% |
| Stage 3 (6 blocks) | ~11.2M | 40% |
| Stage 4 (2 blocks) | ~0.9M | 3% |
| Final Norm + AvgPool | ~2K | 0.01% |
| Linear Projection | ~2M | ~7% |

#### About Parameter Distribution - Simple Explanation

**Stage-by-stage breakdown:**

- **Stage 3 (40%)**: Highest because it has 6 transformer blocks, each with attention + MLP
- **Stage 2 (45%)**: Second highest but with 2 blocks at higher 192 channels
- **Stage 1 (12%)**: Only 2 blocks at smaller 96 channels
- **Stage 4 (3%)**: Only 2 blocks but reduced to 1×7 spatial

**Why attention is the main cost:**
Each transformer block contains:
- 4 "learnable matrices" for Query, Key, Value, and Output
- A feed-forward network that expands features 4x then compresses back

Think of these matrices as "lookup tables" the network learns during training. Each matrix is a grid of numbers that transforms the input features. More channels = larger grids = more parameters.

**Simple math (concept only):**
- 1 attention matrix ≈ channels × channels (e.g., 768 × 768 = ~590K numbers)
- Each Swin block has ~4-5 such matrices
- Multiple blocks stack together = lots of parameters

**Activation Functions:**

| Layer | Activation | Notes |
|-------|----------|-------|
| Swin Block | GELU | Gaussian Error Linear Unit |
| MLP | GELU | Two FC layers |
| Layer Norm | None | No activation (normalization) |
| Final Pool | None | Average pooling |

**Key Swin Concepts:**

- **Window Attention**: 7×7 fixed window splits input
- **Shifted Windows**: Alternating shift by 3 pixels
- **Relative Position Bias**: Learnable positional encoding
- **MLP Ratio**: 4× hidden dimension (3072 for 768)

---

### VRAM Usage - Swin-Tiny

VRAM (Video RAM) is the GPU memory needed to train the model. Here's how it's used:

**What's stored in VRAM during training:**

| Component | Description | Size (approximate) |
|-----------|-------------|-------------------|
| **Model weights** | The learned parameters | 28M × 2 bytes = ~56MB |
| **Gradients** | Derivatives for backpropagation | 28M × 4 bytes = ~112MB |
| **Adam optimizer state** | Momentum for each parameter (2 buffers) | 28M × 8 bytes = ~224MB |
| **Activations** | Intermediate values during forward pass | Varies by layer |

**Activations breakdown by stage:**

| Stage | Activation Size per Sample | Total for Batch 32 |
|-------|---------------------------|---------------------|
| Stage 1 | 12×56×96 × 2 | ~130KB |
| Stage 2 | 6×28×192 × 2 | ~65KB |
| Stage 3 | 3×14×384 × 2 | ~32KB |
| Stage 4 | 1×7×768 × 2 | ~11KB |
| **Total activations** | | ~240KB per sample |

**Estimated VRAM for training:**

| Batch Size | Estimated VRAM |
|-----------|----------------|
| 16 | ~500MB |
| 32 | ~700MB-1GB |
| 64 | ~1.2GB-1.5GB |
| 128 | May exceed 2GB (OOM risk) |

**With AMP (Automatic Mixed Precision):**
- Model weights stored in FP16 (~56MB instead of 112MB)
- Gradients computed in FP32 for stability
- Total VRAM reduced by ~30-40%

| Batch Size with AMP | Estimated VRAM |
|-----------------|-----------------|
| 32 | ~500MB-700MB |
| 64 | ~800MB-1GB |

---

### 4.2 CRNN Architecture

**Why CRNN for OCR?**

1. **Lightweight**: Much fewer parameters than Swin (~5M vs ~28M)
2. **Fast inference**: Single forward pass, no attention complexity
3. **Proven**: CRNN+CTC is the standard baseline for scene text recognition
4. **Simple**: Easy to train, debug, and understand

**CNN Stack Design (Height Compression):**

```python
self.cnn = nn.Sequential(
    # Block 1: 1 -> 64 features, height / 2
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(2, 2),  # 48 -> 24 height
    
    # Block 2: 64 -> 128 features, height / 4
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d(2, 2),  # 24 -> 12 height
    
    # Block 3: 128 -> 256 features, height / 4
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(256),
    nn.ReLU(True),
    nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d((2, 1), (2, 1)),  # height / 2, width preserved
    
    # Block 4: 256 -> 512 features, height / 8
    nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
    nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
    nn.ReLU(True),
    nn.MaxPool2d((2, 1), (2, 1)),  # height / 2, width preserved
    
    # Block 5: 256 -> 512 features, height / 8
    nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0),
    nn.BatchNorm2d(512),
    nn.ReLU(True),
)
```

**CNN Output Shape Calculation:**

For input 32×128:
- Block 1: MaxPool2d(2,2) → 16×64
- Block 2: MaxPool2d(2,2) → 8×32
- Block 3: MaxPool2d((2,1),(2,1)) → 4×32
- Block 4: MaxPool2d((2,1),(2,1)) → 2×32
- Block 5: Conv2d(2,1) → 1×31

Final: [batch, 512, 1, 31] → squeeze height → [31, batch, 512]

**BiLSTM Stack:**

```python
self.rnn = nn.Sequential(
    # First LSTM: 512 -> 256 (bidirectional = 512 total)
    BidirectionalLSTM(512, 256, 256),
    # Second LSTM: 256 -> 256 (bidirectional = 512 total)
    BidirectionalLSTM(256, 256, num_classes),
)
```

```python
class BidirectionalLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, 
            bidirectional=True, batch_first=True
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)
        return self.fc(out)
```

**Full Forward Pass:**

```
Input:              [batch, 1, 32, 128]
    ↓
CNN Stack:         [batch, 512, 1, 31]
    ↓
Squeeze Height:    [31, batch, 512]  (seq_len, batch, features)
    ↓
Permute:           [batch, 31, 512]  (batch, seq_len, features)
    ↓
BiLSTM 1:          [batch, 31, 256]
    ↓
BiLSTM 2:          [batch, 31, num_classes]
    ↓
Permute:           [31, batch, num_classes] (for CTC)
```

**Why height must be 1 before LSTM?**

The assert ensures the CNN properly compresses height:
```python
_, _, height, _ = conv.size()
if height != 1:
    raise RuntimeError(f"CNN output height must be 1, got {height}")
```

### 4.2.1 CRNN Layer Breakdown

**CNN Block Architecture:**

| Block | Layers | Input Ch | Output Ch | Spatial Reduction |
|-------|--------|---------|-----------|------------------|
| Block 1 | Conv + ReLU + Pool | 1 | 64 | H/2 × W/2 |
| Block 2 | Conv + ReLU + Pool | 64 | 128 | H/4 × W/4 |
| Block 3 | 2×Conv + 2×ReLU + Pool | 128 | 256 | H/8 × W/8 |
| Block 4 | 2×Conv + 2×ReLU + Pool | 256 | 512 | H/16 × W/16 |
| Block 5 | Conv + BatchNorm + ReLU | 512 | 512 | H/16 × (W-1) |

**Feature Map Depths (32×128 input):**

| Block | Input Size | Channels | Output Size |
|-------|----------|----------|-----------|
| Input | 32×128 | 1 | 32×128 |
| Block 1 | 32×128 | 64 | 16×64 |
| Block 2 | 16×64 | 128 | 8×32 |
| Block 3 | 8×32 | 256 | 4×32 |
| Block 4 | 4×32 | 512 | 2×32 |
| Block 5 | 2×32 | 512 | 1×31 |

**Parameter Count (~5M total):**

| Layer | Parameters | Percentage |
|-------|-------------|------------|
| Block 1 (Conv 3×3) | 1,728 | 0.03% |
| Block 2 (Conv 3×3) | 73,728 | 1.4% |
| Block 3 (2×Conv 3×3 + BN) | ~590K | 11% |
| Block 4 (2×Conv 3×3 + BN) | ~2.4M | 45% |
| Block 5 (Conv 2×2 + BN) | ~1.0M | 19% |
| BiLSTM Layer 1 (256 hidden) | ~790K | 15% |
| BiLSTM Layer 2 (256 hidden) | ~525K | 10% |

#### About the Conv Layers

Each Conv2d layer learns filters that detect specific patterns:
- **Early layers**: Edges, corners, basic shapes
- **Later layers**: Character components, complex textures

The learning happens through training - filters that help reduce prediction error become stronger.

**MaxPool Purpose:**
Max pooling (taking maximum value in each 2×2 window) makes the network:
- Translation invariant (pixel shifts don't affect detection)
- More robust to variations in character position
- Computationally cheaper for subsequent layers

#### About BiLSTM Parameters

Bidirectional LSTM processes sequences in both forward and backward directions:
- **Forward direction**: Captures left-to-right context
- **Backward direction**: Captures right-to-left context
- **Combined**: Full bidirectional context for each position

This is important for OCR because a character often depends on both preceding and following characters.

**BiLSTM Specifications:**

| Layer | Input | Hidden | Directions | Output |
|-------|-------|--------|------------|---------|
| LSTM 1 | 512 | 256 | 2 (bidirectional) | 256 |
| LSTM 2 | 256 | 256 | 2 (bidirectional) | num_classes |

#### About Parameter Distribution

- **Block 4 (45%)**: Largest conv layer with 512 channels
- **BiLSTM (25%)**: Handles sequential modeling
- **Blocks 3-5**: Feature extraction
- **Block 1-2**: Small initial features

**Why fewer parameters than Swin:**
- CRNN uses standard convolutions (local receptive fields)
- No self-attention matrices
- Single forward pass without attention overhead
- LSTM has fewer parameters than transformer attention

**Activation Functions:**

| Layer | Activation | Notes |
|-------|----------|-------|
| Conv layers | ReLU | After each convolution |
| BatchNorm | None | Normalization only |
| LSTM | tanh (internal) | Cell uses tanh |
| Dropout | None | Not in default config |

**Key CRNN Concepts:**

- **Height Compression**: CNN reduces height from 32→1
- **Sequential Output**: Each column = 1 time step
- **Bidirectional**: Processes both directions for context
- **CTC Ready**: Output [seq, batch, classes] format

---

### VRAM Usage - CRNN

**What's stored in VRAM during training:**

| Component | Description | Size (approximate) |
|-----------|-------------|-------------------|
| **Model weights** | The learned parameters | 5M × 2 bytes = ~10MB |
| **Gradients** | Derivatives for backpropagation | 5M × 4 bytes = ~20MB |
| **Adam optimizer state** | Momentum for each parameter | 5M × 8 bytes = ~40MB |
| **Activations** | Intermediate values | Smaller than Swin |

**Activations breakdown by layer:**

| Layer | Activation Size per Sample | Total for Batch 32 |
|-------|----------------------------|---------------------|
| Block 1-5 | Variable by layer | ~50KB |
| LSTM 1 | 31×512 × 2 | ~32KB |
| LSTM 2 | 31×256 × 2 | ~16KB |
| **Total activations** | | ~100KB per sample |

**Estimated VRAM for training:**

| Batch Size | Estimated VRAM |
|-----------|----------------|
| 16 | ~120MB |
| 32 | ~150MB |
| 64 | ~250MB |
| 128 | ~450MB |

**With AMP:**

| Batch Size | Estimated VRAM |
|-----------|----------------|
| 32 | ~100MB |
| 64 | ~180MB |
| 128 | ~350MB |

---

### Swin vs CRNN VRAM Comparison

| Metric | Swin-Tiny | CRNN |
|--------|----------|------|
| Parameters | ~28M | ~5M |
| Base VRAM (batch 32) | ~700MB | ~150MB |
| VRAM with AMP | ~500MB | ~100MB |
| Can fit in 4GB GPU? | No (needs 6GB+) | Yes |
| Can fit in 6GB GPU? | Yes (with AMP) | Yes |
| Can fit in 8GB GPU? | Yes | Yes |

---

This is critical because RNNs expect sequence data (time major), and height=1 means each column represents a time step.

### 4.3 Architecture Comparison

| Aspect | SwinCTC | CRNN |
|--------|---------|------|
| Parameters | ~28M | ~5M |
| VRAM (batch=32) | ~3GB | ~1GB |
| Inference speed | Slower | Faster |
| Accuracy | Higher | Lower |
| Attention mechanism | Yes (shifted windows) | No |
| Sequential建模 | Pooled (single vector) | BiLSTM (full sequence) |
| Best for | High accuracy tasks | Resource-constrained |

---

## 5) Training Configuration Rationale

### 5.1 Hyperparameter Choices

**Why Image Size 48×224 for Swin?**

```
Aspect ratio consideration:
- Text images are typically wide (more width than height)
- 48 height: enough vertical resolution for character features
- 224 width: accommodates 10-30+ characters at ~7px/char
- Total pixels: 10,752 (much smaller than 224×224 = 50,176)

Trade-offs:
- Larger = more VRAM, slower training
- Smaller = might lose fine-grained character details
- 48×224 is a sweet spot for multilingual text
```

**Why Image Size 32×128 for CRNN?**

```
CRNN is more compact:
- 32 height: sufficient for CNN to compress to height=1
- 128 width: adequate for 5-15 characters
- Total pixels: 4,096 (even smaller than Swin)

Why smaller than Swin?
- CRNN is designed to be lightweight
- Smaller images = faster training
- Trade-off: slightly lower accuracy
```

### Image Size Handling

The training scripts handle images of any size by resizing to the target dimensions while preserving aspect ratio.

#### Resize and Pad Strategy

All input images are resized to a fixed target size (48×224 for Swin, 32×128 for CRNN) using a letterbox approach:

1. **Calculate scaling**: Determine the scale factor to fit within target bounds while preserving aspect ratio
2. **Resize**: Scale the image to fit within target dimensions
3. **Pad**: Add black padding to reach exact target size (centered)

```
Original image: 100×300 → Resize to fit within 48×224 → Pad to 48×224
Original image: 200×400 → Resize to fit within 48×224 → Pad to 48×224
Original image: 24×100 → Resize to fit within 48×224 → Pad to 48×224
```

#### Why This Approach?

| Aspect | Description |
|--------|-------------|
| **No data loss** | All text is preserved - no cropping |
| **Aspect ratio preserved** | Characters maintain proper shape |
| **Consistent input** | All images become exact same size for batch processing |
| **Black padding** | Uses black (0) as padding color - standard for OCR |

#### Image Size Recommendations

| Model | Default Size | Recommended For |
|-------|-------------|-----------------|
| Swin | 48×224 | Multilingual text with complex scripts (Hindi, Arabic, etc.) |
| Swin | 32×224 | Simpler scripts, more VRAM constrained |
| CRNN | 32×128 | Lightweight baseline, resource-constrained |
| CRNN | 48×224 | Better accuracy with more compute |

**Note:** Image dimensions should ideally be multiples of 4 for Swin due to the patch embedding dividing by 4 at each stage.

**Why Batch Size 32 for both Swin and CRNN?**

```
Swin batch=32:
- Swin-T is larger (~28M params)
- Forward+backward activations are memory-heavy
- 32 is conservative, fits in 8GB VRAM

CRNN batch=32:
- CRNN is smaller (~5M params)
- Default kept conservative for consistency
- With AMP, can double/triple batch size
```

### Batch Size Selection Guide

#### How to Determine Optimal Batch Size

Batch size primarily depends on your GPU's VRAM. Here's the breakdown:

**VRAM Breakdown (Swin-Tiny, batch_size=32):**

| Component | Without AMP | With AMP |
|-----------|-------------|----------|
| Model weights | ~112MB | ~56MB |
| Gradients | ~112MB | ~112MB |
| Adam optimizer | ~224MB | ~224MB |
| Activations | ~500-800MB | ~300-500MB |
| **Total per batch** | **~1GB** | **~800MB** |

#### Recommended Batch Sizes by GPU VRAM

| GPU VRAM | Swin (No AMP) | Swin (AMP) | CRNN (No AMP) | CRNN (AMP) |
|---------|---------------|------------|---------------|------------|
| 4GB | 8-12 | 16-20 | 32-48 | 48-64 |
| 6GB | 20-24 | 32-40 | 64-96 | 96-128 |
| 8GB | 28-32 | 48-64 | 96-128 | 128-192 |
| 12GB | 40-48 | 64-96 | 128-192 | 192-256 |
| 24GB+ | 64+ | 96+ | 192+ | 256+ |

#### How to Find Your Optimal Batch Size

1. **Start conservative**: Use batch_size=16 for Swin, 32 for CRNN
2. **Test for OOM**: Run a few training steps and check if CUDA out-of-memory occurs
3. **Scale up gradually**: If no OOM, increase batch size by 4-8 until you approach memory limit
4. **With AMP**: You can typically double the batch size compared to without AMP
5. **Rule of thumb**: Leave ~500MB headroom for variations in input sizes

#### Trade-offs

| Higher Batch Size | Lower Batch Size |
|-----------------|------------------|
| More stable gradients | More noisy gradients |
| Faster convergence | Slower convergence |
| Better GPU utilization | Lower GPU utilization |
| May need learning rate adjustment | May need smaller learning rate |

#### Example Calculation

For 8GB GPU with Swin and AMP:
- Available: ~8GB minus system overhead (~1GB) = ~7GB usable
- Per-sample with AMP: ~25MB
- Maximum batch: 7GB / 25MB ≈ 280 samples
- Conservative: 64-128 (to leave headroom)

For 6GB GPU with Swin and AMP:
- Available: ~6GB minus overhead ≈ 5GB
- Maximum: 5GB / 25MB ≈ 200
- Conservative: 32-48

**Epochs:**

| Model | Default | Rationale |
|-------|---------|-----------|
| Swin | 60 | Larger model learns faster per epoch |
| CRNN | 100 | Smaller model needs more epochs |

### 5.2 Learning Rate Selection

**Why 1e-4 for AdamW?**

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

**Rationale:**
- 1e-4 is the standard starting LR for AdamW on vision tasks
- Too high (>1e-3): training unstable, loss diverges
- Too low (<1e-5): training very slow, might converge to poor local minimum
- 1e-4 works well for both Swin and CRNN

**Weight Decay: 1e-4**

- Moderate regularization
- Prevents overfitting without hurting convergence
- Standard for AdamW (different from Adam's default 0)

### 5.3 Early Stopping Configuration

**Why patience=8?**

```python
patience: int = 8  # Wait 8 epochs without improvement before stopping
```

**Rationale:**
- Too small (2-3): might stop too early, missing better convergence
- Too large (15+): wastes compute on plateau
- 8 is a balanced choice:
  - OCR loss curves can be noisy
  - 8 epochs = reasonable wait time
  - Combined with LR reduction (patience=3), provides good escape from plateaus

---

## 6) CTC Loss Deep Dive

### 6.1 Why CTC for OCR?

**Traditional vs CTC:**

```
Traditional approach:
- Requires alignment between input frames and target characters
- Need to know which frame corresponds to which character
- Infeasible for variable-length text

CTC (Connectionist Temporal Classification):
- No alignment needed
- Learns to collapse repeated characters and blanks
- Works with variable-length input and output
```

**CTC Example:**

```
Input sequence (frames):  [a, a, a, b, b, c, c, c, -, -]
Target text:             "abc"
CTC decoding:            collapse(aaa) + collapse(bb) + collapse(ccc) + collapse(-) = "abc"
                          = remove blanks + merge repeats
```

### 6.2 CTC Loss Configuration

```python
criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
```

**Why blank=0?**

- Index 0 is reserved for the "blank" token (no character)
- CTC uses blank to handle repeated characters
- 0 is the standard convention (works with vocab where 0 = blank)

**Why reduction="mean"?**

- Options: "none", "mean", "sum"
- "mean": average loss over batch (standard)
- "sum": total loss (useful for weighted batches)
- "none": return per-sample losses (debugging)

**Why zero_infinity=True?**

```python
# Without zero_infinity=True:
# If target length > input length, loss = inf
# This causes NaN in gradients, crashes training

# With zero_infinity=True:
# If target length > input length, loss = 0
# Training continues, sample is essentially "impossible"
```

This prevents NaN crashes on edge cases.

### 6.3 CTC Decoding Algorithm

```python
def decode_batch(log_probs, idx_to_char, blank_idx=0):
    """Greedy CTC decoding (used in training scripts)."""
    # log_probs: [seq_len, batch, num_classes]
    preds = log_probs.permute(1, 0, 2).argmax(2)  # [batch, seq_len]
    
    decoded = []
    for b in range(preds.size(0)):
        chars = []
        prev = None
        for t in range(preds.size(1)):
            idx = int(preds[b, t].item())
            # Keep character if: not blank, not same as previous, in vocab
            if idx != blank_idx and idx != prev and idx in idx_to_char:
                chars.append(idx_to_char[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded
```

**Greedy vs Beam Search:**

| Method | Description | Use Case |
|--------|-------------|----------|
| Greedy | Take argmax at each step | Fast, standard for training |
| Beam | Keep top-k paths, choose best | Better accuracy, slower |

The scripts use greedy decoding (faster, sufficient for most cases).

---

## 7) Optimizer and Scheduler Deep Dive

### 7.1 Why AdamW over SGD/Adam?

```python
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
```

**AdamW vs Adam:**

```
Adam:
- Uses momentum and adaptive learning rates
- weight_decay is implemented as L2 regularization (during gradient computation)
- Can be less effective for deep networks

AdamW (Adam with Weight Decay):
- Decoupled weight decay (applied to parameters, not gradients)
- Better generalization than Adam
- Now the standard optimizer for Transformers and CNNs
```

**AdamW vs SGD:**

```
SGD:
- Simple, requires tuning
- Often achieves better final accuracy with proper scheduling
- Requires learning rate warmup and decay

AdamW:
- Adaptive, less sensitive to LR choice
- No warmup needed (usually)
- Slightly lower final accuracy but more stable training
```

For multilingual OCR with noisy CTC loss curves, AdamW's stability is preferred over SGD's potential for better but less consistent results.

### 7.2 ReduceLROnPlateau Configuration

```python
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",        # Minimize validation loss
    factor=0.5,        # Halve LR when triggered
    patience=3,        # Wait 3 epochs without improvement
    min_lr=1e-6,       # Minimum LR (don't go below this)
)
```

**Why ReduceLROnPlateau?**

```
Alternative schedulers:
- StepLR: Fixed schedule (e.g., epoch 30, 60)
- CosineAnnealing: Smooth decay
- CyclicLR: Oscillating LR

ReduceLROnPlateau advantages for OCR:
- CTC loss can be noisy (doesn't decrease smoothly)
- Adaptive: reduces LR only when truly stuck
- No manual schedule tuning needed
```

**Why patience=3?**

```
Too aggressive (patience=1):
- Might reduce LR too quickly
- Could stop before true plateau

Too passive (patience=10):
- Wastes compute waiting
- Might overshoot best LR

3 is a balanced choice:
- Enough time to distinguish noise from real plateau
- Quick enough to not waste compute
```

**Why factor=0.5?**

```
Halving the LR is standard:
- Large enough to escape plateau
- Small enough to not destabilize training
- 0.3-0.5 is the typical range
```

**Why min_lr=1e-6?**

```
At min_lr, model is essentially fine-tuning:
- Large changes unlikely
- Further reductions have minimal impact
- 1e-6 is low enough to allow extensive fine-tuning
- Prevents complete convergence stall
```

---

## 8) Data Pipeline Deep Dive

### 8.1 OCRDataset Implementation

```python
class OCRDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,  # DataFrame with key, ground_truth, language
        img_height: int,
        img_width: int,
        split_name: str,
        skipped_log_dir: Optional[Path] = None,
        lmdb_path: Optional[Path] = None,
    ):
        self.frame = frame.reset_index(drop=True)
        self.split_name = split_name
        self.skipped_log_dir = skipped_log_dir
        self.lmdb_path = lmdb_path
        self.lmdb_env = None
        self.tf = transforms.Compose([
            transforms.Resize((img_height, img_width)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        # Lazy LMDB open (first access only)
        if self.lmdb_path is not None and self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
            )

        row = self.frame.iloc[idx]
        
        if self.lmdb_env is not None:
            # Load from LMDB
            with self.lmdb_env.begin() as txn:
                img_bytes = txn.get(f"{row['key']}.img".encode("utf-8"))
                image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            # Load from file path (TSV format)
            image = Image.open(row["image_path"]).convert("RGB")

        # Transform to tensor
        image = self.tf(image)
        label = row["ground_truth"]
        return image, label, 0  # 0 = not skipped
```

### 8.2 Why No Data Augmentation?

**Current pipeline:**
- Resize to fixed size
- ToTensor (normalize to [0,1])

**Why this simplicity?**

1. **Reproducibility**: Augmentation adds randomness, makes reproduction harder
2. **CTC robustness**: CTC naturally handles slight variations in character position
3. **Dataset already diverse**: Multilingual dataset has natural variation
4. **Debugging**: Simpler pipeline = easier to debug issues

**If augmentation needed later:**
- Geometric: Random rotation (±5°), random perspective
- Color: Random brightness/contrast (for document images, less relevant)
- Implement with `transforms.RandomChoice` or custom collate function

### 8.3 LMDB Loading Strategy

**Why lazy open?**

```python
# In __getitem__:
if self.lmdb_path is not None and self.lmdb_env is None:
    self.lmdb_env = lmdb.open(...)
```

- Dataset `__init__` doesn't open LMDB (fast instantiation)
- First `__getitem__` opens it (lazy)
- Worker init function can also open (see next section)

**Transaction per image:**

```python
with self.lmdb_env.begin() as txn:
    img_bytes = txn.get(key)
```

- Each image read = one transaction
- Works but not optimal for throughput
- Alternative: keep transaction open across multiple reads

### 8.4 Worker LMDB Connection

```python
def _init_worker_lmdb(worker_id):
    """Initialize LMDB environment for each DataLoader worker."""
    worker = get_worker_info()
    if worker is None:
        return
    dataset = worker.dataset
    if dataset.lmdb_path is not None:
        dataset.lmdb_env = lmdb.open(
            str(dataset.lmdb_path),
            readonly=True,
            lock=False,
            readahead=False,
        )
```

**Why per-worker connections?**

```
Problem: LMDB connections are not thread-safe
Solution: Each DataLoader worker opens its own connection

Without worker init:
- Main process opens LMDB
- Workers try to use same connection -> errors/crashes

With worker init:
- Worker 0 opens train.lmdb for itself
- Worker 1 opens train.lmdb for itself
- Each has independent connection
```

---

## 9) Evaluation Metrics Deep Dive

### 9.1 Levenshtein Distance

```python
def levenshtein_distance(s1, s2):
    """Compute Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]
```

**Example:**
```
Reference: "hello"
Hypothesis: "hallo"

Operations:
- h -> h (match, cost 0)
- e -> a (substitute, cost 1)
- l -> l (match, cost 0)
- l -> l (match, cost 0)
- o -> o (match, cost 0)

Levenshtein distance = 1
```

### 9.2 Character Error Rate (CER)

```python
def cer(reference: str, hypothesis: str) -> float:
    if len(reference) == 0:
        return 0.0
    return levenshtein_distance(reference, hypothesis) / len(reference)
```

**Why divide by reference length?**

```
Normalization:
- Raw distance: 5 errors
- Reference "hello": 5/5 = 1.0 (100% error)
- Reference "verylongword": 5/12 = 0.42 (42% error)

Same errors, different text lengths -> different CER
Division makes it comparable
```

**CER Interpretation:**
- 0.0 = perfect (no errors)
- 1.0 = completely wrong (all characters wrong)
- Lower is better

### 9.3 Word Error Rate (WER)

```python
def wer(reference: str, hypothesis: str) -> float:
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)
```

**CER vs WER:**

| Metric | Granularity | Use Case |
|--------|-------------|----------|
| CER | Character | Fine-grained OCR evaluation |
| WER | Word | Document-level OCR, ASR |

For multilingual OCR with varying scripts (not all have "words"), CER is more universally applicable.

### 9.4 Exact Match Accuracy

```python
def exact_match(reference: str, hypothesis: str) -> float:
    return 1.0 if reference == hypothesis else 0.0
```

**Why this metric?**

- Binary: either exact match or not
- Complementary to CER/WER (which measure partial errors)
- Useful for: "What % of samples are 100% correct?"

---

## 10) Checkpoint System Deep Dive

### 10.1 What's Saved in Checkpoint

```python
ckpt = {
    "epoch": epoch + 1,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "char_to_idx": char_to_idx,      # Vocabulary mapping
    "idx_to_char": idx_to_char,      # Reverse vocabulary
    "best_val_loss": best_val_loss,
    "patience_count": patience_count,
    "train_losses": train_losses,    # Historical metrics
    "val_losses": val_losses,
    "train_cers": train_cers,
    "val_cers": val_cers,
    "train_wers": train_wers,
    "val_wers": val_wers,
    "train_accs": train_accs,
    "val_accs": val_accs,
    "config": vars(cfg),             # Training config
}
```

**Why save all this?**

| Component | Why |
|------------|-----|
| model_state_dict | Restore model weights |
| optimizer_state_dict | Continue training from exact point |
| char_to_idx / idx_to_char | Ensure vocabulary consistency |
| best_val_loss | Track best performance |
| patience_count | Continue early stopping logic |
| train/val_* arrays | Keep plots/reports continuous |
| config | Reproducibility |

### 10.2 Best vs Latest Checkpoint

```python
# Save latest every epoch
torch.save(ckpt, ckpt_dir / "latest_checkpoint.pth")

# Save best only on improvement
if val_metrics["loss"] < best_val_loss:
    best_val_loss = val_metrics["loss"]
    patience_count = 0
    torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")
```

**Why two checkpoints?**

```
latest_checkpoint.pth:
- Always available (updated every epoch)
- Used for: resume after crash, continue training
- May not be the best model

best_checkpoint.pth:
- Updated only when val_loss improves
- Used for: final evaluation, deployment
- Always the best model seen
```

### 10.3 Resume Logic

```python
# Key concepts for resume:
# 1. Load checkpoint state dict
# 2. Validate vocabulary matches
# 3. Validate config compatibility
```
- For CLI usage, see README

**Why vocabulary validation?**

If you change dataset (different languages):
- Old checkpoint has vocabulary ["a", "b", "c"]
- New dataset has vocabulary ["x", "y", "z"]
- Model output indices won't match labels
- Solution: fail fast with clear error message

**Edge Cases Handled:**

| Edge Case | Handling |
|----------|---------|
| Missing checkpoint file | Raise FileNotFoundError |
| Vocabulary mismatch | Raise ValueError with details |
| Different image size | Raise ValueError |
| Different model type | Raise ValueError |
| Corrupted checkpoint | Raise RuntimeError |
| Missing optimizer state | Warn, continue with random optimizer |

**What Gets Restored:**

| Component | Restored From |
|-----------|--------------|
| Model weights | `model_state_dict` |
| Optimizer state | `optimizer_state_dict` |
| Epoch number | `epoch` |
| Best val loss | `best_val_loss` |
| Patience counter | `patience_count` |
| Training history | `train_losses`, `val_losses`, etc. |

**What Doesn't Get Restored:**

- Training history (loss curves start fresh, but logs are appended)
- DataLoader state (iterators restart)

---

### 10.4 Checkpoint Best Practices

**When to Use Which Checkpoint:**

| Checkpoint | Use Case |
|------------|----------|
| `latest_checkpoint.pth` | Resume after crash |
| `best_checkpoint.pth` | Final evaluation, deployment |

**Best Practices:**

1. **Always keep latest_checkpoint**: Updated every epoch
2. **Use best_checkpoint for inference**: Best validation performance
3. **Don't delete old checkpoints**: Until new one performs better

**Example Workflow:**

```bash
# Training crashes at epoch 45
# Use latest to resume
python3 train_swin_multilingual.py --resume

# Best was epoch 40
# Use best for inference
python3 inference.py --model ./checkpoints/best_checkpoint.pth
```

---

## 11) Performance Optimization Deep Dive

### 11.1 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
```

**Why needed for OCR?**

```
CTC loss characteristics:
- Can produce very large gradients
- Particularly in early training
- Large gradients -> unstable training -> NaN

How clipping works:
- Compute gradient norm: ||g||
- If ||g|| > max_norm: scale down
- g_clipped = g * (max_norm / ||g||)
```

**Why max_norm=2.0?**

```
Too small (< 1.0):
- Overly aggressive clipping
- Might slow down learning

Too large (> 5.0):
- Insufficient clipping
- Might not prevent explosions

2.0 is a sweet spot:
- OCR-specific value (from literature)
- Allows learning while preventing explosion
- Works well for both Swin and CRNN
```

### 11.2 AMP (Automatic Mixed Precision)

**How AMP works:**

```
Without AMP (FP32):
- All computations in 32-bit floating point
- Standard, accurate, but slower

With AMP (FP16 + FP32):
- Forward pass: FP16 where possible
- Backward pass: FP16 where possible
- FP32 for: loss scaling, optimizer states, master weights
- GradScaler handles FP16 -> FP32 conversion
```

**Code flow:**

```python
scaler = GradScaler()  # Initialize scaler

# Forward pass in FP16
with autocast(device_type='cuda'):
    logits = model(images)
    log_probs = logits.log_softmax(2)
    loss = criterion(log_probs, targets, input_lengths, target_lengths)

# Backward pass with scaling
scaler.scale(loss).backward()
scaler.unscale_(optimizer)  # Unscale for gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
scaler.step(optimizer)     # Unscale and update
scaler.update()            # Update scaler state
```

**Why use AMP?**

| Aspect | FP32 | FP16+AMP |
|--------|------|----------|
| Memory | 100% | ~50-70% |
| Speed | 1x | ~1.5-2x |
| Accuracy | Baseline | ~same |

**When to use AMP:**

- Always beneficial on modern GPUs (Turing+)
- Especially useful when: limited VRAM, want larger batches
- Default: off (for maximum compatibility with older GPUs)

### 11.3 DataLoader Performance

**persistent_workers=True:**

```python
DataLoader(
    dataset,
    num_workers=4,
    persistent_workers=True,  # Workers stay alive between epochs
)
```

**How it works:**
```
Without persistent_workers:
- Epoch 1: spawn workers
- Epoch 1 end: kill workers
- Epoch 2: spawn workers again
- ...repeat

With persistent_workers:
- Epoch 1: spawn workers
- Epoch 1 end: keep workers alive
- Epoch 2: reuse workers (no spawn/kill overhead)
- ...repeat

Savings: ~0.5-2 seconds per epoch (depending on workers)
```

**prefetch_factor=2:**

```
Without prefetch:
- Worker fetches batch 1 -> GPU processes -> Worker fetches batch 2 -> ...

With prefetch_factor=2:
- Worker fetches batch 1, 2, 3 (keeps 2 ahead)
- GPU processes batch 1
- Worker fetches batch 4 (continues prefetching)
- ...

Benefit: hides data loading latency
Best for: fast GPUs that can outpace CPU data loading
```

### 11.4 cuDNN Benchmark

```python
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
```

**How it works:**
- On first run, cuDNN benchmarks different convolution algorithms
- Selects the fastest for your specific input size/model
- Caches the result for subsequent runs

**Trade-offs:**
- First epoch slower (benchmarking)
- Subsequent epochs faster (5-15% speedup)
- Non-deterministic results (different algorithm each run)
- Enable for training, disable for exact reproducibility

---

## 12) ASCII Data Flow Diagrams

### 12.1 Training Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING PIPELINE                                │
└─────────────────────────────────────────────────────────────────────────────┘

   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  LMDB    │────▶│ Dataset  │────▶│DataLoader│────▶│  Model   │
   │  Files   │     │  __getitem__│  │(batches) │     │(Swin/CRNN)│
   └──────────┘     └──────────┘     └──────────┘     └──────────┘
                                                                     │
                                                                     ▼
   ┌──────────┐     ┌──────────┐     ┌──────────┐     ┌──────────┐
   │  Metrics │◀────│  Loss    │◀────│  CTC     │◀────│  Forward │
   │  CER/WER │     │ CTCLoss  │     │ Decoding │     │  Pass    │
   └──────────┘     └──────────┘     └──────────┘     └──────────┘
        │                                    │
        ▼                                    ▼
   ┌──────────┐                        ┌──────────┐
   │  Logger │                        │ Optimizer│
   │ CSV/PNG │                        │  AdamW   │
   └──────────┘                        └──────────┘
```

### 12.2 Swin Model Data Flow

```
INPUT IMAGE                    SWIN BACKBONE                  OUTPUT
─────────────────              ─────────────────              ──────

[batch, 3, 48, 224]           (RGB image)                    
        │                         │                             │
        ▼                         ▼                             │
   ┌─────────────┐          ┌───────────┐                      │
   │   Patch    │          │  Swin     │                      │
   │ Embedding  │─────────▶│ Transformer│                     │
   └─────────────┘          │   Stages  │                      │
        │                   │  1,2,3,4  │                      │
        │                   └───────────┘                      │
        │                         │                             │
        ▼                         ▼                             ▼
[batch, 49, 768]          [batch, 49, 768]            [seq, batch, num_classes]
                              │
                              ▼
                         ┌───────────┐
                         │ Global    │
                         │ Avg Pool  │
                         └───────────┘
                              │
                              ▼
                        [batch, 768]
                              │
                              ▼
                         ┌───────────┐
                         │ Linear    │
                         │ Projection│
                         │ 768→C    │
                         └───────────┘
                              │
                              ▼
                        [batch, C]
                              │
                              ▼
                        ┌───────────┐
                        │ Permute   │
                        │ (T, B, C) │
                        └───────────┘
```

### 12.3 CRNN Model Data Flow

```
INPUT IMAGE                    CNN STACK                      OUTPUT
─────────────────              ──────────                      ──────

[batch, 1, 32, 128]           (grayscale)                    
        │                         │                             │
        ▼                         ▼                             │
   ┌─────────────┐          ┌───────────┐                      │
   │ Conv Block │          │    CNN    │                      │
   │   1-5      │─────────▶│ (6 convs) │                      │
   └─────────────┘          │ + BN +    │                      │
        │                   │ Pooling   │                      │
        │                   └───────────┘                      │
        │                         │                             │
        ▼                         ▼                             ▼
[batch, 512, 1, 31]      [batch, 512, 1, 31]          [seq, batch, num_classes]
   (feature maps)              │                             │
        │                     ▼                             │
        ▼               ┌───────────┐                      │
   ┌───────────┐        │  Squeeze  │                      │
   │  Reshape  │───────▶│ height=1  │                      │
   │ (T, B, F) │        └───────────┘                      │
   └───────────┘              │                             │
        │                     ▼                             │
        ▼               ┌───────────┐                      │
   ┌───────────┐        │ BiLSTM    │                      │
   │ Permute   │───────▶│ Stack     │                      │
   │ (B, T, F) │        │ (2-layer) │                      │
   └───────────┘        └───────────┘                      │
                              │                             │
                              ▼                             │
                        [batch, T, C]                      │
                              │                             │
                              ▼                             │
                        ┌───────────┐                      │
                        │ Permute   │                      │
                        │ (T, B, C) │                      │
                        └───────────┘                      │
```

### 12.4 CTC Decoding Flow

```
MODEL OUTPUT                   CTC DECODING                   FINAL
(log_probs)                    (greedy)                      OUTPUT
─────────────                  ──────────                     ──────

[seq_len, batch, C]                                         
        │                         │                             │
        ▼                         ▼                             │
   ┌─────────────┐          ┌───────────┐                      │
   │ log_softmax │          │   Argmax  │                      │
   └─────────────┘          │  per step │                      │
        │                   └───────────┘                      │
        │                         │                             │
        ▼                         ▼                             │
[seq_len, batch, C]          [batch, seq_len]                 │
                              (class indices)                 │
                                    │                         │
                                    ▼                         │
                              ┌───────────┐                   │
                              │  Collapse │                   │
                              │  repeats  │                   │
                              │ + remove  │                   │
                              │  blanks   │                   │
                              └───────────┘                   │
                                    │                         │
                                    ▼                         │
                              ┌───────────┐                   │
                              │ Map to    │                   │
                              │ characters│                   │
                              └───────────┘                   │
                                    │                         │
                                    ▼                         ▼
                              [batch] strings          ["hello", "world", ...]
```

---

## 13) Research References

For deeper understanding of the models and algorithms used in this project:

### Swin Transformer

- **Paper**: "Swin Transformer: Hierarchical Features for Vision"
- **Authors**: Liu et al., Microsoft Research
- **Key Idea**: Shifted window attention for efficient global context
- **Paper**: https://arxiv.org/abs/2103.14030
- **Official Code**: https://github.com/microsoft/Swin-Transformer

### CRNN (Convolutional Recurrent Neural Network)

- **Paper**: "An End-to-End Trainable Neural Network for Image-based Sequence Recognition"
- **Authors**: Shi et al., Baidu Research
- **Key Idea**: CNN feature extraction + Bidirectional LSTM for sequence modeling
- **Paper**: https://arxiv.org/abs/1507.05717
- **GitHub**: https://github.com/bgshih/crnn

### CTC (Connectionist Temporal Classification)

- **Paper**: "Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with Recurrent Neural Networks"
- **Authors**: Graves et al., IDSIA
- **Key Idea**: No alignment needed, learn to collapse repeats and blanks
- **Paper**: https://arxiv.org/abs/1207.3205
- **PyTorch Docs**: https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html

### Levenshtein Distance (Edit Distance)

- **Original Paper**: "On the Computational Complexity of Metaphonology"
- **Author**: Levenshtein, 1966
- **Note**: The classic dynamic programming algorithm

### Word Error Rate (WER)

- **Standard**: IEEE Speech Recognition Performance Evaluation
- **Note**: Commonly used in ASR (Automatic Speech Recognition)

### LMDB (Lightning Database)

- **Paper**: "LMDB: Lightning Memory-Mapped Database"
- **Author**: Howard, 2015
- **GitHub**: https://github.com/LMDB/lmdb

### AdamW Optimizer

- **Paper**: "Decoupled Weight Decay Regularization"
- **Authors**: Loshchilov & Hutter, 2019
- **Key Idea**: Decoupled weight decay vs L2 regularization
- **Paper**: https://arxiv.org/abs/1711.05101

### Beam Search Decoding

- **Note**: Extension of greedy CTC decoding
- **Key Idea**: Keep top-k paths, choose best final sequence
- **Trade-off**: Better accuracy, slower than greedy
- **Implementation**: PyTorch CTC supports beam decoder

### Gradient Clipping

- **Technique**: Standard deep learning practice
- **Key Idea**: Prevent gradient explosion in recurrent networks
- **Paper**: "Gradient-Based Learning Applied to Document Recognition" (1998)
- **No single seminal paper** - Standard technique since early RNNs

---

### Further Reading

| Topic | Resource | URL |
|-------|----------|-----|
| Attention Mechanisms | "Attention Is All You Need" | https://arxiv.org/abs/1706.03762 |
| Transformer Vision | "Vision Transformers" (ViT) | https://arxiv.org/abs/2010.11922 |
| OCR Survey | "What's in the Dark?" | https://arxiv.org/abs/2203.29579 |
| Deep Learning | Goodfellow et al. | https://www.deeplearningbook.org/ |
| Transformers for OCR | TrOCR (Microsoft) | https://arxiv.org/abs/2109.10242 |
| CRNN Variants | "Robust Scene Text Recognition" | https://arxiv.org/abs/1904.01906 |
