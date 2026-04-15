"""Swin Transformer based multilingual OCR training script."""

import argparse
import csv
import hashlib
import io
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import matplotlib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler, get_worker_info
from torchvision.models import swin_t
from tqdm import tqdm

matplotlib.use("Agg")
from torch.cuda.amp import autocast, GradScaler

import matplotlib.pyplot as plt


@dataclass
class Config:
    img_height: int = 48
    img_width: int = 224
    batch_size: int = 32
    epochs: int = 60
    patience: int = 8
    learning_rate: float = 1e-4
    seed: int = 42
    num_workers: int = 4


def print_table(title: str, headers: List[str], rows: List[List[str]]) -> None:
    """Print a formatted table to console."""
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))

    sep = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    tqdm.write(f"\n{title}")
    tqdm.write(sep)
    tqdm.write(
        "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    )
    tqdm.write(sep)
    for row in rows:
        tqdm.write(
            "| "
            + " | ".join(str(row[i]).ljust(widths[i]) for i in range(len(headers)))
            + " |"
        )
    tqdm.write(sep)


def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def cer(reference: str, hypothesis: str) -> float:
    """Compute Character Error Rate."""
    if len(reference) == 0:
        return 0.0
    return levenshtein_distance(reference, hypothesis) / len(reference)


def wer(reference: str, hypothesis: str) -> float:
    """Compute Word Error Rate."""
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    if len(ref_words) == 0:
        return 0.0
    return levenshtein_distance(ref_words, hyp_words) / len(ref_words)


def exact_match(reference: str, hypothesis: str) -> float:
    """Check if strings are exactly equal."""
    return 1.0 if reference == hypothesis else 0.0


def decode_batch(
    log_probs: torch.Tensor, idx_to_char: Dict[int, str], blank_idx: int = 0
) -> List[str]:
    """Decode CTC log probabilities to text strings."""
    preds = log_probs.permute(1, 0, 2).argmax(2)
    decoded = []
    for b in range(preds.size(0)):
        chars = []
        prev = None
        for t in range(preds.size(1)):
            idx = int(preds[b, t].item())
            if idx != blank_idx and idx != prev and idx in idx_to_char:
                chars.append(idx_to_char[idx])
            prev = idx
        decoded.append("".join(chars))
    return decoded


def encode_labels(
    texts: List[str], char_to_idx: Dict[str, int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode text labels to integer tensor and lengths."""
    encoded = []
    lengths = []
    for text in texts:
        encoded.extend([char_to_idx[ch] for ch in text])
        lengths.append(len(text))
    return torch.tensor(encoded, dtype=torch.long), torch.tensor(
        lengths, dtype=torch.long
    )


def _infer_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Infer column name from candidates (case-insensitive)."""
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _is_abs_path_series(paths: pd.Series) -> pd.Series:
    """Check if paths are absolute."""
    return paths.str.startswith(("/", "\\")) | paths.str.match(
        r"^[A-Za-z]:[\\/]", na=False
    )


def _join_paths(base_dir: Path, rel_paths: pd.Series) -> pd.Series:
    """Convert relative paths to absolute."""
    rel = rel_paths.astype(str).str.replace("\\", "/", regex=False)
    abs_mask = _is_abs_path_series(rel)
    base_abs = str(base_dir.resolve())
    return pd.Series(np.where(abs_mask, rel, base_abs + "/" + rel), index=rel.index)


def _append_skipped_event(
    skipped_log_dir: Optional[Path], split_name: str, image_path: str, exc: Exception
) -> None:
    """Log a failed image load to CSV."""
    if skipped_log_dir is None:
        return

    skipped_log_dir.mkdir(parents=True, exist_ok=True)
    worker = get_worker_info()
    worker_id = worker.id if worker is not None else 0
    pid = os.getpid()
    log_path = skipped_log_dir / f"{split_name}_worker{worker_id}_pid{pid}.csv"
    is_new = not log_path.exists()

    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(
                ["timestamp", "split", "image_path", "error_type", "error_message"]
            )
        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                split_name,
                image_path,
                type(exc).__name__,
                str(exc).replace("\n", " ")[:500],
            ]
        )


def summarize_skipped_images(skipped_log_dir: Path, summary_csv_path: Path):
    """Summarize skipped images from log files."""
    files = sorted(skipped_log_dir.glob("*.csv")) if skipped_log_dir.exists() else []
    if not files:
        return None

    frames = []
    for fp in files:
        try:
            frames.append(pd.read_csv(fp))
        except Exception:
            continue

    if not frames:
        return None

    all_skipped = pd.concat(frames, ignore_index=True)
    by_split = all_skipped["split"].value_counts().to_dict()

    counts = all_skipped.groupby("image_path").size().reset_index(name="count")
    last_err = (
        all_skipped.groupby("image_path")["error_type"]
        .agg(lambda s: s.iloc[-1])
        .reset_index(name="last_error_type")
    )
    summary_df = counts.merge(last_err, on="image_path", how="left").sort_values(
        "count", ascending=False
    )
    summary_df.to_csv(summary_csv_path, index=False)

    return {
        "total_events": int(len(all_skipped)),
        "unique_images": int(all_skipped["image_path"].nunique()),
        "train": int(by_split.get("train", 0)),
        "val": int(by_split.get("val", 0)),
        "test": int(by_split.get("test", 0)),
    }


def load_multilingual_df(
    dataset_root: Path, max_rows_per_tsv: Optional[int] = None
) -> pd.DataFrame:
    """Load multilingual dataset from LMDB or TSV format."""
    # Check if this is an LMDB dataset (has .lmdb files and meta directory)
    lmdb_splits = ["train.lmdb", "val.lmdb", "test.lmdb"]
    meta_dir = dataset_root / "meta"
    is_lmdb_format = meta_dir.exists() and any(
        (dataset_root / split).exists() for split in lmdb_splits
    )

    if is_lmdb_format:
        # LMDB format: return combined dataframe for backward compatibility
        # Actual splits are preserved in the index files
        train_df, val_df, test_df = _load_multilingual_lmdb_df(
            dataset_root, max_rows_per_tsv
        )
        # Combine for now - train function will handle split separation
        all_parts = [train_df]
        if not val_df.empty:
            all_parts.append(val_df)
        if not test_df.empty:
            all_parts.append(test_df)
        return pd.concat(all_parts, ignore_index=True)
    else:
        # Original TSV-based loading
        if not dataset_root.exists():
            raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

        language_dirs = sorted([p for p in dataset_root.iterdir() if p.is_dir()])
        if not language_dirs:
            raise FileNotFoundError(
                f"No language subfolders found under dataset root: {dataset_root}"
            )

        parts = []
        for lang_dir in language_dirs:
            language = lang_dir.name.strip().lower()
            tsv_paths = sorted(lang_dir.rglob("*.tsv"))
            if not tsv_paths:
                continue

            for tsv in tsv_paths:
                read_kwargs = {}
                if max_rows_per_tsv is not None:
                    read_kwargs["nrows"] = max_rows_per_tsv

                try:
                    local_df = pd.read_csv(tsv, sep="\t", **read_kwargs)
                except Exception:
                    local_df = pd.read_csv(tsv, **read_kwargs)

                img_col = _infer_col(
                    list(local_df.columns),
                    ["image_path", "img_path", "image", "path", "filename", "file"],
                )
                txt_col = _infer_col(
                    list(local_df.columns),
                    ["ground_truth", "label", "text", "transcription", "word"],
                )
                if img_col is None or txt_col is None:
                    raise ValueError(
                        f"Could not infer image/text columns in {tsv}. "
                        f"Found columns: {list(local_df.columns)}"
                    )

                part = local_df[[img_col, txt_col]].dropna().copy()
                part.rename(
                    columns={img_col: "image_path", txt_col: "ground_truth"},
                    inplace=True,
                )
                part["image_path"] = _join_paths(tsv.parent, part["image_path"])
                part["ground_truth"] = part["ground_truth"].astype(str)
                part["language"] = language
                part["source_tsv"] = str(tsv)
                part["subgroup"] = tsv.parent.name
                parts.append(
                    part[
                        [
                            "image_path",
                            "ground_truth",
                            "language",
                            "source_tsv",
                            "subgroup",
                        ]
                    ]
                )

        df = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if df.empty:
            raise ValueError(
                f"No valid rows parsed from TSVs under dataset root: {dataset_root}"
            )
        return df


def _load_multilingual_lmdb_df(
    dataset_root: Path, max_rows_per_tsv: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load dataset from LMDB format using index files - preserves original splits"""
    meta_dir = dataset_root / "meta"

    train_df = None
    val_df = None
    test_df = None

    for split_name in ["train", "val", "test"]:
        index_path = meta_dir / f"{split_name}_index.csv"
        if not index_path.exists():
            continue

        df_split = pd.read_csv(index_path)

        # Apply row limit if specified (limit each split proportionally)
        if max_rows_per_tsv is not None and len(df_split) > max_rows_per_tsv:
            df_split = df_split.sample(n=max_rows_per_tsv, random_state=42).reset_index(
                drop=True
            )

        # Add required columns if missing
        if "subgroup" not in df_split.columns:
            df_split["subgroup"] = df_split["language"]

        cols_to_keep = ["key", "ground_truth", "language", "subgroup"]
        df_split = df_split[cols_to_keep]

        if split_name == "train":
            train_df = df_split
        elif split_name == "val":
            val_df = df_split
        elif split_name == "test":
            test_df = df_split

    if train_df is None:
        raise ValueError(f"No LMDB train index file found in {meta_dir}")

    # Return None for val/test if they don't exist
    return (
        train_df,
        val_df if val_df is not None else pd.DataFrame(),
        test_df if test_df is not None else pd.DataFrame(),
    )


def _is_lmdb_format(dataset_root: Path) -> bool:
    """Check if dataset is in LMDB format"""
    lmdb_splits = ["train.lmdb", "val.lmdb", "test.lmdb"]
    meta_dir = dataset_root / "meta"
    return meta_dir.exists() and any(
        (dataset_root / split).exists() for split in lmdb_splits
    )


def _find_dataset_root(user_path: str = "./dataset_lmdb") -> Tuple[Path, bool]:
    """
    Smart dataset root detection with auto-fallback and helpful error.

    Returns:
        Tuple of (dataset_root_path, is_fallback_to_legacy)

    Logic:
    1. If user provided custom path -> use that (detect format automatically)
    2. Else check ./dataset_lmdb exists -> use it
    3. Else check ./dataset exists -> use it (legacy) + note
    4. Else raise helpful error
    """
    default_lmdb = Path("./dataset_lmdb")
    default_legacy = Path("./dataset")
    user_specified = user_path != "./dataset_lmdb"

    if user_specified:
        # User specified custom path - use it directly
        return Path(user_path), False

    # Check default LMDB path first
    if default_lmdb.exists():
        return default_lmdb, False

    # Check legacy TSV path
    if default_legacy.exists():
        return default_legacy, True

    # Neither exists - raise helpful error
    raise FileNotFoundError(
        "Error: Default dataset directories not found!\n"
        "Please run with --dataset-root to specify your dataset location.\n"
        "Example: python train_swin_multilingual.py --dataset-root ./dataset\n"
        "Exiting..."
    )


def load_lmdb_splits(
    dataset_root: Path, max_rows_per_tsv: Optional[int] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load pre-split data from LMDB format - preserves original train/val/test splits"""
    return _load_multilingual_lmdb_df(dataset_root, max_rows_per_tsv)


def _collect_source_tsvs(dataset_root: Path) -> List[Path]:
    """Collect all TSV file paths from dataset root."""
    return sorted(dataset_root.rglob("*.tsv"))


def _manifest_signature(
    dataset_root: Path,
    verify_paths: bool,
    max_rows_per_tsv: Optional[int],
) -> Dict[str, object]:
    """Generate a hash signature for the manifest to detect changes."""
    files = []
    for p in _collect_source_tsvs(dataset_root):
        st = p.stat()
        files.append(
            {
                "path": str(p.resolve()),
                "size": int(st.st_size),
                "mtime_ns": int(st.st_mtime_ns),
            }
        )
    payload = {
        "manifest_version": 3,
        "dataset_root": str(dataset_root.resolve()),
        "verify_paths": bool(verify_paths),
        "max_rows_per_tsv": max_rows_per_tsv,
        "files": files,
    }
    payload_str = json.dumps(payload, sort_keys=True)
    return {
        "hash": hashlib.sha256(payload_str.encode("utf-8")).hexdigest(),
        "payload": payload,
    }


def load_or_build_manifest_cached(
    cache_path: Path,
    dataset_root: Path,
    output_manifest: Optional[Path],
    verify_paths: bool,
    max_rows_per_tsv: Optional[int],
    rebuild_manifest: bool,
) -> pd.DataFrame:
    """Load manifest from cache or build from source files."""
    sig = _manifest_signature(
        dataset_root,
        verify_paths=verify_paths,
        max_rows_per_tsv=max_rows_per_tsv,
    )
    meta_path = cache_path.with_suffix(cache_path.suffix + ".meta.json")

    if not rebuild_manifest and cache_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("hash") == sig["hash"]:
                t0 = time.time()
                df = pd.read_parquet(cache_path)
                tqdm.write(
                    f"Manifest cache loaded from {cache_path} in {time.time() - t0:.2f}s"
                )
                return df
        except Exception:
            pass

    t0 = time.time()
    df = load_multilingual_df(dataset_root, max_rows_per_tsv)
    tqdm.write(f"Manifest built from source in {time.time() - t0:.2f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    meta_path.write_text(json.dumps(sig, indent=2), encoding="utf-8")
    tqdm.write(f"Manifest cache saved to {cache_path}")
    return df


def split_df(df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1):
    """Split dataframe into train/val/test while preserving language proportions."""
    train_parts = []
    val_parts = []
    test_parts = []

    for lang, group in df.groupby("language"):
        group = group.sample(frac=1.0, random_state=42)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_parts.append(group.iloc[:n_train])
        val_parts.append(group.iloc[n_train : n_train + n_val])
        test_parts.append(group.iloc[n_train + n_val :])

    train_df = (
        pd.concat(train_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    )
    test_df = (
        pd.concat(test_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)
    )
    return train_df, val_df, test_df


def cap_dominant_language_ratio(
    df: pd.DataFrame, max_dominant_to_second_ratio: Optional[float]
) -> pd.DataFrame:
    """Limit the ratio between dominant and second language to prevent bias."""
    if max_dominant_to_second_ratio is None:
        return df

    counts = df["language"].value_counts()
    if len(counts) < 2:
        return df

    second_lang = str(counts.index[1])
    dominant_count = int(counts.iloc[0])
    second_count = int(counts.iloc[1])

    if second_count <= 0:
        return df

    max_second = int(dominant_count / max_dominant_to_second_ratio)
    if second_count > max_second:
        second_lang_df = df[df["language"] == second_lang]
        sampled = second_lang_df.sample(n=max_second, random_state=42)
        df = pd.concat([df[df["language"] != second_lang], sampled], ignore_index=True)
        df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    return df


def build_language_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    """Build a weighted sampler that balances languages and subgroups."""
    frame = frame.copy()

    def _subfolder_for_row(row: pd.Series) -> str:
        if pd.notna(row.get("subgroup", None)):
            return str(row.get("subgroup"))
        return "default"

    frame["_subfolder"] = frame.apply(_subfolder_for_row, axis=1)

    lang_counts = frame["language"].value_counts().to_dict()
    sub_counts = frame.groupby(["language", "_subfolder"]).size().to_dict()
    sub_per_lang = frame.groupby("language")["_subfolder"].nunique().to_dict()

    def _w(row: pd.Series) -> float:
        lang = row["language"]
        sub = row["_subfolder"]
        # Equalize language, then equalize subfolders within language, then samples within subfolder.
        return 1.0 / (
            float(lang_counts[lang])
            * float(sub_per_lang[lang])
            * float(sub_counts[(lang, sub)])
        )

    sample_weights = frame.apply(_w, axis=1).to_numpy(dtype=np.float64)
    weights = torch.from_numpy(sample_weights)
    return WeightedRandomSampler(
        weights=weights, num_samples=len(frame), replacement=True
    )


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


class OCRDataset(Dataset):
    def __init__(
        self,
        frame: pd.DataFrame,
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
        self.tf = transforms.Compose(
            [
                transforms.Resize((img_height, img_width)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        if self.lmdb_path is not None and self.lmdb_env is None:
            self.lmdb_env = lmdb.open(
                str(self.lmdb_path),
                readonly=True,
                lock=False,
                readahead=False,
            )

        n = len(self.frame)
        skipped = 0
        for _ in range(n):
            row = self.frame.iloc[idx]
            try:
                if self.lmdb_env is not None:
                    # Load image from LMDB using key
                    with self.lmdb_env.begin() as txn:
                        img_bytes = txn.get(f"{row['key']}.img".encode("utf-8"))
                        if img_bytes is None:
                            raise ValueError(f"Key not found in LMDB: {row['key']}")
                        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    # Load image from file path (legacy TSV format)
                    image = Image.open(row["image_path"]).convert("RGB")
                image = self.tf(image)
                label = row["ground_truth"]
                return image, label, skipped
            except Exception as exc:
                _append_skipped_event(
                    self.skipped_log_dir,
                    self.split_name,
                    str(row.get("key", "")),
                    exc,
                )
                skipped += 1
                idx = (idx + 1) % n

        raise RuntimeError("Could not load any valid image from dataset.")


class SwinCTC(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = swin_t(weights=None)
        self.proj = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone.features(x)
        feat = self.backbone.norm(feat)
        feat = feat.mean(dim=1)
        logits = self.proj(feat)
        return logits.permute(1, 0, 2)


def evaluate(
    model,
    loader,
    criterion,
    char_to_idx,
    idx_to_char,
    device,
    split_name: str = "eval",
    show_progress: bool = False,
):
    """Evaluate model on given data loader."""
    model.eval()
    loss_sum = 0.0
    total = 0
    cer_sum = 0.0
    wer_sum = 0.0
    acc_sum = 0.0
    skipped_sum = 0

    data_iter = loader
    if show_progress:
        data_iter = tqdm(
            loader,
            desc=split_name,
            total=len(loader),
            dynamic_ncols=True,
            leave=True,
            position=0,
            mininterval=0.2,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )

    with torch.no_grad():
        for batch_idx, (images, labels, skipped) in enumerate(data_iter):
            images = images.to(device)
            skipped_sum += int(skipped.sum().item())
            logits = model(images)
            log_probs = logits.log_softmax(2)

            targets, target_lengths = encode_labels(labels, char_to_idx)
            targets = targets.to(device)
            target_lengths = target_lengths.to(device)
            input_lengths = torch.full(
                (images.size(0),), logits.size(0), dtype=torch.long, device=device
            )

            loss = criterion(log_probs, targets, input_lengths, target_lengths)
            loss_sum += float(loss.item()) * images.size(0)

            preds = decode_batch(log_probs, idx_to_char)
            for ref, hyp in zip(labels, preds):
                cer_sum += cer(ref, hyp)
                wer_sum += wer(ref, hyp)
                acc_sum += exact_match(ref, hyp)

            total += images.size(0)

    return {
        "loss": loss_sum / total,
        "cer": cer_sum / total,
        "wer": wer_sum / total,
        "accuracy": acc_sum / total,
        "skipped": skipped_sum,
    }


def run_smoke_test() -> None:
    """Run a quick smoke test to verify model and training setup."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SwinCTC(num_classes=40).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    x = torch.randn(2, 3, 48, 224, device=device)
    logits = model(x)
    log_probs = logits.log_softmax(2)

    targets = torch.tensor([1, 2, 3, 4, 5, 1, 2], dtype=torch.long, device=device)
    target_lengths = torch.tensor([4, 3], dtype=torch.long, device=device)
    input_lengths = torch.full((2,), logits.size(0), dtype=torch.long, device=device)

    loss = criterion(log_probs, targets, input_lengths, target_lengths)
    loss.backward()
    print(
        f"Smoke test OK | loss={loss.item():.4f} | output_shape={tuple(logits.shape)}"
    )


def train(args):
    """Main training function."""
    cfg = Config(
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        img_height=args.img_height,
        img_width=args.img_width,
        num_workers=args.num_workers,
    )

    set_seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        tqdm.write(
            "Run setup | "
            f"device={device} | gpu={torch.cuda.get_device_name(0)} | "
            f"gpu_memory={torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )
    else:
        tqdm.write(f"Run setup | device={device}")
    tqdm.write("")

    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "logs"
    plot_dir = out_dir / "plots"
    skipped_log_dir = log_dir / "skipped_images"
    for p in [ckpt_dir, log_dir, plot_dir]:
        p.mkdir(parents=True, exist_ok=True)

    # Find dataset root with smart auto-detection
    dataset_root, is_fallback = _find_dataset_root(args.dataset_root)
    if is_fallback:
        tqdm.write(
            "Note: dataset_lmdb not found, falling back to legacy dataset folder './dataset'"
        )

    is_lmdb = _is_lmdb_format(dataset_root)

    if is_lmdb:
        # LMDB format: use pre-split data from index files (faster - no re-shuffling)
        tqdm.write("Data | Format: LMDB (using pre-split data from index files)")
        train_df, val_df, test_df = load_lmdb_splits(
            dataset_root, args.max_rows_per_tsv
        )
        # Apply language ratio cap to train split only
        train_df = cap_dominant_language_ratio(
            train_df, args.max_dominant_to_second_ratio
        )
        df = pd.concat(
            [train_df, val_df, test_df], ignore_index=True
        )  # Combined for stats
    else:
        # TSV format: build manifest and split (legacy)
        tqdm.write("Data | Format: TSV (legacy - building manifest from TSV files)")
        manifest_cache_path = (
            Path(args.manifest_cache)
            if args.manifest_cache
            else (out_dir / "manifest_cache.parquet")
        )
        df = load_or_build_manifest_cached(
            cache_path=manifest_cache_path,
            dataset_root=dataset_root,
            output_manifest=Path(args.manifest_out) if args.manifest_out else None,
            verify_paths=args.verify_paths,
            max_rows_per_tsv=args.max_rows_per_tsv,
            rebuild_manifest=args.rebuild_manifest,
        )
        df = cap_dominant_language_ratio(df, args.max_dominant_to_second_ratio)
        train_df, val_df, test_df = split_df(df)

    if args.max_samples is not None:
        train_df = train_df.head(args.max_samples).copy()

    lang_counts = df["language"].value_counts().to_dict()
    tqdm.write(
        "Data | "
        f"total={len(df)} | train={len(train_df)} | val={len(val_df)} | test={len(test_df)} | "
        f"languages={lang_counts}"
    )

    # Build vocabulary from ALL data (train + val + test) to ensure full character coverage
    all_data = pd.concat([train_df, val_df, test_df])
    all_text = "".join(all_data["ground_truth"].tolist())
    chars = sorted(list(set(all_text)))
    char_to_idx = {ch: i + 1 for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for ch, i in char_to_idx.items()}
    num_classes = len(chars) + 1
    tqdm.write(f"Vocab | vocab_no_blank={len(chars)} | num_classes={num_classes}")

    train_ds = OCRDataset(
        train_df,
        cfg.img_height,
        cfg.img_width,
        split_name="train",
        skipped_log_dir=skipped_log_dir,
        lmdb_path=dataset_root / "train.lmdb" if is_lmdb else None,
    )
    val_ds = OCRDataset(
        val_df,
        cfg.img_height,
        cfg.img_width,
        split_name="val",
        skipped_log_dir=skipped_log_dir,
        lmdb_path=dataset_root / "val.lmdb" if is_lmdb and not val_df.empty else None,
    )
    test_ds = OCRDataset(
        test_df,
        cfg.img_height,
        cfg.img_width,
        split_name="test",
        skipped_log_dir=skipped_log_dir,
        lmdb_path=dataset_root / "test.lmdb" if is_lmdb and not test_df.empty else None,
    )

    train_sampler = None
    if args.balance_languages:
        train_sampler = build_language_sampler(train_df)
        tqdm.write("Sampling | language_balanced_sampler=enabled")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=cfg.num_workers,
        worker_init_fn=_init_worker_lmdb,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=_init_worker_lmdb,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        worker_init_fn=_init_worker_lmdb,
        pin_memory=device.type == "cuda",
        persistent_workers=cfg.num_workers > 0,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    model = SwinCTC(num_classes=num_classes).to(device)
    criterion = nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
    )
    scaler = GradScaler() if (args.amp and device.type == "cuda") else None

    start_epoch = 0
    train_losses, val_losses = [], []
    train_cers, val_cers = [], []
    train_wers, val_wers = [], []
    train_accs, val_accs = [], []

    best_val_loss = float("inf")
    patience_count = 0

    resume_path = None
    if args.resume_from:
        resume_path = Path(args.resume_from)
    elif args.resume:
        resume_path = ckpt_dir / "latest_checkpoint.pth"

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")

        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "char_to_idx" in ckpt and ckpt["char_to_idx"] != char_to_idx:
            raise ValueError(
                "Checkpoint vocabulary does not match current training data vocabulary."
            )

        start_epoch = int(ckpt.get("epoch", 0))
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))
        patience_count = int(ckpt.get("patience_count", patience_count))

        train_losses = list(ckpt.get("train_losses", train_losses))
        val_losses = list(ckpt.get("val_losses", val_losses))
        train_cers = list(ckpt.get("train_cers", train_cers))
        val_cers = list(ckpt.get("val_cers", val_cers))
        train_wers = list(ckpt.get("train_wers", train_wers))
        val_wers = list(ckpt.get("val_wers", val_wers))
        train_accs = list(ckpt.get("train_accs", train_accs))
        val_accs = list(ckpt.get("val_accs", val_accs))

        tqdm.write(
            "Resume | "
            f"checkpoint={resume_path} | start_epoch={start_epoch + 1} | "
            f"best_val_loss={best_val_loss:.6f} | patience_count={patience_count}"
        )

    log_path = log_dir / "training_log.csv"
    if start_epoch == 0:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,val_loss,train_cer,val_cer,train_wer,val_wer,train_acc,val_acc,train_skipped,val_skipped\n"
            )
    elif not log_path.exists():
        with open(log_path, "w", encoding="utf-8") as f:
            f.write(
                "epoch,train_loss,val_loss,train_cer,val_cer,train_wer,val_wer,train_acc,val_acc,train_skipped,val_skipped\n"
            )

    train_start = time.time()
    tqdm.write("")
    tqdm.write(f"Training start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    if skipped_log_dir.exists():
        for fp in skipped_log_dir.glob("*.csv"):
            try:
                fp.unlink()
            except Exception:
                pass

    for epoch in range(start_epoch, cfg.epochs):
        epoch_start = time.time()
        model.train()
        tqdm.write("")

        loss_sum = 0.0
        total = 0
        train_cer_sum = 0.0
        train_wer_sum = 0.0
        train_acc_sum = 0.0
        train_skipped_sum = 0

        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}/{cfg.epochs}",
            total=len(train_loader),
            dynamic_ncols=True,
            leave=True,
            position=0,
            mininterval=0.2,
            smoothing=0.1,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        for batch_idx, (images, labels, skipped) in enumerate(progress):
            images = images.to(device)
            train_skipped_sum += int(skipped.sum().item())

            if scaler is not None:
                with autocast():
                    logits = model(images)
                    log_probs = logits.log_softmax(2)
                    targets, target_lengths = encode_labels(labels, char_to_idx)
                    targets = targets.to(device)
                    target_lengths = target_lengths.to(device)
                    input_lengths = torch.full(
                        (images.size(0),), logits.size(0), dtype=torch.long, device=device
                    )
                    loss = criterion(log_probs, targets, input_lengths, target_lengths)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(images)
                log_probs = logits.log_softmax(2)

                targets, target_lengths = encode_labels(labels, char_to_idx)
                targets = targets.to(device)
                target_lengths = target_lengths.to(device)
                input_lengths = torch.full(
                    (images.size(0),), logits.size(0), dtype=torch.long, device=device
                )

                loss = criterion(log_probs, targets, input_lengths, target_lengths)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()

            preds = decode_batch(log_probs.detach(), idx_to_char)
            for ref, hyp in zip(labels, preds):
                train_cer_sum += cer(ref, hyp)
                train_wer_sum += wer(ref, hyp)
                train_acc_sum += exact_match(ref, hyp)

            bs = images.size(0)
            loss_sum += float(loss.item()) * bs
            total += bs

            if batch_idx % 10 == 0:
                progress.set_postfix(loss=f"{loss.item():.4f}")

        progress.close()

        train_metrics = {
            "loss": loss_sum / total,
            "cer": train_cer_sum / total,
            "wer": train_wer_sum / total,
            "accuracy": train_acc_sum / total,
            "skipped": train_skipped_sum,
        }
        tqdm.write("")
        tqdm.write(f"Starting validation for epoch {epoch + 1}/{cfg.epochs}...")
        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            char_to_idx,
            idx_to_char,
            device,
            split_name=f"Validation {epoch + 1}",
            show_progress=True,
        )

        train_losses.append(train_metrics["loss"])
        val_losses.append(val_metrics["loss"])
        train_cers.append(train_metrics["cer"])
        val_cers.append(train_metrics["cer"])
        train_wers.append(train_metrics["wer"])
        val_wers.append(val_metrics["wer"])
        train_accs.append(train_metrics["accuracy"])
        val_accs.append(val_metrics["accuracy"])
        scheduler.step(val_metrics["loss"])

        tqdm.write(
            f"Epoch {epoch + 1}/{cfg.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} | "
            f"train_cer={train_metrics['cer']:.4f} val_cer={val_metrics['cer']:.4f} | "
            f"train_wer={train_metrics['wer']:.4f} val_wer={val_metrics['wer']:.4f} | "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} | "
            f"skipped(train/val)={train_metrics['skipped']}/{val_metrics['skipped']} | "
            f"epoch_time_sec={time.time() - epoch_start:.2f}"
        )

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch + 1},{train_metrics['loss']:.6f},{val_metrics['loss']:.6f},"
                f"{train_metrics['cer']:.6f},{val_metrics['cer']:.6f},"
                f"{train_metrics['wer']:.6f},{val_metrics['wer']:.6f},"
                f"{train_metrics['accuracy']:.6f},{val_metrics['accuracy']:.6f},"
                f"{train_metrics['skipped']},{val_metrics['skipped']}\n"
            )

        ckpt = {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "char_to_idx": char_to_idx,
            "idx_to_char": idx_to_char,
            "best_val_loss": best_val_loss,
            "patience_count": patience_count,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_cers": train_cers,
            "val_cers": val_cers,
            "train_wers": train_wers,
            "val_wers": val_wers,
            "train_accs": train_accs,
            "val_accs": val_accs,
            "config": vars(cfg),
        }
        torch.save(ckpt, ckpt_dir / "latest_checkpoint.pth")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_count = 0
            torch.save(ckpt, ckpt_dir / "best_checkpoint.pth")
            tqdm.write(f"Saved new best checkpoint (val_loss={best_val_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= cfg.patience:
                tqdm.write(f"Early stopping at epoch {epoch + 1}")
                break

    tqdm.write(f"Training done in {(time.time() - train_start) / 60:.2f} minutes")

    tqdm.write("")
    tqdm.write("Training complete.")
    tqdm.write("Starting test evaluation on the best checkpoint...")
    best_ckpt = torch.load(ckpt_dir / "best_checkpoint.pth", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    test_metrics = evaluate(
        model,
        test_loader,
        criterion,
        char_to_idx,
        idx_to_char,
        device,
        split_name="Test",
        show_progress=True,
    )
    tqdm.write(
        "Test metrics | "
        f"loss={test_metrics['loss']:.4f} cer={test_metrics['cer']:.4f} wer={test_metrics['wer']:.4f} "
        f"accuracy={test_metrics['accuracy']:.4f} skipped={test_metrics['skipped']}"
    )

    # Persist final test metrics
    test_metrics_path = log_dir / "test_metrics.csv"
    with open(test_metrics_path, "w", encoding="utf-8") as f:
        f.write("split,loss,cer,wer,accuracy,skipped\n")
        f.write(
            f"test,{test_metrics['loss']:.6f},{test_metrics['cer']:.6f},"
            f"{test_metrics['wer']:.6f},{test_metrics['accuracy']:.6f},{test_metrics['skipped']}\n"
        )

    skipped_summary_path = log_dir / "skipped_images_summary.csv"
    skipped_stats = summarize_skipped_images(skipped_log_dir, skipped_summary_path)

    if skipped_stats and skipped_stats["total_events"] > 0:
        tqdm.write(
            f"Warning: {skipped_stats['total_events']} image load failures ({skipped_stats['unique_images']} unique)"
        )

    # Combined 4-panel plot (same set of metrics as CRNN script)
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(train_cers, label="Train CER")
    plt.plot(val_cers, label="Val CER")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.title("CER Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(train_wers, label="Train WER")
    plt.plot(val_wers, label="Val WER")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.title("WER Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Over Epochs")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(plot_dir / "training_metrics.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "loss.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_cers, label="Train CER")
    plt.plot(val_cers, label="Val CER")
    plt.xlabel("Epoch")
    plt.ylabel("CER")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "cer.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_wers, label="Train WER")
    plt.plot(val_wers, label="Val WER")
    plt.xlabel("Epoch")
    plt.ylabel("WER")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "wer.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(train_accs, label="Train Accuracy")
    plt.plot(val_accs, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / "accuracy.png")
    plt.close()

    # Detailed markdown report
    trained_epochs = len(train_losses)
    best_epoch_idx = int(np.argmin(np.array(val_losses))) if val_losses else 0
    best_epoch = best_epoch_idx + 1
    total_train_seconds = time.time() - train_start
    h = int(total_train_seconds // 3600)
    m = int((total_train_seconds % 3600) // 60)
    s = int(total_train_seconds % 60)

    lang_counts = df["language"].value_counts().to_dict()
    train_lang_counts = train_df["language"].value_counts().to_dict()
    val_lang_counts = val_df["language"].value_counts().to_dict()
    test_lang_counts = test_df["language"].value_counts().to_dict()

    report_path = log_dir / "training_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Swin Multilingual OCR Training Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Run Summary\n\n")
        f.write(f"- Device: `{device}`\n")
        if device.type == "cuda":
            f.write(f"- GPU: `{torch.cuda.get_device_name(0)}`\n")
            f.write(
                f"- GPU memory: `{torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB`\n"
            )
        f.write(
            f"- Total training time: `{h}h {m}m {s}s` ({total_train_seconds:.2f} seconds)\n"
        )
        f.write(f"- Trained epochs: `{trained_epochs}`\n")
        f.write(f"- Early stopping patience: `{cfg.patience}`\n")
        f.write(
            f"- Best validation loss: `{best_val_loss:.6f}` at epoch `{best_epoch}`\n\n"
        )

        # Add skipped images section only if there are any
        if skipped_stats and skipped_stats["total_events"] > 0:
            f.write("## Data Quality (Skipped/Corrupt Images)\n\n")
            f.write(f"- Total skipped events: `{skipped_stats['total_events']}`\n")
            f.write(
                f"- Unique skipped image paths: `{skipped_stats['unique_images']}`\n"
            )
            f.write(f"- Train skipped: `{skipped_stats['train']}`\n")
            f.write(f"- Validation skipped: `{skipped_stats['val']}`\n")
            f.write(f"- Test skipped: `{skipped_stats['test']}`\n\n")

        f.write("## Configuration\n\n")
        f.write(f"- Image size: `{cfg.img_height}x{cfg.img_width}`\n")
        f.write(f"- Batch size: `{cfg.batch_size}`\n")
        f.write(f"- Learning rate: `{cfg.learning_rate}`\n")
        f.write("- Optimizer: `AdamW`\n")
        f.write("- Loss: `CTCLoss`\n")
        f.write(f"- Num workers: `{cfg.num_workers}`\n")
        f.write(f"- Language balancing sampler: `{args.balance_languages}`\n")
        f.write(
            f"- Max dominant:second ratio cap: `{args.max_dominant_to_second_ratio}`\n\n"
        )

        f.write("## Dataset\n\n")
        f.write(f"- Total samples: `{len(df)}`\n")
        f.write(
            f"- Train / Val / Test: `{len(train_df)} / {len(val_df)} / {len(test_df)}`\n"
        )
        f.write(f"- Vocabulary size (without blank): `{len(chars)}`\n")
        f.write(f"- Num classes (with CTC blank): `{num_classes}`\n")
        f.write("\n### Language Distribution\n\n")
        f.write(f"- Full: `{lang_counts}`\n")
        f.write(f"- Train: `{train_lang_counts}`\n")
        f.write(f"- Val: `{val_lang_counts}`\n")
        f.write(f"- Test: `{test_lang_counts}`\n\n")

        f.write("## Final Metrics\n\n")
        f.write("### Training (last epoch)\n\n")
        f.write(f"- Loss: `{train_losses[-1]:.6f}`\n")
        f.write(f"- CER: `{train_cers[-1]:.6f}`\n")
        f.write(f"- WER: `{train_wers[-1]:.6f}`\n")
        f.write(f"- Accuracy: `{train_accs[-1]:.6f}`\n\n")

        f.write("### Validation (last epoch)\n\n")
        f.write(f"- Loss: `{val_losses[-1]:.6f}`\n")
        f.write(f"- CER: `{val_cers[-1]:.6f}`\n")
        f.write(f"- WER: `{val_wers[-1]:.6f}`\n")
        f.write(f"- Accuracy: `{val_accs[-1]:.6f}`\n\n")

        f.write("### Test\n\n")
        f.write(f"- Loss: `{test_metrics['loss']:.6f}`\n")
        f.write(f"- CER: `{test_metrics['cer']:.6f}`\n")
        f.write(f"- WER: `{test_metrics['wer']:.6f}`\n")
        f.write(f"- Accuracy: `{test_metrics['accuracy']:.6f}`\n\n")

        f.write("## Artifacts\n\n")
        f.write("- Checkpoints:\n")
        f.write("  - `checkpoints/best_checkpoint.pth`\n")
        f.write("  - `checkpoints/latest_checkpoint.pth`\n")
        f.write("- Logs:\n")
        f.write("  - `logs/training_log.csv`\n")
        f.write("  - `logs/test_metrics.csv`\n")
        f.write("  - `logs/skipped_images/`\n")
        f.write("  - `logs/skipped_images_summary.csv`\n")
        f.write("  - `logs/training_report.md`\n")
        f.write("- Plots:\n")
        f.write("  - `plots/training_metrics.png` (4-panel)\n")
        f.write("  - `plots/loss.png`\n")
        f.write("  - `plots/cer.png`\n")
        f.write("  - `plots/wer.png`\n")
        f.write("  - `plots/accuracy.png`\n")

    tqdm.write(
        "Run summary | "
        f"best_val_loss={best_val_loss:.6f}(epoch {best_epoch}) | test_loss={test_metrics['loss']:.6f} | "
        f"test_cer={test_metrics['cer']:.6f} | test_wer={test_metrics['wer']:.6f} | "
        f"test_accuracy={test_metrics['accuracy']:.6f} | test_skipped={test_metrics['skipped']}"
    )


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train multilingual OCR with Swin + CTC"
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default="./dataset_lmdb",
        help="Dataset root (default: ./dataset_lmdb for LMDB format)",
    )
    parser.add_argument("--output-dir", type=str, default="./swin_multilingual_outputs")
    parser.add_argument(
        "--manifest-out", type=str, default="./swin_multilingual_outputs/manifest.csv"
    )
    parser.add_argument("--verify-paths", action="store_true")
    parser.add_argument("--img-height", type=int, default=48)
    parser.add_argument("--img-width", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-rows-per-tsv", type=int, default=None)
    parser.add_argument(
        "--manifest-cache",
        type=str,
        default=None,
        help="Parquet cache path for prebuilt multilingual manifest (default: <output-dir>/manifest_cache.parquet)",
    )
    parser.add_argument(
        "--rebuild-manifest",
        action="store_true",
        help="Force rebuild manifest from source TSVs and overwrite cache",
    )
    parser.add_argument(
        "--balance-languages",
        action="store_true",
        help="Use weighted sampling so discovered languages are sampled more evenly in each epoch",
    )
    parser.add_argument(
        "--max-dominant-to-second-ratio",
        type=float,
        default=None,
        help="Optional hard cap before split, e.g. 1.5 keeps dominant language <= 1.5x second-largest language",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from output-dir/checkpoints/latest_checkpoint.pth",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Resume from an explicit checkpoint path",
    )
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision (GPU only)"
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.smoke_test:
        run_smoke_test()
        return

    train(args)


if __name__ == "__main__":
    main()
