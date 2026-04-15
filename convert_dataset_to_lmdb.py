"""Convert multilingual folder dataset to LMDB format."""

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lmdb
import msgpack
import pandas as pd
from tqdm import tqdm


def _make_progress(total: int, desc: str) -> tqdm:
    """Create a tqdm progress bar with custom formatting."""
    return tqdm(
        total=total,
        desc=desc,
        dynamic_ncols=True,
        leave=True,
        position=0,
        mininterval=0.2,
        smoothing=0.1,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )


def _stage_heading(stage_idx: int, stage_total: int, title: str) -> None:
    """Print a stage heading for pipeline progress."""
    tqdm.write("")
    tqdm.write(f"[Stage {stage_idx}/{stage_total}] {title}")


def _infer_col(columns: List[str], candidates: List[str]) -> Optional[str]:
    """Infer column name from a list of candidates (case-insensitive)."""
    lowered = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def _is_abs_path(p: str) -> bool:
    """Check if a path is absolute."""
    s = str(p)
    return s.startswith("/") or (
        len(s) > 2 and s[1] == ":" and (s[2] == "\\" or s[2] == "/")
    )


def _join_local(base_dir: Path, rel_path: str) -> str:
    """Convert relative path to absolute path."""
    rel = str(rel_path).replace("\\", "/")
    if _is_abs_path(rel):
        return str(Path(rel))
    return str((base_dir / rel).resolve())


def _load_manifest_from_dirs(
    source_root: Path,
    max_rows_per_tsv: Optional[int],
    verify_paths: bool,
) -> pd.DataFrame:
    """Load dataset from source directories, parsing TSV files for each language."""
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    language_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()])
    if not language_dirs:
        raise FileNotFoundError(
            f"No language subfolders found under source root: {source_root}"
        )

    parts = []

    for lang_dir in language_dirs:
        language = lang_dir.name.strip().lower()
        tsv_paths = sorted(lang_dir.rglob("*.tsv"))
        if not tsv_paths:
            continue

        pbar_lang = _make_progress(len(tsv_paths), f"Parse TSVs [{language}]")
        for tsv in tsv_paths:
            kwargs = {}
            if max_rows_per_tsv is not None:
                kwargs["nrows"] = max_rows_per_tsv
            try:
                df = pd.read_csv(tsv, sep="\t", **kwargs)
            except Exception:
                df = pd.read_csv(tsv, **kwargs)

            img_col = _infer_col(
                list(df.columns),
                ["image_path", "img_path", "image", "path", "filename", "file"],
            )
            txt_col = _infer_col(
                list(df.columns),
                ["ground_truth", "label", "text", "transcription", "word"],
            )
            if img_col is None or txt_col is None:
                pbar_lang.update(1)
                continue

            part = df[[img_col, txt_col]].dropna().copy()
            part.rename(
                columns={img_col: "image_path", txt_col: "ground_truth"}, inplace=True
            )
            part["image_path"] = (
                part["image_path"]
                .astype(str)
                .apply(lambda p: _join_local(tsv.parent, p))
            )
            part["ground_truth"] = part["ground_truth"].astype(str)
            part["language"] = language
            part["source_tsv"] = str(tsv)
            part["subgroup"] = tsv.parent.name
            parts.append(
                part[
                    ["image_path", "ground_truth", "language", "source_tsv", "subgroup"]
                ]
            )
            pbar_lang.update(1)
        pbar_lang.close()

    if not parts:
        raise ValueError("No valid rows parsed from source dataset")

    frame = pd.concat(parts, ignore_index=True)
    frame = frame[frame["ground_truth"].str.len() > 0].copy()
    if verify_paths:
        pbar_v = _make_progress(len(frame), "Verify image paths")
        mask = []
        for p in frame["image_path"]:
            mask.append(Path(p).exists())
            pbar_v.update(1)
        pbar_v.close()
        frame = frame[pd.Series(mask, index=frame.index)].copy()
    if frame.empty:
        raise ValueError("Manifest is empty after parsing/verification")
    return frame.sample(frac=1.0, random_state=42).reset_index(drop=True)


def _split_per_language(
    df: pd.DataFrame, train_ratio: float = 0.8, val_ratio: float = 0.1
):
    """Split dataframe into train/val/test, maintaining language proportions."""
    tr, va, te = [], [], []
    for _, g in df.groupby("language"):
        g = g.sample(frac=1.0, random_state=42)
        n = len(g)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        tr.append(g.iloc[:n_train])
        va.append(g.iloc[n_train : n_train + n_val])
        te.append(g.iloc[n_train + n_val :])
    return (
        pd.concat(tr).reset_index(drop=True),
        pd.concat(va).reset_index(drop=True),
        pd.concat(te).reset_index(drop=True),
    )


def _put_with_resize(
    env: lmdb.Environment, txn: lmdb.Transaction, key: bytes, value: bytes
) -> lmdb.Transaction:
    """Write key-value to LMDB with automatic resize on full error."""
    while True:
        try:
            txn.put(key, value)
            return txn
        except lmdb.MapFullError:
            txn.abort()
            current = env.info()["map_size"]
            env.set_mapsize(current * 2)
            txn = env.begin(write=True)


def _read_image_bytes(image_path: str) -> Tuple[Optional[bytes], Optional[str]]:
    """Read image bytes. Returns (bytes, None) or (None, error_message) on failure."""
    try:
        return Path(image_path).read_bytes(), None
    except Exception as exc:
        return None, str(exc)


def _write_split_lmdb(
    split_name: str,
    split_df: pd.DataFrame,
    out_root: Path,
    map_size_bytes: int,
    commit_every: int,
    num_workers: int = 4,
) -> Dict[str, int]:
    """Write one split (train/val/test) to LMDB with parallel image reading."""
    lmdb_path = out_root / f"{split_name}.lmdb"
    lmdb_path.mkdir(parents=True, exist_ok=True)
    meta_dir = out_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    index_path = meta_dir / f"{split_name}_index.csv"

    env = lmdb.open(
        str(lmdb_path),
        map_size=map_size_bytes,
        subdir=True,
        readonly=False,
        lock=True,
        readahead=False,
        meminit=False,
        max_dbs=1,
    )

    total_rows = len(split_df)
    written = 0
    skipped = 0
    skipped_by_lang: Dict[str, int] = {}
    progress = _make_progress(total_rows, f"Write {split_name}.lmdb")

    rows_list = list(split_df.itertuples(index=False))

    with open(index_path, "w", encoding="utf-8", newline="") as idx_f:
        idx_w = csv.writer(idx_f)
        idx_w.writerow(["split", "key", "language", "ground_truth", "subgroup"])

        txn = env.begin(write=True)

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(_read_image_bytes, str(row.image_path)): row
                for row in rows_list
            }

            for future in futures:
                row = futures[future]
                image_bytes, error = future.result()

                if image_bytes is None:
                    skipped += 1
                    lang = str(getattr(row, "language", "unknown"))
                    skipped_by_lang[lang] = skipped_by_lang.get(lang, 0) + 1
                    progress.update(1)
                    continue

                key = f"{split_name}-{written:09d}"
                meta_bytes = msgpack.packb(
                    {
                        "language": str(row.language),
                        "source_tsv": str(getattr(row, "source_tsv", "")),
                        "subgroup": str(getattr(row, "subgroup", "")),
                    },
                    use_bin_type=True,
                )

                txn = _put_with_resize(
                    env, txn, f"{key}.img".encode("utf-8"), image_bytes
                )
                txn = _put_with_resize(
                    env,
                    txn,
                    f"{key}.txt".encode("utf-8"),
                    str(row.ground_truth).encode("utf-8"),
                )
                txn = _put_with_resize(
                    env,
                    txn,
                    f"{key}.lang".encode("utf-8"),
                    str(row.language).encode("utf-8"),
                )
                txn = _put_with_resize(
                    env, txn, f"{key}.meta".encode("utf-8"), meta_bytes
                )

                idx_w.writerow(
                    [
                        split_name,
                        key,
                        str(row.language),
                        str(row.ground_truth),
                        str(getattr(row, "subgroup", "")),
                    ]
                )
                written += 1
                progress.update(1)

                if written % commit_every == 0:
                    txn.commit()
                    txn = env.begin(write=True)

        txn = _put_with_resize(env, txn, b"num-samples", str(written).encode("utf-8"))
        txn.commit()

    progress.close()
    env.sync()
    env.close()

    return {
        "written": written,
        "skipped": skipped,
        "skipped_by_language": skipped_by_lang,
    }


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    p = argparse.ArgumentParser(
        description="Convert multilingual folder dataset to LMDB without modifying source data"
    )
    p.add_argument(
        "--source-root",
        type=str,
        default="./dataset",
        help="Dataset root with language subfolders; recursively scans <source-root>/<language>/**/*.tsv",
    )
    p.add_argument(
        "--lmdb-root",
        type=str,
        default="./dataset_lmdb",
        help="Output root for LMDB copy",
    )
    p.add_argument(
        "--verify-paths",
        action="store_true",
        help="Verify source image paths during manifest build",
    )
    p.add_argument("--max-rows-per-tsv", type=int, default=None)
    p.add_argument(
        "--map-size-gb",
        type=int,
        default=256,
        help="Initial map size per split LMDB in GB (auto-expands)",
    )
    p.add_argument(
        "--commit-every",
        type=int,
        default=1000,
        help="Commit LMDB transaction every N written samples",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel workers for image reading",
    )
    return p.parse_args()


def main() -> None:
    """Main pipeline to convert dataset to LMDB format."""
    args = parse_args()
    total_stages = 7
    stage_bar = _make_progress(total_stages, "Pipeline")

    _stage_heading(1, total_stages, "Validate inputs and prepare output")
    source_root = Path(args.source_root)
    if not source_root.exists():
        raise FileNotFoundError(f"Source root not found: {source_root}")

    num_languages = len([p for p in source_root.iterdir() if p.is_dir()])
    out_root = Path(args.lmdb_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "meta").mkdir(parents=True, exist_ok=True)
    tqdm.write(f"Source root | {source_root}")
    tqdm.write(f"LMDB root   | {out_root}")
    tqdm.write(f"Languages   | {num_languages}")
    stage_bar.update(1)

    _stage_heading(2, total_stages, "Build multilingual manifest")
    manifest = _load_manifest_from_dirs(
        source_root=source_root,
        max_rows_per_tsv=args.max_rows_per_tsv,
        verify_paths=args.verify_paths,
    )
    tqdm.write(f"Total images in manifest | {len(manifest):,}")
    stage_bar.update(1)

    _stage_heading(3, total_stages, "Split train/val/test per language")
    train_df, val_df, test_df = _split_per_language(manifest)
    tqdm.write(
        f"Split sizes | train={len(train_df):,} val={len(val_df):,} test={len(test_df):,}"
    )
    stage_bar.update(1)

    map_size_bytes = int(args.map_size_gb) * 1024 * 1024 * 1024

    _stage_heading(4, total_stages, f"Write train LMDB ({len(train_df):,} images)")
    train_stats = _write_split_lmdb(
        "train", train_df, out_root, map_size_bytes, args.commit_every, args.num_workers
    )
    stage_bar.update(1)

    _stage_heading(5, total_stages, f"Write val LMDB ({len(val_df):,} images)")
    val_stats = _write_split_lmdb(
        "val", val_df, out_root, map_size_bytes, args.commit_every, args.num_workers
    )
    stage_bar.update(1)

    _stage_heading(6, total_stages, f"Write test LMDB ({len(test_df):,} images)")
    test_stats = _write_split_lmdb(
        "test", test_df, out_root, map_size_bytes, args.commit_every, args.num_workers
    )
    stage_bar.update(1)

    _stage_heading(7, total_stages, "Write metadata and finalize")
    info = {
        "source_root": str(source_root.resolve()),
        "lmdb_root": str(out_root.resolve()),
        "num_workers": args.num_workers,
        "counts": {
            "full": int(len(manifest)),
            "train": int(train_stats["written"]),
            "val": int(val_stats["written"]),
            "test": int(test_stats["written"]),
        },
        "skipped": {
            "train": int(train_stats["skipped"]),
            "val": int(val_stats["skipped"]),
            "test": int(test_stats["skipped"]),
            "train_by_language": train_stats["skipped_by_language"],
        },
        "language_distribution": {
            "full": manifest["language"].value_counts().to_dict(),
            "train": train_df["language"].value_counts().to_dict(),
            "val": val_df["language"].value_counts().to_dict(),
            "test": test_df["language"].value_counts().to_dict(),
        },
        "lmdb_files": {
            "train": "train.lmdb",
            "val": "val.lmdb",
            "test": "test.lmdb",
        },
    }
    (out_root / "meta" / "dataset_info.json").write_text(
        json.dumps(info, indent=2), encoding="utf-8"
    )
    stage_bar.update(1)
    stage_bar.close()

    tqdm.write(
        "Done | "
        f"train={train_stats['written']:,} val={val_stats['written']:,} test={test_stats['written']:,} | "
        f"skipped(train/val/test)={train_stats['skipped']}/{val_stats['skipped']}/{test_stats['skipped']}"
    )

    if train_stats["skipped"] > 0:
        tqdm.write("Skipped by language:")
        for lang, count in train_stats["skipped_by_language"].items():
            tqdm.write(f"  {lang}: {count}")


if __name__ == "__main__":
    main()
