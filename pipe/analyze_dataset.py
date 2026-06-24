"""
Batch-analyse every image referenced in a pairs CSV.

Reads a CSV with the columns produced by BuildPairsDataset.ipynb:
    path_a, path_b, label, vehicle_id_a, vehicle_id_b

Builds the set of *unique* images across both columns, then runs the full
pipeline on each and writes two aligned artefacts:

    --out      analysis_results.csv      one row per unique image with metadata
    --out-emb  analysis_embeddings.npy   (N, 256) float32, aligned with CSV row order

Resumable: rerun with --resume to skip images already processed in the
existing output CSV (useful for long runs).

Example:
    python analyze_dataset.py \
        --csv analysed_data/pairs_test.csv \
        --images-root /path/to/dataset/pairs \
        --out analysed_data/test_analysis.csv \
        --out-emb analysed_data/test_embeddings.npy
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from pipeline import process_image


CSV_FIELDS = [
    "path",
    "vehicle_id",
    "detected",
    "detection_conf",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "color_name",
    "color_hex",
    "color_share",
    "num_plates",
    "plates",
    "embedding_idx",            # row index in the .npy file
]


def collect_unique_images(pairs_csv: Path) -> pd.DataFrame:
    """Union the path_a/path_b columns, dedup by path, keep one vehicle_id per path."""
    df = pd.read_csv(pairs_csv)
    side_a = df[["path_a", "vehicle_id_a"]].rename(
        columns={"path_a": "path", "vehicle_id_a": "vehicle_id"}
    )
    side_b = df[["path_b", "vehicle_id_b"]].rename(
        columns={"path_b": "path", "vehicle_id_b": "vehicle_id"}
    )
    both = pd.concat([side_a, side_b], ignore_index=True)
    return both.drop_duplicates(subset=["path"]).reset_index(drop=True)


def build_record(path: str, vehicle_id, profile, emb_idx: int) -> dict:
    """Turn a VehicleProfile into a flat dict ready for csv.DictWriter."""
    rec = {f: "" for f in CSV_FIELDS}
    rec["path"]          = path
    rec["vehicle_id"]    = vehicle_id
    rec["embedding_idx"] = emb_idx

    if profile is None:
        rec["detected"]       = False
        rec["detection_conf"] = 0.0
        rec["num_plates"]     = 0
        rec["plates"]         = "[]"
        return rec

    rec["detected"]       = profile.detected
    rec["detection_conf"] = round(profile.detection_conf, 4)

    if profile.bbox is not None:
        x1, y1, x2, y2 = profile.bbox
        rec["bbox_x1"], rec["bbox_y1"], rec["bbox_x2"], rec["bbox_y2"] = x1, y1, x2, y2

    if profile.color is not None:
        rec["color_name"]  = profile.color["name"]
        rec["color_hex"]   = profile.color["hex"]
        rec["color_share"] = round(profile.color["share"], 4)

    plate_texts = [p["text"] for p in profile.plates]
    rec["num_plates"] = len(plate_texts)
    rec["plates"]     = json.dumps(plate_texts, ensure_ascii=False)
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv",          required=True, type=Path, help="input pairs CSV")
    ap.add_argument("--images-root",  required=True, type=Path, help="base dir for relative image paths")
    ap.add_argument("--out",          default=Path("analysis_results.csv"), type=Path)
    ap.add_argument("--out-emb",      default=Path("analysis_embeddings.npy"), type=Path)
    ap.add_argument("--resume",       action="store_true", help="skip images already in --out")
    ap.add_argument("--limit",        type=int, default=None, help="process only first N images (debug)")
    args = ap.parse_args()

    if not args.csv.exists():
        print(f"input CSV not found: {args.csv}", file=sys.stderr)
        return 1
    if not args.images_root.exists():
        print(f"images root not found: {args.images_root}", file=sys.stderr)
        return 1

    todo = collect_unique_images(args.csv)
    print(f"unique images in {args.csv.name}: {len(todo)}")

    # ---------------------------------------------------------------
    # Resume handling
    # ---------------------------------------------------------------
    done_paths: set[str] = set()
    existing_emb: np.ndarray | None = None
    if args.resume and args.out.exists():
        prev = pd.read_csv(args.out)
        done_paths = set(prev["path"].astype(str).tolist())
        if args.out_emb.exists():
            existing_emb = np.load(args.out_emb)
        print(f"resuming: {len(done_paths)} already processed, "
              f"existing_emb={None if existing_emb is None else existing_emb.shape}")
    else:
        # fresh run — wipe outputs
        if args.out.exists():     args.out.unlink()
        if args.out_emb.exists(): args.out_emb.unlink()

    remaining = todo[~todo["path"].isin(done_paths)].reset_index(drop=True)
    if args.limit:
        remaining = remaining.head(args.limit)
    if remaining.empty:
        print("nothing to do.")
        return 0

    # ---------------------------------------------------------------
    # Process
    # ---------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out_emb.parent.mkdir(parents=True, exist_ok=True)

    new_embeddings: list[np.ndarray] = []
    base_idx = 0 if existing_emb is None else int(existing_emb.shape[0])

    write_header = not (args.resume and args.out.exists())
    mode = "a" if (args.resume and args.out.exists()) else "w"

    with open(args.out, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if write_header:
            writer.writeheader()

        for i, row in tqdm(remaining.iterrows(), total=len(remaining), desc="analyzing"):
            rel_path = str(row["path"])
            img_path = args.images_root / rel_path

            profile = None
            try:
                if img_path.exists():
                    profile = process_image(img_path)
                else:
                    print(f"\n[missing] {img_path}", file=sys.stderr)
            except Exception as e:
                print(f"\n[error] {rel_path}: {e}", file=sys.stderr)

            emb = (
                profile.embedding
                if (profile is not None and profile.embedding is not None)
                else np.zeros(256, dtype=np.float32)
            )
            new_embeddings.append(emb.astype(np.float32))

            record = build_record(rel_path, row["vehicle_id"], profile, base_idx + len(new_embeddings) - 1)
            writer.writerow(record)
            f.flush()

    # ---------------------------------------------------------------
    # Save embeddings (combined with previous if resuming)
    # ---------------------------------------------------------------
    new_arr = np.stack(new_embeddings) if new_embeddings else np.zeros((0, 256), dtype=np.float32)
    if existing_emb is not None and existing_emb.shape[0] > 0:
        combined = np.concatenate([existing_emb, new_arr], axis=0)
    else:
        combined = new_arr
    np.save(args.out_emb, combined)

    print(f"\nDone. Wrote {args.out}  ({combined.shape[0]} rows)")
    print(f"      Wrote {args.out_emb}  shape={combined.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
