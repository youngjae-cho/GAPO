#!/usr/bin/env python
# coding=utf-8
"""
Resample noise ratios from an existing flipped dataset (e.g. uf_random_flip_r02).

Guarantees the inclusion property:
    10% flips ⊂ 20% flips ⊂ 30% flips ⊂ 40% flips

For target < base_ratio: un-flip a random subset of currently-flipped samples.
For target > base_ratio: additionally flip a random subset of non-flipped samples.

Usage:
    python scripts/resample_flips_from_base.py \
        --base_dataset datasets/uf_random_flip_r02 \
        --targets 0.1 0.3 0.4 \
        --output_prefix datasets/uf_random_flip \
        --splits train_prefs \
        --seed 42
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

import pyarrow as pa
from datasets import Dataset, DatasetDict, Features, Sequence, Value


def load_legacy_dataset_dict(path: str) -> DatasetDict:
    """Load a DatasetDict saved by an older datasets version (List → Sequence compat)."""
    with open(os.path.join(path, "dataset_dict.json")) as f:
        splits = json.load(f)["splits"]

    dd = DatasetDict()
    for split in splits:
        split_dir = os.path.join(path, split)
        with open(os.path.join(split_dir, "state.json")) as f:
            state = json.load(f)
        arrow_file = os.path.join(split_dir, state["_data_files"][0]["filename"])
        reader = pa.ipc.open_stream(pa.memory_map(arrow_file, "r"))
        table = reader.read_all()
        table = table.replace_schema_metadata({})
        dd[split] = Dataset(table)
    return dd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resample flip ratios from a base flipped dataset."
    )
    parser.add_argument(
        "--base_dataset",
        type=str,
        required=True,
        help="Path to the base dataset (e.g. datasets/uf_random_flip_r02).",
    )
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        required=True,
        help="Target noise ratios, e.g. 0.1 0.3 0.4",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="datasets/uf_random_flip",
        help="Output path prefix. Ratio is appended as _rXX (e.g. _r01, _r03).",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_prefs"],
        help="Splits to resample. Others are copied unchanged.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic resampling.",
    )
    return parser.parse_args()


def swap_chosen_rejected(row: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(row)
    result["chosen"], result["rejected"] = row["rejected"], row["chosen"]
    if "score_chosen" in row and "score_rejected" in row:
        result["score_chosen"], result["score_rejected"] = (
            row["score_rejected"],
            row["score_chosen"],
        )
    return result


def resample_split(
    ds: Dataset,
    target_ratio: float,
    seed: int,
    split_name: str,
) -> Dataset:
    rng = random.Random(seed)
    rows = list(ds)
    n = len(rows)

    flipped_indices = [i for i, r in enumerate(rows) if r["is_flipped"]]
    non_flipped_indices = [i for i, r in enumerate(rows) if not r["is_flipped"]]
    base_flipped = len(flipped_indices)
    target_flipped = round(n * target_ratio)

    if target_flipped <= base_flipped:
        rng.shuffle(flipped_indices)
        to_unflip = flipped_indices[target_flipped:]
        for i in to_unflip:
            rows[i] = swap_chosen_rejected(rows[i])
            rows[i]["is_flipped"] = False
    else:
        n_additional = target_flipped - base_flipped
        rng.shuffle(non_flipped_indices)
        to_flip = non_flipped_indices[:n_additional]
        for i in to_flip:
            rows[i] = swap_chosen_rejected(rows[i])
            rows[i]["is_flipped"] = True

    actual_flipped = sum(1 for r in rows if r["is_flipped"])
    stats = {
        "split": split_name,
        "rows": n,
        "base_flipped": base_flipped,
        "base_ratio": round(base_flipped / max(1, n), 4),
        "target_ratio": target_ratio,
        "target_flipped": target_flipped,
        "actual_flipped": actual_flipped,
        "actual_ratio": round(actual_flipped / max(1, n), 4),
    }
    print(json.dumps(stats))
    return Dataset.from_list(rows)


def ratio_to_suffix(ratio: float) -> str:
    pct = round(ratio * 100)
    return f"_r{pct:02d}"


def main() -> None:
    args = parse_args()

    for t in args.targets:
        if t < 0.0 or t > 1.0:
            raise ValueError(f"Target ratio must be in [0, 1], got {t}")

    base = load_legacy_dataset_dict(args.base_dataset)

    print(f"=== Base dataset: {args.base_dataset} ===")
    for split, ds in base.items():
        has_flip = "is_flipped" in ds.column_names
        if has_flip:
            flipped = sum(1 for r in ds if r["is_flipped"])
            print(f"  {split}: {len(ds)} rows, {flipped} flipped ({flipped/max(1,len(ds)):.2%})")
        else:
            print(f"  {split}: {len(ds)} rows (no is_flipped column)")

    sorted_targets = sorted(args.targets)

    non_flipped_order = None
    flipped_order = None

    for target in sorted_targets:
        suffix = ratio_to_suffix(target)
        output_dir = args.output_prefix + suffix
        print(f"\n=== Generating target={target:.0%} → {output_dir} ===")

        output = DatasetDict()
        for split, ds in base.items():
            if split in args.splits:
                split_seed = args.seed + sum(ord(ch) for ch in split)
                output[split] = resample_split(
                    ds=ds,
                    target_ratio=target,
                    seed=split_seed,
                    split_name=split,
                )
            else:
                output[split] = ds

        os.makedirs(output_dir, exist_ok=True)
        output.save_to_disk(output_dir)
        print(f"Saved → {output_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
