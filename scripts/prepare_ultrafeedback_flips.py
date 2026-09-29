#!/usr/bin/env python
# coding=utf-8
"""
Prepare flipped preference data from HuggingFaceH4/ultrafeedback_binarized.

Supports:
1) random flip: flip chosen/rejected with probability `ratio`
2) length_dependent flip: only when chosen response is longer than rejected,
   then flip with probability `ratio`
"""

import argparse
import json
import os
import random
from typing import Any, Dict, List

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flip chosen/rejected responses and save dataset.")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/mistral-instruct-ultrafeedback",
        help="HF dataset name or local dataset path accepted by datasets.load_dataset.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train"],
        help="Splits to transform. Recommended: only train_* splits.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["random", "length_dependent"],
        default="random",
        help="Flip strategy.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        required=True,
        help="Flip probability in [0, 1].",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic flipping.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output path for DatasetDict.save_to_disk().",
    )
    parser.add_argument(
        "--copy_other_splits",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-target splits unchanged (default: true).",
    )
    parser.add_argument(
        "--allow_test_split_flip",
        action="store_true",
        help="Explicitly allow flipping test/validation splits.",
    )
    return parser.parse_args()


def validate_ratio(ratio: float) -> None:
    if ratio < 0.0 or ratio > 1.0:
        raise ValueError(f"`ratio` must be in [0, 1], got: {ratio}")


def extract_last_assistant_content(messages: List[Dict[str, Any]]) -> str:
    # UltraFeedback uses OpenAI-style message list. We use the final assistant turn.
    for msg in reversed(messages):
        if msg.get("role") == "assistant":
            return str(msg.get("content", ""))
    # Fallback: if role is missing/unexpected, use the last message content.
    if messages:
        return str(messages[-1].get("content", ""))
    return ""


def should_flip(
    example: Dict[str, Any],
    mode: str,
    ratio: float,
    rng: random.Random,
) -> bool:
    if mode == "random":
        return rng.random() < ratio

    if mode == "length_dependent":
        chosen_text = extract_last_assistant_content(example["chosen"])
        rejected_text = extract_last_assistant_content(example["rejected"])
        chosen_is_longer = len(chosen_text) < len(rejected_text)
        return chosen_is_longer and (rng.random() < ratio)

    raise ValueError(f"Unknown mode: {mode}")


def flip_example(example: Dict[str, Any], do_flip: bool) -> Dict[str, Any]:
    result = dict(example)
    result["is_flipped"] = bool(do_flip)
    if not do_flip:
        return result

    result["chosen"], result["rejected"] = example["rejected"], example["chosen"]
    if "score_chosen" in example and "score_rejected" in example:
        result["score_chosen"], result["score_rejected"] = example["score_rejected"], example["score_chosen"]
    return result


def transform_split(dataset: Dataset, mode: str, ratio: float, seed: int, split_name: str) -> Dataset:
    rng = random.Random(seed)
    transformed_rows = []
    flipped_count = 0
    for example in dataset:
        do_flip = should_flip(example, mode=mode, ratio=ratio, rng=rng)
        if do_flip:
            flipped_count += 1
        transformed_rows.append(flip_example(example, do_flip))

    from datasets import Features, Sequence, Value
    original_features = dataset.features.copy()
    original_features["is_flipped"] = Value("bool")
    out_dataset = Dataset.from_list(transformed_rows, features=Features(original_features))
    stats = {
        "split": split_name,
        "rows": len(dataset),
        "mode": mode,
        "ratio": ratio,
        "flipped_count": flipped_count,
        "flipped_fraction": (flipped_count / max(1, len(dataset))),
    }
    print(json.dumps(stats, ensure_ascii=True))
    return out_dataset


def print_dataset_structure(dataset_dict: DatasetDict) -> None:
    print("=== Dataset structure ===")
    for split, ds in dataset_dict.items():
        print(f"- {split}: {len(ds)} rows")
        print(f"  features: {list(ds.features.keys())}")
    # Show one sample's key structure for quick inspection.
    first_split = next(iter(dataset_dict.keys()))
    sample = dataset_dict[first_split][0]
    print("=== Sample field types ===")
    for key, value in sample.items():
        if isinstance(value, list):
            first_type = type(value[0]).__name__ if value else "None"
            print(f"- {key}: list(len={len(value)}, first_type={first_type})")
        else:
            print(f"- {key}: {type(value).__name__}")


def load_dataset_dict(source: str) -> DatasetDict:
    if os.path.isdir(source):
        loaded = load_from_disk(source)
        if not isinstance(loaded, DatasetDict):
            raise ValueError(f"Expected DatasetDict at '{source}', got {type(loaded)}")
        return loaded
    return load_dataset(source)


def main() -> None:
    args = parse_args()
    validate_ratio(args.ratio)
    os.makedirs(args.output_dir, exist_ok=True)

    target_test_like_splits = [s for s in args.splits if ("test" in s.lower() or "val" in s.lower())]
    if target_test_like_splits and not args.allow_test_split_flip:
        raise ValueError(
            f"Refusing to flip eval-like splits: {target_test_like_splits}. "
            "Use --allow_test_split_flip if you really want this."
        )

    dataset_dict = load_dataset_dict(args.dataset_name)
    print_dataset_structure(dataset_dict)

    output = DatasetDict()
    for split, ds in dataset_dict.items():
        if split in args.splits:
            split_seed = args.seed + sum(ord(ch) for ch in split)
            output[split] = transform_split(
                dataset=ds,
                mode=args.mode,
                ratio=args.ratio,
                seed=split_seed,
                split_name=split,
            )
        else:
            if args.copy_other_splits:
                output[split] = ds
            print(json.dumps({"split": split, "rows": len(ds), "skipped": True, "reason": "not_in_target_splits"}, ensure_ascii=True))

    if len(output) == 0:
        raise ValueError(
            "No output splits were produced. Check --splits values or use --copy_other_splits."
        )

    output.save_to_disk(args.output_dir)
    print(f"Saved transformed dataset to: {args.output_dir}")


if __name__ == "__main__":
    main()
