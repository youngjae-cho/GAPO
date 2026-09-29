"""Offline drift diagnostics: run KL / margin / length-drift on existing checkpoints.

Use this when training is already done and checkpoints sit on disk, e.g.

    python scripts/run_drift_offline.py \
        --sft_model mistralai/Mistral-7B-Instruct-v0.2 \
        --checkpoints outputs/mistral-7b-instruct-gapo_noise_len_0.2/checkpoint-116 \
                      outputs/mistral-7b-instruct-gapo_noise_len_0.2/checkpoint-232 \
                      outputs/mistral-7b-instruct-gapo_noise_len_0.2/checkpoint-348 \
                      outputs/mistral-7b-instruct-gapo_noise_len_0.2/checkpoint-466 \
        --dataset princeton-nlp/mistral-instruct-ultrafeedback \
        --output_dir outputs/mistral-7b-instruct-gapo_noise_len_0.2/drift_diagnostics \
        --run_name gapo_noise_len_0.2 \
        --include_sft_step0 \
        --do_generation --gen_n_prompts 200

The held-out / train-subset carve is deterministic on `--heldout_seed`
(default 42) and `--train_subset_seed` (default 43), exactly mirroring the
in-training callback in `run_gapo.py`. Use the SAME seeds across methods so
their diagnostics are computed on identical pairs.
"""

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM

# Local: make `scripts/` (for gapo_trainer, drift_diagnostics) and the
# project root (for the `alignment` package) importable.
THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

from alignment import (  # noqa: E402
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_tokenizer,
)
from drift_diagnostics import compute_kl_and_margin, generate_response_lengths  # noqa: E402
from gapo_trainer import GAPOTrainer  # noqa: E402
from run_gapo import apply_chat_template, MISTRAL_CHAT_TEMPLATE  # noqa: E402
from trl.trainer.utils import DPODataCollatorWithPadding  # noqa: E402

logger = logging.getLogger("drift_offline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
# Trainer-shim: just enough surface for compute_kl_and_margin / generation
# ---------------------------------------------------------------------------
class TrainerShim:
    """Mimics the slice of GAPOTrainer the diagnostic functions call into."""

    # Reuse the exact methods from GAPOTrainer so tokenisation matches training.
    tokenize_row = GAPOTrainer.tokenize_row
    build_tokenized_answer = GAPOTrainer.build_tokenized_answer
    concatenated_inputs = staticmethod(GAPOTrainer.concatenated_inputs)

    def __init__(self, model, tokenizer, *, max_length, max_prompt_length,
                 max_target_length=None, truncation_mode="keep_end",
                 label_pad_token_id=-100, padding_value=None,
                 is_encoder_decoder=False, per_device_eval_batch_size=4,
                 device=None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_prompt_length = max_prompt_length
        self.max_target_length = max_target_length
        self.truncation_mode = truncation_mode
        self.label_pad_token_id = label_pad_token_id
        self.padding_value = padding_value if padding_value is not None else tokenizer.pad_token_id
        self.is_encoder_decoder = is_encoder_decoder
        self.data_collator = DPODataCollatorWithPadding(
            pad_token_id=tokenizer.pad_token_id,
            label_pad_token_id=label_pad_token_id,
            is_encoder_decoder=is_encoder_decoder,
        )
        self.args = SimpleNamespace(per_device_eval_batch_size=per_device_eval_batch_size)
        self.accelerator = SimpleNamespace(
            device=device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")),
            is_main_process=True,
        )


# ---------------------------------------------------------------------------
# Data prep — mirrors run_gapo.main() up to the held-out carve point
# ---------------------------------------------------------------------------
def _prepare_datasets(args, tokenizer):
    raw_datasets = get_datasets(
        DataArguments(dataset_mixer={args.dataset: 1.0}, dataset_splits=["train", "test"]),
        splits=["train", "test"],
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "is_flipped"],
    )

    change_template = "mistral" if "mistral" in args.sft_model.lower() else None
    column_names = list(raw_datasets["train"].features)
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "gapo",
            "auto_insert_empty_system_msg": True,
            "change_template": change_template,
        },
        num_proc=args.preprocessing_num_workers,
        remove_columns=[c for c in column_names if c != "is_flipped"],
        desc="apply_chat_template",
    )
    for split in ["train", "test"]:
        raw_datasets[split] = raw_datasets[split].rename_columns(
            {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
        )
    return raw_datasets


def _carve_splits(train_split: Dataset, args):
    n = len(train_split)
    rng = np.random.default_rng(seed=args.heldout_seed)
    perm = np.arange(n)
    rng.shuffle(perm)
    n_heldout = min(args.heldout_size, n // 4)
    heldout_idx = sorted(perm[:n_heldout].tolist())
    remaining_idx = sorted(perm[n_heldout:].tolist())

    heldout_ds = train_split.select(heldout_idx)
    remaining_ds = train_split.select(remaining_idx)

    rng2 = np.random.default_rng(seed=args.train_subset_seed)
    sub_perm = np.arange(len(remaining_ds))
    rng2.shuffle(sub_perm)
    n_subset = min(args.train_subset_size, len(remaining_ds))
    subset_idx = sorted(sub_perm[:n_subset].tolist())
    subset_ds = remaining_ds.select(subset_idx)

    return heldout_ds, subset_ds, heldout_idx, subset_idx, len(remaining_ds)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_model(path_or_id, *, dtype, attn_impl, device):
    model = AutoModelForCausalLM.from_pretrained(
        path_or_id,
        torch_dtype=dtype,
        attn_implementation=attn_impl,
    )
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def _free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _step_from_path(p: str) -> int:
    name = Path(p).name
    if name.startswith("checkpoint-"):
        try:
            return int(name.split("-", 1)[1])
        except ValueError:
            return -1
    return -1


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_model", required=True,
                   help="HF id or path of the SFT/reference model (e.g. mistralai/Mistral-7B-Instruct-v0.2).")
    p.add_argument("--checkpoints", nargs="+", required=True,
                   help="Checkpoint directories (or HF ids) to evaluate.")
    p.add_argument("--dataset", default="princeton-nlp/mistral-instruct-ultrafeedback")
    p.add_argument("--output_dir", required=True,
                   help="Where summary JSONs are written.")
    p.add_argument("--run_name", default="run")

    p.add_argument("--include_sft_step0", action="store_true",
                   help="Also write a step-0 summary using the SFT model itself as the policy "
                        "(KL=0 baseline, initial margin/length).")
    p.add_argument("--skip_existing", action="store_true",
                   help="Skip a checkpoint if its summary file already exists.")

    p.add_argument("--heldout_size", type=int, default=2000)
    p.add_argument("--train_subset_size", type=int, default=1000)
    p.add_argument("--heldout_seed", type=int, default=0)
    p.add_argument("--train_subset_seed", type=int, default=0)

    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--max_prompt_length", type=int, default=1800)
    p.add_argument("--per_device_eval_batch_size", type=int, default=4)
    p.add_argument("--preprocessing_num_workers", type=int, default=8)

    p.add_argument("--torch_dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32", "auto"])
    p.add_argument("--attn_implementation", default="flash_attention_2")

    p.add_argument("--do_generation", action="store_true")
    p.add_argument("--gen_n_prompts", type=int, default=500)
    p.add_argument("--gen_max_new_tokens", type=int, default=1024)
    p.add_argument("--gen_temperature", type=float, default=0.7)
    p.add_argument("--gen_top_p", type=float, default=0.9)
    p.add_argument("--gen_seed", type=int, default=12345)
    p.add_argument("--gen_batch_size", type=int, default=8,
                   help="Batch size for on-policy generation (set 16-32 on big GPUs).")

    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.torch_dtype == "auto":
        dtype = "auto"
    else:
        dtype = getattr(torch, args.torch_dtype)

    # ---- tokenizer + data --------------------------------------------------
    model_args = ModelArguments(
        model_name_or_path=args.sft_model,
        torch_dtype=args.torch_dtype if args.torch_dtype != "auto" else "auto",
    )
    data_args = DataArguments(
        truncation_side="left",
        preprocessing_num_workers=args.preprocessing_num_workers,
    )
    tokenizer = get_tokenizer(model_args, data_args)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "mistral" in args.sft_model.lower():
        tokenizer.chat_template = MISTRAL_CHAT_TEMPLATE

    raw_datasets = _prepare_datasets(
        SimpleNamespace(
            dataset=args.dataset,
            sft_model=args.sft_model,
            preprocessing_num_workers=args.preprocessing_num_workers,
        ),
        tokenizer,
    )

    heldout_ds, subset_ds, heldout_idx, subset_idx, remaining_n = _carve_splits(
        raw_datasets["train"], args
    )
    logger.info(
        f"[carve] heldout={len(heldout_ds)} train_subset={len(subset_ds)} "
        f"remaining_train={remaining_n} (seeds: heldout={args.heldout_seed}, subset={args.train_subset_seed})"
    )

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w") as f:
        json.dump({
            "heldout_seed": args.heldout_seed,
            "train_subset_seed": args.train_subset_seed,
            "heldout_size": len(heldout_ds),
            "train_subset_size": len(subset_ds),
            "remaining_train_size": remaining_n,
            "heldout_indices_into_original_train": heldout_idx,
            "train_subset_indices_into_remaining_train": subset_idx,
        }, f)
    logger.info(f"[carve] wrote {splits_path}")

    gen_prompts = None
    if args.do_generation:
        n_gen = min(args.gen_n_prompts, len(heldout_ds))
        gen_prompts = [heldout_ds[i]["prompt"] for i in range(n_gen)]

    # ---- Tokenise BEFORE loading any model onto GPU ------------------------
    # tokenize_row depends only on the tokenizer; doing this first avoids
    # `.map(num_proc=...)` forking workers that try to re-init CUDA on the
    # SFT model. We cap num_proc here because the parent process has already
    # imported very heavy modules (deepspeed, trl, transformers, gapo_trainer);
    # forking 8 workers can spend 30-60s in dill startup before any tokenisation
    # actually happens, which makes the script look hung.
    tok_shim = TrainerShim(
        model=None, tokenizer=tokenizer,
        max_length=args.max_length, max_prompt_length=args.max_prompt_length,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        device=torch.device("cpu"),
    )
    tok_nproc = max(1, min(args.preprocessing_num_workers, 4))
    logger.info(
        f"[tokenize] starting heldout ({len(heldout_ds)} pairs, num_proc={tok_nproc}); "
        f"first ~30s is fork warm-up before the progress bar appears."
    )
    tok_heldout = heldout_ds.map(tok_shim.tokenize_row,
                                 num_proc=tok_nproc,
                                 desc="tokenize heldout")
    logger.info(f"[tokenize] heldout done. Starting train_subset ({len(subset_ds)} pairs).")
    tok_subset = subset_ds.map(tok_shim.tokenize_row,
                               num_proc=tok_nproc,
                               desc="tokenize train_subset")
    logger.info("[tokenize] done.")

    # ---- SFT (loaded once, kept on device) ---------------------------------
    logger.info(f"[load] SFT: {args.sft_model}")
    sft_model = _load_model(
        args.sft_model, dtype=dtype, attn_impl=args.attn_implementation, device=device
    )

    # ---- Run on each checkpoint -------------------------------------------
    targets = []
    if args.include_sft_step0:
        targets.append(("__SFT__", 0))
    for c in args.checkpoints:
        targets.append((c, _step_from_path(c)))

    for ckpt_path, step in targets:
        out_path = output_dir / f"checkpoint_{int(step)}_summary.json"
        if args.skip_existing and out_path.exists():
            logger.info(f"[skip] {out_path} exists")
            continue

        if ckpt_path == "__SFT__":
            logger.info(f"[run]  step={step} policy=SFT (initial baseline)")
            policy_model = sft_model  # alias — KL will be ~0
            shim = TrainerShim(
                model=policy_model, tokenizer=tokenizer,
                max_length=args.max_length, max_prompt_length=args.max_prompt_length,
                per_device_eval_batch_size=args.per_device_eval_batch_size, device=device,
            )
        else:
            logger.info(f"[load] policy: {ckpt_path}")
            policy_model = _load_model(
                ckpt_path, dtype=dtype, attn_impl=args.attn_implementation, device=device
            )
            shim = TrainerShim(
                model=policy_model, tokenizer=tokenizer,
                max_length=args.max_length, max_prompt_length=args.max_prompt_length,
                per_device_eval_batch_size=args.per_device_eval_batch_size, device=device,
            )

        summary = {
            "run_name": args.run_name,
            "checkpoint_path": ckpt_path if ckpt_path != "__SFT__" else args.sft_model,
            "global_step": int(step),
        }

        heldout = compute_kl_and_margin(shim, tok_heldout, sft_model,
                                        desc=f"step={step} heldout")
        summary["heldout"] = heldout

        train_eval = compute_kl_and_margin(shim, tok_subset, sft_model,
                                           desc=f"step={step} train_subset")
        summary["train_subset"] = train_eval

        summary["margin_gap_train_minus_heldout"] = (
            train_eval["margin_mean"] - heldout["margin_mean"]
        )

        if args.do_generation and gen_prompts:
            gen = generate_response_lengths(
                shim,
                gen_prompts,
                max_new_tokens=args.gen_max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.gen_top_p,
                base_seed=args.gen_seed,
                batch_size=args.gen_batch_size,
            )
            summary["generation"] = {k: v for k, v in gen.items() if k != "records"}
            with open(output_dir / f"checkpoint_{int(step)}_generations.jsonl", "w") as f:
                for r in gen["records"]:
                    f.write(json.dumps(r) + "\n")

        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(
            f"[done] step={step} kl={heldout['kl_mean']:.4f} "
            f"kl_k1={heldout['kl_k1_mean']:.4f} "
            f"margin_h={heldout['margin_mean']:.4f} "
            f"margin_t={train_eval['margin_mean']:.4f} "
            f"gap={summary['margin_gap_train_minus_heldout']:.4f} "
            f"-> {out_path}"
        )

        if ckpt_path != "__SFT__":
            _free(policy_model)


if __name__ == "__main__":
    main()
