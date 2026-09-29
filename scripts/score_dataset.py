#!/usr/bin/env python
"""
One-pass per-example scoring of a preference dataset under a base model
that mirrors GAPO's adversarial-anchor mechanism.

For each batch we reproduce `GAPOTrainer.concatenated_forward_perturbed`:
    1. clean forward            -> chosen/rejected logps   (m_clean = c - r)
    2. SimPO-style backward     -> per-parameter gradient
    3. SAM weight perturbation  p <- p + rho * p^2 * g / ||p * g||
    4. perturbed forward        -> chosen/rejected logps_perturbed (m_pert)
    5. weight restore (subtract the saved delta)

The Anchor Gap Gamma is then
    Gamma_i = beta * ( (m_clean_i - m_pert_i) - gamma_beta_ratio )
("the GAPO loss logit when the per-example anchor is the perturbed margin").

Usage
-----
    # Single-GPU only. Multi-process distributed scoring is not supported
    # (each rank would need to gather/dedupe and write one shared npz; the
    # SAM perturbation also needs a single, unsharded backward pass).
    CUDA_VISIBLE_DEVICES=0 python scripts/score_dataset.py \\
        training_configs/mistral-instruct-uf-simpo.yaml \\
        --output_npz=experiments/scores/mistral_uf_base_T1.npz \\
        --score_split=train --rho=0.1

    # Quick run on a deterministic 1k subset (matches alpha_iB.py's seed=12345):
    CUDA_VISIBLE_DEVICES=0 python scripts/score_dataset.py \\
        training_configs/mistral-instruct-uf-simpo.yaml \\
        --output_npz=experiments/scores/mistral_uf_base_T1_n1k.npz \\
        --max_samples=1000 --subset_seed=12345

    # If you must use accelerate, force a single process:
    accelerate launch --num_processes=1 scripts/score_dataset.py ...

Outputs an npz with the schema:
    sample_indices:    (N,)  int64
    is_flipped:        (N,)  int64
    margins:           (1, N) float64   # clean (chosen - rejected) logp diff
    margins_perturbed: (1, N) float64   # SAM-perturbed margin
    reward_margins:    (1, N) float64   # beta * margins
    simpo_losses:      (1, N) float64   # at clean weights
    chosen_logps:      (1, N) float64
    rejected_logps:    (1, N) float64
    steps:             (1,)  int64     # [0]
    checkpoint_paths:  (1,)  str       # ["base_model"]

Sidecar `<output>.json` records {beta, gamma_beta_ratio, rho, ...} so that
`select_subset.py --score anchor_gap` can pull them automatically.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Literal

# Make `alignment/` (sibling of scripts/) importable even when launched as
# `accelerate launch scripts/score_dataset.py` (which sets sys.path[0] to
# scripts/, not the repo root).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from transformers import set_seed

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    get_datasets,
    get_kbit_device_map,
    get_quantization_config,
    get_tokenizer,
)

# Reuse the same chat-template + dataset prep used by run_simpo.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_simpo import apply_chat_template  # noqa: E402
from simpo_config import SimPOConfig  # noqa: E402
from simpo_trainer import SimPOTrainer  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class ScoreArguments:
    output_npz: Optional[str] = field(
        default=None,
        metadata={"help": "Path of the output npz to write per-example scores to."},
    )
    score_split: str = field(
        default="train",
        metadata={"help": "Which split to score (key into raw_datasets after loading)."},
    )
    score_batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "Override per_device_eval_batch_size for scoring (else uses training_args)."},
    )
    rho: float = field(
        default=0.1,
        metadata={"help": "SAM perturbation radius (matches GAPO's args.rho; default 0.1)."},
    )
    max_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, score only this many examples (deterministic subset). "
                    "Useful for sweeping checkpoints quickly. None means score the full split."
        },
    )
    subset_seed: int = field(
        default=12345,
        metadata={
            "help": "Seed for deterministic sub-sampling when --max_samples is set. "
                    "Default 12345 matches experiments/alpha_iB.py for cross-experiment parity."
        },
    )


def main():
    parser = H4ArgumentParser((ModelArguments, DataArguments, SimPOConfig, ScoreArguments))
    model_args, data_args, training_args, score_args = parser.parse()

    if not score_args.output_npz:
        sys.exit("--output_npz is required")

    # Refuse to run under multi-process distributed launch. The script writes
    # a single consolidated npz and runs the SAM perturbation on one model
    # copy; there is no gather/dedupe path for ranks > 0.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        sys.exit(
            f"score_dataset.py is single-process only (got WORLD_SIZE={world_size}). "
            f"Re-launch as `python scripts/score_dataset.py ...` (set CUDA_VISIBLE_DEVICES "
            f"to pick a single GPU) or `accelerate launch --num_processes=1 ...`."
        )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)

    set_seed(training_args.seed)

    # Force-disable training side-effects we don't want during scoring.
    training_args.do_train = False
    training_args.do_eval = False
    training_args.report_to = []
    if score_args.score_batch_size is not None:
        training_args.per_device_eval_batch_size = score_args.score_batch_size

    # Load datasets (same call as run_simpo.py)
    raw_datasets = get_datasets(
        data_args,
        splits=data_args.dataset_splits,
        configs=data_args.dataset_configs,
        columns_to_keep=["messages", "chosen", "rejected", "prompt", "completion", "label", "is_flipped"],
    )

    if score_args.score_split not in raw_datasets:
        sys.exit(f"score_split={score_args.score_split} not in {list(raw_datasets.keys())}")

    target = raw_datasets[score_args.score_split]
    n_total = len(target)
    logger.info(f"Loaded split '{score_args.score_split}': {n_total} examples")

    # Optional deterministic sub-sampling. We do this BEFORE chat template /
    # tokenization so we don't pay tokenization cost on rows we will discard.
    if score_args.max_samples is not None and score_args.max_samples < n_total:
        rng = np.random.default_rng(score_args.subset_seed)
        original_indices = np.sort(
            rng.choice(n_total, size=int(score_args.max_samples), replace=False)
        ).astype(np.int64)
        target = target.select(original_indices.tolist())
        logger.info(
            f"Subsampled to {len(target)} examples (seed={score_args.subset_seed}); "
            f"first 5 indices in dataset: {original_indices[:5].tolist()}"
        )
    else:
        original_indices = np.arange(n_total, dtype=np.int64)

    # Tokenizer
    data_args.truncation_side = "left"
    tokenizer = get_tokenizer(model_args, data_args)

    if "mistral" in model_args.model_name_or_path.lower():
        change_template = "mistral"
    else:
        change_template = None

    # Apply chat template (same as run_simpo.py)
    column_names = list(target.features)
    target = target.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "gapo",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
            "change_template": change_template,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=[col for col in column_names if col != "is_flipped"],
        desc="Formatting comparisons with prompt template",
    )
    target = target.rename_columns(
        {"text_prompt": "prompt", "text_chosen": "chosen", "text_rejected": "rejected"}
    )

    # Stash is_flipped before the trainer's tokenize step, since DPODataCollator's
    # tokenize_row does not preserve it. We index back by sample order.
    if "is_flipped" in target.column_names:
        is_flipped = np.asarray(target["is_flipped"], dtype=np.int64)
        target = target.remove_columns(["is_flipped"])
    else:
        is_flipped = np.zeros(len(target), dtype=np.int64)

    # Model
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None]
        else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        # Pass the STRING dtype ("bfloat16"/"auto"/None); SimPOTrainer.__init__ does its own
        # getattr(torch, ...) conversion. Passing a torch.dtype object here double-converts and
        # raises "getattr(): attribute name must be string".
        torch_dtype=model_args.torch_dtype,
        use_cache=False,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
        attn_implementation=model_args.attn_implementation,
    )
    training_args.model_init_kwargs = model_kwargs

    # We need a *non-empty* train_dataset to instantiate SimPOTrainer; reuse target.
    # The trainer only needs it to wire collator/tokenize; we won't call .train().
    trainer = SimPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=target,           # tokenized in __init__
        eval_dataset=None,
        tokenizer=tokenizer,
        peft_config=None,
    )

    model = trainer.model
    # The SAM perturbation requires a backward pass. We run the model in train()
    # mode (dropout has been replaced with Identity by SimPOTrainer when
    # disable_dropout=True, which it is by default).
    model.train()
    device = trainer.accelerator.device

    # SequentialSampler-backed loader; deterministic iteration order matches `target`.
    dataloader = trainer.get_eval_dataloader(eval_dataset=trainer.train_dataset)

    def inner_named_params():
        """Iterate parameters that GAPOTrainer perturbs.

        GAPOTrainer iterates `model.module.model.named_parameters()` -- i.e. the
        transformer body (LlamaModel), excluding lm_head. We mirror that here
        whether or not the model is wrapped by Accelerate/DDP.
        """
        m = model
        if hasattr(m, "module"):
            m = m.module
        if hasattr(m, "model"):
            m = m.model
        return list(m.named_parameters())

    n = len(target)
    chosen_logps_all = np.zeros(n, dtype=np.float64)
    rejected_logps_all = np.zeros(n, dtype=np.float64)
    margins_all = np.zeros(n, dtype=np.float64)
    margins_perturbed_all = np.zeros(n, dtype=np.float64)
    reward_margins_all = np.zeros(n, dtype=np.float64)
    simpo_losses_all = np.zeros(n, dtype=np.float64)

    beta = float(training_args.beta)
    gamma_beta_ratio = float(training_args.gamma_beta_ratio)
    rho = float(score_args.rho)
    eps = 1e-12

    named_params = inner_named_params()

    cursor = 0
    for step, batch in enumerate(dataloader):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

        # 1) Clean forward (with grads) and SimPO loss
        chosen_logps, rejected_logps, _, _, _ = trainer.concatenated_forward(model, batch)
        margin_clean = (chosen_logps - rejected_logps)

        # SimPO logit/loss with anchor=0 (matches GAPOTrainer's call inside concatenated_forward_perturbed).
        simpo_logits = beta * (margin_clean - gamma_beta_ratio)
        loss = -F.logsigmoid(simpo_logits).mean()

        # 2) Backward
        model.zero_grad(set_to_none=False)
        loss.backward()

        # 3) Compute SAM scale = rho / ||p * g||_2 over inner params
        vals = []
        for _, p in named_params:
            g = p.grad
            if g is None or 0 in p.shape or 0 in g.shape or p.shape != g.shape:
                continue
            vals.append((p.detach().abs() * g.detach()).norm(p=2))
        if len(vals) == 0:
            norm = torch.tensor(0.0, device=device)
        else:
            norm = torch.norm(torch.stack(vals), p=2)
        scale = rho / (norm + eps)

        # 4) Apply perturbation in place (save deltas so we can undo)
        deltas = []
        with torch.no_grad():
            for _, p in named_params:
                g = p.grad
                if g is None or 0 in p.shape or 0 in g.shape or p.shape != g.shape:
                    continue
                if getattr(g, "is_sparse", False):
                    g = g.to_dense()
                if g.dtype != p.dtype:
                    g = g.to(dtype=p.dtype)
                if g.device != p.device:
                    g = g.to(p.device)
                scale_t = torch.as_tensor(scale, device=p.device, dtype=p.dtype)
                e_w = p.pow(2).mul(g).mul(scale_t)
                p.add_(e_w)
                deltas.append((p, e_w))

        # 5) Perturbed forward (no grads needed)
        try:
            with torch.no_grad():
                cp_p, rp_p, _, _, _ = trainer.concatenated_forward(model, batch)
            margin_perturbed = (cp_p - rp_p)
        finally:
            # 6) Restore weights regardless of forward outcome
            with torch.no_grad():
                for p, e_w in deltas:
                    p.sub_(e_w)
            model.zero_grad(set_to_none=True)

        # Record
        m_clean_np = margin_clean.detach().float().cpu().numpy()
        m_pert_np = margin_perturbed.detach().float().cpu().numpy()
        bs = m_clean_np.shape[0]
        sl = slice(cursor, cursor + bs)

        chosen_logps_all[sl] = chosen_logps.detach().float().cpu().numpy()
        rejected_logps_all[sl] = rejected_logps.detach().float().cpu().numpy()
        margins_all[sl] = m_clean_np
        margins_perturbed_all[sl] = m_pert_np
        reward_margins_all[sl] = beta * m_clean_np
        simpo_losses_all[sl] = np.logaddexp(0.0, -beta * (m_clean_np - gamma_beta_ratio))
        cursor += bs

        if step % 20 == 0:
            logger.info(f"scored {cursor}/{n} (sam_norm={float(norm):.4g})")

    if cursor != n:
        raise RuntimeError(f"Scored {cursor} examples but expected {n}")

    if len(original_indices) != n:
        raise RuntimeError(
            f"Index bookkeeping mismatch: original_indices has {len(original_indices)} "
            f"but scored {n} examples"
        )
    sample_indices = original_indices

    out = {
        "sample_indices": sample_indices,
        "is_flipped": is_flipped,
        "margins": margins_all[None, :],
        "margins_perturbed": margins_perturbed_all[None, :],
        "reward_margins": reward_margins_all[None, :],
        "simpo_losses": simpo_losses_all[None, :],
        "gapo_losses": simpo_losses_all[None, :],  # placeholder so select_subset works
        "chosen_logps": chosen_logps_all[None, :],
        "rejected_logps": rejected_logps_all[None, :],
        "steps": np.array([0], dtype=np.int64),
        "checkpoint_paths": np.array(["base_model"], dtype="<U64"),
    }
    meta = {
        "model_name_or_path": model_args.model_name_or_path,
        "dataset_mixer": data_args.dataset_mixer,
        "dataset_splits": data_args.dataset_splits,
        "score_split": score_args.score_split,
        "beta": beta,
        "gamma_beta_ratio": gamma_beta_ratio,
        "rho": rho,
        "n": int(n),
        "n_total_split": int(n_total),
        "max_samples": score_args.max_samples,
        "subset_seed": score_args.subset_seed,
    }

    os.makedirs(os.path.dirname(os.path.abspath(score_args.output_npz)) or ".", exist_ok=True)
    np.savez(score_args.output_npz, **out)
    with open(score_args.output_npz + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"wrote {score_args.output_npz} ({n} examples) and metadata sidecar")


if __name__ == "__main__":
    main()
