"""Drift diagnostics for GAPO paper.

Three drift modes are tracked at every checkpoint:
  - KL drift     : teacher-forced KL(policy || SFT) on held-out responses
  - Margin drift : length-normalized log-prob margin on a fixed train subset
                   AND on a held-out set (gap = train - heldout)
  - Length drift : avg response length under controlled decoding on a fixed prompt set

The diagnostics are wired in as a `TrainerCallback` that fires on every
`on_save`. Results are written to:
    {output_dir}/drift_diagnostics/checkpoint_{global_step}.json
"""

import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import TrainerCallback


def _build_dataloader(trainer, dataset):
    """Use the trainer's own collator + padding so batches match training."""
    return DataLoader(
        dataset,
        batch_size=getattr(trainer.args, "per_device_eval_batch_size", 4),
        shuffle=False,
        collate_fn=trainer.data_collator,
        num_workers=0,
        pin_memory=False,
    )


def _per_token_logp_and_kl(p_logits, s_logits, labels, response_mask):
    """Compute per-token label log-probs (policy + ref) and per-token analytic
    KL(policy||SFT), aligned with `labels`. All shapes (B, T-1).

    Inputs are already shifted: p_logits[:, :T-1, :] vs labels[:, 1:].

    KL is computed in the logits' native dtype (typically bf16) to keep memory
    low — for full-vocab (32k) over long sequences this matters a lot. The
    final per-sequence reduction happens in fp32 for stable averaging.
    """
    p_logp = F.log_softmax(p_logits, dim=-1)
    s_logp = F.log_softmax(s_logits, dim=-1)

    # KL(policy || sft) = sum_v p(v) * (log p(v) - log s(v))
    p_prob = p_logp.exp()
    per_tok_kl = (p_prob * (p_logp - s_logp)).sum(dim=-1).float()  # (B, T-1) fp32

    # Per-token label log-prob under policy AND ref (for margin and k1 log-ratio).
    safe_labels = labels.clone()
    safe_labels[~response_mask] = 0
    per_tok_label_logp = torch.gather(
        p_logp, dim=2, index=safe_labels.unsqueeze(2)
    ).squeeze(2).float()  # (B, T-1) fp32
    per_tok_ref_label_logp = torch.gather(
        s_logp, dim=2, index=safe_labels.unsqueeze(2)
    ).squeeze(2).float()  # (B, T-1) fp32

    return per_tok_label_logp, per_tok_ref_label_logp, per_tok_kl


@torch.no_grad()
def compute_kl_and_margin(
    trainer,
    dataset,
    sft_model,
    label_pad_token_id: int = -100,
    desc: str = "drift",
) -> Dict[str, Any]:
    """Single pass over `dataset` collecting per-pair KL + length-normalised margin.

    Reuses `trainer.concatenated_inputs` so the tensorisation matches training
    exactly (padding values, label masking, etc.).

    If `policy_model is sft_model` (i.e. the step-0 / SFT baseline), we short-
    circuit the SFT forward — KL is zero by definition.
    """
    device = trainer.accelerator.device
    policy_model = trainer.model
    policy_was_training = policy_model.training
    sft_was_training = sft_model.training
    policy_model.eval()
    sft_model.eval()

    is_sft_baseline = policy_model is sft_model

    dataloader = _build_dataloader(trainer, dataset)

    kl_chosen, kl_rejected = [], []
    # k1 log-ratio estimator: mean_t [log pi(y_t) - log ref(y_t)] over data tokens.
    # Note: y_t comes from the dataset (teacher-forced), so this is the cross-
    # entropy difference / log-likelihood ratio under data — NOT an unbiased KL
    # estimator (that would require y_t ~ pi). Reported alongside the analytic
    # KL for comparison.
    k1_chosen, k1_rejected = [], []
    chosen_logp_lst, rejected_logp_lst = [], []
    chosen_resp_lens, rejected_resp_lens = [], []
    margins = []

    for batch in tqdm(dataloader, desc=desc, total=len(dataloader), leave=False):
        # Move tensors to device (collator returns CPU tensors)
        batch = {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }
        concatenated = trainer.concatenated_inputs(
            batch,
            is_encoder_decoder=trainer.is_encoder_decoder,
            label_pad_token_id=label_pad_token_id,
            padding_value=trainer.padding_value,
            device=device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        input_ids = concatenated["concatenated_input_ids"]
        attn_mask = concatenated["concatenated_attention_mask"]
        labels = concatenated["concatenated_labels"]

        p_logits = policy_model(input_ids, attention_mask=attn_mask, use_cache=False).logits
        # Shift for next-token prediction.
        p_logits = p_logits[:, :-1, :]
        labels = labels[:, 1:]
        response_mask = labels != label_pad_token_id

        if is_sft_baseline:
            # KL(policy || policy) = 0; only need label log-probs for margin.
            p_logp = F.log_softmax(p_logits, dim=-1)
            safe_labels = labels.clone()
            safe_labels[~response_mask] = 0
            per_tok_label_logp = torch.gather(
                p_logp, dim=2, index=safe_labels.unsqueeze(2)
            ).squeeze(2).float()
            per_tok_ref_label_logp = per_tok_label_logp  # ref == policy at step 0
            per_tok_kl = torch.zeros_like(per_tok_label_logp)
        else:
            s_logits = sft_model(input_ids, attention_mask=attn_mask, use_cache=False).logits
            s_logits = s_logits[:, :-1, :]
            per_tok_label_logp, per_tok_ref_label_logp, per_tok_kl = _per_token_logp_and_kl(
                p_logits, s_logits, labels, response_mask
            )

        n_resp = response_mask.sum(dim=-1).clamp(min=1).float()
        per_seq_kl = (per_tok_kl * response_mask.float()).sum(dim=-1) / n_resp
        per_seq_logp = (per_tok_label_logp * response_mask.float()).sum(dim=-1) / n_resp
        per_tok_logratio = per_tok_label_logp - per_tok_ref_label_logp
        per_seq_k1 = (per_tok_logratio * response_mask.float()).sum(dim=-1) / n_resp

        c_kl = per_seq_kl[:len_chosen].cpu()
        r_kl = per_seq_kl[len_chosen:].cpu()
        c_logp = per_seq_logp[:len_chosen].cpu()
        r_logp = per_seq_logp[len_chosen:].cpu()
        c_k1 = per_seq_k1[:len_chosen].cpu()
        r_k1 = per_seq_k1[len_chosen:].cpu()

        kl_chosen.extend(c_kl.tolist())
        kl_rejected.extend(r_kl.tolist())
        k1_chosen.extend(c_k1.tolist())
        k1_rejected.extend(r_k1.tolist())
        chosen_logp_lst.extend(c_logp.tolist())
        rejected_logp_lst.extend(r_logp.tolist())
        margins.extend((c_logp - r_logp).tolist())
        chosen_resp_lens.extend(response_mask[:len_chosen].sum(dim=-1).cpu().tolist())
        rejected_resp_lens.extend(response_mask[len_chosen:].sum(dim=-1).cpu().tolist())

    if policy_was_training:
        policy_model.train()
    if sft_was_training:
        sft_model.train()

    return {
        "n_pairs": len(margins),
        "kl_chosen_mean": float(np.mean(kl_chosen)),
        "kl_rejected_mean": float(np.mean(kl_rejected)),
        "kl_mean": float(0.5 * (np.mean(kl_chosen) + np.mean(kl_rejected))),
        "kl_k1_chosen_mean": float(np.mean(k1_chosen)),
        "kl_k1_rejected_mean": float(np.mean(k1_rejected)),
        "kl_k1_mean": float(0.5 * (np.mean(k1_chosen) + np.mean(k1_rejected))),
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "chosen_logp_mean": float(np.mean(chosen_logp_lst)),
        "rejected_logp_mean": float(np.mean(rejected_logp_lst)),
        "accuracy": float(np.mean([m > 0 for m in margins])),
        "chosen_resp_len_mean": float(np.mean(chosen_resp_lens)),
        "rejected_resp_len_mean": float(np.mean(rejected_resp_lens)),
    }


@torch.no_grad()
def generate_response_lengths(
    trainer,
    prompts: List[str],
    *,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.9,
    base_seed: int = 12345,
    batch_size: int = 8,
) -> Dict[str, Any]:
    """On-policy generation on a fixed prompt list, BATCHED for throughput.

    A single seed is set per chunk (`base_seed + chunk_idx`). Per-prompt seeds
    are sacrificed for ~batch_size× speed; for length-drift trajectory analysis
    this is fine — what matters is identical decoding settings across methods/
    checkpoints, which we keep.
    """
    model = trainer.model
    tokenizer = trainer.tokenizer
    device = trainer.accelerator.device

    was_training = model.training
    model.eval()

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    # Tokenizer must left-pad for generation; restore after.
    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"

    records: List[Dict[str, Any]] = []

    n = len(prompts)
    n_chunks = (n + batch_size - 1) // batch_size
    pbar = tqdm(total=n, desc="generate", leave=False)
    for ci in range(n_chunks):
        chunk_start = ci * batch_size
        chunk = prompts[chunk_start: chunk_start + batch_size]

        seed = base_seed + ci
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        random.seed(seed)
        np.random.seed(seed)

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=trainer.max_prompt_length,
            padding=True,
            add_special_tokens=False,
        ).to(device)

        # Prepend BOS where missing — match training tokenisation.
        if tokenizer.bos_token_id is not None:
            ids = enc["input_ids"]
            if ids.shape[1] == 0 or (ids[:, 0] != tokenizer.bos_token_id).any():
                bos_col = torch.full(
                    (ids.shape[0], 1), tokenizer.bos_token_id,
                    device=device, dtype=ids.dtype,
                )
                enc["input_ids"] = torch.cat([bos_col, ids], dim=1)
                enc["attention_mask"] = torch.cat(
                    [torch.ones_like(bos_col), enc["attention_mask"]], dim=1
                )

        prompt_len = enc["input_ids"].shape[1]
        out = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            use_cache=True,
        )
        gen_block = out[:, prompt_len:]  # (B, T_new)

        for i, gen_ids in enumerate(gen_block):
            if pad_id is not None:
                non_pad = (gen_ids != pad_id).nonzero(as_tuple=True)[0]
                if non_pad.numel() > 0:
                    gen_ids = gen_ids[: non_pad[-1].item() + 1]
                else:
                    gen_ids = gen_ids[:0]
            text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            records.append({
                "prompt_idx": chunk_start + i,
                "response_length_tokens": int(gen_ids.shape[0]),
                "response_length_chars": len(text),
                "response": text,
            })
        pbar.update(len(chunk))
    pbar.close()

    tokenizer.padding_side = orig_padding_side
    if was_training:
        model.train()

    lens_tok = [r["response_length_tokens"] for r in records]
    lens_chr = [r["response_length_chars"] for r in records]
    return {
        "n_prompts": len(records),
        "avg_response_length_tokens": float(np.mean(lens_tok)),
        "avg_response_length_chars": float(np.mean(lens_chr)),
        "median_response_length_tokens": float(np.median(lens_tok)),
        "records": records,
    }


class DriftDiagnosticsCallback(TrainerCallback):
    """Fires KL+margin (and optionally generation) every checkpoint save."""

    def __init__(
        self,
        trainer,
        sft_model,
        heldout_dataset,
        train_subset_dataset,
        *,
        output_dir: Optional[str] = None,
        run_name: str = "run",
        do_generation: bool = False,
        gen_prompts: Optional[List[str]] = None,
        gen_max_new_tokens: int = 1024,
        gen_temperature: float = 0.7,
        gen_top_p: float = 0.9,
        gen_seed: int = 12345,
        run_at_step_zero: bool = True,
    ):
        super().__init__()
        self.trainer = trainer
        self.sft_model = sft_model
        self.heldout_dataset = heldout_dataset
        self.train_subset_dataset = train_subset_dataset
        self.output_dir = Path(output_dir or os.path.join(trainer.args.output_dir, "drift_diagnostics"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_name = run_name
        self.do_generation = do_generation
        self.gen_prompts = gen_prompts or []
        self.gen_max_new_tokens = gen_max_new_tokens
        self.gen_temperature = gen_temperature
        self.gen_top_p = gen_top_p
        self.gen_seed = gen_seed
        self.run_at_step_zero = run_at_step_zero
        self._zero_run_done = False

        # Freeze SFT once.
        self.sft_model.eval()
        for p in self.sft_model.parameters():
            p.requires_grad_(False)

    # ---- internals ----------------------------------------------------------
    def _run(self, global_step: int, epoch: float):
        is_main = self.trainer.accelerator.is_main_process

        summary = {
            "run_name": self.run_name,
            "global_step": int(global_step),
            "epoch": float(epoch) if epoch is not None else None,
        }

        heldout = compute_kl_and_margin(self.trainer, self.heldout_dataset, self.sft_model)
        summary["heldout"] = heldout

        train_eval = compute_kl_and_margin(self.trainer, self.train_subset_dataset, self.sft_model)
        summary["train_subset"] = train_eval

        summary["margin_gap_train_minus_heldout"] = (
            train_eval["margin_mean"] - heldout["margin_mean"]
        )

        if self.do_generation and len(self.gen_prompts) > 0 and is_main:
            gen = generate_response_lengths(
                self.trainer,
                self.gen_prompts,
                max_new_tokens=self.gen_max_new_tokens,
                temperature=self.gen_temperature,
                top_p=self.gen_top_p,
                base_seed=self.gen_seed,
            )
            summary["generation"] = {
                k: v for k, v in gen.items() if k != "records"
            }
            # Also dump full generations alongside the summary.
            gen_path = self.output_dir / f"checkpoint_{int(global_step)}_generations.jsonl"
            with open(gen_path, "w") as f:
                for r in gen["records"]:
                    f.write(json.dumps(r) + "\n")

        if is_main:
            out_path = self.output_dir / f"checkpoint_{int(global_step)}_summary.json"
            with open(out_path, "w") as f:
                json.dump(summary, f, indent=2)
            try:
                # Also push to wandb if active.
                import wandb
                if wandb.run is not None:
                    flat = {
                        "drift/heldout/kl_mean": heldout["kl_mean"],
                        "drift/heldout/kl_k1_mean": heldout["kl_k1_mean"],
                        "drift/heldout/margin_mean": heldout["margin_mean"],
                        "drift/heldout/accuracy": heldout["accuracy"],
                        "drift/heldout/chosen_resp_len_mean": heldout["chosen_resp_len_mean"],
                        "drift/train/margin_mean": train_eval["margin_mean"],
                        "drift/train/kl_k1_mean": train_eval["kl_k1_mean"],
                        "drift/margin_gap": summary["margin_gap_train_minus_heldout"],
                    }
                    if "generation" in summary:
                        flat["drift/gen/avg_resp_len_tokens"] = summary["generation"]["avg_response_length_tokens"]
                    wandb.log(flat, step=int(global_step))
            except Exception:
                pass

    # ---- callback hooks -----------------------------------------------------
    def on_train_begin(self, args, state, control, **kwargs):
        if self.run_at_step_zero and not self._zero_run_done:
            self._run(global_step=0, epoch=0.0)
            self._zero_run_done = True

    def on_save(self, args, state, control, **kwargs):
        self._run(global_step=state.global_step, epoch=state.epoch)
