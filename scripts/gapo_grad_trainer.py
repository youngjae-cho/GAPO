
import inspect
import random
import warnings
from collections import defaultdict
from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from trl.trainer import CPOTrainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput
from transformers.utils import is_torch_fx_proxy
import torch.distributed as dist

from trl.import_utils import is_peft_available, is_wandb_available
from gapo_config import GAPOConfig

from dataclasses import dataclass
from typing import Dict, Literal, Optional

from transformers import TrainingArguments

from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)
import deepspeed
import copy

from gapo_trainer import GAPOTrainer

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class GAPOGradTrainer(GAPOTrainer):
    r"""
    GAPOGradTrainer is a variant of GAPOTrainer that, instead of perturbing the
    parameters in a SAM-style step, computes a per-instance gradient norm for
    each sample in the batch and uses those norms directly as the `margin` term
    fed into the GAPO loss.

    Concretely, for each sample i in the batch we compute its individual
    preference loss l_i = gapo_loss(chosen_logps_i, rejected_logps_i), call
    `l_i.backward(retain_graph=True)`, and aggregate the gradient norm
    `(|p| * g).norm(p=2)` across all model parameters (matching the norm
    definition used by `concatenated_forward_perturbed` in the parent class).
    The resulting per-instance scalar is detached and returned as the margin.
    """

    _tag_names = ["trl", "gapo", "gapo-grad"]

    def concatenated_forward_grad(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]]
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.FloatTensor]:
        """Run model on the batch and compute per-instance gradient norms as `margin`.

        Unlike `concatenated_forward_perturbed`, this does NOT modify model
        parameters. It performs:
            1) a forward pass and a per-sample backward to collect per-instance
               gradient norms,
            2) a clean second forward pass whose logps will be used by the
               outer training step's backward.
        """

        concatenated_batch = self.concatenated_inputs(
            batch,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
            padding_value=self.padding_value,
            device=self.accelerator.device,
        )
        len_chosen = batch["chosen_labels"].shape[0]

        model_kwargs = (
            {
                "labels": concatenated_batch["concatenated_labels"],
                "decoder_input_ids": concatenated_batch.pop("concatenated_decoder_input_ids", None),
            }
            if self.is_encoder_decoder
            else {}
        )

        # ---- Pass 1: gradient norm (mirrors gapo_trainer's pattern) -------------
        # NOTE: do NOT call model.zero_grad() AFTER the inner backward — that
        # corrupts DeepSpeed ZeRO-3's IPG bucket state and breaks the outer
        # backward. zero_grad must only be called BEFORE the inner backward,
        # exactly as `concatenated_forward_perturbed` does it.
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        model.zero_grad()

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]
        loss, _, _ = self.gapo_loss(chosen_logps, rejected_logps)
        loss.mean().backward()

        vals = []
        for _, p in model.module.model.named_parameters():
            g = p.grad
            if g is None:
                continue
            if 0 in p.shape or 0 in g.shape or p.shape != g.shape:
                continue
            if getattr(g, "is_sparse", False):
                g = g.to_dense()
            vals.append((p.abs() * g).norm(p=2).detach().cpu())

        if len(vals) == 0:
            norm = torch.tensor(0.0).detach()
        else:
            norm = torch.norm(torch.stack(vals), p=2).detach()

        margin = norm.to(self.accelerator.device).view(1).expand(len_chosen).clone().detach()

        # ---- Pass 2: clean forward used by the outer .backward() ----------------
        # NO zero_grad here; matches the parent's perturbed flow.
        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=True,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )
        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]
        del concatenated_batch

        return (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, margin)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        """Compute the GAPO-Grad loss and metrics. Margin is per-instance grad norm."""
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        if train_eval == "train":
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                chosen_labels,
                margin,
            ) = self.concatenated_forward_grad(model, batch)
            losses, chosen_rewards, rejected_rewards = self.gapo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                margin=margin,
            )
        else:
            (
                policy_chosen_logps,
                policy_rejected_logps,
                policy_chosen_logits,
                policy_rejected_logits,
                chosen_labels,
            ) = self.concatenated_forward(model, batch)
            losses, chosen_rewards, rejected_rewards = self.gapo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
            )

        loss = losses.mean()

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels = chosen_labels[..., 1:].clone()
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(policy_chosen_logits.view(-1, policy_chosen_logits.shape[-1]), chosen_labels.view(-1))
            loss = self.sft_weight * sft_loss + loss
            metrics[f"{prefix}sft_loss"] = sft_loss.detach().cpu()

        reward_accuracies = (chosen_rewards > rejected_rewards).float()
        flipped_indicator = batch.get("is_flipped", None)
        if flipped_indicator is None:
            flipped_mask = torch.zeros_like(reward_accuracies, dtype=torch.bool, device=reward_accuracies.device)
        else:
            if isinstance(flipped_indicator, torch.Tensor):
                flipped_mask = flipped_indicator.to(reward_accuracies.device).bool().view(-1)
            else:
                flipped_mask = torch.tensor(flipped_indicator, device=reward_accuracies.device).bool().view(-1)
            if flipped_mask.numel() != reward_accuracies.numel():
                raise ValueError(
                    f"is_flipped length ({flipped_mask.numel()}) does not match batch size ({reward_accuracies.numel()})"
                )

        original_gt_reward_accuracies = torch.where(
            flipped_mask,
            (chosen_rewards < rejected_rewards).float(),
            (chosen_rewards > rejected_rewards).float(),
        )
        reward_margins = chosen_rewards - rejected_rewards

        clean_mask = ~flipped_mask
        noisy_mask = flipped_mask

        def _masked_mean_std(values: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            if mask.any():
                subset = values[mask]
                return subset.mean(), subset.std(unbiased=False)
            zero = torch.tensor(0.0, device=values.device)
            return zero, zero

        clean_margin_mean, clean_margin_std = _masked_mean_std(reward_margins, clean_mask)
        noisy_margin_mean, noisy_margin_std = _masked_mean_std(reward_margins, noisy_mask)

        if train_eval == "train":
            metrics[f"{prefix}grad_norm/std"] = (
                margin.std(unbiased=False).cpu() if margin.numel() > 1
                else torch.tensor(0.0)
            )
            metrics[f"{prefix}grad_norm/mean"] = margin.mean().cpu()
            grad_clean_mean, grad_clean_std = _masked_mean_std(margin, clean_mask)
            grad_noisy_mean, grad_noisy_std = _masked_mean_std(margin, noisy_mask)
            metrics[f"{prefix}grad_norm/clean_mean"] = grad_clean_mean.cpu()
            metrics[f"{prefix}grad_norm/clean_std"] = grad_clean_std.cpu()
            metrics[f"{prefix}grad_norm/noisy_mean"] = grad_noisy_mean.cpu()
            metrics[f"{prefix}grad_norm/noisy_std"] = grad_noisy_std.cpu()

        metrics[f"{prefix}rewards/chosen"] = chosen_rewards.mean().cpu()
        metrics[f"{prefix}rewards/rejected"] = rejected_rewards.mean().cpu()
        metrics[f"{prefix}rewards/accuracies"] = reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/accuracies_original_gt"] = original_gt_reward_accuracies.mean().cpu()
        metrics[f"{prefix}rewards/flip_fraction"] = flipped_mask.float().mean().cpu()
        metrics[f"{prefix}rewards/margins"] = reward_margins.mean().cpu()
        metrics[f"{prefix}rewards/margins_clean_mean"] = clean_margin_mean.cpu()
        metrics[f"{prefix}rewards/margins_clean_std"] = clean_margin_std.cpu()
        metrics[f"{prefix}rewards/margins_noisy_mean"] = noisy_margin_mean.cpu()
        metrics[f"{prefix}rewards/margins_noisy_std"] = noisy_margin_std.cpu()
        metrics[f"{prefix}rewards/margins_clean_count"] = clean_mask.float().sum().cpu()
        metrics[f"{prefix}rewards/margins_noisy_count"] = noisy_mask.float().sum().cpu()
        metrics[f"{prefix}logps/rejected"] = policy_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/chosen"] = policy_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return loss, metrics
