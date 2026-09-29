from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class DrDPOConfig(TrainingArguments):
    r"""
    DrDPOConfig collects all training arguments related to the [`DrDPOTrainer`] class.

    Dr_DPO (Wu et al., 2024) augments standard DPO with a Distributionally Robust
    Optimization aggregation across the batch:

        loss = -beta_prime * log( mean( exp( -per_sample_dpo_loss / beta_prime ) ) )

    Parameters:
        max_length, max_prompt_length, max_target_length: usual DPO truncation knobs.
        beta (`float`, defaults to 0.1):
            The DPO temperature on the implicit reward (policy_logps - ref_logps).
        beta_prime (`float`, defaults to 1.0):
            The Dr_DPO log-sum-exp temperature (beta' in the paper). Smaller values are
            more pessimistic / robust; as beta_prime -> infinity the aggregation reduces
            to the standard `losses.mean()`.
        label_smoothing (`float`, defaults to 0):
            cDPO-style label smoothing on the per-sample loss.
        loss_type (`str`, defaults to `sigmoid`):
            Per-sample DPO loss type: 'sigmoid' or 'hinge'.
        sft_weight (`float`, defaults to 0.0):
            Optional SFT loss weight added on the chosen response.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in the policy and reference models.
        ref_model_init_kwargs (`Optional[Dict]`, *optional*):
            Kwargs forwarded to `AutoModelForCausalLM.from_pretrained` when the
            reference model is loaded from a string. Falls back to `model_init_kwargs`.
        precompute_ref_log_probs (`bool`, defaults to `False`):
            Reserved; not implemented yet. Reference logps are computed on the fly.
    """

    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 0.1
    beta_prime: float = 1.0

    sft_weight: float = 0.0
    label_smoothing: float = 0
    loss_type: Literal["sigmoid", "hinge"] = "sigmoid"
    disable_dropout: bool = True

    label_pad_token_id: int = -100
    padding_value: int = None
    truncation_mode: str = "keep_end"
    generate_during_eval: bool = False
    is_encoder_decoder: Optional[bool] = None

    model_init_kwargs: Optional[Dict] = None
    ref_model_init_kwargs: Optional[Dict] = None
    precompute_ref_log_probs: bool = False

    dataset_num_proc: Optional[int] = None
