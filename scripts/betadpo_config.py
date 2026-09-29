from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class BetaDPOConfig(TrainingArguments):
    r"""
    BetaDPOConfig collects all training arguments related to the [`BetaDPOTrainer`] class.

    β-DPO (Wu et al., 2024) makes the DPO temperature `beta` dynamic and filters
    out unreliable / outlier preference pairs at the batch level, using a running
    (EMA) estimate of the reward-gap distribution:

        A_i         = (pi_chosen - ref_chosen) - (pi_rejected - ref_rejected)
        weight_i    = exp(-0.5 * ((A_i - gap_mean) / gap_std) ** 2)
        selected    = multinomial(weight, int(N * (1 - mode_weight)))
        beta_used   = beta * (1 + a * (mean(A[selected]) - gap_mean)), clamp(min=1e-3)
        loss        = mean_{i in selected} -log_sigmoid(beta_used * logits_i)

    Parameters:
        max_length, max_prompt_length, max_target_length: usual DPO truncation knobs.
        beta (`float`, defaults to 0.1):
            The base DPO temperature on the implicit reward.
        a (`float`, defaults to 0.6):
            Coefficient scaling the dynamic-beta adjustment around `gap_mean`.
        mode_weight (`float`, defaults to 0.2):
            Fraction of the batch to drop; `int(N * (1 - mode_weight))` samples are kept.
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
    a: float = 0.6
    mode_weight: float = 0.2

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
