from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class EpsilonDPOConfig(TrainingArguments):
    r"""
    EpsilonDPOConfig collects all training arguments related to the [`EpsilonDPOTrainer`] class.

    ε-DPO (Epsilon-DPO) perturbs the policy logits toward / away from the reference
    logits before computing the DPO implicit reward:

        p_epsilon_logits = (1 + epsilon) * policy_logits - epsilon * ref_logits

    The ε-perturbed policy log-probs (summed over the response tokens) replace the
    standard policy log-probs inside the DPO logsigmoid loss, while the reference
    log-probs used for the reference term stay the standard ones:

        logits = (eps_pi_chosen - eps_pi_rejected) - (ref_chosen - ref_rejected)
        loss   = -log_sigmoid(beta * logits)

    Parameters:
        max_length, max_prompt_length, max_target_length: usual DPO truncation knobs.
        beta (`float`, defaults to 0.01):
            The DPO temperature on the implicit reward.
        epsilon (`float`, defaults to 0.01):
            Strength of the ε perturbation on the policy logits.
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

    beta: float = 0.01
    epsilon: float = 0.01

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
