from dataclasses import dataclass
from typing import Dict, Literal, Optional
from transformers import TrainingArguments


@dataclass
class GAPOConfig(TrainingArguments):
    r"""
    GAPOConfig collects all training arguments related to the [`GAPOTrainer`] class.

    Using [`HfArgumentParser`] we can turn this class into
    [argparse](https://docs.python.org/3/library/argparse#module-argparse) arguments that can be specified on the
    command line.

    Parameters:
        max_length (`int`, defaults to `None`):
            The maximum length of the sequences in the batch. This argument is required if you want to use the default data collator.
        max_prompt_length (`int`, defaults to `None`):
            The maximum length of the prompt. This argument is required if you want to use the default data collator.
        max_target_length (`int`, defaults to `None`):
            The maximum length of the target. This argument is required if you want to use the default data collator and your model is an encoder-decoder.
        beta (`float`, defaults to 2.0):
            The beta factor in GAPO loss.
        gamma_beta_ratio (`float`, defaults to 0.25):
            The ratio between the target reward margin (gamma) and beta in GAPO loss.
        rho (`float`, defaults to 0.05):
            The scaling factor for gradient normalization in GAPO.
        perturb_start_layer (`int`, defaults to 0):
            The layer index from which to start applying perturbation. Layers before this
            index are not perturbed. The lm_head and final layer norm are always perturbed.
        sft_weight (`float`, defaults to 0.0):
            SFT loss weight added to the GAPO loss (0.0 is not using SFT).
        label_smoothing (`float`, defaults to 0):
            The label smoothing factor. This argument is required if you want to use the default data collator.
        loss_type (`str`, defaults to `sigmoid`):
            The type of loss to use. This argument is required if you want to use the default data collator.
        label_pad_token_id (`int`, defaults to `-100`):
            The label pad token id. This argument is required if you want to use the default data collator.
        padding_value (`int`, defaults to `None`):
            The padding value if it is different to the tokenizer's pad_token_id.
        truncation_mode (`str`, defaults to `keep_end`):
            The truncation mode to use, either `keep_end` or `keep_start`. This argument is required if you want to use the default data collator.
        generate_during_eval (`bool`, defaults to `False`):
            Whether to sample and log generations during evaluation step.
        is_encoder_decoder (`Optional[bool]`, `optional`, defaults to `None`):
            If no model is provided, we need to know if the model_init returns an encoder-decoder.
        disable_dropout (`bool`, defaults to `True`):
            Whether or not to disable dropouts in `model`.
        model_init_kwargs (`Optional[Dict]`, *optional*):
            Dict of Optional kwargs to pass when instantiating the model from a string
        dataset_num_proc (`Optional[int]`, *optional*):
            The number of workers to use to tokenize the data. Defaults to None.
    """
    
    max_length: Optional[int] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    max_target_length: Optional[int] = None

    beta: float = 2.0
    rho: float = 0.05
    perturb_start_layer: int = 0
    # Perturbation direction within the rho-ball:
    #   "pessimistic" (default) = worst-case ASAM-style gradient ascent direction (GAPO).
    #   "gaussian" = random Gaussian direction at the same rho radius (E4 ablation / AC4 control).
    perturb_mode: str = "pessimistic"
    perturb_require_fp32: bool = False
    """Require FP32 model weights and CUDA computation for perturbation runs."""
    perturb_diagnostics: bool = False
    """Log intended/applied perturbation size and the effective anchor gap."""
    gamma_beta_ratio: float = 0.25
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

    dataset_num_proc: Optional[int] = None

    # ---- Drift diagnostics (KL / Length / Margin) ---------------------------
    drift_diagnostics: bool = False
    """Enable the drift-diagnostics callback (KL-to-SFT + train/heldout margin)."""

    drift_generation: bool = False
    """Also run on-policy generation at every checkpoint to track length drift."""

    drift_heldout_size: int = 2000
    """Number of preference pairs carved out of the train split for held-out diagnostics (seed=42)."""

    drift_train_subset_size: int = 1000
    """Size of the fixed train-subset used for the train-margin trajectory (seed=43)."""

    drift_gen_n_prompts: int = 500
    """How many held-out prompts to generate from for the length-drift panel."""

    drift_gen_max_new_tokens: int = 1024
    drift_gen_temperature: float = 0.7
    drift_gen_top_p: float = 0.9
    drift_gen_seed: int = 12345

    drift_heldout_seed: int = 42
    drift_train_subset_seed: int = 43
