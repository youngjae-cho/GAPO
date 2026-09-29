import inspect
import random
import warnings
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from functools import wraps
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import is_deepspeed_available
from datasets import Dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, DataCollator, PreTrainedModel, PreTrainedTokenizerBase, Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalLoopOutput

from trl.import_utils import is_peft_available, is_wandb_available
from trl.trainer.utils import (
    DPODataCollatorWithPadding,
    disable_dropout_in_model,
    pad_to_length,
    peft_module_casting_to_bf16,
    trl_sanitze_kwargs_for_tagging,
)

from edpo_config import EpsilonDPOConfig

if is_deepspeed_available():
    import deepspeed

if is_peft_available():
    from peft import PeftModel, get_peft_model, prepare_model_for_kbit_training

if is_wandb_available():
    import wandb


class EpsilonDPOTrainer(Trainer):
    r"""
    Faithful implementation of ε-DPO (Epsilon-DPO) with a frozen reference model.

    The policy logits are perturbed toward / away from the reference logits before
    the DPO implicit reward is computed:

        p_epsilon_logits = (1 + epsilon) * policy_logits - epsilon * ref_logits

    The ε-perturbed policy log-probs (log_softmax -> gather at labels -> mask padding
    -> sum over the response tokens) replace the standard policy log-probs inside the
    DPO logsigmoid loss, while the reference term keeps the standard ref log-probs:

        logits = (eps_pi_chosen - eps_pi_rejected) - (ref_chosen - ref_rejected)
        l_i    = -log_sigmoid( beta * logits )
        L      = mean_i( l_i )
    """

    _tag_names = ["trl", "edpo"]

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        ref_model: Optional[Union[PreTrainedModel, nn.Module, str]] = None,
        args: Optional[EpsilonDPOConfig] = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional[Dict] = None,
        compute_metrics: Optional[Callable[[EvalLoopOutput], Dict]] = None,
    ):
        if args.model_init_kwargs is None:
            model_init_kwargs = {}
        elif not isinstance(model, str):
            raise ValueError("You passed model_kwargs to the EpsilonDPOTrainer. But your model is already instantiated.")
        else:
            model_init_kwargs = args.model_init_kwargs
            model_init_kwargs["torch_dtype"] = (
                model_init_kwargs["torch_dtype"]
                if model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, model_init_kwargs["torch_dtype"])
            )

        if args.ref_model_init_kwargs is None:
            ref_model_init_kwargs = model_init_kwargs.copy() if model_init_kwargs else {}
        elif not isinstance(ref_model, str):
            raise ValueError("You passed ref_model_kwargs to the EpsilonDPOTrainer. But your ref_model is already instantiated.")
        else:
            ref_model_init_kwargs = args.ref_model_init_kwargs
            ref_model_init_kwargs["torch_dtype"] = (
                ref_model_init_kwargs["torch_dtype"]
                if ref_model_init_kwargs["torch_dtype"] in ["auto", None]
                else getattr(torch, ref_model_init_kwargs["torch_dtype"])
            )

        if isinstance(model, str):
            warnings.warn(
                "You passed a model_id to the EpsilonDPOTrainer. This will automatically create an "
                "`AutoModelForCausalLM` or a `PeftModel` (if you passed a `peft_config`) for you."
            )
            model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)

        if isinstance(ref_model, str):
            ref_model = AutoModelForCausalLM.from_pretrained(ref_model, **ref_model_init_kwargs)

        self._peft_has_been_casted_to_bf16 = False

        if not is_peft_available() and peft_config is not None:
            raise ValueError(
                "PEFT is not installed and you passed a `peft_config` in the trainer's kwargs, please install it to use the PEFT models"
            )
        elif is_peft_available() and peft_config is not None:
            if isinstance(model, PeftModel):
                model = model.merge_and_unload()

            if getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False):
                _support_gc_kwargs = hasattr(
                    args, "gradient_checkpointing_kwargs"
                ) and "gradient_checkpointing_kwargs" in list(
                    inspect.signature(prepare_model_for_kbit_training).parameters
                )

                prepare_model_kwargs = {"use_gradient_checkpointing": args.gradient_checkpointing}

                if _support_gc_kwargs:
                    prepare_model_kwargs["gradient_checkpointing_kwargs"] = args.gradient_checkpointing_kwargs

                model = prepare_model_for_kbit_training(model, **prepare_model_kwargs)
            elif getattr(args, "gradient_checkpointing", False):
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()
                else:

                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)

                    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

            model = get_peft_model(model, peft_config)
            if args.bf16 and getattr(model, "is_loaded_in_4bit", False):
                peft_module_casting_to_bf16(model)
                self._peft_has_been_casted_to_bf16 = True

        elif getattr(args, "gradient_checkpointing", False):
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if args.generate_during_eval and not is_wandb_available():
            raise ValueError(
                "`generate_during_eval=True` requires Weights and Biases to be installed."
                " Please install `wandb` to resolve."
            )

        if model is not None:
            self.is_encoder_decoder = model.config.is_encoder_decoder
        elif args.is_encoder_decoder is None:
            raise ValueError("When no model is provided, you need to pass the parameter is_encoder_decoder.")
        else:
            self.is_encoder_decoder = args.is_encoder_decoder

        # PEFT path: no separate reference model, use disable_adapter context.
        self.is_peft_model = is_peft_available() and isinstance(model, PeftModel)
        if self.is_peft_model and ref_model is not None:
            warnings.warn(
                "ref_model was provided alongside a PEFT model. Ignoring ref_model and using the base model "
                "via `disable_adapter()` as the reference."
            )
            ref_model = None

        if not self.is_peft_model and ref_model is None:
            raise ValueError(
                "EpsilonDPOTrainer requires a reference model. Pass `ref_model` (instance or hub id), or use a "
                "`peft_config` so that the base model can serve as the reference via adapter disabling."
            )

        if self.is_encoder_decoder:
            self.decoder_start_token_id = model.config.decoder_start_token_id
            self.pad_token_id = model.config.pad_token_id

        if tokenizer is None:
            raise ValueError("tokenizer must be specified to tokenize an ε-DPO dataset.")
        if args.max_length is None:
            warnings.warn(
                "`max_length` is not set in the EpsilonDPOConfig's init"
                " it will default to `512` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_length = 512
        else:
            max_length = args.max_length
        if args.max_prompt_length is None:
            warnings.warn(
                "`max_prompt_length` is not set in the EpsilonDPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_prompt_length = 128
        else:
            max_prompt_length = args.max_prompt_length

        if args.max_target_length is None and self.is_encoder_decoder:
            warnings.warn(
                "When using an encoder decoder architecture, you should set `max_target_length` in the EpsilonDPOConfig's init"
                " it will default to `128` by default, but you should do it yourself in the future.",
                UserWarning,
            )
            max_target_length = 128
        else:
            max_target_length = args.max_target_length

        if data_collator is None:
            data_collator = DPODataCollatorWithPadding(
                pad_token_id=tokenizer.pad_token_id,
                label_pad_token_id=args.label_pad_token_id,
                is_encoder_decoder=self.is_encoder_decoder,
            )

            if args.remove_unused_columns:
                args.remove_unused_columns = False
                warnings.warn(
                    "When using DPODataCollatorWithPadding, you should set `remove_unused_columns=False` in your TrainingArguments"
                    " we have set it for you, but you should do it yourself in the future.",
                    UserWarning,
                )

            self.use_dpo_data_collator = True
        else:
            self.use_dpo_data_collator = False

        if args.disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.max_length = max_length
        self.generate_during_eval = args.generate_during_eval
        self.label_pad_token_id = args.label_pad_token_id
        self.padding_value = args.padding_value if args.padding_value is not None else tokenizer.pad_token_id
        self.max_prompt_length = max_prompt_length
        self.truncation_mode = args.truncation_mode
        self.max_target_length = max_target_length
        self.tokenizer = tokenizer

        if args.loss_type in ["hinge"] and args.label_smoothing > 0:
            warnings.warn(
                "You are using a loss type that does not support label smoothing. Ignoring label_smoothing parameter."
            )

        self.beta = args.beta
        self.epsilon = args.epsilon
        self.sft_weight = args.sft_weight
        self.label_smoothing = args.label_smoothing
        self.loss_type = args.loss_type

        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        with PartialState().local_main_process_first():
            train_dataset = train_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)
            if eval_dataset is not None:
                eval_dataset = eval_dataset.map(self.tokenize_row, num_proc=args.dataset_num_proc)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        if hasattr(self.model, "add_model_tags"):
            self.model.add_model_tags(self._tag_names)

        if not hasattr(self, "accelerator"):
            raise AttributeError(
                "Your `Trainer` does not have an `accelerator` object. Consider upgrading `transformers`."
            )

        if ref_model is not None:
            ref_model.eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)
            if self.is_deepspeed_enabled:
                self.ref_model = self._prepare_deepspeed(ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(ref_model, evaluation_mode=True)
        else:
            self.ref_model = None

    def _prepare_deepspeed(self, model: PreTrainedModel):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepspeed_plugin.deepspeed_config.copy()
        stage = config_kwargs.get("zero_optimization", {}).get("stage", 0)

        if model is not None and hasattr(model, "config"):
            hidden_size = (
                max(model.config.hidden_sizes) if getattr(model.config, "hidden_sizes", None) else getattr(model.config, "hidden_size", None)
            )
            if hidden_size is not None and stage == 3:
                config_kwargs.update(
                    {
                        "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                        "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                        "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                    }
                )

        if stage != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
        model.eval()
        return model

    @contextmanager
    def null_ref_context(self):
        """For PEFT, the reference is the base model with adapters disabled."""
        with self.accelerator.unwrap_model(self.model).disable_adapter() if self.is_peft_model else nullcontext():
            yield

    def build_tokenized_answer(self, prompt, answer):
        full_tokenized = self.tokenizer(prompt + answer, add_special_tokens=False)
        prompt_input_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]

        answer_input_ids = full_tokenized["input_ids"][len(prompt_input_ids):]
        answer_attention_mask = full_tokenized["attention_mask"][len(prompt_input_ids):]

        full_concat_input_ids = np.concatenate([prompt_input_ids, answer_input_ids])
        full_input_ids = np.array(full_tokenized["input_ids"])

        if len(full_input_ids) != len(full_concat_input_ids):
            raise ValueError("Prompt input ids and answer input ids should have the same length.")

        response_token_ids_start_idx = len(prompt_input_ids)
        if prompt_input_ids != full_tokenized["input_ids"][:response_token_ids_start_idx]:
            response_token_ids_start_idx -= 1

        prompt_input_ids = full_tokenized["input_ids"][:response_token_ids_start_idx]
        prompt_attention_mask = full_tokenized["attention_mask"][:response_token_ids_start_idx]

        if len(prompt_input_ids) != len(prompt_attention_mask):
            raise ValueError("Prompt input ids and attention mask should have the same length.")

        answer_input_ids = full_tokenized["input_ids"][response_token_ids_start_idx:]
        answer_attention_mask = full_tokenized["attention_mask"][response_token_ids_start_idx:]

        return dict(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=answer_input_ids,
            attention_mask=answer_attention_mask,
        )

    def tokenize_row(self, feature, model: Optional[Union[PreTrainedModel, nn.Module]] = None) -> Dict:
        batch = {}
        is_flipped = int(bool(feature.get("is_flipped", False)))
        prompt = feature["prompt"]
        chosen = feature["chosen"]
        rejected = feature["rejected"]

        if not self.is_encoder_decoder:
            if not isinstance(prompt, str):
                raise ValueError(f"prompt should be an str but got {type(prompt)}")
            prompt_tokens = self.tokenizer(prompt, add_special_tokens=False)
            prompt_tokens = {f"prompt_{k}": v for k, v in prompt_tokens.items()}

            if not isinstance(chosen, str):
                raise ValueError(f"chosen should be an str but got {type(chosen)}")
            chosen_tokens = self.build_tokenized_answer(prompt, chosen)

            if not isinstance(rejected, str):
                raise ValueError(f"rejected should be an str but got {type(rejected)}")
            rejected_tokens = self.build_tokenized_answer(prompt, rejected)

            chosen_prompt_len_input_ids = len(chosen_tokens["prompt_input_ids"])
            rejected_prompt_len_input_ids = len(rejected_tokens["prompt_input_ids"])
            prompt_len_input_ids = min(chosen_prompt_len_input_ids, rejected_prompt_len_input_ids)

            for k, v in prompt_tokens.items():
                prompt_tokens[k] = v[:prompt_len_input_ids]

            num_diff_tokens = sum(
                [a != b for a, b in zip(chosen_tokens["prompt_input_ids"], rejected_tokens["prompt_input_ids"])]
            )
            num_diff_len = abs(chosen_prompt_len_input_ids - rejected_prompt_len_input_ids)
            if num_diff_tokens > 1 or num_diff_len > 1:
                raise ValueError(
                    "Chosen and rejected prompt_input_ids might only differ on the "
                    "last token due to tokenizer merge ops."
                )

            bos_token_id = self.tokenizer.bos_token_id
            if prompt_len_input_ids == 0 or bos_token_id != prompt_tokens["prompt_input_ids"][0]:
                prompt_tokens["prompt_input_ids"] = [bos_token_id] + prompt_tokens["prompt_input_ids"]
                prompt_tokens["prompt_attention_mask"] = [1] + prompt_tokens["prompt_attention_mask"]
            if chosen_prompt_len_input_ids == 0 or bos_token_id != chosen_tokens["prompt_input_ids"][0]:
                chosen_tokens["prompt_input_ids"] = [bos_token_id] + chosen_tokens["prompt_input_ids"]
                chosen_tokens["prompt_attention_mask"] = [1] + chosen_tokens["prompt_attention_mask"]
            if rejected_prompt_len_input_ids == 0 or bos_token_id != rejected_tokens["prompt_input_ids"][0]:
                rejected_tokens["prompt_input_ids"] = [bos_token_id] + rejected_tokens["prompt_input_ids"]
                rejected_tokens["prompt_attention_mask"] = [1] + rejected_tokens["prompt_attention_mask"]

            eos_token_id = self.tokenizer.eos_token_id
            if len(chosen_tokens["input_ids"]) == 0 or eos_token_id != chosen_tokens["input_ids"][-1]:
                chosen_tokens["input_ids"].append(eos_token_id)
                chosen_tokens["attention_mask"].append(1)
            if len(rejected_tokens["input_ids"]) == 0 or eos_token_id != rejected_tokens["input_ids"][-1]:
                rejected_tokens["input_ids"].append(eos_token_id)
                rejected_tokens["attention_mask"].append(1)

            longer_response_length = max(len(chosen_tokens["input_ids"]), len(rejected_tokens["input_ids"]))

            for answer_tokens in [chosen_tokens, rejected_tokens, prompt_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    if self.truncation_mode == "keep_start":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][: self.max_prompt_length]
                    elif self.truncation_mode == "keep_end":
                        for k in ["prompt_input_ids", "prompt_attention_mask"]:
                            answer_tokens[k] = answer_tokens[k][-self.max_prompt_length:]
                    else:
                        raise ValueError(f"Unknown truncation mode: {self.truncation_mode}")

            for answer_tokens in [chosen_tokens, rejected_tokens]:
                if len(answer_tokens["prompt_input_ids"]) + longer_response_length > self.max_length:
                    for k in ["input_ids", "attention_mask"]:
                        answer_tokens[k] = answer_tokens[k][: self.max_length - self.max_prompt_length]

            chosen_sequence_tokens = {
                k: chosen_tokens[f"prompt_{k}"] + chosen_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            rejected_sequence_tokens = {
                k: rejected_tokens[f"prompt_{k}"] + rejected_tokens[k] for k in ["input_ids", "attention_mask"]
            }
            chosen_sequence_tokens["labels"] = chosen_sequence_tokens["input_ids"][:]
            chosen_sequence_tokens["labels"][: len(chosen_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(chosen_tokens["prompt_input_ids"])
            rejected_sequence_tokens["labels"] = rejected_sequence_tokens["input_ids"][:]
            rejected_sequence_tokens["labels"][: len(rejected_tokens["prompt_input_ids"])] = [
                self.label_pad_token_id
            ] * len(rejected_tokens["prompt_input_ids"])

            for k, toks in {
                "chosen_": chosen_sequence_tokens,
                "rejected_": rejected_sequence_tokens,
                "": prompt_tokens,
            }.items():
                for type_key, tokens in toks.items():
                    if type_key == "token_type_ids":
                        continue
                    batch[f"{k}{type_key}"] = tokens

        else:
            chosen_tokens = self.tokenizer(
                chosen, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            rejected_tokens = self.tokenizer(
                rejected, truncation=True, max_length=self.max_target_length, add_special_tokens=True
            )
            prompt_tokens = self.tokenizer(
                prompt, truncation=True, max_length=self.max_prompt_length, add_special_tokens=True
            )

            batch["chosen_labels"] = chosen_tokens["input_ids"]
            batch["rejected_labels"] = rejected_tokens["input_ids"]
            batch["prompt_input_ids"] = prompt_tokens["input_ids"]
            batch["prompt_attention_mask"] = prompt_tokens["attention_mask"]

            if model is not None and hasattr(model, "prepare_decoder_input_ids_from_labels"):
                batch["rejected_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["rejected_labels"])
                )
                batch["chosen_decoder_input_ids"] = model.prepare_decoder_input_ids_from_labels(
                    labels=torch.tensor(batch["chosen_labels"])
                )

        batch["is_flipped"] = is_flipped
        return batch

    @staticmethod
    def concatenated_inputs(
        batch: Dict[str, Union[List, torch.LongTensor]],
        is_encoder_decoder: bool = False,
        label_pad_token_id: int = -100,
        padding_value: int = 0,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.LongTensor]:
        concatenated_batch = {}

        if is_encoder_decoder:
            max_length = max(batch["chosen_labels"].shape[1], batch["rejected_labels"].shape[1])
        else:
            max_length = max(batch["chosen_input_ids"].shape[1], batch["rejected_input_ids"].shape[1])

        for k in batch:
            if k.startswith("chosen") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("chosen", "concatenated")
                concatenated_batch[concatenated_key] = pad_to_length(batch[k], max_length, pad_value=pad_value)
        for k in batch:
            if k.startswith("rejected") and isinstance(batch[k], torch.Tensor):
                if "labels" in k or is_encoder_decoder:
                    pad_value = label_pad_token_id
                elif k.endswith("_input_ids"):
                    pad_value = padding_value
                elif k.endswith("_attention_mask"):
                    pad_value = 0
                concatenated_key = k.replace("rejected", "concatenated")
                concatenated_batch[concatenated_key] = torch.cat(
                    (
                        concatenated_batch[concatenated_key],
                        pad_to_length(batch[k], max_length, pad_value=pad_value),
                    ),
                    dim=0,
                ).to(device=device)

        if is_encoder_decoder:
            concatenated_batch["concatenated_input_ids"] = batch["prompt_input_ids"].repeat(2, 1).to(device=device)
            concatenated_batch["concatenated_attention_mask"] = (
                batch["prompt_attention_mask"].repeat(2, 1).to(device=device)
            )

        return concatenated_batch

    def edpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
        steps: torch.Tensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Per-sample ε-DPO losses with a per-pair adaptive KL penalty.

        The KL penalty is adapted per preference pair via ``updated_beta = beta / (1 + epsilon*steps)``
        where ``steps in {-1, 0, +1}`` is the monotonicity decision from the ±epsilon perturbations
        (computed in ``get_batch_loss_metrics``). The DPO log-ratio uses the STANDARD (unperturbed)
        policy log-probs — the perturbations only decide the step.

        Returns: (aggregated_loss, per_sample_losses, chosen_rewards, rejected_rewards)
        """
        pi_logratios = (policy_chosen_logps - policy_rejected_logps).to(self.accelerator.device)
        ref_logratios = (reference_chosen_logps - reference_rejected_logps).to(self.accelerator.device)
        logits = pi_logratios - ref_logratios

        steps = steps.to(logits.device).to(logits.dtype)
        updated_beta = self.beta / (1.0 + self.epsilon * steps)

        if self.loss_type == "sigmoid":
            per_sample_losses = (
                -F.logsigmoid(updated_beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-updated_beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            per_sample_losses = torch.relu(1 - updated_beta * logits)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge']")

        aggregated = per_sample_losses.mean()

        chosen_rewards = (
            updated_beta * (policy_chosen_logps - reference_chosen_logps).to(self.accelerator.device)
        ).detach()
        rejected_rewards = (
            updated_beta * (policy_rejected_logps - reference_rejected_logps).to(self.accelerator.device)
        ).detach()

        return aggregated, per_sample_losses, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: nn.Module, batch: Dict[str, Union[List, torch.LongTensor]], ref_logits: Optional[torch.FloatTensor] = None
    ):
        """Run the given model on the concatenated (chosen + rejected) inputs.

        Returns (chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, all_logits).
        When ``ref_logits`` is provided (the policy pass), two extra tensors are appended:
        the ε-perturbed chosen/rejected log-probs computed against ``ref_logits``.
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

        all_logits = model(
            concatenated_batch["concatenated_input_ids"],
            attention_mask=concatenated_batch["concatenated_attention_mask"],
            use_cache=False,
            **model_kwargs,
        ).logits

        # Standard DPO uses sum log-probs (not averaged), unlike GAPO/SimPO.
        all_logps = self.get_batch_logps(
            all_logits,
            concatenated_batch["concatenated_labels"],
            average_log_prob=False,
            is_encoder_decoder=self.is_encoder_decoder,
            label_pad_token_id=self.label_pad_token_id,
        )

        chosen_logps = all_logps[:len_chosen]
        rejected_logps = all_logps[len_chosen:]

        chosen_logits = all_logits[:len_chosen]
        rejected_logits = all_logits[len_chosen:]

        chosen_labels = concatenated_batch["concatenated_labels"][:len_chosen]

        if ref_logits is None:
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, chosen_labels, all_logits

        # Both-direction ε-perturbed policy log-probs (used only to decide the KL step):
        #   p_eps = (1+eps)*policy_logits - eps*ref_logits    (sign=+1)
        #   n_eps = (1-eps)*policy_logits + eps*ref_logits    (sign=-1)
        p_eps_all = self.get_epsilon_batch_logps(
            all_logits, ref_logits, concatenated_batch["concatenated_labels"],
            epsilon=self.epsilon, sign=1.0,
            label_pad_token_id=self.label_pad_token_id, is_encoder_decoder=self.is_encoder_decoder,
        )
        n_eps_all = self.get_epsilon_batch_logps(
            all_logits, ref_logits, concatenated_batch["concatenated_labels"],
            epsilon=self.epsilon, sign=-1.0,
            label_pad_token_id=self.label_pad_token_id, is_encoder_decoder=self.is_encoder_decoder,
        )

        return (
            chosen_logps,
            rejected_logps,
            chosen_logits,
            rejected_logits,
            chosen_labels,
            all_logits,
            p_eps_all[:len_chosen],
            p_eps_all[len_chosen:],
            n_eps_all[:len_chosen],
            n_eps_all[len_chosen:],
        )

    def compute_reference_log_probs(self, batch: Dict[str, Union[List, torch.LongTensor]]):
        compute_ref_context_manager = (
            torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext
        )
        with torch.no_grad(), compute_ref_context_manager():
            if self.ref_model is None:
                with self.null_ref_context():
                    ref_chosen_logps, ref_rejected_logps, _, _, _, ref_all_logits = self.concatenated_forward(
                        self.model, batch
                    )
            else:
                ref_chosen_logps, ref_rejected_logps, _, _, _, ref_all_logits = self.concatenated_forward(
                    self.ref_model, batch
                )
        return ref_chosen_logps.detach(), ref_rejected_logps.detach(), ref_all_logits.detach()

    @staticmethod
    def get_batch_logps(
        logits: torch.FloatTensor,
        labels: torch.LongTensor,
        average_log_prob: bool = False,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        if logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            logits = logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0

        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

        if average_log_prob:
            return (per_token_logps * loss_mask).sum(-1) / loss_mask.sum(-1)
        else:
            return (per_token_logps * loss_mask).sum(-1)

    @staticmethod
    def get_epsilon_batch_logps(
        policy_logits: torch.FloatTensor,
        ref_logits: torch.FloatTensor,
        labels: torch.LongTensor,
        epsilon: float,
        sign: float = 1.0,
        label_pad_token_id: int = -100,
        is_encoder_decoder: bool = False,
    ) -> torch.FloatTensor:
        """Summed ε-perturbed policy log-probs at the label tokens.

        eps_logits = (1 + sign*epsilon) * policy_logits - sign*epsilon * ref_logits,
        then log_softmax -> gather at labels -> zero padding -> sum per sequence.
        sign=+1 gives the positive perturbation p_eps = (1+eps)*pol - eps*ref;
        sign=-1 gives the negative perturbation n_eps = (1-eps)*pol + eps*ref.
        """
        if policy_logits.shape[:-1] != labels.shape:
            raise ValueError("Logits (batch and sequence length dim) and labels must have the same shape.")

        if not is_encoder_decoder:
            labels = labels[:, 1:].clone()
            policy_logits = policy_logits[:, :-1, :]
            ref_logits = ref_logits[:, :-1, :]
        loss_mask = labels != label_pad_token_id

        labels[labels == label_pad_token_id] = 0

        eps_logits = (1 + sign * epsilon) * policy_logits - sign * epsilon * ref_logits
        per_token_logps = torch.gather(
            eps_logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)
        ).squeeze(2)

        return (per_token_logps * loss_mask).sum(-1)

    def get_batch_loss_metrics(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
        train_eval: Literal["train", "eval"] = "train",
    ):
        metrics = {}
        prefix = "eval_" if train_eval == "eval" else ""

        # Reference forward first, so we can feed the raw ref logits into the
        # policy forward for the ε perturbation.
        ref_chosen_logps, ref_rejected_logps, ref_all_logits = self.compute_reference_log_probs(batch)

        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            chosen_labels,
            _policy_all_logits,
            p_eps_chosen_logps,
            p_eps_rejected_logps,
            n_eps_chosen_logps,
            n_eps_rejected_logps,
        ) = self.concatenated_forward(model, batch, ref_logits=ref_all_logits)

        # KL step decision per pair via monotonicity of the log-ratio under ±epsilon perturbation.
        base_logratios = policy_chosen_logps - policy_rejected_logps
        p_eps_logratios = p_eps_chosen_logps - p_eps_rejected_logps
        n_eps_logratios = n_eps_chosen_logps - n_eps_rejected_logps
        p_steps = (p_eps_logratios > base_logratios) & (base_logratios > n_eps_logratios)
        n_steps = (n_eps_logratios > base_logratios) & (base_logratios > p_eps_logratios)
        steps = p_steps.float() - n_steps.float()  # in {-1, 0, +1}

        loss, per_sample_losses, chosen_rewards, rejected_rewards = self.edpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            steps,
        )

        if self.sft_weight > 0.0:
            if not self.is_encoder_decoder:
                policy_chosen_logits_shift = policy_chosen_logits[..., :-1, :].contiguous()
                chosen_labels_shift = chosen_labels[..., 1:].clone()
            else:
                policy_chosen_logits_shift = policy_chosen_logits
                chosen_labels_shift = chosen_labels
            loss_func = nn.CrossEntropyLoss()
            sft_loss = loss_func(
                policy_chosen_logits_shift.view(-1, policy_chosen_logits_shift.shape[-1]),
                chosen_labels_shift.view(-1),
            )
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

        with torch.no_grad():
            metrics[f"{prefix}loss/per_sample_mean"] = per_sample_losses.mean().detach().cpu()
            metrics[f"{prefix}loss/per_sample_max"] = per_sample_losses.max().detach().cpu()
            metrics[f"{prefix}loss/per_sample_std"] = per_sample_losses.std(unbiased=False).detach().cpu()
            metrics[f"{prefix}loss/mean"] = loss.detach().cpu()

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
        with torch.no_grad():
            updated_beta = self.beta / (1.0 + self.epsilon * steps.to(policy_chosen_logps.device))
            metrics[f"{prefix}beta/updated_beta_mean"] = updated_beta.mean().cpu()
            metrics[f"{prefix}steps/p_frac"] = (steps > 0).float().mean().cpu()
            metrics[f"{prefix}steps/n_frac"] = (steps < 0).float().mean().cpu()
            metrics[f"{prefix}steps/zero_frac"] = (steps == 0).float().mean().cpu()
        metrics[f"{prefix}logps/ref_rejected"] = ref_rejected_logps.detach().mean().cpu()
        metrics[f"{prefix}logps/ref_chosen"] = ref_chosen_logps.detach().mean().cpu()
        metrics[f"{prefix}logits/rejected"] = policy_rejected_logits.detach().mean().cpu()
        metrics[f"{prefix}logits/chosen"] = policy_chosen_logits.detach().mean().cpu()

        return loss, metrics

    def compute_loss(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        return_outputs=False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        if not self.use_dpo_data_collator:
            warnings.warn(
                "compute_loss is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior."
            )

        compute_loss_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with compute_loss_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="train")

        self.store_metrics(metrics, train_eval="train")

        if return_outputs:
            return (loss, metrics)
        return loss

    def get_batch_samples(self, model, batch: Dict[str, torch.LongTensor]) -> Tuple[str, str]:
        generate_context_manager = nullcontext if not self._peft_has_been_casted_to_bf16 else torch.cuda.amp.autocast

        with generate_context_manager():
            policy_output = model.generate(
                input_ids=batch["prompt_input_ids"],
                attention_mask=batch["prompt_attention_mask"],
                max_length=self.max_length,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        policy_output = pad_to_length(policy_output, self.max_length, self.tokenizer.pad_token_id)
        policy_output_decoded = self.tokenizer.batch_decode(policy_output, skip_special_tokens=True)

        return policy_output_decoded

    def prediction_step(
        self,
        model: Union[PreTrainedModel, nn.Module],
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ):
        if not self.use_dpo_data_collator:
            warnings.warn(
                "prediction_step is only implemented for DPODataCollatorWithPadding, and you passed a datacollator that is different than "
                "DPODataCollatorWithPadding - you might see unexpected behavior."
            )
        if ignore_keys is None:
            if hasattr(model, "config"):
                ignore_keys = getattr(model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        prediction_context_manager = torch.cuda.amp.autocast if self._peft_has_been_casted_to_bf16 else nullcontext

        with torch.no_grad(), prediction_context_manager():
            loss, metrics = self.get_batch_loss_metrics(model, inputs, train_eval="eval")

        self.store_metrics(metrics, train_eval="eval")

        if prediction_loss_only:
            return (loss.detach(), None, None)

        logits_dict = {
            "eval_logits/chosen": metrics["eval_logits/chosen"],
            "eval_logits/rejected": metrics["eval_logits/rejected"],
        }
        logits = tuple(v.unsqueeze(dim=0) for k, v in logits_dict.items() if k not in ignore_keys)
        logits = torch.stack(logits).mean(axis=1).to(self.accelerator.device)
        labels = torch.zeros(logits.shape[0], device=self.accelerator.device)

        return (loss.detach(), logits, labels)

    def store_metrics(self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train") -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        if self.generate_during_eval:
            num_samples = len(dataloader.dataset)
            random_indices = random.sample(range(num_samples), k=self.args.eval_batch_size)
            random_batch_dataset = dataloader.dataset.select(random_indices)
            random_batch = self.data_collator(random_batch_dataset)
            random_batch = self._prepare_inputs(random_batch)

            policy_output_decoded = self.get_batch_samples(self.model, random_batch)

            self.log(
                {
                    "game_log": wandb.Table(
                        columns=["Prompt", "Policy"],
                        rows=[
                            [prompt, pol[len(prompt):]]
                            for prompt, pol in zip(random_batch["prompt"], policy_output_decoded)
                        ],
                    )
                }
            )
            self.state.log_history.pop()

        initial_output = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )

        return initial_output

    def log(self, logs: Dict[str, float]) -> None:
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return super().log(logs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, commit_message: Optional[str] = "End of training", blocking: bool = True, **kwargs) -> str:
        kwargs = trl_sanitze_kwargs_for_tagging(model=self.model, tag_names=self._tag_names, kwargs=kwargs)
        return super().push_to_hub(commit_message=commit_message, blocking=blocking, **kwargs)
