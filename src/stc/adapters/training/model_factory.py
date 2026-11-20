"""Backbone construction for LoRA, full fine-tuning and the tiny test model."""

from __future__ import annotations

from typing import Any

import peft
import torch
from torch import nn
from transformers import (
    AutoModelForSequenceClassification,
    BertConfig,
    BertForSequenceClassification,
)

from stc.adapters.text.hf_tokenizer import HfTokenizer
from stc.adapters.training.config import TrainingConfig
from stc.domain.errors import InvalidInputError

__all__ = [
    "TINY_BACKBONE",
    "build_base_model",
    "build_model",
    "device_for_training",
    "parameter_counts",
]

TINY_BACKBONE = "stc-tiny-test"

_TINY_HIDDEN = 64
_TINY_LAYERS = 2
_TINY_HEADS = 4
_TINY_INTERMEDIATE = 128


def _build_tiny_model(n_labels: int, tokenizer: HfTokenizer) -> nn.Module:
    # The fake test backbone must be reproducible across processes so exporting
    # it (which reloads the base weights) yields bit-identical logits.
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        intermediate_size=_TINY_INTERMEDIATE,
        max_position_embeddings=max(64, tokenizer.max_length),
        type_vocab_size=2,
        num_labels=n_labels,
        pad_token_id=tokenizer.pad_id,
    )
    return BertForSequenceClassification(config)


def _ensure_trainable_head(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        if "classifier" in name or "modules_to_save" in name:
            parameter.requires_grad = True


def build_base_model(
    backbone: str,
    n_labels: int,
    *,
    tokenizer: HfTokenizer,
) -> nn.Module:
    """Build the unwrapped sequence-classification model."""
    if backbone == TINY_BACKBONE:
        return _build_tiny_model(n_labels, tokenizer)
    kwargs: dict[str, Any] = {
        "num_labels": n_labels,
        "attn_implementation": "eager",
    }
    model = AutoModelForSequenceClassification.from_pretrained(backbone, **kwargs)
    model.config.pad_token_id = tokenizer.pad_id
    return model


def build_model(
    backbone: str,
    n_labels: int,
    method: str,
    config: TrainingConfig,
    *,
    tokenizer: HfTokenizer,
) -> nn.Module:
    """Build the base model and, for LoRA runs, wrap it with PEFT."""
    if method not in {"lora", "full"}:
        raise InvalidInputError("method must be 'lora' or 'full'")
    model = build_base_model(backbone, n_labels, tokenizer=tokenizer)
    if method == "full":
        return model
    lora_config = peft.LoraConfig(
        task_type=peft.TaskType.SEQ_CLS,
        r=config.r,
        lora_alpha=config.alpha,
        lora_dropout=config.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    wrapped = peft.get_peft_model(model, lora_config)
    _ensure_trainable_head(wrapped)
    return wrapped


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    """Return ``(trainable, total)`` parameter counts."""
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total = sum(parameter.numel() for parameter in model.parameters())
    return int(trainable), int(total)


def device_for_training() -> torch.device:
    """Use CUDA when available, otherwise CPU (tests and CI run on CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
