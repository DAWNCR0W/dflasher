from __future__ import annotations

from dataclasses import dataclass

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


@dataclass(frozen=True)
class TargetShape:
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    max_position_embeddings: int


def resolve_device(device: str) -> torch.device:
    if device != "auto":
        resolved = torch.device(device)
        if resolved.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but torch.cuda.is_available() is false.")
        if resolved.type == "mps":
            mps_backend = getattr(torch.backends, "mps", None)
            if not mps_backend or not mps_backend.is_available():
                raise ValueError(
                    "MPS was requested but torch.backends.mps.is_available() is false."
                )
        return resolved
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def dtype_from_name(name: str) -> torch.dtype:
    if name == "auto":
        return torch.float32
    aliases = {
        "fp32": "float32",
        "fp16": "float16",
        "bf16": "bfloat16",
    }
    name = aliases.get(name, name)
    try:
        dtype = getattr(torch, name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported torch dtype: {name}") from exc
    if not isinstance(dtype, torch.dtype):
        raise ValueError(f"Unsupported torch dtype: {name}")
    return dtype


def get_target_shape(source_model: str, trust_remote_code: bool = False) -> TargetShape:
    config = AutoConfig.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    if hasattr(config, "text_config"):
        config = config.text_config
    hidden_size = getattr(config, "hidden_size", getattr(config, "n_embd", None))
    num_hidden_layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", None))
    max_position_embeddings = getattr(
        config, "max_position_embeddings", getattr(config, "n_positions", 2048)
    )
    if hidden_size is None or num_hidden_layers is None:
        raise ValueError("Could not infer hidden size or layer count from the source model config.")
    return TargetShape(
        hidden_size=int(hidden_size),
        num_hidden_layers=int(num_hidden_layers),
        vocab_size=int(config.vocab_size),
        max_position_embeddings=int(max_position_embeddings),
    )


def load_tokenizer(source_model: str, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        source_model,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.bos_token
    if tokenizer.pad_token_id is None:
        raise ValueError(
            "Tokenizer has no pad/eos/bos token. Set a pad token in the tokenizer before "
            "using dflasher."
        )
    return tokenizer


def load_target_model(
    source_model: str,
    device: torch.device,
    dtype: torch.dtype,
    trust_remote_code: bool = False,
):
    try:
        model = AutoModelForCausalLM.from_pretrained(
            source_model,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            source_model,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
        )
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model
