from __future__ import annotations

import json
import os
import random
import re
import shutil
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from tqdm.auto import trange

from dflasher.data import load_texts
from dflasher.model_profile import (
    family_defaults,
    infer_family,
    target_layer_ids_for_policy,
)

console = Console()

OMLX_DRAFT_FORMAT = "dflasher.omlx-dflash"
OMLX_CACHE_FORMAT = "dflasher.omlx-hidden-cache"
OMLX_MODEL_ROOT = Path.home() / ".omlx" / "models"
OMLX_MODEL_SETTINGS_PATH = Path.home() / ".omlx" / "model_settings.json"
DEFAULT_OMLX_APP_PATH = Path("/Applications/oMLX.app")
MINIMAX_M2_TARGET_BACKEND = "dflash_mlx.engine.target_minimax_m2:MiniMaxM2TargetOps"

OMLX_LOSS_HIDDEN_MSE = "hidden-mse"
OMLX_LOSS_CE = "ce"
OMLX_LOSS_CE_HIDDEN = "ce-hidden"
OMLX_LOSS_ALIASES = {
    "hidden_mse": OMLX_LOSS_HIDDEN_MSE,
    "mse": OMLX_LOSS_HIDDEN_MSE,
    "hidden-mse": OMLX_LOSS_HIDDEN_MSE,
    "ce": OMLX_LOSS_CE,
    "cross-entropy": OMLX_LOSS_CE,
    "cross_entropy": OMLX_LOSS_CE,
    "ce-hidden": OMLX_LOSS_CE_HIDDEN,
    "ce_hidden": OMLX_LOSS_CE_HIDDEN,
}

MINIMAX_M2_TARGET_OPS_SOURCE = '''# Copyright 2026 dflasher contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.base import create_attention_mask

from dflash_mlx.engine.target_ops import TargetCapabilities


class MiniMaxM2TargetOps:
    backend_name = "minimax_m2"

    def model_type(self, target_model: Any) -> str:
        args = getattr(target_model, "args", None)
        value = getattr(args, "model_type", None)
        if value is not None:
            return str(value).lower()
        config = getattr(target_model, "config", None)
        if isinstance(config, dict):
            return str(config.get("model_type", "")).lower()
        return str(getattr(target_model, "model_type", "")).lower()

    def supports_model(self, target_model: Any) -> bool:
        if self.model_type(target_model) != "minimax_m2":
            return False
        try:
            inner = self.text_model(target_model)
        except AttributeError:
            return False
        return hasattr(inner, "layers") and hasattr(inner, "embed_tokens")

    def text_wrapper(self, target_model: Any) -> Any:
        if hasattr(target_model, "model"):
            return target_model
        raise AttributeError(f"Unsupported MiniMax-M2 target wrapper: {type(target_model)!r}")

    def text_model(self, target_model: Any) -> Any:
        wrapper = self.text_wrapper(target_model)
        if hasattr(wrapper, "model"):
            return wrapper.model
        raise AttributeError(f"Unsupported MiniMax-M2 text model: {type(wrapper)!r}")

    def embed_tokens(self, target_model: Any) -> Any:
        return self.text_model(target_model).embed_tokens

    def logits_from_hidden(self, target_model: Any, hidden_states: mx.array) -> mx.array:
        wrapper = self.text_wrapper(target_model)
        if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", False):
            return wrapper.model.embed_tokens.as_linear(hidden_states)
        return wrapper.lm_head(hidden_states)

    def family(self, target_model: Any) -> str:
        del target_model
        return "pure_attention"

    def capabilities_for(self, target_model: Any) -> TargetCapabilities:
        del target_model
        return TargetCapabilities(
            supports_dflash=True,
            supports_recurrent_rollback=False,
            supports_kv_trim=True,
            supports_prefix_snapshot=True,
            supports_rotating_cache_snapshot=False,
            supports_shared_kv=False,
            supports_target_hidden_capture=True,
            supports_verify_linear=False,
            supports_tree_verify=False,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        del cache_entries
        return False

    def extract_context_feature(
        self,
        captured_dict: dict[int, mx.array],
        target_layer_ids: list[int],
    ) -> mx.array:
        selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
        return mx.concatenate(selected, axis=-1)

    def forward_with_hidden_capture(
        self,
        target_model: Any,
        *,
        input_ids: Optional[mx.array] = None,
        cache: Optional[list[Any]] = None,
        input_embeddings: Optional[mx.array] = None,
        capture_layer_ids: Optional[set[int]] = None,
        logits_last_only: bool = False,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        inner = self.text_model(target_model)
        hidden_states = (
            input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
        )
        if cache is None:
            cache = [None] * len(inner.layers)
        capture_all = capture_layer_ids is None
        if capture_all:
            captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
        else:
            capture_layer_ids = set(capture_layer_ids)
            captured = {0: hidden_states} if 0 in capture_layer_ids else {}

        mask = create_attention_mask(hidden_states, cache[0])
        h = hidden_states
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            h = layer(h, mask, layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h

        normalized = inner.norm(h)
        if logits_last_only and isinstance(captured, dict):
            captured[-1] = normalized
        logits_hidden = normalized[:, -1:, :] if logits_last_only else normalized
        logits = self.logits_from_hidden(target_model, logits_hidden)
        return logits, captured

    def verify_block(
        self,
        *,
        target_model: Any,
        verify_ids: mx.array,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        if int(verify_ids.shape[1]) <= 0:
            raise ValueError("verify block must contain at least one token")
        return self.forward_with_hidden_capture(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )

    def verify_tree_block(
        self,
        *,
        target_model: Any,
        tree_inputs: Any,
        target_cache: list[Any],
        capture_layer_ids: Optional[set[int]] = None,
    ) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
        del target_model, tree_inputs, target_cache, capture_layer_ids
        raise NotImplementedError("MiniMax-M2 DFlash target ops do not support tree verify.")

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        del cache_entries, accepted_tree_indices
        raise NotImplementedError("MiniMax-M2 DFlash target ops do not support tree verify.")

    def install_speculative_hooks(self, target_model: Any) -> None:
        self.text_model(target_model)._dflash_speculative_hooks_installed = True

    def make_cache(
        self,
        target_model: Any,
        *,
        enable_speculative_linear_cache: bool,
        quantize_kv_cache: bool = False,
        target_fa_window: Optional[int] = None,
    ) -> list[Any]:
        del enable_speculative_linear_cache
        fa_window = 0 if target_fa_window is None else int(target_fa_window)
        if fa_window < 0:
            raise ValueError("target_fa_window must be >= 0")
        if fa_window > 0 and quantize_kv_cache:
            raise ValueError("target_fa_window does not support quantized target KV cache")
        text_model = self.text_model(target_model)
        caches: list[Any] = []
        for _layer in text_model.layers:
            if fa_window > 0:
                caches.append(cache_mod.RotatingKVCache(max_size=fa_window))
            elif quantize_kv_cache:
                caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
            else:
                caches.append(cache_mod.KVCache())
        return caches

    def arm_rollback(self, cache_entries: list[Any], *, prefix_len: int) -> None:
        del cache_entries, prefix_len

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        del acceptance_length, drafted_tokens
        replay_ns_total = 0
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "trim"):
                offset = int(getattr(cache_entry, "offset", 0) or 0)
                if offset > target_len:
                    replay_start_ns = time.perf_counter_ns()
                    cache_entry.trim(offset - target_len)
                    replay_ns_total += time.perf_counter_ns() - replay_start_ns
            elif hasattr(cache_entry, "offset"):
                offset = int(getattr(cache_entry, "offset", 0) or 0)
                if offset > target_len:
                    cache_entry.offset = target_len
            elif hasattr(cache_entry, "crop"):
                cache_entry.crop(target_len)
        return replay_ns_total

    def cleanup_generation_caches(
        self,
        target_cache: list[Any],
        draft_cache: list[Any],
    ) -> None:
        draft_cache.clear()
        target_cache.clear()
'''


@dataclass(frozen=True)
class OmlxDraftConfig:
    source_model: str
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    block_size: int
    target_layer_ids: tuple[int, ...]
    num_target_layers: int
    mask_token_id: int = 0
    rope_scaling: dict[str, Any] | None = None
    layer_types: tuple[str, ...] = ()
    sliding_window: int | None = None
    final_logit_softcapping: float | None = None
    draft_format: str = OMLX_DRAFT_FORMAT
    training_data_source: str = "unknown"
    training_objective: str = "hidden_mse"

    def __post_init__(self) -> None:
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive.")
        if self.num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1.")
        if self.num_key_value_heads < 1:
            raise ValueError("num_key_value_heads must be at least 1.")
        if self.head_dim < 1:
            raise ValueError("head_dim must be at least 1.")
        if self.intermediate_size < 1:
            raise ValueError("intermediate_size must be at least 1.")
        if self.vocab_size < 1:
            raise ValueError("vocab_size must be positive.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.num_target_layers < 1:
            raise ValueError("num_target_layers must be positive.")
        if not self.target_layer_ids:
            raise ValueError("target_layer_ids must not be empty.")
        invalid = [
            layer_id
            for layer_id in self.target_layer_ids
            if layer_id < 0 or layer_id >= self.num_target_layers
        ]
        if invalid:
            raise ValueError(f"target_layer_ids out of range: {invalid}")


@dataclass(frozen=True)
class OmlxCacheMetadata:
    source_model: str
    cache_format: str
    selected_layer_ids: tuple[int, ...]
    hidden_size: int
    context_width: int
    vocab_size: int
    block_size: int
    mask_token_id: int
    max_length: int
    samples: int
    files: tuple[str, ...]
    dtype: str

    def save(self, cache_dir: Path) -> None:
        payload = asdict(self)
        payload["selected_layer_ids"] = list(self.selected_layer_ids)
        payload["files"] = list(self.files)
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / "dflasher_omlx_cache.json").write_text(
            json.dumps(payload, indent=2) + "\n"
        )

    @classmethod
    def load(cls, cache_dir: str | Path) -> OmlxCacheMetadata:
        path = Path(cache_dir)
        payload = json.loads((path / "dflasher_omlx_cache.json").read_text())
        payload["selected_layer_ids"] = tuple(payload["selected_layer_ids"])
        payload["files"] = tuple(payload["files"])
        return cls(**payload)


@dataclass(frozen=True)
class OmlxBuildOptions:
    source_model: str
    output_dir: Path
    texts_file: str | None = None
    dataset_name: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    data_limit: int | None = None
    allow_builtin_data: bool = False
    cache_dir: Path | None = None
    max_samples: int = 8
    max_length: int = 128
    block_size: int = 8
    draft_layers: int = 2
    layer_policy: str = "auto"
    target_layer_ids: tuple[int, ...] | None = None
    mask_token_id: int = 0
    max_steps: int = 20
    learning_rate: float = 1e-4
    loss_fn: str = OMLX_LOSS_CE_HIDDEN
    hidden_loss_weight: float = 0.01
    seed: int = 13
    overwrite: bool = False
    train: bool = True

    def __post_init__(self) -> None:
        if self.max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        if self.max_length < self.block_size + 1:
            raise ValueError("max_length must be larger than block_size.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.draft_layers < 1:
            raise ValueError("draft_layers must be at least 1.")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        normalize_omlx_loss_fn(self.loss_fn)
        if self.hidden_loss_weight < 0:
            raise ValueError("hidden_loss_weight must be non-negative.")
        if not self.texts_file and not self.dataset_name and not self.allow_builtin_data:
            raise ValueError(
                "OMLX build requires --texts-file or --dataset. "
                "Pass --allow-builtin-data only for smoke/debug builds."
            )


@dataclass(frozen=True)
class OmlxEvalResult:
    prompts: int
    exact_matches: int
    mean_acceptance: float
    mean_draft_acceptance: float
    max_prompt_tokens: int


@dataclass(frozen=True)
class OmlxInstallResult:
    source_model: str
    draft_model: str
    settings_path: Path
    compatible_with_native_omlx: bool
    compatibility_reason: str


@dataclass(frozen=True)
class OmlxAppPatchResult:
    app_path: Path
    target_ops_path: Path
    target_backend_path: Path
    dflash_engine_path: Path
    changed_paths: tuple[Path, ...]


class _LayerHook:
    def __init__(self, layer: Any, index: int, storage: list[Any]) -> None:
        self._layer = layer
        self._index = index
        self._storage = storage

    def __call__(self, *args, **kwargs):
        output = self._layer(*args, **kwargs)
        self._storage[self._index] = output[0] if isinstance(output, tuple) else output
        return output

    def __getattr__(self, name: str) -> Any:
        return getattr(self._layer, name)


def _import_mlx():
    try:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx_lm import load as mlx_lm_load
        from mlx_lm.models.cache import make_prompt_cache
        from mlx_lm.sample_utils import make_sampler
    except ImportError as exc:
        raise RuntimeError(
            "OMLX/MLX backend requires the mac extra: pip install -e '.[mac]'"
        ) from exc
    return mx, nn, optim, mlx_lm_load, make_prompt_cache, make_sampler


def _import_dflash_mlx():
    try:
        from dflash.model_mlx import DFlashConfig, DFlashDraftModel, stream_generate
    except ImportError as exc:
        raise RuntimeError(
            "OMLX DFlash runtime requires z-lab dflash MLX support: "
            "pip install -e '.[zlab-mlx]'"
        ) from exc
    return DFlashConfig, DFlashDraftModel, stream_generate


def _model_name_tokens(value: str) -> set[str]:
    ignored = {"bf16", "mlx", "model", "models", "local"}
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if token and token not in ignored
    }


def normalize_omlx_loss_fn(loss_fn: str) -> str:
    normalized = OMLX_LOSS_ALIASES.get(loss_fn.strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(set(OMLX_LOSS_ALIASES.values())))
        raise ValueError(f"Unsupported OMLX loss function {loss_fn!r}; choose one of: {choices}")
    return normalized


def resolve_omlx_source_model(source_model: str | Path) -> str:
    raw = Path(str(source_model)).expanduser()
    if raw.exists():
        return str(raw)
    if not raw.is_absolute():
        direct = OMLX_MODEL_ROOT / str(source_model)
        if direct.exists():
            return str(direct)
    if not OMLX_MODEL_ROOT.exists():
        return str(raw)
    wanted = _model_name_tokens(raw.name or str(source_model))
    if not wanted:
        return str(raw)
    matches = [
        candidate
        for candidate in OMLX_MODEL_ROOT.iterdir()
        if candidate.is_dir() and wanted.issubset(_model_name_tokens(candidate.name))
    ]
    if len(matches) == 1:
        return str(matches[0])
    return str(raw)


def read_model_config(source_model: str | Path) -> dict[str, Any]:
    config_path = Path(resolve_omlx_source_model(source_model)) / "config.json"
    if not config_path.exists():
        raise ValueError(
            "OMLX source must be a local model directory with config.json: "
            f"{source_model}"
        )
    return json.loads(config_path.read_text())


def _config_value(config: dict[str, Any], name: str, default: Any = None) -> Any:
    text_config = config.get("text_config") or {}
    return config.get(name, text_config.get(name, default))


def _num_attention_heads(config: dict[str, Any]) -> int:
    return int(_config_value(config, "num_attention_heads"))


def _num_key_value_heads(config: dict[str, Any]) -> int:
    return int(_config_value(config, "num_key_value_heads", _num_attention_heads(config)))


def _head_dim(config: dict[str, Any]) -> int:
    hidden_size = int(_config_value(config, "hidden_size"))
    return int(_config_value(config, "head_dim", hidden_size // _num_attention_heads(config)))


def resolve_omlx_layer_ids(
    source_model: str,
    layer_policy: str,
    explicit_layer_ids: tuple[int, ...] | None = None,
) -> tuple[int, ...]:
    config = read_model_config(source_model)
    num_layers = int(_config_value(config, "num_hidden_layers"))
    if explicit_layer_ids:
        invalid = [
            layer_id
            for layer_id in explicit_layer_ids
            if layer_id < 0 or layer_id >= num_layers
        ]
        if invalid:
            raise ValueError(f"target layer ids out of range for {num_layers} layers: {invalid}")
        return explicit_layer_ids
    model_type = str(_config_value(config, "model_type", "unknown"))
    family = infer_family(str(source_model).lower(), model_type)
    if layer_policy == "auto":
        policy = family_defaults(family).zlab_layer_policy
    else:
        policy = layer_policy
    return target_layer_ids_for_policy(num_layers, family, policy)  # type: ignore[arg-type]


def make_omlx_draft_config(
    source_model: str,
    block_size: int,
    draft_layers: int,
    layer_policy: str = "auto",
    target_layer_ids: tuple[int, ...] | None = None,
    mask_token_id: int = 0,
    training_data_source: str = "unknown",
    training_objective: str = OMLX_LOSS_HIDDEN_MSE,
) -> OmlxDraftConfig:
    source_model = resolve_omlx_source_model(source_model)
    config = read_model_config(source_model)
    selected_layer_ids = resolve_omlx_layer_ids(source_model, layer_policy, target_layer_ids)
    hidden_size = int(_config_value(config, "hidden_size"))
    layer_types = tuple("full_attention" for _ in range(draft_layers))
    return OmlxDraftConfig(
        source_model=source_model,
        hidden_size=hidden_size,
        num_hidden_layers=draft_layers,
        num_attention_heads=_num_attention_heads(config),
        num_key_value_heads=_num_key_value_heads(config),
        head_dim=_head_dim(config),
        intermediate_size=int(_config_value(config, "intermediate_size", hidden_size * 4)),
        vocab_size=int(_config_value(config, "vocab_size")),
        rms_norm_eps=float(_config_value(config, "rms_norm_eps", 1e-6)),
        rope_theta=float(_config_value(config, "rope_theta", 1_000_000.0)),
        max_position_embeddings=int(_config_value(config, "max_position_embeddings", 2048)),
        block_size=block_size,
        target_layer_ids=selected_layer_ids,
        num_target_layers=int(_config_value(config, "num_hidden_layers")),
        mask_token_id=mask_token_id,
        rope_scaling=_config_value(config, "rope_scaling"),
        layer_types=layer_types,
        sliding_window=_config_value(config, "sliding_window"),
        final_logit_softcapping=_config_value(config, "final_logit_softcapping"),
        training_data_source=training_data_source,
        training_objective=normalize_omlx_loss_fn(training_objective),
    )


def _to_dflash_config(config: OmlxDraftConfig):
    DFlashConfig, _DFlashDraftModel, _stream_generate = _import_dflash_mlx()
    return DFlashConfig(
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        intermediate_size=config.intermediate_size,
        vocab_size=config.vocab_size,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        block_size=config.block_size,
        target_layer_ids=config.target_layer_ids,
        num_target_layers=config.num_target_layers,
        mask_token_id=config.mask_token_id,
        rope_scaling=config.rope_scaling,
        layer_types=config.layer_types,
        sliding_window=config.sliding_window,
        final_logit_softcapping=config.final_logit_softcapping,
    )


def _config_payload(config: OmlxDraftConfig) -> dict[str, Any]:
    source_config = read_model_config(config.source_model)
    source_model_type = str(_config_value(source_config, "model_type", "unknown"))
    payload = {
        "architectures": ["DFlashDraftModel"],
        "auto_map": {"AutoModel": "dflash.DFlashDraftModel"},
        "model_type": source_model_type,
        "dtype": "bfloat16",
        "attention_bias": bool(_config_value(source_config, "attention_bias", False)),
        "attention_dropout": float(_config_value(source_config, "attention_dropout", 0.0)),
        "hidden_act": _config_value(source_config, "hidden_act", "silu"),
        "initializer_range": float(_config_value(source_config, "initializer_range", 0.02)),
        "tie_word_embeddings": bool(_config_value(source_config, "tie_word_embeddings", False)),
        "use_cache": False,
        "bos_token_id": _config_value(source_config, "bos_token_id"),
        "eos_token_id": _config_value(source_config, "eos_token_id"),
        "pad_token_id": _config_value(source_config, "pad_token_id"),
        "hidden_size": config.hidden_size,
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "intermediate_size": config.intermediate_size,
        "vocab_size": config.vocab_size,
        "rms_norm_eps": config.rms_norm_eps,
        "rope_theta": config.rope_theta,
        "max_position_embeddings": config.max_position_embeddings,
        "block_size": config.block_size,
        "num_target_layers": config.num_target_layers,
        "mask_token_id": config.mask_token_id,
        "rope_scaling": config.rope_scaling,
        "layer_types": list(config.layer_types),
        "sliding_window": config.sliding_window,
        "final_logit_softcapping": config.final_logit_softcapping,
        "dflash_config": {
            "target_layer_ids": list(config.target_layer_ids),
            "mask_token_id": config.mask_token_id,
        },
        "dflasher": {
            "draft_format": config.draft_format,
            "source_model": config.source_model,
            "training_data_source": config.training_data_source,
            "training_objective": config.training_objective,
        },
    }
    return {key: value for key, value in payload.items() if value is not None}


def init_omlx_draft(
    source_model: str,
    output_dir: Path,
    block_size: int = 8,
    draft_layers: int = 2,
    layer_policy: str = "auto",
    target_layer_ids: tuple[int, ...] | None = None,
    mask_token_id: int = 0,
    training_data_source: str = "unknown",
    training_objective: str = OMLX_LOSS_HIDDEN_MSE,
    overwrite: bool = False,
) -> Path:
    source_model = resolve_omlx_source_model(source_model)
    if output_dir.exists():
        if not overwrite:
            raise ValueError(f"Output path already exists. Pass --overwrite: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = make_omlx_draft_config(
        source_model=source_model,
        block_size=block_size,
        draft_layers=draft_layers,
        layer_policy=layer_policy,
        target_layer_ids=target_layer_ids,
        mask_token_id=mask_token_id,
        training_data_source=training_data_source,
        training_objective=training_objective,
    )
    _DFlashConfig, DFlashDraftModel, _stream_generate = _import_dflash_mlx()
    draft = DFlashDraftModel(_to_dflash_config(config))
    (output_dir / "config.json").write_text(json.dumps(_config_payload(config), indent=2) + "\n")
    draft.save_weights(str(output_dir / "model.safetensors"))
    return output_dir


def read_omlx_draft_config(draft_dir: str | Path) -> OmlxDraftConfig:
    path = Path(draft_dir)
    cfg = json.loads((path / "config.json").read_text())
    layer_types = tuple(cfg.get("layer_types") or ["full_attention"] * cfg["num_hidden_layers"])
    dflasher_cfg = cfg.get("dflasher") or {}
    return OmlxDraftConfig(
        source_model=dflasher_cfg.get("source_model", ""),
        hidden_size=cfg["hidden_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg["num_key_value_heads"],
        head_dim=cfg["head_dim"],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg["rms_norm_eps"],
        rope_theta=cfg["rope_theta"],
        max_position_embeddings=cfg["max_position_embeddings"],
        block_size=cfg["block_size"],
        target_layer_ids=tuple(cfg["dflash_config"]["target_layer_ids"]),
        num_target_layers=cfg["num_target_layers"],
        mask_token_id=cfg["dflash_config"].get("mask_token_id", cfg.get("mask_token_id", 0)),
        rope_scaling=cfg.get("rope_scaling"),
        layer_types=layer_types,
        sliding_window=cfg.get("sliding_window"),
        final_logit_softcapping=cfg.get("final_logit_softcapping"),
        training_data_source=dflasher_cfg.get("training_data_source", "unknown"),
        training_objective=dflasher_cfg.get("training_objective", OMLX_LOSS_HIDDEN_MSE),
    )


def load_omlx_draft(draft_dir: str | Path):
    path = Path(draft_dir)
    config = read_omlx_draft_config(path)
    _DFlashConfig, DFlashDraftModel, _stream_generate = _import_dflash_mlx()
    draft = DFlashDraftModel(_to_dflash_config(config))
    draft.load_weights(str(path / "model.safetensors"))
    return draft


def _get_layers(model) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "language_model") and hasattr(model.language_model, "layers"):
        return model.language_model.layers
    if hasattr(model, "layers"):
        return model.layers
    raise AttributeError(f"Cannot find layers in {type(model).__name__}")


def _get_embed_tokens(model):
    if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        return model.model.embed_tokens
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
        if hasattr(inner, "embed_tokens"):
            return inner.embed_tokens
    if hasattr(model, "embed_tokens"):
        return model.embed_tokens
    raise AttributeError(f"Cannot find embed_tokens in {type(model).__name__}")


def _get_lm_head(model):
    lm = getattr(model, "language_model", model)
    if hasattr(model, "lm_head"):
        return model.lm_head
    if hasattr(lm, "lm_head"):
        return lm.lm_head
    return _get_embed_tokens(model).as_linear


@contextmanager
def capture_selected_layers(model, layer_ids: tuple[int, ...]) -> Iterator[list[Any]]:
    layers = _get_layers(model)
    originals = {layer_id: layers[layer_id] for layer_id in layer_ids}
    storage = [None] * len(layer_ids)
    try:
        for index, layer_id in enumerate(layer_ids):
            layers[layer_id] = _LayerHook(layers[layer_id], index, storage)
        yield storage
    finally:
        for layer_id, original in originals.items():
            layers[layer_id] = original


def _encode_text(tokenizer, text: str, max_length: int) -> list[int]:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    return [int(token) for token in tokens[:max_length]]


def _as_numpy(array, dtype: str = "float16") -> np.ndarray:
    import mlx.core as mx

    out = np.array(array.astype(mx.float32))
    if dtype == "float16":
        return out.astype(np.float16)
    if dtype == "float32":
        return out.astype(np.float32)
    raise ValueError("cache dtype must be float16 or float32.")


def extract_omlx_hidden_cache(
    source_model: str,
    cache_dir: Path,
    texts_file: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    text_column: str = "text",
    data_limit: int | None = None,
    allow_builtin_data: bool = False,
    max_samples: int = 8,
    max_length: int = 128,
    block_size: int = 8,
    layer_policy: str = "auto",
    target_layer_ids: tuple[int, ...] | None = None,
    mask_token_id: int = 0,
    dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    source_model = resolve_omlx_source_model(source_model)
    if cache_dir.exists():
        if not overwrite:
            raise ValueError(f"Cache path already exists. Pass --overwrite: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    work_dir.mkdir(parents=True)
    mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    selected_layer_ids = resolve_omlx_layer_ids(source_model, layer_policy, target_layer_ids)
    config = read_model_config(source_model)
    hidden_size = int(_config_value(config, "hidden_size"))
    vocab_size = int(_config_value(config, "vocab_size"))
    texts = load_texts(
        texts_file=texts_file,
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        text_column=text_column,
        limit=data_limit,
        allow_builtin_data=allow_builtin_data,
    )
    console.print(f"[bold]Loading OMLX source model[/bold] {source_model}")
    model, tokenizer = mlx_lm_load(source_model, lazy=True)
    if hasattr(model, "eval"):
        model.eval()
    embed_tokens = _get_embed_tokens(model)
    mask_embedding = embed_tokens(mx.array([mask_token_id]))[0]
    sample_files: list[str] = []
    started = time.perf_counter()
    try:
        for text in texts:
            if len(sample_files) >= max_samples:
                break
            token_ids = _encode_text(tokenizer, text, max_length)
            if len(token_ids) < block_size + 1:
                continue
            input_ids = mx.array(token_ids, dtype=mx.uint32)[None, :]
            with capture_selected_layers(model, selected_layer_ids) as captured:
                logits = model(input_ids)
                selected = [hidden for hidden in captured if hidden is not None]
                if len(selected) != len(selected_layer_ids):
                    raise RuntimeError("Failed to capture every selected hidden-state layer.")
                context_hidden = mx.concatenate(selected, axis=-1)
                target_hidden = selected[-1]
                token_embeddings = embed_tokens(input_ids)
            mx.eval(logits, context_hidden, target_hidden, token_embeddings, mask_embedding)
            sample_name = f"sample_{len(sample_files):05d}.npz"
            np.savez_compressed(
                work_dir / sample_name,
                tokens=np.asarray(token_ids, dtype=np.int64),
                context_hidden=_as_numpy(context_hidden[0], dtype),
                target_hidden=_as_numpy(target_hidden[0], dtype),
                token_embeddings=_as_numpy(token_embeddings[0], dtype),
                mask_embedding=_as_numpy(mask_embedding, dtype),
            )
            sample_files.append(sample_name)
            console.print(f"[dim]cached sample {len(sample_files)} tokens={len(token_ids)}[/dim]")
        if not sample_files:
            raise ValueError("No cache samples were written; provide longer training text.")
        metadata = OmlxCacheMetadata(
            source_model=source_model,
            cache_format=OMLX_CACHE_FORMAT,
            selected_layer_ids=selected_layer_ids,
            hidden_size=hidden_size,
            context_width=hidden_size * len(selected_layer_ids),
            vocab_size=vocab_size,
            block_size=block_size,
            mask_token_id=mask_token_id,
            max_length=max_length,
            samples=len(sample_files),
            files=tuple(sample_files),
            dtype=dtype,
        )
        metadata.save(work_dir)
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        work_dir.rename(cache_dir)
    except Exception:
        shutil.rmtree(work_dir, ignore_errors=True)
        raise
    console.print(
        f"[green]Saved OMLX hidden cache[/green] {cache_dir} "
        f"({len(sample_files)} samples, {time.perf_counter() - started:.1f}s)"
    )
    return cache_dir


def _load_cache_arrays(cache_dir: Path, metadata: OmlxCacheMetadata) -> list[dict[str, np.ndarray]]:
    samples = []
    for file_name in metadata.files:
        with np.load(cache_dir / file_name) as data:
            samples.append({name: data[name].copy() for name in data.files})
    return samples


def _direct_draft_hidden(draft, block_embeddings, target_hidden, cache):
    h = block_embeddings * getattr(draft, "embed_scale", 1.0)
    h_ctx = draft.hidden_norm(draft.fc(target_hidden))
    for layer, layer_cache in zip(draft.layers, cache, strict=True):
        h = layer(h, h_ctx, draft.rope, layer_cache)
    return draft.norm(h)


def _softcap_logits(logits, cap: float | None):
    if cap is None:
        return logits
    mx, _nn, _optim, _mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    return mx.tanh(logits / cap) * cap


def _verify_cache_matches_draft(metadata: OmlxCacheMetadata, draft_config: OmlxDraftConfig) -> None:
    problems: list[str] = []
    if draft_config.source_model:
        cache_source = Path(resolve_omlx_source_model(metadata.source_model)).resolve(strict=False)
        draft_source = Path(resolve_omlx_source_model(draft_config.source_model)).resolve(
            strict=False
        )
        if cache_source != draft_source:
            problems.append(f"source_model {cache_source} != {draft_source}")
    if metadata.hidden_size != draft_config.hidden_size:
        problems.append(f"hidden_size {metadata.hidden_size} != {draft_config.hidden_size}")
    if metadata.context_width != draft_config.hidden_size * len(draft_config.target_layer_ids):
        problems.append(
            "context_width "
            f"{metadata.context_width} != "
            f"{draft_config.hidden_size * len(draft_config.target_layer_ids)}"
        )
    if metadata.block_size != draft_config.block_size:
        problems.append(f"block_size {metadata.block_size} != {draft_config.block_size}")
    if metadata.mask_token_id != draft_config.mask_token_id:
        problems.append(f"mask_token_id {metadata.mask_token_id} != {draft_config.mask_token_id}")
    if metadata.vocab_size != draft_config.vocab_size:
        problems.append(f"vocab_size {metadata.vocab_size} != {draft_config.vocab_size}")
    if metadata.selected_layer_ids != draft_config.target_layer_ids:
        problems.append(
            f"selected_layer_ids {metadata.selected_layer_ids} != {draft_config.target_layer_ids}"
        )
    if problems:
        raise ValueError("OMLX cache and draft are incompatible: " + "; ".join(problems))


def train_omlx_draft_from_cache(
    cache_dir: Path,
    draft_dir: Path,
    max_steps: int = 20,
    learning_rate: float = 1e-4,
    source_model: str | None = None,
    loss_fn: str = OMLX_LOSS_CE_HIDDEN,
    hidden_loss_weight: float = 0.01,
    seed: int = 13,
) -> Path:
    if max_steps < 1:
        return draft_dir
    loss_name = normalize_omlx_loss_fn(loss_fn)
    if hidden_loss_weight < 0:
        raise ValueError("hidden_loss_weight must be non-negative.")
    mx, nn, optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    metadata = OmlxCacheMetadata.load(cache_dir)
    if metadata.cache_format != OMLX_CACHE_FORMAT:
        raise ValueError(f"Unsupported OMLX cache format: {metadata.cache_format}")
    draft_config = read_omlx_draft_config(draft_dir)
    _verify_cache_matches_draft(metadata, draft_config)
    draft = load_omlx_draft(draft_dir)
    samples = _load_cache_arrays(cache_dir, metadata)
    rng = random.Random(seed)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    lm_head = None
    if loss_name in {OMLX_LOSS_CE, OMLX_LOSS_CE_HIDDEN}:
        target_ref = resolve_omlx_source_model(source_model or metadata.source_model)
        console.print(f"[bold]Loading OMLX target lm_head for CE loss[/bold] {target_ref}")
        target_model, _tokenizer = mlx_lm_load(target_ref, lazy=True)
        if hasattr(target_model, "eval"):
            target_model.eval()
        lm_head = _get_lm_head(target_model)

    def loss_fn_inner(model, context_hidden, block_embeddings, label_hidden, label_tokens):
        hidden = _direct_draft_hidden(model, block_embeddings, context_hidden, model.make_cache())
        pred = hidden[:, 1:, :]
        hidden_loss = ((pred - label_hidden) ** 2).mean()
        if loss_name == OMLX_LOSS_HIDDEN_MSE:
            return hidden_loss
        if lm_head is None:
            raise RuntimeError("CE training requires a loaded target lm_head.")
        logits = _softcap_logits(lm_head(pred), draft_config.final_logit_softcapping)
        ce_loss = nn.losses.cross_entropy(logits, label_tokens, reduction="mean")
        if loss_name == OMLX_LOSS_CE:
            return ce_loss
        return ce_loss + (hidden_loss_weight * hidden_loss)

    value_and_grad = nn.value_and_grad(draft, loss_fn_inner)
    progress = trange(max_steps, desc="omlx-draft-training", leave=True)
    for _ in progress:
        sample = rng.choice(samples)
        token_count = int(sample["tokens"].shape[0])
        max_anchor = token_count - metadata.block_size
        anchor = rng.randint(0, max_anchor)
        context_hidden = mx.array(sample["context_hidden"][: anchor + 1][None, :, :])
        target_hidden = mx.array(
            sample["target_hidden"][anchor + 1 : anchor + metadata.block_size][None, :, :]
        )
        label_tokens = mx.array(
            sample["tokens"][anchor + 1 : anchor + metadata.block_size][None, :],
            dtype=mx.uint32,
        )
        anchor_embedding = mx.array(sample["token_embeddings"][anchor])
        mask_embedding = mx.array(sample["mask_embedding"])
        mask_block = mx.broadcast_to(
            mask_embedding,
            (metadata.block_size - 1, metadata.hidden_size),
        )
        block_embeddings = mx.concatenate([anchor_embedding[None, :], mask_block], axis=0)[
            None, :, :
        ]
        loss, grads = value_and_grad(
            draft,
            context_hidden,
            block_embeddings,
            target_hidden,
            label_tokens,
        )
        optimizer.update(draft, grads)
        mx.eval(draft.parameters(), optimizer.state, loss)
        progress.set_postfix(loss=f"{float(loss.item()):.5f}")
    draft.save_weights(str(draft_dir / "model.safetensors"))
    manifest = {
        "source_model": metadata.source_model,
        "cache_dir": str(cache_dir),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "objective": loss_name,
        "hidden_loss_weight": hidden_loss_weight,
        "format": OMLX_DRAFT_FORMAT,
    }
    (draft_dir / "dflasher_omlx_train_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    console.print(f"[green]Updated OMLX DFlash draft[/green] {draft_dir}")
    return draft_dir


def _validate_omlx_build_paths(output_dir: Path, cache_dir: Path, overwrite: bool) -> None:
    for label, path in (("output", output_dir), ("cache", cache_dir)):
        if path.exists() and not overwrite:
            raise ValueError(f"{label} path already exists. Pass --overwrite: {path}")


def build_omlx_draft(options: OmlxBuildOptions) -> Path:
    source_model = resolve_omlx_source_model(options.source_model)
    cache_dir = (
        options.cache_dir
        or options.output_dir.parent / f"{options.output_dir.name}.omlx-cache"
    )
    _validate_omlx_build_paths(options.output_dir, cache_dir, options.overwrite)
    training_source = describe_omlx_training_data(options)
    extract_omlx_hidden_cache(
        source_model=source_model,
        cache_dir=cache_dir,
        texts_file=options.texts_file,
        dataset_name=options.dataset_name,
        dataset_split=options.dataset_split,
        text_column=options.text_column,
        data_limit=options.data_limit,
        allow_builtin_data=options.allow_builtin_data,
        max_samples=options.max_samples,
        max_length=options.max_length,
        block_size=options.block_size,
        layer_policy=options.layer_policy,
        target_layer_ids=options.target_layer_ids,
        mask_token_id=options.mask_token_id,
        overwrite=options.overwrite,
    )
    init_omlx_draft(
        source_model=source_model,
        output_dir=options.output_dir,
        block_size=options.block_size,
        draft_layers=options.draft_layers,
        layer_policy=options.layer_policy,
        target_layer_ids=options.target_layer_ids,
        mask_token_id=options.mask_token_id,
        training_data_source=training_source,
        training_objective=options.loss_fn,
        overwrite=options.overwrite,
    )
    if options.train and options.max_steps > 0:
        train_omlx_draft_from_cache(
            cache_dir=cache_dir,
            draft_dir=options.output_dir,
            max_steps=options.max_steps,
            learning_rate=options.learning_rate,
            source_model=source_model,
            loss_fn=options.loss_fn,
            hidden_loss_weight=options.hidden_loss_weight,
            seed=options.seed,
        )
    build_manifest = {
        "source_model": source_model,
        "backend": "omlx",
        "output": str(options.output_dir),
        "cache_dir": str(cache_dir),
        "training_objective": normalize_omlx_loss_fn(options.loss_fn),
        "hidden_loss_weight": options.hidden_loss_weight,
        "format": OMLX_DRAFT_FORMAT,
    }
    (options.output_dir / "dflasher_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2) + "\n"
    )
    return options.output_dir


def _prompt_tokens(tokenizer, prompt: str):
    _mx, _nn, _optim, _mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    add_special_tokens = getattr(tokenizer, "bos_token", None) is None or not prompt.startswith(
        str(getattr(tokenizer, "bos_token", ""))
    )
    return tokenizer.encode(prompt, add_special_tokens=add_special_tokens)


def _eos_token_ids(tokenizer) -> set[int]:
    eos_ids: set[int] = set()
    plural = getattr(tokenizer, "eos_token_ids", None)
    if plural is not None:
        eos_ids.update(int(token_id) for token_id in plural)
    singular = getattr(tokenizer, "eos_token_id", None)
    if singular is not None:
        eos_ids.add(int(singular))
    raw = getattr(tokenizer, "_tokenizer", None)
    raw_singular = getattr(raw, "eos_token_id", None)
    if raw_singular is not None:
        eos_ids.add(int(raw_singular))
    return eos_ids


def _validate_prompt_context(
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    max_position_embeddings: int,
) -> int:
    prompt_len = len(_prompt_tokens(tokenizer, prompt))
    if prompt_len + max_new_tokens > max_position_embeddings:
        raise ValueError(
            "Prompt exceeds source context window: "
            f"prompt_tokens={prompt_len}, max_new_tokens={max_new_tokens}, "
            f"max_position_embeddings={max_position_embeddings}"
        )
    return prompt_len


def greedy_omlx_tokens(model, tokenizer, prompt: str, max_new_tokens: int) -> list[int]:
    mx, _nn, _optim, _mlx_lm_load, make_prompt_cache, make_sampler = _import_mlx()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_new_tokens == 0:
        return []
    prompt_ids = mx.array(_prompt_tokens(tokenizer, prompt), dtype=mx.uint32)
    cache = make_prompt_cache(model)
    sampler = make_sampler(temp=0.0)
    logits = model(prompt_ids[None, :], cache)
    token = int(sampler(logits[:, -1:])[0, 0].item())
    tokens = [token]
    eos_ids = _eos_token_ids(tokenizer)
    for _ in range(max_new_tokens - 1):
        if token in eos_ids:
            break
        logits = model(mx.array([[token]], dtype=mx.uint32), cache)
        token = int(sampler(logits[:, -1:])[0, 0].item())
        tokens.append(token)
    return tokens


def generate_omlx_dflash(
    source_model: str,
    draft_dir: Path,
    prompt: str,
    max_new_tokens: int = 64,
) -> tuple[str, list[int], float]:
    source_model = resolve_omlx_source_model(source_model)
    _mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    _DFlashConfig, _DFlashDraftModel, stream_generate = _import_dflash_mlx()
    model, tokenizer = mlx_lm_load(source_model, lazy=True)
    source_config = read_model_config(source_model)
    _validate_prompt_context(
        tokenizer,
        prompt,
        max_new_tokens,
        int(_config_value(source_config, "max_position_embeddings", 2048)),
    )
    draft = load_omlx_draft(draft_dir)
    text_parts: list[str] = []
    tokens: list[int] = []
    accepted: list[int] = []
    for response in stream_generate(
        model,
        draft,
        tokenizer,
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
    ):
        text_parts.append(response.text)
        tokens.extend(response.tokens)
        if response.tokens:
            accepted.append(response.accepted)
    mean_acceptance = sum(accepted) / len(accepted) if accepted else 0.0
    return "".join(text_parts), tokens, mean_acceptance


def evaluate_omlx_dflash(
    source_model: str,
    draft_dir: Path,
    prompts: list[str],
    max_new_tokens: int = 32,
) -> OmlxEvalResult:
    source_model = resolve_omlx_source_model(source_model)
    _mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    _DFlashConfig, _DFlashDraftModel, stream_generate = _import_dflash_mlx()
    model, tokenizer = mlx_lm_load(source_model, lazy=True)
    source_config = read_model_config(source_model)
    max_context = int(_config_value(source_config, "max_position_embeddings", 2048))
    draft = load_omlx_draft(draft_dir)
    exact = 0
    acceptances: list[float] = []
    draft_accepted_total = 0
    draft_committed_total = 0
    max_prompt_tokens = 0
    for prompt in prompts:
        prompt_tokens = _validate_prompt_context(tokenizer, prompt, max_new_tokens, max_context)
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        expected = greedy_omlx_tokens(model, tokenizer, prompt, max_new_tokens)
        actual: list[int] = []
        accepted: list[int] = []
        for response in stream_generate(
            model,
            draft,
            tokenizer,
            prompt,
            max_tokens=max_new_tokens,
            temperature=0.0,
        ):
            actual.extend(response.tokens)
            if response.tokens:
                accepted.append(response.accepted)
                draft_accepted_total += max(0, int(response.accepted) - 1)
                draft_committed_total += max(0, len(response.tokens) - 1)
        actual = actual[: len(expected)]
        if actual == expected:
            exact += 1
        if accepted:
            acceptances.append(sum(accepted) / len(accepted))
    return OmlxEvalResult(
        prompts=len(prompts),
        exact_matches=exact,
        mean_acceptance=sum(acceptances) / len(acceptances) if acceptances else 0.0,
        mean_draft_acceptance=(
            draft_accepted_total / draft_committed_total if draft_committed_total else 0.0
        ),
        max_prompt_tokens=max_prompt_tokens,
    )


def _resolve_omlx_app_patch_paths(app_path: str | Path) -> tuple[Path, Path, Path]:
    app = Path(app_path).expanduser()
    contents = app / "Contents"
    if not contents.exists():
        raise ValueError(f"oMLX app bundle was not found: {app}")
    site_packages = sorted(
        contents.glob("Python/framework-mlx-framework/lib/python*/site-packages")
    )
    for site_package in site_packages:
        target_ops_path = site_package / "dflash_mlx" / "engine" / "target_ops.py"
        target_backend_path = site_package / "dflash_mlx" / "engine" / "target_minimax_m2.py"
        dflash_engine_path = contents / "Resources" / "omlx" / "engine" / "dflash.py"
        if target_ops_path.exists() and dflash_engine_path.exists():
            return target_ops_path, target_backend_path, dflash_engine_path
    raise ValueError(
        "Could not find dflash_mlx/omlx engine files inside the oMLX app bundle: "
        f"{app}"
    )


def _patch_target_ops_text(text: str) -> str:
    if MINIMAX_M2_TARGET_BACKEND in text:
        return text
    gemma_backend = '    "dflash_mlx.engine.target_gemma4:Gemma4TargetOps",\n'
    minimax_backend = f'    "{MINIMAX_M2_TARGET_BACKEND}",\n'
    if gemma_backend in text:
        return text.replace(gemma_backend, gemma_backend + minimax_backend, 1)
    marker = "TARGET_BACKENDS = [\n"
    if marker in text:
        return text.replace(marker, marker + minimax_backend, 1)
    raise ValueError("Could not patch target_ops.py: TARGET_BACKENDS list was not found.")


def _patch_omlx_dflash_text(text: str) -> str:
    patched = text
    gemma_line = '    is_gemma4 = model_type in ("gemma4", "gemma4_text")\n'
    minimax_line = '    is_minimax_m2 = model_type == "minimax_m2"\n'
    if minimax_line not in patched:
        if gemma_line not in patched:
            raise ValueError("Could not patch oMLX dflash.py: compatibility gate not found.")
        patched = patched.replace(gemma_line, gemma_line + minimax_line, 1)
    patched = patched.replace(
        "if not (is_qwen or is_gemma4):",
        "if not (is_qwen or is_gemma4 or is_minimax_m2):",
        1,
    )
    patched = patched.replace(
        "DFlash supports only Qwen and Gemma4 models",
        "DFlash supports only Qwen, Gemma4, and MiniMax-M2 models",
    )
    if "is_minimax_m2" not in patched:
        raise ValueError("Could not patch oMLX dflash.py for MiniMax-M2.")
    return patched


def _write_text_with_backup(path: Path, text: str) -> None:
    backup_path = path.with_suffix(path.suffix + ".dflasher.bak")
    if path.exists() and not backup_path.exists():
        shutil.copy2(path, backup_path)
    temp_path = path.with_name(f".{path.name}.dflasher-tmp-{os.getpid()}-{time.time_ns()}")
    temp_path.write_text(text)
    temp_path.replace(path)


def patch_omlx_app_for_minimax(
    app_path: str | Path = DEFAULT_OMLX_APP_PATH,
    *,
    overwrite_target_backend: bool = False,
    dry_run: bool = False,
) -> OmlxAppPatchResult:
    target_ops_path, target_backend_path, dflash_engine_path = _resolve_omlx_app_patch_paths(
        app_path
    )
    changed: list[Path] = []
    originals: dict[Path, str | None] = {}

    target_backend_source = MINIMAX_M2_TARGET_OPS_SOURCE.rstrip() + "\n"
    target_backend_changed = (
        overwrite_target_backend
        or not target_backend_path.exists()
        or target_backend_path.read_text() != target_backend_source
    )
    if target_backend_changed:
        changed.append(target_backend_path)

    target_ops_text = target_ops_path.read_text()
    patched_target_ops = _patch_target_ops_text(target_ops_text)
    if patched_target_ops != target_ops_text:
        changed.append(target_ops_path)

    dflash_engine_text = dflash_engine_path.read_text()
    patched_dflash_engine = _patch_omlx_dflash_text(dflash_engine_text)
    if patched_dflash_engine != dflash_engine_text:
        changed.append(dflash_engine_path)

    if not dry_run:
        for path in changed:
            originals[path] = path.read_text() if path.exists() else None
        try:
            if target_backend_changed:
                _write_text_with_backup(target_backend_path, target_backend_source)
            if patched_target_ops != target_ops_text:
                _write_text_with_backup(target_ops_path, patched_target_ops)
            if patched_dflash_engine != dflash_engine_text:
                _write_text_with_backup(dflash_engine_path, patched_dflash_engine)
        except Exception:
            for path, original in originals.items():
                if original is None:
                    path.unlink(missing_ok=True)
                else:
                    path.write_text(original)
            raise

    return OmlxAppPatchResult(
        app_path=Path(app_path).expanduser(),
        target_ops_path=target_ops_path,
        target_backend_path=target_backend_path,
        dflash_engine_path=dflash_engine_path,
        changed_paths=tuple(changed),
    )


def native_omlx_dflash_compatibility(source_model: str | Path) -> tuple[bool, str]:
    cfg = read_model_config(source_model)
    model_type = str(_config_value(cfg, "model_type", "")).lower()
    if model_type == "minimax_m2":
        return True, (
            "MiniMax-M2 app serving requires: "
            "dflasher omlx patch-app --app-path /Applications/oMLX.app"
        )
    if "qwen" in model_type or model_type in {"gemma4", "gemma4_text"}:
        return True, ""
    return False, (
        "Installed oMLX DFlash target support is available for Qwen, Gemma4, "
        f"and MiniMax-M2 after the dflasher local patch (model_type='{model_type}')."
    )


def _safe_omlx_model_dir(model_root: Path, installed_name: str) -> Path:
    raw = Path(installed_name)
    if raw.is_absolute() or raw.name != installed_name or installed_name in {".", ".."}:
        raise ValueError("--installed-name must be a single directory name, not a path.")
    target_dir = (model_root / installed_name).resolve(strict=False)
    root = model_root.resolve(strict=False)
    if target_dir != root and root not in target_dir.parents:
        raise ValueError(f"Installed draft path escapes model root: {target_dir}")
    return target_dir


def _validate_omlx_draft_for_source(source_path: Path, draft_dir: Path) -> None:
    draft_config = read_omlx_draft_config(draft_dir)
    source_reference = make_omlx_draft_config(
        source_model=str(source_path),
        block_size=draft_config.block_size,
        draft_layers=draft_config.num_hidden_layers,
        target_layer_ids=draft_config.target_layer_ids,
        mask_token_id=draft_config.mask_token_id,
        training_objective=draft_config.training_objective,
    )
    if draft_config.source_model:
        draft_source = Path(resolve_omlx_source_model(draft_config.source_model)).resolve(
            strict=False
        )
        resolved_source = source_path.resolve(strict=False)
        if draft_source != resolved_source:
            raise ValueError(
                "Draft source_model does not match install target: "
                f"{draft_source} != {resolved_source}"
            )
    checks = {
        "hidden_size": (draft_config.hidden_size, source_reference.hidden_size),
        "vocab_size": (draft_config.vocab_size, source_reference.vocab_size),
        "num_target_layers": (
            draft_config.num_target_layers,
            source_reference.num_target_layers,
        ),
        "target_layer_ids": (
            draft_config.target_layer_ids,
            source_reference.target_layer_ids,
        ),
        "block_size": (draft_config.block_size, source_reference.block_size),
        "mask_token_id": (draft_config.mask_token_id, source_reference.mask_token_id),
    }
    mismatches = [
        f"{name} {actual!r} != {expected!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Draft config is incompatible with source model: " + "; ".join(mismatches))


def install_omlx_draft_for_app(
    source_model: str,
    draft_dir: Path,
    installed_name: str | None = None,
    settings_path: Path | None = None,
    model_root: Path | None = None,
    copy_draft: bool = True,
    overwrite: bool = False,
    in_memory_cache_max_entries: int = 4,
    in_memory_cache_max_bytes: int = 8 * 1024**3,
    ssd_cache: bool = True,
    ssd_cache_max_bytes: int = 20 * 1024**3,
    verify_mode: str = "adaptive",
) -> OmlxInstallResult:
    source_path = Path(resolve_omlx_source_model(source_model))
    draft_dir = Path(draft_dir).expanduser()
    if not (draft_dir / "config.json").exists() or not (draft_dir / "model.safetensors").exists():
        raise ValueError(f"Draft directory is missing config.json/model.safetensors: {draft_dir}")
    _validate_omlx_draft_for_source(source_path, draft_dir)
    compatible, reason = native_omlx_dflash_compatibility(source_path)
    source_key = source_path.name
    target_name = installed_name or f"{source_key}-DFlash-dflasher"
    settings_path = Path(settings_path or OMLX_MODEL_SETTINGS_PATH).expanduser()
    model_root = Path(model_root or OMLX_MODEL_ROOT).expanduser()
    target_dir = _safe_omlx_model_dir(model_root, target_name)
    if copy_draft:
        model_root.mkdir(parents=True, exist_ok=True)
        if target_dir.exists():
            if not overwrite:
                raise ValueError(f"Installed draft already exists. Pass --overwrite: {target_dir}")
            shutil.rmtree(target_dir)
        shutil.copytree(draft_dir, target_dir)
        draft_model_ref = str(target_dir)
    else:
        draft_model_ref = str(draft_dir.resolve(strict=False))
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    if settings_path.exists():
        payload = json.loads(settings_path.read_text())
    else:
        payload = {"version": 1, "models": {}}
    models = payload.setdefault("models", {})
    if not isinstance(models, dict):
        raise ValueError(f"Invalid OMLX model settings payload: {settings_path}")
    current = dict(models.get(source_key) or {})
    current.update(
        {
            "dflash_enabled": True,
            "dflash_draft_model": draft_model_ref,
            "dflash_in_memory_cache": True,
            "dflash_in_memory_cache_max_entries": int(in_memory_cache_max_entries),
            "dflash_in_memory_cache_max_bytes": int(in_memory_cache_max_bytes),
            "dflash_ssd_cache": bool(ssd_cache),
            "dflash_ssd_cache_max_bytes": int(ssd_cache_max_bytes),
            "dflash_verify_mode": verify_mode,
            "dflash_draft_quant_enabled": False,
            "trust_remote_code": True,
        }
    )
    models[source_key] = current
    backup_path = settings_path.with_suffix(settings_path.suffix + ".dflasher.bak")
    if settings_path.exists() and not backup_path.exists():
        shutil.copy2(settings_path, backup_path)
    settings_path.write_text(json.dumps(payload, indent=2) + "\n")
    return OmlxInstallResult(
        source_model=str(source_path),
        draft_model=draft_model_ref,
        settings_path=settings_path,
        compatible_with_native_omlx=compatible,
        compatibility_reason=reason,
    )


def describe_omlx_training_data(options: OmlxBuildOptions) -> str:
    if options.texts_file:
        return f"texts_file:{options.texts_file}"
    if options.dataset_name:
        return (
            f"dataset:{options.dataset_name};split={options.dataset_split};"
            f"column={options.text_column};limit={options.data_limit}"
        )
    if options.allow_builtin_data:
        return "builtin-smoke"
    return "unknown"
