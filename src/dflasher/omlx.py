from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import sys
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
PROTECTED_OMLX_DRAFT_NAMES = frozenset(
    {
        "Qwen3.6-35B-A3B-DFlash",
        "Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-text-oQ8",
    }
)

OMLX_LOSS_HIDDEN_MSE = "hidden-mse"
OMLX_LOSS_CE = "ce"
OMLX_LOSS_CE_HIDDEN = "ce-hidden"
OMLX_LOSS_TOPK_KL = "topk-kl"
OMLX_LOSS_CE_TOPK_KL = "ce-topk-kl"
OMLX_LOSS_CE_HIDDEN_TOPK_KL = "ce-hidden-topk-kl"
OMLX_LABEL_RAW_NEXT_TOKEN = "raw-next-token"
OMLX_LABEL_TARGET_GREEDY = "target-greedy"
OMLX_HIDDEN_TARGET_SELECTED = "selected"
OMLX_HIDDEN_TARGET_FINAL = "final"
OMLX_LOSS_ALIASES = {
    "hidden_mse": OMLX_LOSS_HIDDEN_MSE,
    "mse": OMLX_LOSS_HIDDEN_MSE,
    "hidden-mse": OMLX_LOSS_HIDDEN_MSE,
    "ce": OMLX_LOSS_CE,
    "cross-entropy": OMLX_LOSS_CE,
    "cross_entropy": OMLX_LOSS_CE,
    "ce-hidden": OMLX_LOSS_CE_HIDDEN,
    "ce_hidden": OMLX_LOSS_CE_HIDDEN,
    "topk-kl": OMLX_LOSS_TOPK_KL,
    "topk_kl": OMLX_LOSS_TOPK_KL,
    "kl-topk": OMLX_LOSS_TOPK_KL,
    "kl_topk": OMLX_LOSS_TOPK_KL,
    "ce-topk-kl": OMLX_LOSS_CE_TOPK_KL,
    "ce_topk_kl": OMLX_LOSS_CE_TOPK_KL,
    "ce-kl": OMLX_LOSS_CE_TOPK_KL,
    "ce_kl": OMLX_LOSS_CE_TOPK_KL,
    "ce-hidden-topk-kl": OMLX_LOSS_CE_HIDDEN_TOPK_KL,
    "ce_hidden_topk_kl": OMLX_LOSS_CE_HIDDEN_TOPK_KL,
    "ce-hidden-kl": OMLX_LOSS_CE_HIDDEN_TOPK_KL,
    "ce_hidden_kl": OMLX_LOSS_CE_HIDDEN_TOPK_KL,
}
OMLX_LOSSES_WITH_CE = {
    OMLX_LOSS_CE,
    OMLX_LOSS_CE_HIDDEN,
    OMLX_LOSS_CE_TOPK_KL,
    OMLX_LOSS_CE_HIDDEN_TOPK_KL,
}
OMLX_LOSSES_WITH_TOPK_KL = {
    OMLX_LOSS_TOPK_KL,
    OMLX_LOSS_CE_TOPK_KL,
    OMLX_LOSS_CE_HIDDEN_TOPK_KL,
}
OMLX_LABEL_SOURCE_ALIASES = {
    "raw": OMLX_LABEL_RAW_NEXT_TOKEN,
    "raw-next-token": OMLX_LABEL_RAW_NEXT_TOKEN,
    "raw_next_token": OMLX_LABEL_RAW_NEXT_TOKEN,
    "target-greedy": OMLX_LABEL_TARGET_GREEDY,
    "target_greedy": OMLX_LABEL_TARGET_GREEDY,
    "greedy": OMLX_LABEL_TARGET_GREEDY,
}
OMLX_HIDDEN_TARGET_ALIASES = {
    "selected": OMLX_HIDDEN_TARGET_SELECTED,
    "selected-layer": OMLX_HIDDEN_TARGET_SELECTED,
    "selected_layer": OMLX_HIDDEN_TARGET_SELECTED,
    "final": OMLX_HIDDEN_TARGET_FINAL,
    "final-hidden": OMLX_HIDDEN_TARGET_FINAL,
    "final_hidden": OMLX_HIDDEN_TARGET_FINAL,
    "final-norm": OMLX_HIDDEN_TARGET_FINAL,
    "final_norm": OMLX_HIDDEN_TARGET_FINAL,
}
OMLX_MASK_TOKEN_CANDIDATES = (
    "<fim_pad>",
    "<|fim_pad|>",
    "<mask>",
    "[MASK]",
)

MINIMAX_M2_TARGET_OPS_SOURCE = '''# Copyright 2026 dflasher contributors
# Licensed under the Apache License, Version 2.0

from __future__ import annotations

import time
from typing import Any, Optional

import mlx.core as mx
from mlx_lm.models import cache as cache_mod
from mlx_lm.models.base import create_attention_mask, scaled_dot_product_attention

from dflash_mlx.engine.target_qwen_gdn import (
    _HYBRID_SDPA_EXACT_KV_THRESHOLD,
    _TREE_POSITIONS_ATTR,
    _apply_rope_positions,
    _clear_tree_cache_context,
    _commit_kv_tree_path,
    _gqa_reshape_sdpa,
    _set_tree_cache_context,
)
from dflash_mlx.engine.sampling import greedy_tokens_with_mask
from dflash_mlx.engine.target_ops import TargetCapabilities


def _minimax_project_qkv(attn: Any, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    batch_size, seq_len, _ = x.shape
    queries = attn.q_proj(x)
    keys = attn.k_proj(x)
    values = attn.v_proj(x)
    if getattr(attn, "use_qk_norm", False):
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys)
    queries = queries.reshape(batch_size, seq_len, attn.num_attention_heads, -1).transpose(
        0, 2, 1, 3
    )
    keys = keys.reshape(batch_size, seq_len, attn.num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )
    values = values.reshape(batch_size, seq_len, attn.num_key_value_heads, -1).transpose(
        0, 2, 1, 3
    )
    return queries, keys, values


def _minimax_gqa_sdpa(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    *,
    cache: Optional[Any],
    scale: float,
    mask: Optional[Any],
) -> mx.array:
    _, query_heads, q_len, head_dim = queries.shape
    _, kv_heads, _kv_len, _ = keys.shape
    can_use_native_minimax_gqa = (
        cache is None or not hasattr(cache, "bits")
    ) and (
        kv_heads > 0
        and query_heads != kv_heads
        and query_heads % kv_heads == 0
        and int(head_dim) == 128
        and int(q_len) in (1, 2, 3, 4)
        and queries.dtype in (mx.bfloat16, mx.float16)
    )
    if can_use_native_minimax_gqa:
        return scaled_dot_product_attention(
            queries,
            keys,
            values,
            cache=cache,
            scale=scale,
            mask=mask,
        )
    return _gqa_reshape_sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=scale,
        mask=mask,
    )


def _minimax_tree_attention_call(
    attn: Any,
    x: mx.array,
    *,
    mask: Optional[mx.array],
    cache: Any,
) -> Optional[mx.array]:
    if cache is None or not hasattr(cache, _TREE_POSITIONS_ATTR):
        return None
    batch_size, seq_len, _ = x.shape
    queries, keys, values = _minimax_project_qkv(attn, x)
    positions = getattr(cache, _TREE_POSITIONS_ATTR)
    queries = _apply_rope_positions(attn.rope, queries, positions)
    keys = _apply_rope_positions(attn.rope, keys, positions)
    keys, values = cache.update_and_fetch(keys, values)
    tree_mask = getattr(cache, "_dflash_tree_attention_mask", mask)
    output = _minimax_gqa_sdpa(
        queries,
        keys,
        values,
        cache=cache,
        scale=attn.scale,
        mask=tree_mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    return attn.o_proj(output)


def _install_minimax_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflasher_minimax_attention_hook_installed", False):
        return

    original_call = cls.__call__

    def attention_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        tree_output = _minimax_tree_attention_call(self, x, mask=mask, cache=cache)
        if tree_output is not None:
            return tree_output
        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        can_route_gqa = (
            cache is not None
            and not isinstance(cache, cache_mod.QuantizedKVCache)
            and cached_prefix_len >= _HYBRID_SDPA_EXACT_KV_THRESHOLD
            and (mask is None or isinstance(mask, mx.array) or mask == "causal")
            and 0 < int(x.shape[1]) <= 16
        )
        if not can_route_gqa:
            return original_call(self, x, mask=mask, cache=cache)
        batch_size, seq_len, _ = x.shape
        queries, keys, values = _minimax_project_qkv(self, x)
        can_use_gqa_fast_path = (
            queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if not can_use_gqa_fast_path:
            return original_call(self, x, mask=mask, cache=cache)
        queries = self.rope(queries, offset=cached_prefix_len)
        keys = self.rope(keys, offset=cached_prefix_len)
        keys, values = cache.update_and_fetch(keys, values)
        output = _minimax_gqa_sdpa(
            queries,
            keys,
            values,
            cache=cache,
            scale=self.scale,
            mask=mask,
        )
        output = output.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
        return self.o_proj(output)

    cls.__call__ = attention_call
    cls._dflasher_minimax_attention_hook_installed = True


def _minimax_moe_sequential_from_layer() -> int:
    raw = os.environ.get("DFLASH_MINIMAX_MOE_SEQUENTIAL_FROM_LAYER", "0").strip()
    try:
        return int(raw)
    except ValueError:
        return 0


def _minimax_moe_verify_mode() -> str:
    return os.environ.get("DFLASH_MINIMAX_MOE_VERIFY_MODE", "sequential").strip().lower()


def _minimax_switch_glu_sorted(switch_mlp: Any, x: mx.array, indices: mx.array) -> mx.array:
    x = mx.expand_dims(x, (-2, -3))
    *_, expert_count = indices.shape
    flat_indices = indices.flatten()
    order = mx.argsort(flat_indices)
    inv_order = mx.argsort(order)
    x_sorted = x.flatten(0, -3)[order // expert_count]
    idx_sorted = flat_indices[order]
    if getattr(switch_mlp, "training", False):
        idx_sorted = mx.stop_gradient(idx_sorted)
    x_up = switch_mlp.up_proj(x_sorted, idx_sorted, sorted_indices=True)
    x_gate = switch_mlp.gate_proj(x_sorted, idx_sorted, sorted_indices=True)
    x_out = switch_mlp.down_proj(
        switch_mlp.activation(x_up, x_gate),
        idx_sorted,
        sorted_indices=True,
    )
    x_out = x_out[inv_order]
    x_out = mx.unflatten(x_out, 0, indices.shape)
    return x_out.squeeze(-2)


def _minimax_moe_sorted_call(moe: Any, x: mx.array) -> mx.array | None:
    if getattr(moe, "sharding_group", None) is not None:
        return None
    gates = moe.gate(x.astype(mx.float32))
    scores = mx.sigmoid(gates)
    orig_scores = scores
    scores = scores + moe.e_score_correction_bias
    k = moe.num_experts_per_tok
    inds = mx.argpartition(-scores, kth=k - 1, axis=-1)[..., :k]
    scores = mx.take_along_axis(orig_scores, inds, axis=-1)
    scores = scores / (mx.sum(scores, axis=-1, keepdims=True) + 1e-20)
    scores = scores.astype(x.dtype)
    y = _minimax_switch_glu_sorted(moe.switch_mlp, x, inds)
    return (y * scores[..., None]).sum(axis=-2)


def _install_minimax_moe_hook(moe: Any, *, layer_index: int) -> None:
    setattr(moe, "_dflasher_minimax_layer_index", int(layer_index))
    cls = type(moe)
    if getattr(cls, "_dflasher_minimax_moe_hook_installed", False):
        return

    original_call = cls.__call__

    def moe_call(self, x: mx.array) -> mx.array:
        seq_len = int(x.shape[1])
        if seq_len <= 1 or seq_len > 16:
            return original_call(self, x)
        layer_idx = int(getattr(self, "_dflasher_minimax_layer_index", 0) or 0)
        if layer_idx < _minimax_moe_sequential_from_layer():
            return original_call(self, x)
        mode = _minimax_moe_verify_mode()
        if mode == "batched":
            return original_call(self, x)
        if mode == "sorted":
            sorted_out = _minimax_moe_sorted_call(self, x)
            if sorted_out is not None:
                return sorted_out
        chunks = [
            original_call(self, x[:, index : index + 1, :])
            for index in range(seq_len)
        ]
        return mx.concatenate(chunks, axis=1)

    cls.__call__ = moe_call
    cls._dflasher_minimax_moe_hook_installed = True


class MiniMaxM2TargetOps:
    backend_name = "minimax_m2"

    def __init__(self) -> None:
        self._last_verify_target_model: Any | None = None
        self._last_verify_ids: mx.array | None = None

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
            supports_tree_verify=True,
        )

    def supports_tree_cache(self, cache_entries: list[Any]) -> bool:
        return all(
            not isinstance(
                cache_entry,
                (cache_mod.QuantizedKVCache, cache_mod.RotatingKVCache),
            )
            for cache_entry in cache_entries
        )

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
        self._last_verify_target_model = target_model
        self._last_verify_ids = verify_ids
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
        tree_size = int(tree_inputs.token_ids.shape[0])
        if tree_size <= 0:
            raise ValueError("DDTree target tree must contain at least the root slot")
        self.install_speculative_hooks(target_model)
        _set_tree_cache_context(target_cache, tree_inputs)
        try:
            return self.forward_with_hidden_capture(
                target_model,
                input_ids=tree_inputs.token_ids[None],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
        except Exception as exc:
            for cache_entry in target_cache:
                _clear_tree_cache_context(cache_entry)
            raise RuntimeError("MiniMax-M2 DDTree target-tree verify failed") from exc

    def restore_after_tree_acceptance(
        self,
        cache_entries: list[Any],
        *,
        accepted_tree_indices: list[int],
    ) -> int:
        if not accepted_tree_indices:
            raise ValueError("accepted_tree_indices must not be empty")
        replay_start_ns = time.perf_counter_ns()
        for cache_entry in cache_entries:
            if hasattr(cache_entry, "keys") and hasattr(cache_entry, "values"):
                _commit_kv_tree_path(cache_entry, accepted_tree_indices)
            else:
                _clear_tree_cache_context(cache_entry)
                raise NotImplementedError(
                    f"DDTree cache commit unsupported for {type(cache_entry).__name__}"
                )
        return time.perf_counter_ns() - replay_start_ns

    def install_speculative_hooks(self, target_model: Any) -> None:
        text_model = self.text_model(target_model)
        if getattr(text_model, "_dflash_speculative_hooks_installed", False):
            return
        for layer_index, layer in enumerate(text_model.layers):
            attn = getattr(layer, "self_attn", None)
            if attn is not None:
                _install_minimax_attention_hook(attn)
            moe = getattr(layer, "block_sparse_moe", None)
            if moe is not None:
                _install_minimax_moe_hook(moe, layer_index=layer_index)
        text_model._dflash_speculative_hooks_installed = True

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

    def _trim_target_cache(self, cache_entries: list[Any], target_len: int) -> int:
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

    def _replay_accepted_draft_tokens(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
    ) -> int:
        if acceptance_length <= 0 or self._last_verify_ids is None:
            return self._trim_target_cache(cache_entries, target_len)
        target_model = self._last_verify_target_model
        if target_model is None:
            return self._trim_target_cache(cache_entries, target_len)
        commit_count = 1 + int(acceptance_length)
        keep_len = max(0, int(target_len) - commit_count)
        replay_ns_total = self._trim_target_cache(cache_entries, keep_len)
        committed_ids = self._last_verify_ids[:, :commit_count]
        for index in range(commit_count):
            replay_start_ns = time.perf_counter_ns()
            logits, _captured = self.forward_with_hidden_capture(
                target_model,
                input_ids=committed_ids[:, index : index + 1],
                cache=cache_entries,
                capture_layer_ids=set(),
            )
            mx.eval(logits)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        return replay_ns_total

    def restore_after_acceptance(
        self,
        cache_entries: list[Any],
        *,
        target_len: int,
        acceptance_length: int,
        drafted_tokens: int = 0,
    ) -> int:
        try:
            if (
                os.environ.get("DFLASH_MINIMAX_FULL_COMMIT_CORRECTION") == "1"
                and acceptance_length > 0
                and drafted_tokens > 0
            ):
                return self._replay_accepted_draft_tokens(
                    cache_entries,
                    target_len=target_len,
                    acceptance_length=acceptance_length,
                )
            return self._trim_target_cache(cache_entries, target_len)
        finally:
            self._last_verify_target_model = None
            self._last_verify_ids = None

    def correct_committed_block_after_acceptance(
        self,
        *,
        target_model: Any,
        target_cache: list[Any],
        verify_token_ids: mx.array,
        target_layer_ids: list[int],
        capture_layer_ids: set[int],
        prefix_len: int,
        acceptance_length: int,
        suppress_token_mask: Optional[mx.array] = None,
    ) -> dict[str, Any] | None:
        commit_count = 1 + int(acceptance_length)
        if commit_count <= 0:
            raise ValueError("MiniMax-M2 commit correction requires committed tokens")
        if os.environ.get("DFLASH_MINIMAX_FULL_COMMIT_CORRECTION") != "1":
            return None
        replay_start_ns = time.perf_counter_ns()
        self._trim_target_cache(target_cache, int(prefix_len))
        committed_ids = verify_token_ids[:, :commit_count].astype(mx.uint32)
        hidden_chunks: list[mx.array] = []
        last_logits = None
        for index in range(commit_count):
            logits, hidden_states = self.forward_with_hidden_capture(
                target_model,
                input_ids=committed_ids[:, index : index + 1],
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            hidden_chunks.append(self.extract_context_feature(hidden_states, target_layer_ids))
            last_logits = logits[:, -1, :]
        if last_logits is None:
            raise RuntimeError("MiniMax-M2 commit correction did not produce logits")
        committed_hidden = mx.concatenate(hidden_chunks, axis=1)
        staged_first_next = greedy_tokens_with_mask(
            last_logits,
            suppress_token_mask,
        ).reshape(-1)
        mx.eval(committed_hidden, last_logits, staged_first_next)
        return {
            "committed_hidden": committed_hidden,
            "last_cycle_logits": last_logits,
            "staged_first_next": staged_first_next,
            "replay_ns": time.perf_counter_ns() - replay_start_ns,
        }

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
    label_source: str = OMLX_LABEL_RAW_NEXT_TOKEN
    generated_continuation_tokens: int = 0
    use_chat_template: bool = False
    include_prefill_anchors: bool = False
    target_top_k: int = 0
    hidden_target: str = OMLX_HIDDEN_TARGET_SELECTED

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
        payload["label_source"] = normalize_omlx_label_source(
            payload.get("label_source", OMLX_LABEL_RAW_NEXT_TOKEN)
        )
        payload["use_chat_template"] = bool(payload.get("use_chat_template", False))
        payload["include_prefill_anchors"] = bool(
            payload.get("include_prefill_anchors", False)
        )
        payload["target_top_k"] = int(payload.get("target_top_k", 0))
        payload["hidden_target"] = normalize_omlx_hidden_target(
            payload.get("hidden_target", OMLX_HIDDEN_TARGET_SELECTED)
        )
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
    intermediate_size: int | None = None
    layer_policy: str = "auto"
    target_layer_ids: tuple[int, ...] | None = None
    mask_token_id: int | None = None
    max_steps: int = 20
    learning_rate: float = 1e-4
    loss_fn: str = OMLX_LOSS_CE_HIDDEN
    hidden_loss_weight: float = 0.01
    label_source: str = OMLX_LABEL_RAW_NEXT_TOKEN
    generated_continuation_tokens: int = 0
    use_chat_template: bool = False
    include_prefill_anchors: bool = False
    target_top_k: int = 0
    hidden_target: str = OMLX_HIDDEN_TARGET_SELECTED
    topk_loss_weight: float = 1.0
    topk_temperature: float = 1.0
    anchor_span_tokens: int = 0
    first_anchor_probability: float = 0.0
    anchor_margin_min: float = 0.0
    anchor_margin_top_fraction: float = 0.0
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
        if self.intermediate_size is not None and self.intermediate_size < 1:
            raise ValueError("intermediate_size must be positive.")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        normalize_omlx_loss_fn(self.loss_fn)
        if self.hidden_loss_weight < 0:
            raise ValueError("hidden_loss_weight must be non-negative.")
        normalize_omlx_label_source(self.label_source)
        normalize_omlx_hidden_target(self.hidden_target)
        if self.target_top_k < 0:
            raise ValueError("target_top_k must be non-negative.")
        if self.topk_loss_weight < 0:
            raise ValueError("topk_loss_weight must be non-negative.")
        if self.topk_temperature <= 0:
            raise ValueError("topk_temperature must be positive.")
        if self.anchor_span_tokens < 0:
            raise ValueError("anchor_span_tokens must be non-negative.")
        if self.first_anchor_probability < 0 or self.first_anchor_probability > 1:
            raise ValueError("first_anchor_probability must be between 0 and 1.")
        if self.anchor_margin_min < 0:
            raise ValueError("anchor_margin_min must be non-negative.")
        if self.anchor_margin_top_fraction < 0 or self.anchor_margin_top_fraction > 1:
            raise ValueError("anchor_margin_top_fraction must be between 0 and 1.")
        if (
            normalize_omlx_loss_fn(self.loss_fn) in OMLX_LOSSES_WITH_TOPK_KL
            and self.target_top_k < 1
        ):
            raise ValueError("top-k KL losses require target_top_k >= 1.")
        if self.generated_continuation_tokens < 0:
            raise ValueError("generated_continuation_tokens must be non-negative.")
        if (
            self.generated_continuation_tokens
            and self.max_length <= self.generated_continuation_tokens
        ):
            raise ValueError("max_length must be larger than generated_continuation_tokens.")
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
    spec_epoch_path: Path
    dflash_engine_path: Path
    model_settings_path: Path
    dflash_lifecycle_path: Path
    changed_paths: tuple[Path, ...]


@dataclass(frozen=True)
class _DFlashMlxRuntime:
    flavor: str
    config_class: Any
    draft_model_class: Any
    stream_generate: Any | None


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


def _omlx_app_site_packages(app_path: str | Path = DEFAULT_OMLX_APP_PATH) -> tuple[Path, ...]:
    app = Path(app_path).expanduser()
    contents = app / "Contents"
    if not contents.exists():
        return ()
    return tuple(
        sorted(contents.glob("Python/framework-mlx-framework/lib/python*/site-packages"))
    )


def _ensure_omlx_app_site_packages() -> None:
    explicit = os.environ.get("DFLASHER_DFLASH_MLX_SITE_PACKAGES")
    candidates = [Path(explicit).expanduser()] if explicit else list(_omlx_app_site_packages())
    for candidate in candidates:
        if candidate.exists() and str(candidate) not in sys.path:
            sys.path.insert(0, str(candidate))


def _import_dflash_mlx() -> _DFlashMlxRuntime:
    _ensure_omlx_app_site_packages()
    try:
        from dflash_mlx.model import DFlashDraftModel, DFlashDraftModelArgs

        return _DFlashMlxRuntime(
            flavor="dflash_mlx",
            config_class=DFlashDraftModelArgs,
            draft_model_class=DFlashDraftModel,
            stream_generate=None,
        )
    except ImportError:
        pass
    try:
        from dflash.model_mlx import DFlashConfig, DFlashDraftModel, stream_generate
    except ImportError as exc:
        raise RuntimeError(
            "OMLX DFlash runtime requires either the oMLX bundled dflash_mlx runtime "
            "or z-lab dflash MLX support. Install oMLX.app, set "
            "DFLASHER_DFLASH_MLX_SITE_PACKAGES, or run: pip install -e '.[zlab-mlx]'"
        ) from exc
    return _DFlashMlxRuntime(
        flavor="zlab",
        config_class=DFlashConfig,
        draft_model_class=DFlashDraftModel,
        stream_generate=stream_generate,
    )


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


def normalize_omlx_label_source(label_source: str) -> str:
    normalized = OMLX_LABEL_SOURCE_ALIASES.get(label_source.strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(set(OMLX_LABEL_SOURCE_ALIASES.values())))
        raise ValueError(
            f"Unsupported OMLX label source {label_source!r}; choose one of: {choices}"
        )
    return normalized


def normalize_omlx_hidden_target(hidden_target: str) -> str:
    normalized = OMLX_HIDDEN_TARGET_ALIASES.get(hidden_target.strip().lower())
    if normalized is None:
        choices = ", ".join(sorted(set(OMLX_HIDDEN_TARGET_ALIASES.values())))
        raise ValueError(
            f"Unsupported OMLX hidden target {hidden_target!r}; choose one of: {choices}"
        )
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


def _added_token_id_from_decoder(payload: dict[str, Any], token: str) -> int | None:
    decoder = payload.get("added_tokens_decoder") or {}
    if isinstance(decoder, dict):
        for key, value in decoder.items():
            if isinstance(value, dict) and value.get("content") == token:
                return int(key)
    added_tokens = payload.get("added_tokens") or []
    if isinstance(added_tokens, list):
        for value in added_tokens:
            if isinstance(value, dict) and value.get("content") == token:
                token_id = value.get("id")
                if token_id is not None:
                    return int(token_id)
    return None


def _added_token_id_from_files(model_dir: Path, token: str) -> int | None:
    for file_name in ("tokenizer_config.json", "tokenizer.json"):
        path = model_dir / file_name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        token_id = _added_token_id_from_decoder(payload, token)
        if token_id is not None:
            return token_id
    added_tokens_path = model_dir / "added_tokens.json"
    if added_tokens_path.exists():
        try:
            added_tokens = json.loads(added_tokens_path.read_text())
        except json.JSONDecodeError:
            added_tokens = {}
        if isinstance(added_tokens, dict) and token in added_tokens:
            return int(added_tokens[token])
    return None


def resolve_omlx_mask_token_id(
    source_model: str | Path,
    mask_token_id: int | None = None,
) -> int:
    if mask_token_id is not None:
        return int(mask_token_id)
    model_dir = Path(resolve_omlx_source_model(source_model))
    if model_dir.exists():
        for token in OMLX_MASK_TOKEN_CANDIDATES:
            token_id = _added_token_id_from_files(model_dir, token)
            if token_id is not None:
                return token_id
    return 0


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
    intermediate_size: int | None = None,
    layer_policy: str = "auto",
    target_layer_ids: tuple[int, ...] | None = None,
    mask_token_id: int | None = None,
    training_data_source: str = "unknown",
    training_objective: str = OMLX_LOSS_HIDDEN_MSE,
) -> OmlxDraftConfig:
    source_model = resolve_omlx_source_model(source_model)
    config = read_model_config(source_model)
    selected_layer_ids = resolve_omlx_layer_ids(source_model, layer_policy, target_layer_ids)
    resolved_mask_token_id = resolve_omlx_mask_token_id(source_model, mask_token_id)
    hidden_size = int(_config_value(config, "hidden_size"))
    resolved_intermediate_size = (
        int(intermediate_size)
        if intermediate_size is not None
        else int(_config_value(config, "intermediate_size", hidden_size * 4))
    )
    layer_types = tuple("full_attention" for _ in range(draft_layers))
    return OmlxDraftConfig(
        source_model=source_model,
        hidden_size=hidden_size,
        num_hidden_layers=draft_layers,
        num_attention_heads=_num_attention_heads(config),
        num_key_value_heads=_num_key_value_heads(config),
        head_dim=_head_dim(config),
        intermediate_size=resolved_intermediate_size,
        vocab_size=int(_config_value(config, "vocab_size")),
        rms_norm_eps=float(_config_value(config, "rms_norm_eps", 1e-6)),
        rope_theta=float(_config_value(config, "rope_theta", 1_000_000.0)),
        max_position_embeddings=int(_config_value(config, "max_position_embeddings", 2048)),
        block_size=block_size,
        target_layer_ids=selected_layer_ids,
        num_target_layers=int(_config_value(config, "num_hidden_layers")),
        mask_token_id=resolved_mask_token_id,
        rope_scaling=_config_value(config, "rope_scaling"),
        layer_types=layer_types,
        sliding_window=_config_value(config, "sliding_window"),
        final_logit_softcapping=_config_value(config, "final_logit_softcapping"),
        training_data_source=training_data_source,
        training_objective=normalize_omlx_loss_fn(training_objective),
    )


def _to_dflash_config(config: OmlxDraftConfig):
    runtime = _import_dflash_mlx()
    if runtime.flavor == "dflash_mlx":
        return runtime.config_class.from_dict(_config_payload(config))
    return runtime.config_class(
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
    intermediate_size: int | None = None,
    layer_policy: str = "auto",
    target_layer_ids: tuple[int, ...] | None = None,
    mask_token_id: int | None = None,
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
        intermediate_size=intermediate_size,
        layer_policy=layer_policy,
        target_layer_ids=target_layer_ids,
        mask_token_id=mask_token_id,
        training_data_source=training_data_source,
        training_objective=training_objective,
    )
    runtime = _import_dflash_mlx()
    draft = runtime.draft_model_class(_to_dflash_config(config))
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
    runtime = _import_dflash_mlx()
    draft = runtime.draft_model_class(_to_dflash_config(config))
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


def _get_text_model(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model
    if hasattr(model, "language_model") and hasattr(model.language_model, "model"):
        inner = model.language_model.model
        if hasattr(inner, "layers"):
            return inner
    if hasattr(model, "layers"):
        return model
    raise AttributeError(f"Cannot find text model in {type(model).__name__}")


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


def _final_normalized_hidden(model, hidden_states):
    norm = getattr(_get_text_model(model), "norm", None)
    if norm is None:
        return hidden_states
    return norm(hidden_states)


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


def _encode_text(
    tokenizer,
    text: str,
    max_length: int,
    *,
    use_chat_template: bool = False,
) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        tokens = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True,
            add_generation_prompt=True,
        )
        return [int(token) for token in list(tokens)[:max_length]]
    bos_token = getattr(tokenizer, "bos_token", None)
    add_special_tokens = bos_token is None or not text.startswith(str(bos_token))
    tokens = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    return [int(token) for token in tokens[:max_length]]


def _extend_with_target_greedy_tokens(
    model,
    tokenizer,
    prompt_token_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    runtime_tokens = _extend_with_dflash_mlx_target_ops_greedy_tokens(
        model,
        tokenizer,
        prompt_token_ids,
        max_new_tokens,
    )
    if runtime_tokens is not None:
        return runtime_tokens
    return _extend_with_raw_target_greedy_tokens(
        model,
        tokenizer,
        prompt_token_ids,
        max_new_tokens,
    )


def _extend_with_dflash_mlx_target_ops_greedy_tokens(
    model,
    tokenizer,
    prompt_token_ids: list[int],
    max_new_tokens: int,
) -> list[int] | None:
    if max_new_tokens <= 0:
        return list(prompt_token_ids)
    if not prompt_token_ids:
        raise ValueError("target-generated cache extraction requires a non-empty prompt")
    _ensure_omlx_app_site_packages()
    try:
        import mlx.core as mx
        from dflash_mlx.engine.sampling import greedy_tokens_with_mask
        from dflash_mlx.engine.target_ops import resolve_target_ops
    except ImportError:
        return None
    try:
        target_ops = resolve_target_ops(model)
        target_ops.install_speculative_hooks(model)
        target_cache = target_ops.make_cache(
            model,
            enable_speculative_linear_cache=True,
        )
        logits = _dflash_mlx_runtime_prefill_logits(
            target_ops,
            model,
            target_cache,
            prompt_token_ids,
            mx=mx,
        )
    except Exception:
        return None

    token = greedy_tokens_with_mask(logits[:, -1, :], None).reshape(-1)
    mx.eval(token)
    generated = [int(token.item())]
    eos_ids = _eos_token_ids(tokenizer)
    for _ in range(max_new_tokens - 1):
        if generated[-1] in eos_ids:
            break
        logits, _captured = target_ops.verify_block(
            target_model=model,
            verify_ids=token.reshape(1, 1).astype(mx.uint32),
            target_cache=target_cache,
            capture_layer_ids=set(),
        )
        token = greedy_tokens_with_mask(logits[:, -1, :], None).reshape(-1)
        mx.eval(token)
        generated.append(int(token.item()))
    return list(prompt_token_ids) + generated


def _dflash_mlx_runtime_prefill_logits(
    target_ops,
    target_model,
    target_cache,
    prompt_token_ids: list[int],
    *,
    mx,
):
    """Prefill target cache using the same final-token boundary as DFlash serving."""

    if not prompt_token_ids:
        raise ValueError("DFlash target prefill requires at least one prompt token")
    prompt_array = mx.array(prompt_token_ids, dtype=mx.uint32)[None, :]
    if len(prompt_token_ids) > 1:
        target_ops.forward_with_hidden_capture(
            target_model,
            input_ids=prompt_array[:, :-1],
            cache=target_cache,
            capture_layer_ids=set(),
            logits_last_only=True,
        )
    logits, _captured = target_ops.forward_with_hidden_capture(
        target_model,
        input_ids=prompt_array[:, -1:],
        cache=target_cache,
        capture_layer_ids=set(),
        logits_last_only=True,
    )
    return logits


def _extend_with_raw_target_greedy_tokens(
    model,
    tokenizer,
    prompt_token_ids: list[int],
    max_new_tokens: int,
) -> list[int]:
    if max_new_tokens <= 0:
        return list(prompt_token_ids)
    if not prompt_token_ids:
        raise ValueError("target-generated cache extraction requires a non-empty prompt")
    mx, _nn, _optim, _mlx_lm_load, make_prompt_cache, make_sampler = _import_mlx()
    prompt_ids = mx.array(prompt_token_ids, dtype=mx.uint32)
    cache = make_prompt_cache(model)
    sampler = make_sampler(temp=0.0)
    logits = model(prompt_ids[None, :], cache)
    token = int(sampler(logits[:, -1:])[0, 0].item())
    generated = [token]
    eos_ids = _eos_token_ids(tokenizer)
    for _ in range(max_new_tokens - 1):
        if token in eos_ids:
            break
        logits = model(mx.array([[token]], dtype=mx.uint32), cache)
        token = int(sampler(logits[:, -1:])[0, 0].item())
        generated.append(token)
    return list(prompt_token_ids) + generated


def _captured_layer_hidden(captured: Any, layer_id: int) -> Any:
    capture_key = int(layer_id) + 1
    if isinstance(captured, dict):
        return captured[capture_key]
    return captured[capture_key]


def _extract_target_ops_sequence_features(
    model,
    embed_tokens,
    token_ids: list[int],
    prompt_len: int,
    selected_layer_ids: tuple[int, ...],
    hidden_target: str = OMLX_HIDDEN_TARGET_SELECTED,
) -> tuple[Any, Any, Any, Any]:
    _ensure_omlx_app_site_packages()
    try:
        import mlx.core as mx
        from dflash_mlx.engine.target_ops import resolve_target_ops
    except ImportError as exc:
        raise RuntimeError(
            "dflash_mlx target_ops is required for runtime-aligned cache extraction."
        ) from exc

    if not token_ids:
        raise ValueError("runtime-aligned cache extraction requires at least one token")

    prompt_len = max(1, min(int(prompt_len), len(token_ids)))
    hidden_target = normalize_omlx_hidden_target(hidden_target)
    target_ops = resolve_target_ops(model)
    target_ops.install_speculative_hooks(model)
    target_cache = target_ops.make_cache(
        model,
        enable_speculative_linear_cache=True,
    )
    final_layer_id = len(_get_layers(model)) - 1
    captured_layer_ids = tuple(dict.fromkeys((*selected_layer_ids, final_layer_id)))
    capture_layer_ids = {int(layer_id) + 1 for layer_id in captured_layer_ids}
    context_chunks = []
    target_chunks = []
    logits_chunks = []

    def append_features(ids: list[int]) -> None:
        if not ids:
            return
        logits, captured = target_ops.verify_block(
            target_model=model,
            verify_ids=mx.array(ids, dtype=mx.uint32)[None, :],
            target_cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(logits)
        context_chunks.append(
            target_ops.extract_context_feature(captured, list(selected_layer_ids))
        )
        if hidden_target == OMLX_HIDDEN_TARGET_FINAL:
            target_chunks.append(
                _final_normalized_hidden(
                    model,
                    _captured_layer_hidden(captured, final_layer_id),
                )
            )
        else:
            target_chunks.append(_captured_layer_hidden(captured, selected_layer_ids[-1]))

    if prompt_len > 1:
        append_features(token_ids[: prompt_len - 1])
        append_features(token_ids[prompt_len - 1 : prompt_len])
    else:
        append_features(token_ids[:prompt_len])
    for token_id in token_ids[prompt_len:]:
        append_features([int(token_id)])

    context_hidden = (
        context_chunks[0]
        if len(context_chunks) == 1
        else mx.concatenate(context_chunks, axis=1)
    )
    target_hidden = (
        target_chunks[0]
        if len(target_chunks) == 1
        else mx.concatenate(target_chunks, axis=1)
    )
    logits = (
        logits_chunks[0]
        if len(logits_chunks) == 1
        else mx.concatenate(logits_chunks, axis=1)
    )
    token_embeddings = embed_tokens(mx.array(token_ids, dtype=mx.uint32)[None, :])
    return context_hidden, target_hidden, token_embeddings, logits


def _as_numpy(array, dtype: str = "float16") -> np.ndarray:
    import mlx.core as mx

    out = np.array(array.astype(mx.float32))
    if dtype == "float16":
        return out.astype(np.float16)
    if dtype == "float32":
        return out.astype(np.float32)
    raise ValueError("cache dtype must be float16 or float32.")


def _target_topk_payload(logits, top_k: int, dtype: str) -> dict[str, np.ndarray]:
    import mlx.core as mx

    top_k = int(top_k)
    if top_k < 1:
        return {}
    next_token_logits = logits[:, :-1, :]
    if int(next_token_logits.shape[1]) < 1:
        return {}
    vocab_size = int(next_token_logits.shape[-1])
    top_k = min(top_k, vocab_size)
    if top_k == vocab_size:
        indices = mx.argsort(-next_token_logits, axis=-1)
    else:
        partitioned = mx.argpartition(
            next_token_logits,
            kth=vocab_size - top_k,
            axis=-1,
        )
        indices = partitioned[:, :, -top_k:]
        values = mx.take_along_axis(next_token_logits, indices, axis=-1)
        order = mx.argsort(-values, axis=-1)
        indices = mx.take_along_axis(indices, order, axis=-1)
    values = mx.take_along_axis(next_token_logits, indices, axis=-1)
    mx.eval(indices, values)
    return {
        "target_topk_indices": np.asarray(indices[0], dtype=np.int64),
        "target_topk_logits": _as_numpy(values[0], dtype),
    }


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
    mask_token_id: int | None = None,
    label_source: str = OMLX_LABEL_RAW_NEXT_TOKEN,
    generated_continuation_tokens: int = 0,
    use_chat_template: bool = False,
    include_prefill_anchors: bool = False,
    target_top_k: int = 0,
    hidden_target: str = OMLX_HIDDEN_TARGET_SELECTED,
    dtype: str = "float16",
    overwrite: bool = False,
) -> Path:
    source_model = resolve_omlx_source_model(source_model)
    resolved_mask_token_id = resolve_omlx_mask_token_id(source_model, mask_token_id)
    if cache_dir.exists():
        if not overwrite:
            raise ValueError(f"Cache path already exists. Pass --overwrite: {cache_dir}")
    cache_dir.parent.mkdir(parents=True, exist_ok=True)
    work_dir = cache_dir.parent / f".{cache_dir.name}.tmp-{os.getpid()}-{time.time_ns()}"
    work_dir.mkdir(parents=True)
    mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    label_source = normalize_omlx_label_source(label_source)
    hidden_target = normalize_omlx_hidden_target(hidden_target)
    generated_continuation_tokens = int(generated_continuation_tokens)
    if generated_continuation_tokens < 0:
        raise ValueError("generated_continuation_tokens must be non-negative.")
    if target_top_k < 0:
        raise ValueError("target_top_k must be non-negative.")
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
    mask_embedding = embed_tokens(mx.array([resolved_mask_token_id]))[0]
    sample_files: list[str] = []
    started = time.perf_counter()
    try:
        for text in texts:
            if len(sample_files) >= max_samples:
                break
            if generated_continuation_tokens > 0:
                prompt_max_length = max(1, max_length - generated_continuation_tokens)
                prompt_ids = _encode_text(
                    tokenizer,
                    text,
                    prompt_max_length,
                    use_chat_template=use_chat_template,
                )
                token_ids = _extend_with_target_greedy_tokens(
                    model,
                    tokenizer,
                    prompt_ids,
                    generated_continuation_tokens,
                )[:max_length]
                prompt_len = len(prompt_ids)
                min_anchor = 1 if include_prefill_anchors else prompt_len
            else:
                token_ids = _encode_text(
                    tokenizer,
                    text,
                    max_length,
                    use_chat_template=use_chat_template,
                )
                min_anchor = None
            if len(token_ids) < block_size + 1:
                continue
            if min_anchor is not None and len(token_ids) < min_anchor + block_size + 1:
                continue
            if min_anchor is not None:
                (
                    context_hidden,
                    target_hidden,
                    token_embeddings,
                    logits,
                ) = _extract_target_ops_sequence_features(
                    model,
                    embed_tokens,
                    token_ids,
                    prompt_len,
                    selected_layer_ids,
                    hidden_target=hidden_target,
                )
                target_greedy_tokens = (
                    mx.argmax(logits[:, :-1, :], axis=-1)[0]
                    if label_source == OMLX_LABEL_TARGET_GREEDY
                    else None
                )
                eval_values = [
                    logits,
                    context_hidden,
                    target_hidden,
                    token_embeddings,
                    mask_embedding,
                ]
            else:
                input_ids = mx.array(token_ids, dtype=mx.uint32)[None, :]
                final_layer_id = len(_get_layers(model)) - 1
                captured_layer_ids = tuple(dict.fromkeys((*selected_layer_ids, final_layer_id)))
                with capture_selected_layers(model, captured_layer_ids) as captured:
                    logits = model(input_ids)
                    captured_by_layer = dict(zip(captured_layer_ids, captured, strict=True))
                    selected = [
                        captured_by_layer[layer_id]
                        for layer_id in selected_layer_ids
                        if captured_by_layer[layer_id] is not None
                    ]
                    if len(selected) != len(selected_layer_ids):
                        raise RuntimeError("Failed to capture every selected hidden-state layer.")
                    context_hidden = mx.concatenate(selected, axis=-1)
                    if hidden_target == OMLX_HIDDEN_TARGET_FINAL:
                        target_hidden = _final_normalized_hidden(
                            model,
                            captured_by_layer[final_layer_id],
                        )
                    else:
                        target_hidden = selected[-1]
                    token_embeddings = embed_tokens(input_ids)
                    target_greedy_tokens = (
                        mx.argmax(logits[:, :-1, :], axis=-1)[0]
                        if label_source == OMLX_LABEL_TARGET_GREEDY
                        else None
                    )
                eval_values = [
                    logits,
                    context_hidden,
                    target_hidden,
                    token_embeddings,
                    mask_embedding,
                ]
            if target_greedy_tokens is not None and not isinstance(
                target_greedy_tokens, np.ndarray
            ):
                eval_values.append(target_greedy_tokens)
            topk_payload = _target_topk_payload(logits, target_top_k, dtype)
            mx.eval(*eval_values)
            sample_name = f"sample_{len(sample_files):05d}.npz"
            sample_payload = {
                "tokens": np.asarray(token_ids, dtype=np.int64),
                "context_hidden": _as_numpy(context_hidden[0], dtype),
                "target_hidden": _as_numpy(target_hidden[0], dtype),
                "token_embeddings": _as_numpy(token_embeddings[0], dtype),
                "mask_embedding": _as_numpy(mask_embedding, dtype),
            }
            sample_payload.update(topk_payload)
            if target_greedy_tokens is not None:
                sample_payload["target_greedy_tokens"] = np.asarray(
                    target_greedy_tokens,
                    dtype=np.int64,
                )
            if min_anchor is not None:
                sample_payload["min_anchor"] = np.asarray(min_anchor, dtype=np.int64)
            np.savez_compressed(work_dir / sample_name, **sample_payload)
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
            mask_token_id=resolved_mask_token_id,
            max_length=max_length,
            samples=len(sample_files),
            files=tuple(sample_files),
            dtype=dtype,
            label_source=label_source,
            generated_continuation_tokens=generated_continuation_tokens,
            use_chat_template=bool(use_chat_template),
            include_prefill_anchors=bool(include_prefill_anchors),
            target_top_k=int(target_top_k),
            hidden_target=hidden_target,
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


def _runtime_aligned_anchor_bounds(
    token_count: int,
    block_size: int,
    min_anchor: int = 1,
) -> tuple[int, int]:
    min_anchor = max(1, int(min_anchor))
    max_anchor = token_count - block_size
    if max_anchor < min_anchor:
        raise ValueError(
            "Cache sample is too short for runtime-aligned OMLX DFlash training: "
            f"token_count={token_count}, block_size={block_size}, min_anchor={min_anchor}"
        )
    return min_anchor, max_anchor


def _sample_min_anchor(sample: dict[str, np.ndarray]) -> int:
    raw = sample.get("min_anchor")
    if raw is None:
        return 1
    return int(np.asarray(raw).reshape(()).item())


def _runtime_aligned_training_arrays(
    sample: dict[str, np.ndarray],
    metadata: OmlxCacheMetadata,
    anchor: int,
    segment_start: int | None = None,
) -> dict[str, np.ndarray]:
    """Build the same draft inputs used by live DFlash decoding.

    During serving, DFlash has verifier hidden states only for the prefix before
    the staged anchor token. The anchor token itself is passed as an embedding to
    the draft. Training must mirror that boundary; including the anchor hidden
    state leaks information that the runtime does not have and produces drafts
    that look good only under teacher forcing.

    Live oMLX keeps older verifier context in the draft KV cache. After the first
    cycle it passes only the newly committed context segment into the draft. The
    optional ``segment_start`` separates those two parts so training sees the
    same cache/context split as serving.
    """

    min_anchor, max_anchor = _runtime_aligned_anchor_bounds(
        int(sample["tokens"].shape[0]),
        metadata.block_size,
        _sample_min_anchor(sample),
    )
    if anchor < min_anchor or anchor > max_anchor:
        raise ValueError(f"anchor must be in [{min_anchor}, {max_anchor}], got {anchor}")
    if segment_start is None:
        segment_start = 0 if anchor == min_anchor else max(min_anchor, anchor - metadata.block_size)
    segment_start = int(segment_start)
    if segment_start < 0 or segment_start > anchor:
        raise ValueError(f"segment_start must be in [0, {anchor}], got {segment_start}")
    if anchor == min_anchor and segment_start != 0:
        raise ValueError(
            "the first DFlash cycle must use the prompt as the current context segment"
        )
    if anchor > min_anchor and segment_start < min_anchor:
        raise ValueError("generated DFlash cycles must keep the prompt context in the draft cache")
    label_start = anchor + 1
    label_end = anchor + metadata.block_size
    return {
        "cache_context_hidden": sample["context_hidden"][:segment_start],
        "context_hidden": sample["context_hidden"][segment_start:anchor],
        "target_hidden": sample["target_hidden"][label_start:label_end],
        "label_tokens": _runtime_aligned_label_tokens(sample, metadata, anchor),
        "anchor_embedding": sample["token_embeddings"][anchor],
        "mask_embedding": sample["mask_embedding"],
        **_runtime_aligned_topk_arrays(sample, metadata, anchor),
    }


def _runtime_aligned_label_tokens(
    sample: dict[str, np.ndarray],
    metadata: OmlxCacheMetadata,
    anchor: int,
) -> np.ndarray:
    greedy_tokens = sample.get("target_greedy_tokens")
    if metadata.label_source == OMLX_LABEL_TARGET_GREEDY and greedy_tokens is not None:
        label_start = anchor
        label_end = anchor + metadata.block_size - 1
        return greedy_tokens[label_start:label_end]
    label_start = anchor + 1
    label_end = anchor + metadata.block_size
    return sample["tokens"][label_start:label_end]


def _runtime_aligned_topk_arrays(
    sample: dict[str, np.ndarray],
    metadata: OmlxCacheMetadata,
    anchor: int,
) -> dict[str, np.ndarray]:
    if metadata.target_top_k < 1:
        return {}
    indices = sample.get("target_topk_indices")
    logits = sample.get("target_topk_logits")
    if indices is None or logits is None:
        return {}
    label_start = anchor
    label_end = anchor + metadata.block_size - 1
    return {
        "target_topk_indices": indices[label_start:label_end],
        "target_topk_logits": logits[label_start:label_end],
    }


def _runtime_aligned_segment_start(
    sample: dict[str, np.ndarray],
    metadata: OmlxCacheMetadata,
    anchor: int,
    rng: random.Random,
) -> int:
    min_anchor = _sample_min_anchor(sample)
    if anchor == min_anchor:
        return 0
    max_segment_len = min(metadata.block_size, anchor - min_anchor)
    segment_len = rng.randint(1, max_segment_len)
    return anchor - segment_len


def _runtime_aligned_anchor_sample_bounds(
    min_anchor: int,
    max_anchor: int,
    anchor_span_tokens: int = 0,
) -> tuple[int, int]:
    anchor_span_tokens = int(anchor_span_tokens)
    if anchor_span_tokens < 0:
        raise ValueError("anchor_span_tokens must be non-negative.")
    if anchor_span_tokens <= 0:
        return min_anchor, max_anchor
    return min_anchor, min(max_anchor, min_anchor + anchor_span_tokens)


def _sample_runtime_aligned_anchor(
    min_anchor: int,
    max_anchor: int,
    metadata: OmlxCacheMetadata,
    rng: random.Random,
    *,
    anchor_span_tokens: int = 0,
    first_anchor_probability: float = 0.0,
) -> int:
    first_anchor_probability = float(first_anchor_probability)
    if first_anchor_probability < 0 or first_anchor_probability > 1:
        raise ValueError("first_anchor_probability must be between 0 and 1.")
    sampled_min, sampled_max = _runtime_aligned_anchor_sample_bounds(
        min_anchor,
        max_anchor,
        anchor_span_tokens,
    )
    if sampled_max == sampled_min:
        return sampled_min
    if metadata.generated_continuation_tokens > 0:
        if first_anchor_probability > 0 and rng.random() < first_anchor_probability:
            return sampled_min
        return rng.randint(sampled_min, sampled_max)
    if rng.random() < 0.5:
        return sampled_min
    return rng.randint(sampled_min + 1, sampled_max)


def _sample_margin_aligned_anchor(
    sample: dict[str, np.ndarray],
    min_anchor: int,
    max_anchor: int,
    metadata: OmlxCacheMetadata,
    rng: random.Random,
    *,
    anchor_span_tokens: int = 0,
    first_anchor_probability: float = 0.0,
    anchor_margin_min: float = 0.0,
    anchor_margin_top_fraction: float = 0.0,
) -> int:
    anchor_margin_min = float(anchor_margin_min)
    anchor_margin_top_fraction = float(anchor_margin_top_fraction)
    if anchor_margin_min <= 0 and anchor_margin_top_fraction <= 0:
        return _sample_runtime_aligned_anchor(
            min_anchor,
            max_anchor,
            metadata,
            rng,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=first_anchor_probability,
        )
    sampled_min, sampled_max = _runtime_aligned_anchor_sample_bounds(
        min_anchor,
        max_anchor,
        anchor_span_tokens,
    )
    if sampled_max == sampled_min:
        return sampled_min
    if (
        metadata.generated_continuation_tokens > 0
        and first_anchor_probability > 0
        and rng.random() < first_anchor_probability
    ):
        return sampled_min
    topk_logits = sample.get("target_topk_logits")
    if topk_logits is None or int(topk_logits.shape[-1]) < 2:
        return _sample_runtime_aligned_anchor(
            min_anchor,
            max_anchor,
            metadata,
            rng,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=0.0,
        )
    anchor_start = max(0, sampled_min)
    anchor_end = min(sampled_max, int(topk_logits.shape[0]) - 1)
    if anchor_end < anchor_start:
        return _sample_runtime_aligned_anchor(
            min_anchor,
            max_anchor,
            metadata,
            rng,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=0.0,
        )
    margins = np.asarray(topk_logits[anchor_start : anchor_end + 1, :2], dtype=np.float32)
    margins = margins[:, 0] - margins[:, 1]
    candidate_offsets = np.arange(margins.shape[0])
    if anchor_margin_min > 0:
        candidate_offsets = candidate_offsets[margins >= anchor_margin_min]
    if candidate_offsets.size == 0:
        candidate_offsets = np.arange(margins.shape[0])
    if anchor_margin_top_fraction > 0 and candidate_offsets.size > 1:
        keep_count = max(1, math.ceil(candidate_offsets.size * anchor_margin_top_fraction))
        ranked_offsets = candidate_offsets[np.argsort(-margins[candidate_offsets])]
        candidate_offsets = ranked_offsets[:keep_count]
    return int(anchor_start + int(rng.choice(candidate_offsets.tolist())))


def _make_draft_cache(draft):
    if hasattr(draft, "make_cache"):
        return draft.make_cache()
    _ensure_omlx_app_site_packages()
    try:
        from dflash_mlx.draft_backend import EagerDraftBackend
    except ImportError as exc:
        raise RuntimeError("dflash_mlx draft cache support is unavailable.") from exc
    return EagerDraftBackend().make_cache(
        draft_model=draft,
        sink_size=64,
        window_size=1024,
        allow_full_context_layers=False,
    )


def _advance_draft_cache_with_context(draft, cache, target_hidden) -> None:
    if int(target_hidden.shape[1]) <= 0:
        return
    if hasattr(draft, "advance_projected_context_cache") and hasattr(
        draft,
        "project_target_hidden",
    ):
        draft.advance_projected_context_cache(
            draft_context=draft.project_target_hidden(target_hidden),
            cache=cache,
        )
        return
    raise RuntimeError("This DFlash runtime cannot train with split draft context caching.")


def _direct_draft_hidden(draft, block_embeddings, target_hidden, cache):
    if hasattr(draft, "forward_projected_context") and hasattr(draft, "project_target_hidden"):
        return draft.forward_projected_context(
            noise_embedding=block_embeddings,
            draft_context=draft.project_target_hidden(target_hidden),
            cache=cache,
        )
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


def _topk_soft_cross_entropy(logits, topk_indices, topk_logits, temperature: float):
    mx, _nn, _optim, _mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    temperature = float(temperature)
    student_logits = logits / temperature
    teacher_logits = topk_logits / temperature
    gathered_student = mx.take_along_axis(student_logits, topk_indices, axis=-1)
    teacher_log_norm = mx.logsumexp(teacher_logits, axis=-1, keepdims=True)
    teacher_probs = mx.exp(teacher_logits - teacher_log_norm)
    student_log_norm = mx.logsumexp(gathered_student, axis=-1, keepdims=True)
    student_log_probs = gathered_student - student_log_norm
    return -(teacher_probs * student_log_probs).sum(axis=-1).mean() * (temperature**2)


def _bind_dflash_mlx_draft_to_target(draft, target_model) -> None:
    if not hasattr(draft, "bind_target_model"):
        return
    _ensure_omlx_app_site_packages()
    try:
        from dflash_mlx.engine.target_ops import resolve_target_ops
    except ImportError as exc:
        raise RuntimeError("dflash_mlx target binding support is unavailable.") from exc
    target_ops = resolve_target_ops(target_model)
    draft.bind_target_model(target_model, target_ops=target_ops)


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
    topk_loss_weight: float = 1.0,
    topk_temperature: float = 1.0,
    anchor_span_tokens: int = 0,
    first_anchor_probability: float = 0.0,
    anchor_margin_min: float = 0.0,
    anchor_margin_top_fraction: float = 0.0,
    expected_label_source: str | None = None,
    seed: int = 13,
) -> Path:
    if max_steps < 1:
        return draft_dir
    loss_name = normalize_omlx_loss_fn(loss_fn)
    if hidden_loss_weight < 0:
        raise ValueError("hidden_loss_weight must be non-negative.")
    if topk_loss_weight < 0:
        raise ValueError("topk_loss_weight must be non-negative.")
    if topk_temperature <= 0:
        raise ValueError("topk_temperature must be positive.")
    if anchor_span_tokens < 0:
        raise ValueError("anchor_span_tokens must be non-negative.")
    if first_anchor_probability < 0 or first_anchor_probability > 1:
        raise ValueError("first_anchor_probability must be between 0 and 1.")
    if anchor_margin_min < 0:
        raise ValueError("anchor_margin_min must be non-negative.")
    if anchor_margin_top_fraction < 0 or anchor_margin_top_fraction > 1:
        raise ValueError("anchor_margin_top_fraction must be between 0 and 1.")
    mx, nn, optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    metadata = OmlxCacheMetadata.load(cache_dir)
    if expected_label_source is not None:
        expected = normalize_omlx_label_source(expected_label_source)
        if metadata.label_source != expected:
            raise ValueError(
                "OMLX cache label_source mismatch: "
                f"{metadata.label_source} != expected {expected}"
            )
    if (
        metadata.label_source == OMLX_LABEL_TARGET_GREEDY
        and metadata.generated_continuation_tokens <= 0
        and loss_name == OMLX_LOSS_CE_HIDDEN
        and hidden_loss_weight > 0
    ):
        raise ValueError(
            "target-greedy label caches must use --loss-fn ce or --hidden-loss-weight 0. "
            "Raw-sequence hidden states can conflict with target-greedy token labels."
        )
    if metadata.cache_format != OMLX_CACHE_FORMAT:
        raise ValueError(f"Unsupported OMLX cache format: {metadata.cache_format}")
    if loss_name in OMLX_LOSSES_WITH_TOPK_KL and metadata.target_top_k < 1:
        raise ValueError("top-k KL losses require an OMLX cache extracted with target_top_k >= 1.")
    if (anchor_margin_min > 0 or anchor_margin_top_fraction > 0) and metadata.target_top_k < 2:
        raise ValueError("margin-aware anchor sampling requires a cache with target_top_k >= 2.")
    draft_config = read_omlx_draft_config(draft_dir)
    _verify_cache_matches_draft(metadata, draft_config)
    draft = load_omlx_draft(draft_dir)
    samples = _load_cache_arrays(cache_dir, metadata)
    rng = random.Random(seed)
    optimizer = optim.AdamW(learning_rate=learning_rate)
    lm_head = None
    target_model = None
    if loss_name in OMLX_LOSSES_WITH_CE or loss_name in OMLX_LOSSES_WITH_TOPK_KL:
        target_ref = resolve_omlx_source_model(source_model or metadata.source_model)
        console.print(f"[bold]Loading OMLX target lm_head for token loss[/bold] {target_ref}")
        target_model, _tokenizer = mlx_lm_load(target_ref, lazy=True)
        if hasattr(target_model, "eval"):
            target_model.eval()
        lm_head = _get_lm_head(target_model)
    if target_model is not None:
        _bind_dflash_mlx_draft_to_target(draft, target_model)

    supports_split_context_cache = hasattr(draft, "advance_projected_context_cache") and hasattr(
        draft,
        "project_target_hidden",
    )

    def loss_fn_inner(
        model,
        cache_context_hidden,
        context_hidden,
        block_embeddings,
        label_hidden,
        label_tokens,
        target_topk_indices,
        target_topk_logits,
    ):
        draft_cache = _make_draft_cache(model)
        if int(cache_context_hidden.shape[1]) > 0:
            _advance_draft_cache_with_context(model, draft_cache, cache_context_hidden)
        hidden = _direct_draft_hidden(
            model,
            block_embeddings,
            context_hidden,
            draft_cache,
        )
        pred = hidden[:, 1:, :]
        hidden_loss = ((pred - label_hidden) ** 2).mean()
        if loss_name == OMLX_LOSS_HIDDEN_MSE:
            return hidden_loss
        if lm_head is None:
            raise RuntimeError("token training requires a loaded target lm_head.")
        logits = _softcap_logits(lm_head(pred), draft_config.final_logit_softcapping)
        ce_loss = nn.losses.cross_entropy(logits, label_tokens, reduction="mean")
        if loss_name == OMLX_LOSS_CE:
            return ce_loss
        topk_loss = None
        if loss_name in OMLX_LOSSES_WITH_TOPK_KL:
            if target_topk_indices is None or target_topk_logits is None:
                raise RuntimeError("top-k KL loss requires target_topk arrays in the cache sample.")
            topk_loss = _topk_soft_cross_entropy(
                logits,
                target_topk_indices,
                target_topk_logits,
                topk_temperature,
            )
        if loss_name == OMLX_LOSS_TOPK_KL:
            return topk_loss
        if loss_name == OMLX_LOSS_CE_TOPK_KL:
            return ce_loss + (topk_loss_weight * topk_loss)
        if loss_name == OMLX_LOSS_CE_HIDDEN_TOPK_KL:
            return (
                ce_loss
                + (topk_loss_weight * topk_loss)
                + (hidden_loss_weight * hidden_loss)
            )
        return ce_loss + (hidden_loss_weight * hidden_loss)

    value_and_grad = nn.value_and_grad(draft, loss_fn_inner)
    progress = trange(max_steps, desc="omlx-draft-training", leave=True)
    for step in progress:
        sample = rng.choice(samples)
        token_count = int(sample["tokens"].shape[0])
        min_anchor, max_anchor = _runtime_aligned_anchor_bounds(
            token_count,
            metadata.block_size,
            _sample_min_anchor(sample),
        )
        anchor = _sample_margin_aligned_anchor(
            sample,
            min_anchor,
            max_anchor,
            metadata,
            rng,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=first_anchor_probability,
            anchor_margin_min=anchor_margin_min,
            anchor_margin_top_fraction=anchor_margin_top_fraction,
        )
        segment_start = (
            _runtime_aligned_segment_start(sample, metadata, anchor, rng)
            if supports_split_context_cache
            else 0
        )
        window = _runtime_aligned_training_arrays(
            sample,
            metadata,
            anchor,
            segment_start=segment_start,
        )
        cache_context_hidden = mx.array(window["cache_context_hidden"][None, :, :])
        context_hidden = mx.array(window["context_hidden"][None, :, :])
        target_hidden = mx.array(window["target_hidden"][None, :, :])
        label_tokens = mx.array(window["label_tokens"][None, :], dtype=mx.uint32)
        if loss_name in OMLX_LOSSES_WITH_TOPK_KL:
            target_topk_indices = mx.array(
                window["target_topk_indices"][None, :, :],
                dtype=mx.uint32,
            )
            target_topk_logits = mx.array(window["target_topk_logits"][None, :, :])
        else:
            target_topk_indices = None
            target_topk_logits = None
        anchor_embedding = mx.array(window["anchor_embedding"])
        mask_embedding = mx.array(window["mask_embedding"])
        mask_block = mx.broadcast_to(
            mask_embedding,
            (metadata.block_size - 1, metadata.hidden_size),
        )
        block_embeddings = mx.concatenate([anchor_embedding[None, :], mask_block], axis=0)[
            None, :, :
        ]
        loss, grads = value_and_grad(
            draft,
            cache_context_hidden,
            context_hidden,
            block_embeddings,
            target_hidden,
            label_tokens,
            target_topk_indices,
            target_topk_logits,
        )
        optimizer.update(draft, grads)
        mx.eval(draft.parameters(), optimizer.state, loss)
        if (int(step) + 1) % 64 == 0 and hasattr(mx, "clear_cache"):
            mx.clear_cache()
        progress.set_postfix(loss=f"{float(loss.item()):.5f}")
    draft.save_weights(str(draft_dir / "model.safetensors"))
    config_path = draft_dir / "config.json"
    if config_path.exists():
        config_payload = json.loads(config_path.read_text())
        config_payload.setdefault("dflasher", {})["training_objective"] = loss_name
        config_path.write_text(json.dumps(config_payload, indent=2) + "\n")
    manifest = {
        "source_model": metadata.source_model,
        "cache_dir": str(cache_dir),
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "objective": loss_name,
        "hidden_loss_weight": hidden_loss_weight,
        "topk_loss_weight": topk_loss_weight,
        "topk_temperature": topk_temperature,
        "anchor_span_tokens": anchor_span_tokens,
        "first_anchor_probability": first_anchor_probability,
        "anchor_margin_min": anchor_margin_min,
        "anchor_margin_top_fraction": anchor_margin_top_fraction,
        "hidden_target": metadata.hidden_target,
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
        label_source=options.label_source,
        generated_continuation_tokens=options.generated_continuation_tokens,
        use_chat_template=options.use_chat_template,
        include_prefill_anchors=options.include_prefill_anchors,
        target_top_k=options.target_top_k,
        hidden_target=options.hidden_target,
        overwrite=options.overwrite,
    )
    init_omlx_draft(
        source_model=source_model,
        output_dir=options.output_dir,
        block_size=options.block_size,
        draft_layers=options.draft_layers,
        intermediate_size=options.intermediate_size,
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
            topk_loss_weight=options.topk_loss_weight,
            topk_temperature=options.topk_temperature,
            anchor_span_tokens=options.anchor_span_tokens,
            first_anchor_probability=options.first_anchor_probability,
            anchor_margin_min=options.anchor_margin_min,
            anchor_margin_top_fraction=options.anchor_margin_top_fraction,
            expected_label_source=options.label_source,
            seed=options.seed,
        )
    build_manifest = {
        "source_model": source_model,
        "backend": "omlx",
        "output": str(options.output_dir),
        "cache_dir": str(cache_dir),
        "training_objective": normalize_omlx_loss_fn(options.loss_fn),
        "label_source": normalize_omlx_label_source(options.label_source),
        "generated_continuation_tokens": options.generated_continuation_tokens,
        "use_chat_template": options.use_chat_template,
        "include_prefill_anchors": options.include_prefill_anchors,
        "hidden_loss_weight": options.hidden_loss_weight,
        "target_top_k": options.target_top_k,
        "hidden_target": normalize_omlx_hidden_target(options.hidden_target),
        "topk_loss_weight": options.topk_loss_weight,
        "topk_temperature": options.topk_temperature,
        "anchor_span_tokens": options.anchor_span_tokens,
        "first_anchor_probability": options.first_anchor_probability,
        "anchor_margin_min": options.anchor_margin_min,
        "anchor_margin_top_fraction": options.anchor_margin_top_fraction,
        "format": OMLX_DRAFT_FORMAT,
    }
    (options.output_dir / "dflasher_build_manifest.json").write_text(
        json.dumps(build_manifest, indent=2) + "\n"
    )
    return options.output_dir


def _prompt_tokens(tokenizer, prompt: str, *, use_chat_template: bool = False):
    _mx, _nn, _optim, _mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
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
    *,
    use_chat_template: bool = False,
) -> int:
    prompt_len = len(_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template))
    if prompt_len + max_new_tokens > max_position_embeddings:
        raise ValueError(
            "Prompt exceeds source context window: "
            f"prompt_tokens={prompt_len}, max_new_tokens={max_new_tokens}, "
            f"max_position_embeddings={max_position_embeddings}"
        )
    return prompt_len


def greedy_omlx_tokens(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    *,
    use_chat_template: bool = False,
) -> list[int]:
    mx, _nn, _optim, _mlx_lm_load, make_prompt_cache, make_sampler = _import_mlx()
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    if max_new_tokens == 0:
        return []
    prompt_ids = mx.array(
        _prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template),
        dtype=mx.uint32,
    )
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


def _decode_omlx_tokens(tokenizer, tokens: list[int]) -> str:
    try:
        return str(tokenizer.decode(tokens))
    except TypeError:
        return "".join(str(tokenizer.decode(int(token))) for token in tokens)


def _generate_with_dflash_mlx_runtime(
    source_model: str,
    draft_dir: Path,
    prompt: str,
    max_new_tokens: int,
    verify_mode: str | None = "dflash",
    draft_window_size: int | None = None,
    draft_sink_size: int | None = None,
    verify_len_cap: int | None = None,
    block_tokens: int | None = None,
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    use_chat_template: bool = False,
) -> tuple[str, list[int], float]:
    _ensure_omlx_app_site_packages()
    try:
        from dflash_mlx.runtime.bundle import load_runtime_bundle
        from dflash_mlx.runtime.context import build_offline_runtime_context
    except ImportError as exc:
        raise RuntimeError("dflash_mlx generation support is unavailable.") from exc

    runtime_context = build_offline_runtime_context(
        verify_mode=verify_mode,
        draft_window_size=draft_window_size,
        draft_sink_size=draft_sink_size,
        verify_len_cap=verify_len_cap,
        target_fa_window=target_fa_window,
        prefill_step_size=prefill_step_size,
    )
    bundle = load_runtime_bundle(
        model_ref=source_model,
        draft_ref=str(draft_dir),
        draft_quant="none",
        verify_config=runtime_context.verify,
        lazy=True,
    )
    return _generate_with_dflash_mlx_bundle(
        bundle,
        runtime_context,
        prompt,
        max_new_tokens,
        block_tokens=block_tokens,
        use_chat_template=use_chat_template,
    )


def _generate_with_dflash_mlx_bundle(
    bundle,
    runtime_context,
    prompt: str,
    max_new_tokens: int,
    block_tokens: int | None = None,
    use_chat_template: bool = False,
) -> tuple[str, list[int], float]:
    try:
        from dflash_mlx.engine.events import SummaryEvent, TokenEvent
        from dflash_mlx.runtime import get_stop_token_ids, stream_dflash_generate
    except ImportError as exc:
        raise RuntimeError("dflash_mlx generation support is unavailable.") from exc

    stream = stream_dflash_generate(
        target_model=bundle.target_model,
        target_ops=bundle.target_ops,
        tokenizer=bundle.tokenizer,
        draft_model=bundle.draft_model,
        draft_backend=bundle.draft_backend,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        block_tokens=block_tokens,
        use_chat_template=use_chat_template,
        stop_token_ids=get_stop_token_ids(bundle.tokenizer),
        runtime_context=runtime_context,
    )
    tokens: list[int] = []
    summary = None
    try:
        for event in stream:
            if isinstance(event, TokenEvent):
                tokens.append(int(event.token_id))
            elif isinstance(event, SummaryEvent):
                summary = event
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
    acceptance = float(getattr(summary, "acceptance_ratio", 0.0) or 0.0)
    return _decode_omlx_tokens(bundle.tokenizer, tokens), tokens, acceptance


def _greedy_dflash_mlx_tokens(
    bundle,
    prompt: str,
    max_new_tokens: int,
    *,
    use_chat_template: bool = False,
) -> list[int]:
    if max_new_tokens < 1:
        return []
    try:
        import mlx.core as mx
        from dflash_mlx.engine.sampling import greedy_tokens_with_mask, prepare_prompt_tokens
    except ImportError as exc:
        raise RuntimeError("dflash_mlx greedy support is unavailable.") from exc

    prompt_tokens = prepare_prompt_tokens(
        bundle.tokenizer,
        prompt,
        use_chat_template=use_chat_template,
    )
    target_cache = bundle.target_ops.make_cache(
        bundle.target_model,
        enable_speculative_linear_cache=True,
    )
    logits = _dflash_mlx_runtime_prefill_logits(
        bundle.target_ops,
        bundle.target_model,
        target_cache,
        prompt_tokens,
        mx=mx,
    )
    token = greedy_tokens_with_mask(logits[:, -1, :], None).reshape(-1)
    mx.eval(token)
    tokens = [int(token.item())]
    try:
        from dflash_mlx.runtime import get_stop_token_ids

        stop_ids = set(int(token_id) for token_id in get_stop_token_ids(bundle.tokenizer))
    except ImportError:
        stop_ids = set(_eos_token_ids(bundle.tokenizer))
    for _ in range(max_new_tokens - 1):
        if tokens[-1] in stop_ids:
            break
        logits, _captured = bundle.target_ops.verify_block(
            target_model=bundle.target_model,
            verify_ids=token.reshape(1, 1).astype(mx.uint32),
            target_cache=target_cache,
            capture_layer_ids=set(),
        )
        token = greedy_tokens_with_mask(logits[:, -1, :], None).reshape(-1)
        mx.eval(token)
        tokens.append(int(token.item()))
    return tokens


def generate_omlx_dflash(
    source_model: str,
    draft_dir: Path,
    prompt: str,
    max_new_tokens: int = 64,
    verify_mode: str | None = "dflash",
    draft_window_size: int | None = None,
    draft_sink_size: int | None = None,
    verify_len_cap: int | None = None,
    block_tokens: int | None = None,
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    use_chat_template: bool = False,
) -> tuple[str, list[int], float]:
    source_model = resolve_omlx_source_model(source_model)
    _mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    runtime = _import_dflash_mlx()
    if runtime.flavor == "dflash_mlx":
        return _generate_with_dflash_mlx_runtime(
            source_model,
            draft_dir,
            prompt,
            max_new_tokens,
            verify_mode=verify_mode,
            draft_window_size=draft_window_size,
            draft_sink_size=draft_sink_size,
            verify_len_cap=verify_len_cap,
            block_tokens=block_tokens,
            target_fa_window=target_fa_window,
            prefill_step_size=prefill_step_size,
            use_chat_template=use_chat_template,
        )
    if runtime.stream_generate is None:
        raise RuntimeError("DFlash stream_generate is unavailable.")
    stream_generate = runtime.stream_generate
    model, tokenizer = mlx_lm_load(source_model, lazy=True)
    source_config = read_model_config(source_model)
    _validate_prompt_context(
        tokenizer,
        prompt,
        max_new_tokens,
        int(_config_value(source_config, "max_position_embeddings", 2048)),
        use_chat_template=use_chat_template,
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
    verify_mode: str | None = "dflash",
    draft_window_size: int | None = None,
    draft_sink_size: int | None = None,
    verify_len_cap: int | None = None,
    block_tokens: int | None = None,
    target_fa_window: int | None = None,
    prefill_step_size: int | None = None,
    use_chat_template: bool = False,
) -> OmlxEvalResult:
    source_model = resolve_omlx_source_model(source_model)
    _mx, _nn, _optim, mlx_lm_load, _make_prompt_cache, _make_sampler = _import_mlx()
    runtime = _import_dflash_mlx()
    if runtime.flavor == "dflash_mlx":
        _ensure_omlx_app_site_packages()
        try:
            from dflash_mlx.runtime.bundle import load_runtime_bundle
            from dflash_mlx.runtime.context import build_offline_runtime_context
        except ImportError as exc:
            raise RuntimeError("dflash_mlx generation support is unavailable.") from exc
        runtime_context = build_offline_runtime_context(
            verify_mode=verify_mode,
            draft_window_size=draft_window_size,
            draft_sink_size=draft_sink_size,
            verify_len_cap=verify_len_cap,
            target_fa_window=target_fa_window,
            prefill_step_size=prefill_step_size,
        )
        bundle = load_runtime_bundle(
            model_ref=source_model,
            draft_ref=str(draft_dir),
            draft_quant="none",
            verify_config=runtime_context.verify,
            lazy=True,
        )
        tokenizer = bundle.tokenizer
        source_config = read_model_config(source_model)
        max_context = int(_config_value(source_config, "max_position_embeddings", 2048))
        exact = 0
        acceptances: list[float] = []
        max_prompt_tokens = 0
        for prompt in prompts:
            prompt_tokens = _validate_prompt_context(
                tokenizer,
                prompt,
                max_new_tokens,
                max_context,
                use_chat_template=use_chat_template,
            )
            max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
            expected = _greedy_dflash_mlx_tokens(
                bundle,
                prompt,
                max_new_tokens,
                use_chat_template=use_chat_template,
            )
            _text, actual, acceptance = _generate_with_dflash_mlx_bundle(
                bundle,
                runtime_context,
                prompt,
                max_new_tokens,
                block_tokens=block_tokens,
                use_chat_template=use_chat_template,
            )
            actual = actual[: len(expected)]
            if actual == expected:
                exact += 1
            acceptances.append(acceptance)
        mean_acceptance = sum(acceptances) / len(acceptances) if acceptances else 0.0
        return OmlxEvalResult(
            prompts=len(prompts),
            exact_matches=exact,
            mean_acceptance=mean_acceptance,
            mean_draft_acceptance=mean_acceptance,
            max_prompt_tokens=max_prompt_tokens,
        )
    if runtime.stream_generate is None:
        raise RuntimeError("DFlash stream_generate is unavailable.")
    stream_generate = runtime.stream_generate
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
        prompt_tokens = _validate_prompt_context(
            tokenizer,
            prompt,
            max_new_tokens,
            max_context,
            use_chat_template=use_chat_template,
        )
        max_prompt_tokens = max(max_prompt_tokens, prompt_tokens)
        expected = greedy_omlx_tokens(
            model,
            tokenizer,
            prompt,
            max_new_tokens,
            use_chat_template=use_chat_template,
        )
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


def _resolve_omlx_app_patch_paths(
    app_path: str | Path,
) -> tuple[Path, Path, Path, Path, Path, Path]:
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
        spec_epoch_path = site_package / "dflash_mlx" / "engine" / "spec_epoch.py"
        dflash_engine_path = contents / "Resources" / "omlx" / "engine" / "dflash.py"
        model_settings_path = contents / "Resources" / "omlx" / "model_settings.py"
        dflash_lifecycle_path = contents / "Resources" / "omlx" / "patches" / "dflash_lifecycle.py"
        if (
            target_ops_path.exists()
            and spec_epoch_path.exists()
            and dflash_engine_path.exists()
            and model_settings_path.exists()
            and dflash_lifecycle_path.exists()
        ):
            return (
                target_ops_path,
                target_backend_path,
                spec_epoch_path,
                dflash_engine_path,
                model_settings_path,
                dflash_lifecycle_path,
            )
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
    verify_mode_block = """        self._verify_mode = (
            getattr(model_settings, "dflash_verify_mode", None)
            if model_settings
            else None
        )
"""
    verify_len_cap_block = """        self._verify_len_cap = (
            getattr(model_settings, "dflash_verify_len_cap", None)
            if model_settings
            else None
        )
"""
    block_tokens_block = """        self._block_tokens = (
            getattr(model_settings, "dflash_block_tokens", None)
            if model_settings
            else None
        )
"""
    if 'getattr(model_settings, "dflash_verify_len_cap", None)' not in patched:
        if verify_mode_block not in patched:
            raise ValueError("Could not patch oMLX dflash.py: verify mode block not found.")
        patched = patched.replace(verify_mode_block, verify_mode_block + verify_len_cap_block, 1)
    if 'getattr(model_settings, "dflash_block_tokens", None)' not in patched:
        if verify_len_cap_block in patched:
            patched = patched.replace(
                verify_len_cap_block,
                verify_len_cap_block + block_tokens_block,
                1,
            )
        elif verify_mode_block in patched:
            patched = patched.replace(verify_mode_block, verify_mode_block + block_tokens_block, 1)
        else:
            raise ValueError("Could not patch oMLX dflash.py: block token anchor not found.")
    if "verify_len_cap=self._verify_len_cap" not in patched:
        patched = patched.replace(
            "            verify_mode=self._verify_mode,\n",
            "            verify_mode=self._verify_mode,\n"
            "            verify_len_cap=self._verify_len_cap,\n",
            1,
        )
    if "block_tokens=self._block_tokens" not in patched:
        patched = patched.replace(
            "            publish_generation_snapshot=prefix_flow.publish_generation_snapshot,\n"
            "            runtime_context=self._runtime_context,\n",
            "            publish_generation_snapshot=prefix_flow.publish_generation_snapshot,\n"
            "            block_tokens=self._block_tokens,\n"
            "            runtime_context=self._runtime_context,\n",
            1,
        )
    if (
        'getattr(model_settings, "dflash_verify_len_cap", None)' not in patched
        or 'getattr(model_settings, "dflash_block_tokens", None)' not in patched
        or "verify_len_cap=self._verify_len_cap" not in patched
        or "block_tokens=self._block_tokens" not in patched
    ):
        raise ValueError("Could not patch oMLX dflash.py runtime knobs.")
    return patched


def _patch_omlx_model_settings_text(text: str) -> str:
    patched = text
    doc_marker = (
        "        dflash_draft_sink_size: Attention-sink tokens always kept regardless "
        "of window\n"
        "            (None = dflash default 64).\n"
    )
    doc_insert = doc_marker + (
        "        dflash_verify_len_cap: Max target tokens verified per DFlash cycle\n"
        "            (None = dflash default, equal to block size).\n"
        "        dflash_block_tokens: Max DFlash block tokens requested at runtime\n"
        "            (None = draft checkpoint block size).\n"
    )
    if "dflash_verify_len_cap:" not in patched:
        if doc_marker not in patched:
            raise ValueError("Could not patch oMLX model_settings.py docs.")
        patched = patched.replace(doc_marker, doc_insert, 1)
    field_marker = """    dflash_draft_window_size: Optional[int] = None
    dflash_draft_sink_size: Optional[int] = None
    dflash_verify_mode: Optional[str] = None  # "dflash" | "adaptive" | "ddtree" | "off"
"""
    field_insert = """    dflash_draft_window_size: Optional[int] = None
    dflash_draft_sink_size: Optional[int] = None
    dflash_verify_len_cap: Optional[int] = None
    dflash_block_tokens: Optional[int] = None
    dflash_verify_mode: Optional[str] = None  # "dflash" | "adaptive" | "ddtree" | "off"
"""
    if "dflash_block_tokens: Optional[int]" not in patched:
        if field_marker not in patched:
            raise ValueError("Could not patch oMLX model_settings.py fields.")
        patched = patched.replace(field_marker, field_insert, 1)
    return patched


def _patch_omlx_dflash_lifecycle_text(text: str) -> str:
    if "target_minimax_m2" in text and "_dflasher_minimax_attention_hook_installed" in text:
        return text
    marker = """    try:
        from dflash_mlx.engine import target_gemma4 as _gemma4
    except ImportError:
        logger.debug("dflash_mlx.engine.target_gemma4 not importable")
    else:
        wrapped_any |= _wrap_installer(
            _gemma4,
            "_install_full_attention_gqa_hook",
            "_dflash_full_attention_gqa_installed",
        )

"""
    insert = marker + """    try:
        from dflash_mlx.engine import target_minimax_m2 as _minimax_m2
    except ImportError:
        logger.debug("dflash_mlx.engine.target_minimax_m2 not importable")
    else:
        wrapped_any |= _wrap_installer(
            _minimax_m2,
            "_install_minimax_attention_hook",
            "_dflasher_minimax_attention_hook_installed",
        )

"""
    if marker not in text:
        raise ValueError("Could not patch oMLX dflash_lifecycle.py.")
    return text.replace(marker, insert, 1)


def _patch_dflash_spec_epoch_text(text: str) -> str:
    if (
        "correct_committed_block_after_acceptance" in text
        and "commit_correction" in text
        and "DFLASH_MINIMAX_BLOCK2_EARLY_REJECT" in text
    ):
        return text
    patched = text
    needs_commit_correction_patch = not (
        "correct_committed_block_after_acceptance" in patched
        and "commit_correction" in patched
    )
    hidden_block = """            if profile_cycles:
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            committed_hidden = target_ops.extract_context_feature(
                verify_hidden_states,
                target_layer_id_list,
            )[:, : (1 + acceptance_len), :]
            if profile_cycles:
                mx.eval(committed_hidden, posterior)
            else:
                mx.async_eval(committed_hidden)
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
"""
    hidden_replacement = """            if profile_cycles:
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            hidden_extract_start_ns = time.perf_counter_ns() if profile_cycles else 0
            commit_correction = None
            commit_corrector = getattr(
                target_ops,
                "correct_committed_block_after_acceptance",
                None,
            )
            if callable(commit_corrector):
                commit_correction = commit_corrector(
                    target_model=target_model,
                    target_cache=target_cache,
                    verify_token_ids=verify_ids,
                    target_layer_ids=target_layer_id_list,
                    capture_layer_ids=capture_layer_ids,
                    prefix_len=cycle_prefix_len,
                    acceptance_length=acceptance_len,
                    suppress_token_mask=suppress_token_mask,
                )
            if commit_correction is None:
                committed_hidden = target_ops.extract_context_feature(
                    verify_hidden_states,
                    target_layer_id_list,
                )[:, :commit_count, :]
                if profile_cycles:
                    mx.eval(committed_hidden, posterior)
                else:
                    mx.async_eval(committed_hidden)
            else:
                committed_hidden = commit_correction["committed_hidden"]
            if profile_cycles:
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

"""
    if needs_commit_correction_patch:
        if hidden_block not in patched:
            raise ValueError("Could not patch spec_epoch.py hidden commit block.")
        patched = patched.replace(hidden_block, hidden_replacement, 1)
        patched = patched.replace(
            "            state.last_cycle_logits = verify_logits[:, acceptance_len, :]\n",
            """            if commit_correction is None:
                state.last_cycle_logits = verify_logits[:, acceptance_len, :]
            else:
                state.last_cycle_logits = commit_correction["last_cycle_logits"]
""",
            1,
        )
    replay_block = """            replay_cycle_ns = target_ops.restore_after_acceptance(
                target_cache,
                target_len=state.start,
                acceptance_length=acceptance_len,
                drafted_tokens=max(0, verify_token_count - 1),
            )
"""
    replay_replacement = """            if commit_correction is None:
                replay_cycle_ns = target_ops.restore_after_acceptance(
                    target_cache,
                    target_len=state.start,
                    acceptance_length=acceptance_len,
                    drafted_tokens=max(0, verify_token_count - 1),
                )
            else:
                replay_cycle_ns = int(commit_correction.get("replay_ns", 0) or 0)
"""
    if needs_commit_correction_patch:
        if replay_block not in patched:
            raise ValueError("Could not patch spec_epoch.py rollback block.")
        patched = patched.replace(replay_block, replay_replacement, 1)
        patched = patched.replace(
            "            staged_first_next = posterior[acceptance_len : acceptance_len + 1]\n",
            """            if commit_correction is None:
                staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            else:
                staged_first_next = commit_correction["staged_first_next"]
""",
            1,
        )
    if "DFLASH_MINIMAX_BLOCK2_EARLY_REJECT" not in patched:
        verify_block = """            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
"""
        verify_replacement = """            verify_token_count = verify_token_count_for_block(block_len, verify_len_cap)
            serial_block2_verify = (
                int(block_len) == 2
                and int(verify_token_count) == 2
                and not profile_cycles
                and drafted is not None
                and getattr(target_ops, "backend_name", "") == "minimax_m2"
                and _omlx_env_flag("DFLASH_MINIMAX_BLOCK2_EARLY_REJECT", True)
            )
            if profile_cycles or block_len <= 1:
                verify_token_ids = block_token_ids[:verify_token_count]
            elif verify_token_count <= 1:
                verify_token_ids = current_staged_first[:1]
            elif serial_block2_verify:
                verify_token_ids = current_staged_first[:1]
            else:
                verify_token_ids = mx.concatenate(
                    [current_staged_first[:1], drafted[: verify_token_count - 1]],
                    axis=0,
                )
"""
        if verify_block not in patched:
            raise ValueError("Could not patch spec_epoch.py MiniMax block2 verify input.")
        patched = patched.replace(verify_block, verify_replacement, 1)
        acceptance_block = """            acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            if not profile_cycles:
                mx.async_eval(posterior)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            state.acceptance_history.append(acceptance_len)
"""
        acceptance_replacement = """            acceptance_start_ns = time.perf_counter_ns() if profile_cycles else 0
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            if not profile_cycles:
                mx.async_eval(posterior)
            serial_committed_hidden = None
            serial_last_cycle_logits = None
            serial_staged_first_next = None
            if serial_block2_verify:
                acceptance_len = int((drafted[:1] == posterior[:1]).item())
                if acceptance_len > 0:
                    second_verify_ids = drafted[:1][None]
                    second_verify_start_ns = time.perf_counter_ns()
                    second_verify_logits, second_verify_hidden_states = target_ops.verify_block(
                        target_model=target_model,
                        verify_ids=second_verify_ids,
                        target_cache=target_cache,
                        capture_layer_ids=capture_layer_ids,
                    )
                    if profile_cycles:
                        eval_logits_and_captured(
                            second_verify_logits,
                            second_verify_hidden_states,
                        )
                    second_verify_cycle_ns = time.perf_counter_ns() - second_verify_start_ns
                    verify_cycle_ns += second_verify_cycle_ns
                    verify_ns_total += second_verify_cycle_ns
                    first_committed_hidden = target_ops.extract_context_feature(
                        verify_hidden_states,
                        target_layer_id_list,
                    )[:, :1, :]
                    second_committed_hidden = target_ops.extract_context_feature(
                        second_verify_hidden_states,
                        target_layer_id_list,
                    )[:, :1, :]
                    serial_committed_hidden = mx.concatenate(
                        [first_committed_hidden, second_committed_hidden],
                        axis=1,
                    )
                    second_posterior = greedy_tokens_with_mask(
                        second_verify_logits[0],
                        suppress_token_mask,
                    )
                    posterior = mx.concatenate([posterior[:1], second_posterior[:1]], axis=0)
                    verify_token_ids = mx.concatenate(
                        [current_staged_first[:1], drafted[:1]],
                        axis=0,
                    )
                    serial_last_cycle_logits = second_verify_logits[:, 0, :]
                    serial_staged_first_next = second_posterior[:1]
                else:
                    serial_committed_hidden = target_ops.extract_context_feature(
                        verify_hidden_states,
                        target_layer_id_list,
                    )[:, :1, :]
                    serial_last_cycle_logits = verify_logits[:, 0, :]
                    serial_staged_first_next = posterior[:1]
            else:
                acceptance_len = int(
                    _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
                )
            state.acceptance_history.append(acceptance_len)
"""
        if acceptance_block not in patched:
            raise ValueError("Could not patch spec_epoch.py MiniMax block2 acceptance.")
        patched = patched.replace(acceptance_block, acceptance_replacement, 1)
        patched = patched.replace(
            "            if callable(commit_corrector):\n",
            "            if callable(commit_corrector) and not serial_block2_verify:\n",
            1,
        )
        committed_block = """            if commit_correction is None:
                committed_hidden = target_ops.extract_context_feature(
                    verify_hidden_states,
                    target_layer_id_list,
                )[:, :commit_count, :]
                if profile_cycles:
                    mx.eval(committed_hidden, posterior)
                else:
                    mx.async_eval(committed_hidden)
"""
        committed_replacement = """            if commit_correction is None:
                if serial_committed_hidden is not None:
                    committed_hidden = serial_committed_hidden
                else:
                    committed_hidden = target_ops.extract_context_feature(
                        verify_hidden_states,
                        target_layer_id_list,
                    )[:, :commit_count, :]
                if profile_cycles:
                    mx.eval(committed_hidden, posterior)
                else:
                    mx.async_eval(committed_hidden)
"""
        if committed_block not in patched:
            raise ValueError("Could not patch spec_epoch.py MiniMax block2 hidden commit.")
        patched = patched.replace(committed_block, committed_replacement, 1)
        patched = patched.replace(
            """            if commit_correction is None:
                state.last_cycle_logits = verify_logits[:, acceptance_len, :]
            else:
                state.last_cycle_logits = commit_correction["last_cycle_logits"]
""",
            """            if commit_correction is None:
                if serial_last_cycle_logits is not None:
                    state.last_cycle_logits = serial_last_cycle_logits
                else:
                    state.last_cycle_logits = verify_logits[:, acceptance_len, :]
            else:
                state.last_cycle_logits = commit_correction["last_cycle_logits"]
""",
            1,
        )
        patched = patched.replace(
            "                    drafted_tokens=max(0, verify_token_count - 1),\n",
            """                    drafted_tokens=(
                        0 if serial_block2_verify else max(0, verify_token_count - 1)
                    ),
""",
            1,
        )
        patched = patched.replace(
            """            if commit_correction is None:
                staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            else:
                staged_first_next = commit_correction["staged_first_next"]
""",
            """            if commit_correction is None:
                if serial_staged_first_next is not None:
                    staged_first_next = serial_staged_first_next
                else:
                    staged_first_next = posterior[acceptance_len : acceptance_len + 1]
            else:
                staged_first_next = commit_correction["staged_first_next"]
""",
            1,
        )
    if (
        "correct_committed_block_after_acceptance" not in patched
        or "commit_correction" not in patched
        or "DFLASH_MINIMAX_BLOCK2_EARLY_REJECT" not in patched
    ):
        raise ValueError("Could not patch spec_epoch.py MiniMax commit correction.")
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
    (
        target_ops_path,
        target_backend_path,
        spec_epoch_path,
        dflash_engine_path,
        model_settings_path,
        dflash_lifecycle_path,
    ) = _resolve_omlx_app_patch_paths(app_path)
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

    spec_epoch_text = spec_epoch_path.read_text()
    patched_spec_epoch = _patch_dflash_spec_epoch_text(spec_epoch_text)
    if patched_spec_epoch != spec_epoch_text:
        changed.append(spec_epoch_path)

    dflash_engine_text = dflash_engine_path.read_text()
    patched_dflash_engine = _patch_omlx_dflash_text(dflash_engine_text)
    if patched_dflash_engine != dflash_engine_text:
        changed.append(dflash_engine_path)

    model_settings_text = model_settings_path.read_text()
    patched_model_settings = _patch_omlx_model_settings_text(model_settings_text)
    if patched_model_settings != model_settings_text:
        changed.append(model_settings_path)

    dflash_lifecycle_text = dflash_lifecycle_path.read_text()
    patched_dflash_lifecycle = _patch_omlx_dflash_lifecycle_text(dflash_lifecycle_text)
    if patched_dflash_lifecycle != dflash_lifecycle_text:
        changed.append(dflash_lifecycle_path)

    if not dry_run:
        for path in changed:
            originals[path] = path.read_text() if path.exists() else None
        try:
            if target_backend_changed:
                _write_text_with_backup(target_backend_path, target_backend_source)
            if patched_target_ops != target_ops_text:
                _write_text_with_backup(target_ops_path, patched_target_ops)
            if patched_spec_epoch != spec_epoch_text:
                _write_text_with_backup(spec_epoch_path, patched_spec_epoch)
            if patched_dflash_engine != dflash_engine_text:
                _write_text_with_backup(dflash_engine_path, patched_dflash_engine)
            if patched_model_settings != model_settings_text:
                _write_text_with_backup(model_settings_path, patched_model_settings)
            if patched_dflash_lifecycle != dflash_lifecycle_text:
                _write_text_with_backup(dflash_lifecycle_path, patched_dflash_lifecycle)
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
        spec_epoch_path=spec_epoch_path,
        dflash_engine_path=dflash_engine_path,
        model_settings_path=model_settings_path,
        dflash_lifecycle_path=dflash_lifecycle_path,
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
    if installed_name in PROTECTED_OMLX_DRAFT_NAMES:
        raise ValueError(f"Refusing to overwrite protected oMLX draft: {installed_name}")
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
    draft_quant_enabled: bool = False,
    draft_quant_weight_bits: int = 4,
    draft_quant_activation_bits: int = 16,
    draft_quant_group_size: int = 64,
    dflash_max_ctx: int | None = None,
    dflash_draft_window_size: int | None = None,
    dflash_draft_sink_size: int | None = None,
    dflash_verify_len_cap: int | None = None,
    dflash_block_tokens: int | None = None,
) -> OmlxInstallResult:
    source_path = Path(resolve_omlx_source_model(source_model))
    draft_dir = Path(draft_dir).expanduser()
    if not (draft_dir / "config.json").exists() or not (draft_dir / "model.safetensors").exists():
        raise ValueError(f"Draft directory is missing config.json/model.safetensors: {draft_dir}")
    _validate_omlx_draft_for_source(source_path, draft_dir)
    for name, value in {
        "draft_quant_weight_bits": draft_quant_weight_bits,
        "draft_quant_activation_bits": draft_quant_activation_bits,
        "draft_quant_group_size": draft_quant_group_size,
        "dflash_max_ctx": dflash_max_ctx,
        "dflash_draft_window_size": dflash_draft_window_size,
        "dflash_draft_sink_size": dflash_draft_sink_size,
        "dflash_verify_len_cap": dflash_verify_len_cap,
        "dflash_block_tokens": dflash_block_tokens,
    }.items():
        if value is not None and int(value) < 1:
            raise ValueError(f"{name} must be positive when provided.")
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
            "dflash_draft_quant_enabled": bool(draft_quant_enabled),
            "dflash_draft_quant_weight_bits": int(draft_quant_weight_bits),
            "dflash_draft_quant_activation_bits": int(draft_quant_activation_bits),
            "dflash_draft_quant_group_size": int(draft_quant_group_size),
            "turboquant_kv_enabled": False,
            "specprefill_enabled": False,
            "mtp_enabled": False,
            "trust_remote_code": True,
        }
    )
    optional_runtime_settings = {
        "dflash_max_ctx": dflash_max_ctx,
        "dflash_draft_window_size": dflash_draft_window_size,
        "dflash_draft_sink_size": dflash_draft_sink_size,
        "dflash_verify_len_cap": dflash_verify_len_cap,
        "dflash_block_tokens": dflash_block_tokens,
    }
    for key, value in optional_runtime_settings.items():
        if value is None:
            current.pop(key, None)
        else:
            current[key] = int(value)
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
