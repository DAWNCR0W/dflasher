from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dflasher.config import DFlasherConfig
from dflasher.hf import (
    dtype_from_name,
    get_target_shape,
    load_target_model,
    load_tokenizer,
    resolve_device,
)
from dflasher.modeling import DFlashLiteDraftModel, extract_context_feature


@dataclass(frozen=True)
class GenerationStats:
    accepted_tokens: int
    drafted_tokens: int
    target_steps: int
    generated_tokens: int = 0
    target_tokens: int = 0

    @property
    def mean_acceptance(self) -> float:
        """Mean number of accepted draft tokens per speculative verification step."""
        if self.target_steps == 0:
            return 0.0
        return self.accepted_tokens / self.target_steps

    @property
    def mean_generated_per_step(self) -> float:
        if self.target_steps == 0:
            return 0.0
        return self.generated_tokens / self.target_steps


def normalize_eos_token_ids(eos_token_id: int | list[int] | tuple[int, ...] | None) -> set[int]:
    if eos_token_id is None:
        return set()
    if isinstance(eos_token_id, int):
        return {eos_token_id}
    return {int(token_id) for token_id in eos_token_id}


def greedy_generate(
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | list[int] | tuple[int, ...] | None = None,
) -> torch.Tensor:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    eos_token_ids = normalize_eos_token_ids(eos_token_id)
    output_ids = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = target(output_ids, use_cache=False).logits[:, -1, :]
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        output_ids = torch.cat([output_ids, next_token], dim=1)
        if eos_token_ids and int(next_token[0, 0]) in eos_token_ids:
            break
    return output_ids


def dflash_generate(
    draft: DFlashLiteDraftModel,
    target,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: int | list[int] | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, GenerationStats]:
    if input_ids.shape[0] != 1:
        raise ValueError("dflash_generate currently supports batch size 1.")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    eos_token_ids = normalize_eos_token_ids(eos_token_id)

    draft.eval()
    target.eval()
    input_embeddings = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The source model must expose output embeddings for draft logits.")

    output_ids = input_ids.clone()
    generated = 0
    accepted_tokens = 0
    target_tokens = 0
    drafted_tokens = 0
    target_steps = 0

    while generated < max_new_tokens:
        with torch.no_grad():
            target_outputs = target(
                output_ids,
                output_hidden_states=True,
                use_cache=False,
            )
            next_token = torch.argmax(target_outputs.logits[:, -1, :], dim=-1)
            context_features = extract_context_feature(
                target_outputs.hidden_states,
                draft.config.selected_layer_ids,
            )
            block_embeddings = draft.build_block_embeddings(input_embeddings, next_token)
            draft_hidden = draft(context_features, block_embeddings)
            draft_logits = lm_head(draft_hidden[:, 1:, :])
            draft_ids = torch.argmax(draft_logits, dim=-1)

            block_ids = torch.cat([next_token.unsqueeze(1), draft_ids], dim=1)
            verify_ids = torch.cat([output_ids, block_ids], dim=1)
            verify_logits = target(verify_ids, use_cache=False).logits
            start = output_ids.shape[1]
            posterior = torch.argmax(
                verify_logits[:, start : start + draft.config.block_size, :],
                dim=-1,
            )

        target_steps += 1
        drafted_tokens += max(0, draft.config.block_size - 1)

        accepted_this_step = [(int(block_ids[0, 0]), "target")]
        mismatch_index = None
        for idx in range(1, draft.config.block_size):
            if int(block_ids[0, idx]) == int(posterior[0, idx - 1]):
                accepted_this_step.append((int(block_ids[0, idx]), "draft"))
                continue
            mismatch_index = idx - 1
            accepted_this_step.append((int(posterior[0, mismatch_index]), "correction"))
            break

        if mismatch_index is None:
            accepted_this_step.append((int(posterior[0, draft.config.block_size - 1]), "bonus"))

        room = max_new_tokens - generated
        accepted_this_step = accepted_this_step[:room]
        token_values = [token for token, _source in accepted_this_step]
        eos_matches = [index for index, token in enumerate(token_values) if token in eos_token_ids]
        if eos_matches:
            eos_offset = eos_matches[0] + 1
            accepted_this_step = accepted_this_step[:eos_offset]
            token_values = token_values[:eos_offset]

        accepted = torch.tensor(
            [token_values], dtype=output_ids.dtype, device=output_ids.device
        )
        output_ids = torch.cat([output_ids, accepted], dim=1)
        accepted_tokens += sum(1 for _token, source in accepted_this_step if source == "draft")
        target_tokens += sum(1 for _token, source in accepted_this_step if source != "draft")
        generated += len(accepted_this_step)

        if any(token in eos_token_ids for token in token_values):
            break

    return output_ids, GenerationStats(
        accepted_tokens=accepted_tokens,
        drafted_tokens=drafted_tokens,
        target_steps=target_steps,
        generated_tokens=generated,
        target_tokens=target_tokens,
    )


def load_runtime(
    source_model: str,
    draft_dir: str | Path,
    device_name: str = "auto",
    torch_dtype: str = "float32",
    trust_remote_code: bool = False,
    strict_config: bool = True,
    prefer_draft_tokenizer: bool = True,
):
    device = resolve_device(device_name)
    dtype = dtype_from_name(torch_dtype)
    draft_path = Path(draft_dir)
    tokenizer_source = source_model
    if prefer_draft_tokenizer:
        config_path = draft_path / "dflasher_config.json"
        tokenizer_path = draft_path / "tokenizer"
        if config_path.exists():
            try:
                config = DFlasherConfig.load(draft_path)
                tokenizer_path = draft_path / config.tokenizer_path
            except Exception:
                tokenizer_path = draft_path / "tokenizer"
        if tokenizer_path.exists():
            tokenizer_source = str(tokenizer_path)
    tokenizer = load_tokenizer(tokenizer_source, trust_remote_code=trust_remote_code)
    target = load_target_model(
        source_model,
        device,
        dtype,
        trust_remote_code=trust_remote_code,
    )
    draft = DFlashLiteDraftModel.from_pretrained(draft_path, map_location=device).to(
        device=device,
        dtype=dtype,
    )
    if strict_config:
        target_shape = get_target_shape(source_model, trust_remote_code=trust_remote_code)
        validate_runtime_compatibility(source_model, draft, target_shape)
    return tokenizer, target, draft, device


def validate_runtime_compatibility(source_model: str, draft: DFlashLiteDraftModel, shape) -> None:
    if draft.config.source_model != source_model:
        raise ValueError(
            "Draft model was trained for a different source model: "
            f"{draft.config.source_model!r} != {source_model!r}"
        )
    if draft.config.target_hidden_size != shape.hidden_size:
        raise ValueError(
            "Draft hidden size does not match target hidden size: "
            f"{draft.config.target_hidden_size} != {shape.hidden_size}"
        )
    if draft.config.vocab_size != shape.vocab_size:
        raise ValueError(
            "Draft vocab size does not match target vocab size: "
            f"{draft.config.vocab_size} != {shape.vocab_size}"
        )
    invalid_layers = [
        layer_id
        for layer_id in draft.config.selected_layer_ids
        if layer_id < 0 or layer_id >= shape.num_hidden_layers
    ]
    if invalid_layers:
        raise ValueError(
            "Draft selected_layer_ids are outside the target layer range: "
            + ", ".join(map(str, invalid_layers))
        )
