from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from rich.console import Console
from tqdm.auto import trange

from dflasher.config import DFlasherConfig
from dflasher.data import load_texts, sample_batch, tokenize_texts
from dflasher.hf import (
    dtype_from_name,
    get_target_shape,
    load_target_model,
    load_tokenizer,
    resolve_device,
)
from dflasher.modeling import DFlashLiteDraftModel, build_target_layer_ids, extract_context_feature

console = Console()


@dataclass(frozen=True)
class TrainOptions:
    source_model: str
    output_dir: Path
    texts_file: str | None = None
    dataset_name: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    data_limit: int | None = None
    max_length: int = 256
    block_size: int = 4
    draft_hidden_size: int = 128
    num_draft_layers: int = 2
    num_attention_heads: int = 4
    batch_size: int = 4
    max_steps: int = 100
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    loss_decay: float = 4.0
    loss_fn: str = "kl_div"
    device: str = "auto"
    torch_dtype: str = "float32"
    trust_remote_code: bool = False
    seed: int = 13
    allow_builtin_data: bool = False

    def __post_init__(self) -> None:
        if self.max_length < 4:
            raise ValueError("max_length must be at least 4.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.draft_hidden_size <= 0:
            raise ValueError("draft_hidden_size must be positive.")
        if self.num_draft_layers < 1:
            raise ValueError("num_draft_layers must be at least 1.")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1.")
        if self.draft_hidden_size % self.num_attention_heads != 0:
            raise ValueError("draft_hidden_size must be divisible by num_attention_heads.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.loss_decay <= 0:
            raise ValueError("loss_decay must be positive.")
        if self.loss_fn not in {"kl_div", "ce"}:
            raise ValueError("loss_fn must be either 'kl_div' or 'ce'.")
        if not self.texts_file and not self.dataset_name and not self.allow_builtin_data:
            raise ValueError(
                "training data is required; pass texts_file, dataset_name, "
                "or allow_builtin_data for smoke tests."
            )


def train(options: TrainOptions) -> Path:
    torch.manual_seed(options.seed)
    random.seed(options.seed)
    device = resolve_device(options.device)
    dtype = dtype_from_name(options.torch_dtype)

    console.print(f"[bold]Loading source model[/bold] {options.source_model} on {device}")
    tokenizer = load_tokenizer(options.source_model, trust_remote_code=options.trust_remote_code)
    target = load_target_model(
        options.source_model,
        device,
        dtype,
        trust_remote_code=options.trust_remote_code,
    )
    shape = get_target_shape(options.source_model, trust_remote_code=options.trust_remote_code)
    selected_layer_ids = build_target_layer_ids(shape.num_hidden_layers, options.num_draft_layers)

    config = DFlasherConfig(
        source_model=options.source_model,
        target_hidden_size=shape.hidden_size,
        vocab_size=shape.vocab_size,
        block_size=options.block_size,
        draft_hidden_size=options.draft_hidden_size,
        num_draft_layers=options.num_draft_layers,
        num_attention_heads=options.num_attention_heads,
        selected_layer_ids=selected_layer_ids,
        max_position_embeddings=shape.max_position_embeddings,
        torch_dtype=str(dtype).replace("torch.", ""),
        training_data_source=describe_training_data_source(options),
    )
    draft = DFlashLiteDraftModel(config).to(device=device, dtype=dtype)
    input_embeddings = target.get_input_embeddings()
    lm_head = target.get_output_embeddings()
    if lm_head is None:
        raise ValueError("The source model must expose output embeddings for draft logits.")

    texts = load_texts(
        texts_file=options.texts_file,
        dataset_name=options.dataset_name,
        dataset_split=options.dataset_split,
        text_column=options.text_column,
        limit=options.data_limit,
        allow_builtin_data=options.allow_builtin_data,
    )
    sequences = tokenize_texts(tokenizer, texts, options.max_length)

    optimizer = torch.optim.AdamW(
        draft.parameters(),
        lr=options.learning_rate,
        weight_decay=options.weight_decay,
    )
    weights = torch.exp(
        -torch.arange(options.block_size - 1, device=device, dtype=torch.float32)
        / options.loss_decay
    )
    weights = weights / weights.mean()

    draft.train()
    progress = trange(options.max_steps, desc="training", leave=True)
    for _ in progress:
        prefix_ids, attention_mask, anchor_ids, labels = sample_batch(
            sequences=sequences,
            batch_size=options.batch_size,
            block_size=options.block_size,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
        )

        with torch.no_grad():
            target_outputs = target(
                input_ids=prefix_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
            context_features = extract_context_feature(
                target_outputs.hidden_states,
                selected_layer_ids,
            )
            block_embeddings = draft.build_block_embeddings(input_embeddings, anchor_ids)
            clean_future = labels.masked_fill(labels.eq(-100), tokenizer.pad_token_id)
            verify_ids = torch.cat([prefix_ids, anchor_ids.unsqueeze(1), clean_future], dim=1)
            verify_attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=device
                    ),
                    labels.ne(-100).to(attention_mask.dtype),
                ],
                dim=1,
            )
            verify_outputs = target(
                input_ids=verify_ids,
                attention_mask=verify_attention_mask,
                use_cache=False,
            )

        hidden_states = draft(
            context_features=context_features,
            block_embeddings=block_embeddings,
            context_attention_mask=attention_mask,
        )
        logits = lm_head(hidden_states[:, 1:, :])
        prefix_width = prefix_ids.shape[1]
        target_logits = verify_outputs.logits[
            :, prefix_width : prefix_width + options.block_size - 1, :
        ]
        if options.loss_fn == "ce":
            target_ids = torch.argmax(target_logits, dim=-1).masked_fill(labels.eq(-100), -100)
            raw_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                target_ids.reshape(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
        elif options.loss_fn == "kl_div":
            raw_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(target_logits, dim=-1),
                reduction="none",
                log_target=False,
            ).sum(dim=-1)
        else:
            raise ValueError("loss_fn must be either 'kl_div' or 'ce'.")
        valid = labels.ne(-100)
        weighted_loss = raw_loss * weights.unsqueeze(0)
        loss = weighted_loss[valid].mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(draft.parameters(), 1.0)
        optimizer.step()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    options.output_dir.mkdir(parents=True, exist_ok=True)
    draft.save_pretrained(options.output_dir)
    tokenizer.save_pretrained(options.output_dir / "tokenizer")
    console.print(f"[green]Saved draft model to[/green] {options.output_dir}")
    return options.output_dir


def describe_training_data_source(options: TrainOptions) -> str:
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
