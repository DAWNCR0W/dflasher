from __future__ import annotations

import random
from pathlib import Path

import torch
from datasets import load_dataset

BUILTIN_TEXTS = [
    "DFlash draft models propose future tokens, and a target model verifies them.",
    "Speculative decoding is correct when every accepted draft token matches the target model.",
    "Small language models are useful for smoke tests because they download quickly.",
    "A command line tool should save its configuration and weights in a reproducible format.",
    "The draft model predicts a block of future tokens from one clean anchor token.",
    "Careful tests compare speculative decoding against the target model greedy baseline.",
    "Research prototypes should clearly state exact parts and approximations.",
    "Hidden states from the frozen target model provide context for the lightweight drafter.",
]


def load_texts(
    texts_file: str | None = None,
    dataset_name: str | None = None,
    dataset_split: str = "train",
    text_column: str = "text",
    limit: int | None = None,
    allow_builtin_data: bool = False,
) -> list[str]:
    texts: list[str]
    if texts_file:
        texts = [line.strip() for line in Path(texts_file).read_text().splitlines() if line.strip()]
    elif dataset_name:
        dataset = load_dataset(dataset_name, split=dataset_split)
        if limit is not None:
            dataset = dataset.select(range(min(limit, len(dataset))))
        texts = [
            str(item[text_column]).strip() for item in dataset if str(item[text_column]).strip()
        ]
    elif allow_builtin_data:
        texts = BUILTIN_TEXTS
    else:
        raise ValueError(
            "No training data was provided. Pass --texts-file or --dataset, "
            "or use --allow-builtin-data for smoke tests only."
        )

    if limit is not None:
        texts = texts[:limit]
    if not texts:
        raise ValueError("No training texts were loaded.")
    return texts


def tokenize_texts(tokenizer, texts: list[str], max_length: int) -> list[torch.Tensor]:
    sequences = []
    for text in texts:
        encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        input_ids = encoded["input_ids"][0]
        if input_ids.numel() >= 4:
            sequences.append(input_ids)
    if not sequences:
        raise ValueError("No tokenized sequence is long enough for training.")
    return sequences


def sample_batch(
    sequences: list[torch.Tensor],
    batch_size: int,
    block_size: int,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_size < 2:
        raise ValueError("block_size must be at least 2.")

    prefix_items: list[torch.Tensor] = []
    anchor_ids: list[int] = []
    label_items: list[torch.Tensor] = []

    for _ in range(batch_size):
        sequence = random.choice(sequences)
        max_anchor = max(1, sequence.numel() - 2)
        anchor_index = random.randint(1, max_anchor)
        prefix_items.append(sequence[:anchor_index])
        anchor_ids.append(int(sequence[anchor_index]))

        label = torch.full((block_size - 1,), -100, dtype=torch.long)
        available = sequence[anchor_index + 1 : anchor_index + block_size]
        label[: available.numel()] = available
        label_items.append(label)

    max_prefix_len = max(item.numel() for item in prefix_items)
    input_ids = torch.full((batch_size, max_prefix_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_prefix_len), dtype=torch.long)
    for row_idx, item in enumerate(prefix_items):
        start = max_prefix_len - item.numel()
        input_ids[row_idx, start:] = item
        attention_mask[row_idx, start:] = 1

    return (
        input_ids.to(device),
        attention_mask.to(device),
        torch.tensor(anchor_ids, dtype=torch.long, device=device),
        torch.stack(label_items).to(device),
    )
