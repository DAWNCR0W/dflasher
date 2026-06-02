from __future__ import annotations

import pytest
import torch

from dflasher.config import DFlasherConfig
from dflasher.data import load_texts, sample_batch
from dflasher.training import TrainOptions


def test_sample_batch_left_pads_prefix_before_appending_anchor(monkeypatch):
    sequences = [torch.tensor([10, 11, 12, 13, 14]), torch.tensor([20, 21, 22, 23, 24])]
    choices = iter(sequences)
    anchors = iter([1, 3])
    monkeypatch.setattr("random.choice", lambda _items: next(choices))
    monkeypatch.setattr("random.randint", lambda _start, _end: next(anchors))

    input_ids, attention_mask, anchor_ids, labels = sample_batch(
        sequences=sequences,
        batch_size=2,
        block_size=3,
        pad_token_id=0,
        device=torch.device("cpu"),
    )

    assert input_ids.tolist() == [[0, 0, 10], [20, 21, 22]]
    assert attention_mask.tolist() == [[0, 0, 1], [1, 1, 1]]
    assert anchor_ids.tolist() == [11, 23]
    assert labels.tolist() == [[12, 13], [24, -100]]


def test_load_texts_decodes_line_encoded_newlines(tmp_path):
    texts_file = tmp_path / "texts.txt"
    texts_file.write_text("SYSTEM:\\nhello\\n\\tUSER: hi\n")

    texts = load_texts(texts_file=str(texts_file))

    assert texts == ["SYSTEM:\nhello\n\tUSER: hi"]


def test_invalid_block_size_is_rejected_before_training():
    with pytest.raises(ValueError, match="block_size"):
        TrainOptions(source_model="tiny", output_dir="out", block_size=1)


def test_invalid_head_geometry_is_rejected_before_model_construction():
    with pytest.raises(ValueError, match="divisible"):
        DFlasherConfig(
            source_model="tiny",
            target_hidden_size=8,
            vocab_size=16,
            draft_hidden_size=10,
            num_attention_heads=3,
        )
