from __future__ import annotations

import pytest
import torch

from dflasher.config import DFlasherConfig
from dflasher.generation import dflash_generate, greedy_generate
from dflasher.modeling import DFlashLiteDraftModel


class TinyTarget(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = type("Config", (), {"vocab_size": 8})()
        self.embed = torch.nn.Embedding(8, 6)
        self.head = torch.nn.Linear(6, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head

    def eval(self):
        super().eval()
        return self

    def forward(self, input_ids, output_hidden_states=False, use_cache=False):
        hidden = self.embed(input_ids)
        logits = torch.full((*input_ids.shape, 8), -100.0)
        next_ids = (input_ids + 1) % 8
        logits.scatter_(-1, next_ids.unsqueeze(-1), 100.0)
        if output_hidden_states:
            return type("Output", (), {"logits": logits, "hidden_states": (hidden, hidden)})()
        return type("Output", (), {"logits": logits})()


def test_dflash_generate_matches_target_greedy_with_untrained_draft():
    target = TinyTarget()
    config = DFlasherConfig(
        source_model="tiny",
        target_hidden_size=6,
        vocab_size=8,
        block_size=3,
        draft_hidden_size=12,
        num_draft_layers=1,
        num_attention_heads=2,
        selected_layer_ids=(0,),
    )
    draft = DFlashLiteDraftModel(config)
    input_ids = torch.tensor([[1, 2]])

    baseline = greedy_generate(target, input_ids, max_new_tokens=5)
    speculative, _ = dflash_generate(draft, target, input_ids, max_new_tokens=5)

    assert torch.equal(speculative, baseline)


def test_dflash_generate_rejects_batch_inputs():
    target = TinyTarget()
    draft = DFlashLiteDraftModel(
        DFlasherConfig(
            source_model="tiny",
            target_hidden_size=6,
            vocab_size=8,
            block_size=3,
            draft_hidden_size=12,
            num_draft_layers=1,
            num_attention_heads=2,
            selected_layer_ids=(0,),
        )
    )

    with pytest.raises(ValueError, match="batch size 1"):
        dflash_generate(draft, target, torch.tensor([[1, 2], [3, 4]]), max_new_tokens=2)


def test_generate_supports_eos_token_lists_and_rejects_negative_lengths():
    target = TinyTarget()
    draft = DFlashLiteDraftModel(
        DFlasherConfig(
            source_model="tiny",
            target_hidden_size=6,
            vocab_size=8,
            block_size=3,
            draft_hidden_size=12,
            num_draft_layers=1,
            num_attention_heads=2,
            selected_layer_ids=(0,),
        )
    )
    input_ids = torch.tensor([[1, 2]])

    baseline = greedy_generate(target, input_ids, max_new_tokens=5, eos_token_id=[3])
    speculative, stats = dflash_generate(
        draft,
        target,
        input_ids,
        max_new_tokens=5,
        eos_token_id=[3],
    )

    assert baseline.tolist() == [[1, 2, 3]]
    assert torch.equal(speculative, baseline)
    assert stats.generated_tokens == 1
    with pytest.raises(ValueError, match="max_new_tokens"):
        greedy_generate(target, input_ids, max_new_tokens=-1)
    with pytest.raises(ValueError, match="max_new_tokens"):
        dflash_generate(draft, target, input_ids, max_new_tokens=-1)
