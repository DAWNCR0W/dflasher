from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from dflasher.config import DFlasherConfig

WEIGHTS_NAME = "model.safetensors"


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int) -> tuple[int, ...]:
    if num_target_layers <= 0:
        return ()
    if num_draft_layers <= 1:
        return (num_target_layers // 2,)

    start = 1 if num_target_layers > 3 else 0
    end = max(start, num_target_layers - 3)
    span = end - start
    return tuple(
        int(round(start + (layer_idx * span) / (num_draft_layers - 1)))
        for layer_idx in range(num_draft_layers)
    )


def extract_context_feature(
    hidden_states: tuple[torch.Tensor, ...],
    selected_layer_ids: tuple[int, ...],
) -> torch.Tensor:
    offset = 1
    selected = [hidden_states[layer_id + offset] for layer_id in selected_layer_ids]
    return torch.cat(selected, dim=-1)


class DFlashLiteDraftModel(nn.Module):
    """Small block drafter conditioned on frozen target hidden states."""

    def __init__(self, config: DFlasherConfig) -> None:
        super().__init__()
        self.config = config
        context_width = max(1, len(config.selected_layer_ids)) * config.target_hidden_size

        self.context_projection = nn.Linear(context_width, config.draft_hidden_size)
        self.input_projection = nn.Linear(config.target_hidden_size, config.draft_hidden_size)
        self.output_projection = nn.Linear(config.draft_hidden_size, config.target_hidden_size)
        self.mask_embedding = nn.Parameter(torch.empty(config.target_hidden_size))
        self.position_embedding = nn.Embedding(config.block_size, config.draft_hidden_size)

        layer = nn.TransformerDecoderLayer(
            d_model=config.draft_hidden_size,
            nhead=config.num_attention_heads,
            dim_feedforward=config.draft_hidden_size * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.TransformerDecoder(layer, num_layers=config.num_draft_layers)
        self.final_norm = nn.LayerNorm(config.draft_hidden_size)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.mask_embedding, mean=0.0, std=0.02)

    def forward(
        self,
        context_features: torch.Tensor,
        block_embeddings: torch.Tensor,
        context_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, block_size, _ = block_embeddings.shape
        memory = self.context_projection(context_features)
        hidden_states = self.input_projection(block_embeddings)

        positions = torch.arange(block_size, device=block_embeddings.device)
        hidden_states = hidden_states + self.position_embedding(positions).unsqueeze(0)

        memory_key_padding_mask = None
        if context_attention_mask is not None:
            memory_key_padding_mask = context_attention_mask == 0

        hidden_states = self.layers(
            hidden_states,
            memory,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return self.output_projection(self.final_norm(hidden_states))

    def build_block_embeddings(
        self,
        input_embeddings: nn.Module,
        anchor_ids: torch.Tensor,
    ) -> torch.Tensor:
        anchor_embeddings = input_embeddings(anchor_ids).unsqueeze(1)
        mask_embeddings = self.mask_embedding.to(anchor_embeddings.dtype).view(1, 1, -1)
        mask_embeddings = mask_embeddings.expand(
            anchor_ids.shape[0], self.config.block_size - 1, -1
        )
        return torch.cat([anchor_embeddings, mask_embeddings], dim=1)

    def save_pretrained(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self.config.save(output_path)
        save_file(self.state_dict(), str(output_path / WEIGHTS_NAME))

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> DFlashLiteDraftModel:
        config = DFlasherConfig.load(model_dir)
        model = cls(config)
        state_dict = load_file(str(Path(model_dir) / WEIGHTS_NAME), device=str(map_location))
        model.load_state_dict(state_dict)
        return model
