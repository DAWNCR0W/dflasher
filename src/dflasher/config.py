from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

CONFIG_NAME = "dflasher_config.json"


@dataclass(frozen=True)
class DFlasherConfig:
    source_model: str
    target_hidden_size: int
    vocab_size: int
    format_version: int = 1
    draft_format: str = "dflasher.dflash-lite"
    block_size: int = 4
    draft_hidden_size: int = 128
    num_draft_layers: int = 2
    num_attention_heads: int = 4
    dropout: float = 0.0
    selected_layer_ids: tuple[int, ...] = ()
    max_position_embeddings: int = 2048
    torch_dtype: str = "float32"
    architecture: str = "dflash-lite-generic"
    tokenizer_path: str = "tokenizer"
    training_data_source: str = "unknown"
    source_revision: str | None = None
    compatible_runtimes: tuple[str, ...] = ("dflasher.generate", "dflasher.eval")

    def __post_init__(self) -> None:
        if self.target_hidden_size <= 0:
            raise ValueError("target_hidden_size must be positive.")
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
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
        if any(layer_id < 0 for layer_id in self.selected_layer_ids):
            raise ValueError("selected_layer_ids must be non-negative.")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in the range [0, 1).")

    def save(self, output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["selected_layer_ids"] = list(self.selected_layer_ids)
        payload["compatible_runtimes"] = list(self.compatible_runtimes)
        (output_path / CONFIG_NAME).write_text(json.dumps(payload, indent=2) + "\n")

    @classmethod
    def load(cls, model_dir: str | Path) -> DFlasherConfig:
        payload = json.loads((Path(model_dir) / CONFIG_NAME).read_text())
        payload["selected_layer_ids"] = tuple(payload.get("selected_layer_ids", ()))
        payload["compatible_runtimes"] = tuple(payload.get("compatible_runtimes", ()))
        return cls(**payload)
