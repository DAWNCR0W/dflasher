from __future__ import annotations

import importlib.util
import json
import platform
import shlex
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from dflasher.model_profile import resolve_model_profile
from dflasher.training import TrainOptions

MacStage = Literal["train", "eval"]


@dataclass(frozen=True)
class MacPreflightItem:
    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class MacPipelineConfig:
    source_model: str = "Qwen/Qwen3-0.6B"
    workspace: Path = Path("runs/mac-dflash-lite")
    texts_file: str | None = None
    dataset: str | None = None
    dataset_split: str = "train"
    text_column: str = "text"
    data_limit: int | None = None
    max_length: int = 256
    block_size: int = 4
    draft_hidden_size: int = 128
    draft_layers: int = 2
    heads: int = 4
    batch_size: int = 2
    max_steps: int = 100
    learning_rate: float = 5e-4
    loss_fn: str = "kl_div"
    device: str = "mps"
    torch_dtype: str = "float32"
    trust_remote_code: bool = False
    seed: int = 13
    eval_max_new_tokens: int = 12
    allow_builtin_data: bool = False
    python_bin: str = "python"

    def __post_init__(self) -> None:
        if self.eval_max_new_tokens < 1:
            raise ValueError("eval_max_new_tokens must be at least 1.")
        if not self.python_bin:
            raise ValueError("python_bin must not be empty.")
        self.train_options()

    @property
    def draft_dir(self) -> Path:
        return self.workspace / "draft"

    @property
    def script_path(self) -> Path:
        return self.workspace / "run_mac_dflash_lite.sh"

    @property
    def manifest_path(self) -> Path:
        return self.workspace / "dflasher_mac_manifest.json"

    def train_options(self) -> TrainOptions:
        return TrainOptions(
            source_model=self.source_model,
            output_dir=self.draft_dir,
            texts_file=self.texts_file,
            dataset_name=self.dataset,
            dataset_split=self.dataset_split,
            text_column=self.text_column,
            data_limit=self.data_limit,
            max_length=self.max_length,
            block_size=self.block_size,
            draft_hidden_size=self.draft_hidden_size,
            num_draft_layers=self.draft_layers,
            num_attention_heads=self.heads,
            batch_size=self.batch_size,
            max_steps=self.max_steps,
            learning_rate=self.learning_rate,
            loss_fn=self.loss_fn,
            device=self.device,
            torch_dtype=self.torch_dtype,
            trust_remote_code=self.trust_remote_code,
            seed=self.seed,
            allow_builtin_data=self.allow_builtin_data,
        )


def _quote_args(args: list[str]) -> str:
    return shlex.join(args)


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        return False


def run_mac_preflight(
    source_model: str,
    *,
    device: str = "mps",
    trust_remote_code: bool = False,
) -> list[MacPreflightItem]:
    items: list[MacPreflightItem] = []
    system = platform.system()
    machine = platform.machine()
    needs_apple_gpu = device in {"auto", "mps"}
    items.append(
        MacPreflightItem(
            "macOS",
            system == "Darwin",
            f"platform={system}; required for MPS/MLX acceleration",
            required=needs_apple_gpu,
        )
    )
    items.append(
        MacPreflightItem(
            "Apple Silicon",
            machine in {"arm64", "aarch64"},
            f"machine={machine}; required for practical MLX/MPS runs",
            required=needs_apple_gpu,
        )
    )
    items.append(
        MacPreflightItem(
            "torch import",
            _module_available("torch"),
            "required for dflasher train/eval on MPS",
        )
    )
    try:
        import torch

        mps_available = bool(
            getattr(torch.backends, "mps", None)
            and torch.backends.mps.is_available()
            and torch.backends.mps.is_built()
        )
        items.append(
            MacPreflightItem(
                "torch MPS",
                mps_available,
                "required when --device mps is used",
                required=needs_apple_gpu,
            )
        )
    except Exception as exc:
        items.append(MacPreflightItem("torch MPS", False, str(exc), required=needs_apple_gpu))

    items.append(
        MacPreflightItem(
            "mlx import",
            _module_available("mlx"),
            "optional: required for z-lab MLX runtime",
            required=False,
        )
    )
    items.append(
        MacPreflightItem(
            "mlx_lm import",
            _module_available("mlx_lm"),
            "optional: required for z-lab MLX runtime",
            required=False,
        )
    )
    items.append(
        MacPreflightItem(
            "z-lab dflash MLX import",
            _module_available("dflash.model_mlx"),
            "optional: install z-lab/dflash[mlx] to run generated MLX scripts",
            required=False,
        )
    )

    try:
        profile = resolve_model_profile(
            source_model,
            trust_remote_code=trust_remote_code,
            allow_name_fallback=True,
        )
        mlx_support = profile.backend_support["zlab_mlx"]
        items.append(
            MacPreflightItem(
                "model family",
                True,
                (
                    f"family={profile.family}; zlab_mlx={mlx_support.status}; "
                    f"{mlx_support.detail}"
                ),
                required=False,
            )
        )
        items.append(
            MacPreflightItem(
                "known z-lab draft",
                profile.known_zlab_draft_model is not None,
                profile.known_zlab_draft_model or "pass --draft-model for MLX scripts",
                required=False,
            )
        )
    except Exception as exc:
        items.append(MacPreflightItem("model family", False, str(exc), required=False))
    return items


def mac_required_preflight_ok(items: list[MacPreflightItem]) -> bool:
    return all(item.ok for item in items if item.required)


def mac_command_plan(config: MacPipelineConfig) -> dict[MacStage, list[str]]:
    base_command = [config.python_bin, "-m", "dflasher.cli"]
    train_cmd = [
        *base_command,
        "train",
        config.source_model,
        "--out",
        str(config.draft_dir),
        "--max-length",
        str(config.max_length),
        "--block-size",
        str(config.block_size),
        "--draft-hidden-size",
        str(config.draft_hidden_size),
        "--draft-layers",
        str(config.draft_layers),
        "--heads",
        str(config.heads),
        "--batch-size",
        str(config.batch_size),
        "--max-steps",
        str(config.max_steps),
        "--learning-rate",
        str(config.learning_rate),
        "--loss-fn",
        config.loss_fn,
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
        "--seed",
        str(config.seed),
    ]
    if config.texts_file:
        train_cmd.extend(["--texts-file", config.texts_file])
    if config.dataset:
        train_cmd.extend(
            [
                "--dataset",
                config.dataset,
                "--dataset-split",
                config.dataset_split,
                "--text-column",
                config.text_column,
            ]
        )
    if config.data_limit is not None:
        train_cmd.extend(["--data-limit", str(config.data_limit)])
    if config.trust_remote_code:
        train_cmd.append("--trust-remote-code")
    if config.allow_builtin_data:
        train_cmd.append("--allow-builtin-data")

    eval_cmd = [
        *base_command,
        "eval",
        config.source_model,
        str(config.draft_dir),
        "--max-new-tokens",
        str(config.eval_max_new_tokens),
        "--device",
        config.device,
        "--torch-dtype",
        config.torch_dtype,
    ]
    if config.trust_remote_code:
        eval_cmd.append("--trust-remote-code")

    return {"train": train_cmd, "eval": eval_cmd}


def write_mac_manifest(config: MacPipelineConfig) -> Path:
    config.workspace.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["workspace"] = str(config.workspace)
    payload["commands"] = {
        name: _quote_args(args) for name, args in mac_command_plan(config).items()
    }
    config.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return config.manifest_path


def render_mac_pipeline_script(config: MacPipelineConfig) -> str:
    commands = mac_command_plan(config)
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            "",
            "# Generated by dflasher mac pipeline.",
            f"# source model: {config.source_model}",
            f"# device: {config.device}",
            f"# draft dir: {config.draft_dir}",
            "",
            "echo '==> Training local DFlash-lite draft on Mac'",
            _quote_args(commands["train"]),
            "",
            "echo '==> Verifying exact greedy equivalence'",
            _quote_args(commands["eval"]),
            "",
            "echo '==> Done'",
            f"echo 'Draft model: {config.draft_dir}'",
            "",
        ]
    )


def write_mac_pipeline_script(config: MacPipelineConfig) -> Path:
    config.workspace.mkdir(parents=True, exist_ok=True)
    script = render_mac_pipeline_script(config)
    config.script_path.write_text(script)
    config.script_path.chmod(0o755)
    write_mac_manifest(config)
    return config.script_path


def load_mac_manifest(manifest_path: Path) -> MacPipelineConfig:
    payload = json.loads(manifest_path.read_text())
    payload.pop("commands", None)
    payload["workspace"] = Path(payload["workspace"])
    return MacPipelineConfig(**payload)


def run_mac_stage(config: MacPipelineConfig, stage: MacStage) -> None:
    commands = mac_command_plan(config)
    if stage not in commands:
        raise ValueError(f"Unknown Mac stage: {stage}")
    process = subprocess.run(commands[stage], check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: "
            f"{_quote_args(commands[stage])}"
        )
