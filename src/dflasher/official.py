from __future__ import annotations

import asyncio
import importlib.metadata
import importlib.util
import json
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import httpx
from safetensors import safe_open

from dflasher.model_profile import (
    LayerPolicy,
    resolve_model_profile,
    target_layer_ids_for_policy,
)

PipelineMode = Literal["online-delete", "online-cache", "offline-cache"]
Stage = Literal["prepare", "hidden-server", "cache", "train", "serve", "benchmark"]
PIPELINE_MODES: tuple[str, ...] = ("online-delete", "online-cache", "offline-cache")
LAYER_POLICIES: tuple[str, ...] = (
    "auto",
    "speculators",
    "dflash5",
    "zlab5",
    "zlab6",
    "zlab-linspace5",
    "zlab-linspace6",
)
REQUIRED_SPECULATORS_SCRIPTS: tuple[str, ...] = (
    "prepare_data.py",
    "data_generation_offline.py",
    "train.py",
)


@dataclass(frozen=True)
class OfficialPipelineConfig:
    source_model: str = "Qwen/Qwen3-0.6B"
    workspace: Path = Path("runs/qwen3-0.6b-official")
    speculators_repo: Path = Path("/tmp/vllm-speculators-reference")
    data: str = "sharegpt"
    max_samples: int = 5000
    seq_length: int = 8192
    epochs: int = 5
    learning_rate: float = 3e-4
    block_size: int = 8
    max_anchors: int = 3072
    draft_layers: int = 5
    draft_arch: str | None = None
    draft_vocab_size: int = 8192
    target_layer_ids: tuple[int, ...] | None = None
    layer_policy: LayerPolicy = "auto"
    trust_remote_code: bool = False
    mode: PipelineMode = "offline-cache"
    vllm_port: int = 8000
    vllm_gpus: str = "0"
    train_gpus: str = "0"
    train_processes: int = 1
    benchmark_prompts: Path | None = None
    benchmark_max_tokens: int = 256
    benchmark_concurrency: int = 1
    validate_hidden_states: bool = True
    python_bin: str = "python"
    server_start_timeout: int = 900
    allow_experimental: bool = False
    prepare_trust_remote_code: bool = True
    vllm_args: tuple[str, ...] = ()
    serve_args: tuple[str, ...] = ()
    train_args: tuple[str, ...] = ()
    prepare_args: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.mode not in PIPELINE_MODES:
            raise ValueError(
                "mode must be one of: " + ", ".join(PIPELINE_MODES)
            )
        if self.layer_policy not in LAYER_POLICIES:
            raise ValueError(
                "layer_policy must be one of: " + ", ".join(LAYER_POLICIES)
            )
        if self.max_samples < 1:
            raise ValueError("max_samples must be at least 1.")
        if self.seq_length < 2:
            raise ValueError("seq_length must be at least 2.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.block_size < 2:
            raise ValueError("block_size must be at least 2.")
        if self.max_anchors < 1:
            raise ValueError("max_anchors must be at least 1.")
        if self.draft_layers < 1:
            raise ValueError("draft_layers must be at least 1.")
        if self.draft_vocab_size < 1:
            raise ValueError("draft_vocab_size must be at least 1.")
        if self.train_processes < 1:
            raise ValueError("train_processes must be at least 1.")
        if self.benchmark_max_tokens < 1:
            raise ValueError("benchmark_max_tokens must be at least 1.")
        if self.benchmark_concurrency < 1:
            raise ValueError("benchmark_concurrency must be at least 1.")
        if self.server_start_timeout < 1:
            raise ValueError("server_start_timeout must be at least 1.")
        if not self.python_bin:
            raise ValueError("python_bin must not be empty.")
        if self.target_layer_ids and any(layer_id < 0 for layer_id in self.target_layer_ids):
            raise ValueError("target_layer_ids must be non-negative.")
        for name, values in (
            ("vllm_args", self.vllm_args),
            ("serve_args", self.serve_args),
            ("train_args", self.train_args),
            ("prepare_args", self.prepare_args),
        ):
            if any(not value for value in values):
                raise ValueError(f"{name} must not contain empty values.")
            if isinstance(values, list):
                object.__setattr__(self, name, tuple(values))

    @property
    def data_dir(self) -> Path:
        return self.workspace / "data"

    @property
    def hidden_states_dir(self) -> Path:
        return self.workspace / "hidden_states"

    @property
    def checkpoints_dir(self) -> Path:
        return self.workspace / "checkpoints"

    @property
    def best_checkpoint(self) -> Path:
        return self.checkpoints_dir / "checkpoint_best"

    @property
    def logs_dir(self) -> Path:
        return self.workspace / "logs"

    @property
    def script_path(self) -> Path:
        return self.workspace / "run_official_dflash.sh"

    @property
    def manifest_path(self) -> Path:
        return self.workspace / "dflasher_official_manifest.json"

    def resolved_target_layer_ids(self) -> tuple[int, ...]:
        if self.target_layer_ids:
            return self.target_layer_ids
        profile = resolve_model_profile(self.source_model, self.trust_remote_code)
        return target_layer_ids_for_policy(
            profile.num_hidden_layers or 0,
            profile.family,
            self.layer_policy,
        )

    def resolved_draft_arch(self) -> str:
        if self.draft_arch:
            return self.draft_arch
        return "llama"


@dataclass(frozen=True)
class PreflightItem:
    name: str
    ok: bool
    detail: str


def _quote_args(args: list[str]) -> str:
    import shlex

    return " ".join(shlex.quote(arg) for arg in args)


def _parse_version_parts(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple(parts)


def _version_at_least(version: str, minimum: str) -> bool:
    left = _parse_version_parts(version)
    right = _parse_version_parts(minimum)
    width = max(len(left), len(right))
    return left + (0,) * (width - len(left)) >= right + (0,) * (width - len(right))


def _package_version_item(package: str, minimum: str) -> PreflightItem:
    try:
        version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return PreflightItem(f"{package} version", False, "not installed")
    return PreflightItem(
        f"{package} version",
        _version_at_least(version, minimum),
        f"{version}; required >= {minimum}",
    )


def script_contains_flag(repo: Path, script_name: str, flag: str) -> bool:
    script = repo / "scripts" / script_name
    if not script.exists():
        return False
    try:
        return flag in script.read_text()
    except UnicodeDecodeError:
        return False


def _script_path(repo: Path, name: str) -> Path:
    return repo / "scripts" / name


def require_speculators_repo(repo: Path) -> None:
    required = [_script_path(repo, name) for name in REQUIRED_SPECULATORS_SCRIPTS]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Speculators repository is missing required scripts: " + ", ".join(missing)
        )


def install_speculators_reference(repo: Path, update: bool = False) -> Path:
    if repo.exists():
        if update:
            run_command(["git", "fetch", "origin", "--prune"], cwd=repo)
            run_command(["git", "pull", "--ff-only"], cwd=repo)
        require_speculators_repo(repo)
        return repo

    repo.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "https://github.com/vllm-project/speculators.git",
            str(repo),
        ]
    )
    require_speculators_repo(repo)
    return repo


def check_model_compatibility(
    source_model: str,
    trust_remote_code: bool,
    *,
    allow_experimental: bool = False,
) -> list[PreflightItem]:
    try:
        profile = resolve_model_profile(source_model, trust_remote_code)
        support = profile.backend_support["speculators"]
        support_ok = support.status == "supported" or (
            allow_experimental and support.status in {"experimental", "preview"}
        )
        return [
            PreflightItem(
                "model profile",
                True,
                (
                    f"family={profile.family}, model_type={profile.model_type}, "
                    f"hidden={profile.hidden_size}, layers={profile.num_hidden_layers}, "
                    f"vocab={profile.vocab_size}"
                ),
            ),
            PreflightItem(
                "speculators family support",
                support_ok,
                f"{support.status}: {support.detail}"
                + ("" if support.status == "supported" else "; pass --allow-experimental"),
            ),
            PreflightItem(
                "speculators config fields",
                not profile.missing_speculators_fields,
                (
                    "all required decoder fields present"
                    if not profile.missing_speculators_fields
                    else "missing: " + ", ".join(profile.missing_speculators_fields)
                ),
            ),
            PreflightItem(
                "draft architecture",
                True,
                profile.recommended_draft_arch,
            ),
        ]
    except Exception as exc:
        return [PreflightItem("model profile", False, str(exc))]


def run_preflight(
    config: OfficialPipelineConfig,
    *,
    include_environment: bool = True,
) -> list[PreflightItem]:
    items: list[PreflightItem] = []
    items.extend(
        check_model_compatibility(
            config.source_model,
            config.trust_remote_code,
            allow_experimental=config.allow_experimental,
        )
    )
    items.append(
        PreflightItem(
            "speculators repo",
            all(
                _script_path(config.speculators_repo, name).exists()
                for name in REQUIRED_SPECULATORS_SCRIPTS
            ),
            str(config.speculators_repo),
        )
    )
    items.append(
        PreflightItem(
            "prepare_data trust-remote-code",
            (
                not config.trust_remote_code
                or not config.prepare_trust_remote_code
                or script_contains_flag(
                    config.speculators_repo, "prepare_data.py", "--trust-remote-code"
                )
            ),
            (
                "not requested"
                if not config.trust_remote_code
                else "prepare_data.py supports --trust-remote-code"
                if script_contains_flag(
                    config.speculators_repo, "prepare_data.py", "--trust-remote-code"
                )
                else "prepare_data.py does not expose --trust-remote-code"
            ),
        )
    )
    items.append(
        PreflightItem(
            "train dflash options",
            script_contains_flag(config.speculators_repo, "train.py", "--speculator-type")
            and script_contains_flag(config.speculators_repo, "train.py", "dflash")
            and script_contains_flag(config.speculators_repo, "train.py", "--draft-arch"),
            "train.py must support --speculator-type dflash and --draft-arch",
        )
    )
    draft_arch = config.resolved_draft_arch()
    items.append(
        PreflightItem(
            "draft arch vLLM serving",
            draft_arch == "llama",
            (
                f"{draft_arch}; current Speculators train.py docs warn that only "
                "llama draft architecture is supported in vLLM inference"
            ),
        )
    )
    if not include_environment:
        return items
    items.append(
        PreflightItem(
            "torch installed",
            importlib.util.find_spec("torch") is not None,
            "import torch",
        )
    )
    try:
        import torch

        items.append(
            PreflightItem(
                "cuda available",
                torch.cuda.is_available(),
                "required for vLLM Speculators training/serving",
            )
        )
    except Exception as exc:
        items.append(PreflightItem("cuda available", False, str(exc)))

    items.append(
        _package_version_item("vllm", "0.20.1")
    )
    items.append(
        _package_version_item("speculators", "0.5.0")
    )
    items.append(
        PreflightItem(
            "vllm import",
            importlib.util.find_spec("vllm") is not None,
            "required for hidden-state server and serving",
        )
    )
    items.append(
        PreflightItem(
            "speculators import",
            importlib.util.find_spec("speculators") is not None,
            "required for official DFlash training",
        )
    )
    items.append(
        PreflightItem(
            "openai import",
            importlib.util.find_spec("openai") is not None,
            "required by Speculators data generation clients",
        )
    )
    items.append(
        PreflightItem(
            "pydantic import",
            importlib.util.find_spec("pydantic") is not None,
            "required by Speculators configs",
        )
    )
    items.append(
        PreflightItem(
            "curl command",
            shutil.which("curl") is not None,
            shutil.which("curl") or "not found",
        )
    )
    return items


def command_plan(config: OfficialPipelineConfig) -> dict[str, list[str]]:
    target_layers = [str(layer_id) for layer_id in config.resolved_target_layer_ids()]
    endpoint = f"http://localhost:{config.vllm_port}/v1"
    draft_arch = config.resolved_draft_arch()
    prepare_cmd = [
        config.python_bin,
        str(_script_path(config.speculators_repo, "prepare_data.py")),
        "--model",
        config.source_model,
        "--data",
        config.data,
        "--output",
        str(config.data_dir),
        "--max-samples",
        str(config.max_samples),
        "--seq-length",
        str(config.seq_length),
    ]
    if config.trust_remote_code and config.prepare_trust_remote_code:
        prepare_cmd.append("--trust-remote-code")
    prepare_cmd.extend(config.prepare_args)
    hidden_server_cmd = [
        config.python_bin,
        "-m",
        "dflasher.hidden_server",
        config.source_model,
        "--hidden-states-path",
        str(config.hidden_states_dir),
        "--target-layer-ids",
        *target_layers,
    ]
    if config.trust_remote_code:
        hidden_server_cmd.append("--trust-remote-code")
    hidden_server_cmd.extend(
        [
            "--",
            "--port",
            str(config.vllm_port),
        ]
    )
    if config.trust_remote_code:
        hidden_server_cmd.append("--trust-remote-code")
    hidden_server_cmd.extend(config.vllm_args)
    cache_cmd = [
        config.python_bin,
        str(_script_path(config.speculators_repo, "data_generation_offline.py")),
        "--endpoint",
        endpoint,
        "--model",
        config.source_model,
        "--preprocessed-data",
        str(config.data_dir),
        "--output",
        str(config.hidden_states_dir),
        "--max-samples",
        str(config.max_samples),
    ]
    if config.validate_hidden_states:
        cache_cmd.append("--validate-outputs")

    train_cmd = []
    if config.train_processes > 1:
        train_cmd.extend(
            [
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                str(config.train_processes),
            ]
        )
    else:
        train_cmd.append(config.python_bin)
    train_cmd.extend(
        [
            str(_script_path(config.speculators_repo, "train.py")),
            "--verifier-name-or-path",
            config.source_model,
            "--data-path",
            str(config.data_dir),
            "--save-path",
            str(config.checkpoints_dir),
            "--draft-vocab-size",
            str(config.draft_vocab_size),
            "--epochs",
            str(config.epochs),
            "--lr",
            str(config.learning_rate),
            "--total-seq-len",
            str(config.seq_length),
            "--speculator-type",
            "dflash",
            "--block-size",
            str(config.block_size),
            "--max-anchors",
            str(config.max_anchors),
            "--num-layers",
            str(config.draft_layers),
            "--draft-arch",
            draft_arch,
            "--target-layer-ids",
            *target_layers,
            "--save-best",
        ]
    )
    if config.trust_remote_code:
        train_cmd.append("--trust-remote-code")
    train_cmd.extend(config.train_args)
    if config.mode == "offline-cache":
        train_cmd.extend(
            [
                "--hidden-states-path",
                str(config.hidden_states_dir),
                "--on-missing",
                "raise",
            ]
        )
    elif config.mode == "online-cache":
        train_cmd.extend(
            [
                "--vllm-endpoint",
                endpoint,
                "--hidden-states-path",
                str(config.hidden_states_dir),
                "--on-missing",
                "generate",
                "--on-generate",
                "cache",
            ]
        )
    else:
        train_cmd.extend(
            [
                "--vllm-endpoint",
                endpoint,
                "--on-missing",
                "generate",
                "--on-generate",
                "delete",
            ]
        )

    serve_cmd = [
        "vllm",
        "serve",
        str(config.best_checkpoint),
        "--port",
        str(config.vllm_port),
    ]
    if config.trust_remote_code:
        serve_cmd.append("--trust-remote-code")
    serve_cmd.extend(config.serve_args or config.vllm_args)
    benchmark_cmd = [
        config.python_bin,
        "-m",
        "dflasher.cli",
        "official",
        "benchmark",
        str(config.best_checkpoint),
        "--base-url",
        endpoint,
        "--model",
        str(config.best_checkpoint),
        "--max-tokens",
        str(config.benchmark_max_tokens),
        "--concurrency",
        str(config.benchmark_concurrency),
    ]
    if config.benchmark_prompts:
        benchmark_cmd.extend(["--prompts-file", str(config.benchmark_prompts)])

    return {
        "prepare": prepare_cmd,
        "hidden-server": hidden_server_cmd,
        "cache": cache_cmd,
        "train": train_cmd,
        "serve": serve_cmd,
        "benchmark": benchmark_cmd,
    }


def write_manifest(config: OfficialPipelineConfig) -> Path:
    config.workspace.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["workspace"] = str(config.workspace)
    payload["speculators_repo"] = str(config.speculators_repo)
    payload["benchmark_prompts"] = (
        str(config.benchmark_prompts) if config.benchmark_prompts else None
    )
    payload["target_layer_ids"] = list(config.resolved_target_layer_ids())
    payload["draft_arch"] = config.resolved_draft_arch()
    payload["commands"] = {name: _quote_args(args) for name, args in command_plan(config).items()}
    config.manifest_path.write_text(json.dumps(payload, indent=2) + "\n")
    return config.manifest_path


def render_pipeline_script(config: OfficialPipelineConfig) -> str:
    commands = command_plan(config)
    target_layers = " ".join(str(layer_id) for layer_id in config.resolved_target_layer_ids())
    hidden_server_log = config.logs_dir / "hidden_server.log"
    hidden_server_log_arg = _quote_args([str(hidden_server_log)])
    health_wait_block = [
        "SECONDS_WAITED=0",
        f"until curl -sf http://localhost:{config.vllm_port}/health >/dev/null 2>&1; do",
        '  if ! kill -0 "$HIDDEN_SERVER_PID" 2>/dev/null; then',
        "    echo 'Hidden-state server exited before becoming healthy.' >&2",
        f"    tail -n 200 {hidden_server_log_arg} >&2 || true",
        "    exit 1",
        "  fi",
        f"  if [ \"$SECONDS_WAITED\" -ge {config.server_start_timeout} ]; then",
        "    echo 'Timed out waiting for hidden-state server health check.' >&2",
        f"    tail -n 200 {hidden_server_log_arg} >&2 || true",
        "    exit 1",
        "  fi",
        "  sleep 2",
        "  SECONDS_WAITED=$((SECONDS_WAITED + 2))",
        "done",
    ]
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "",
        "# Generated by dflasher official pipeline.",
        f"# source model: {config.source_model}",
        f"# draft architecture: {config.resolved_draft_arch()}",
        f"# target layers: {target_layers}",
        f"# layer policy: {config.layer_policy}",
        f"# mode: {config.mode}",
        "",
        f"export PYTHONPATH={config.speculators_repo / 'src'}:${{PYTHONPATH:-}}",
        f"mkdir -p {_quote_args([str(config.logs_dir)])}",
        "",
        "echo '==> Preprocessed dataset'",
        _quote_args(commands["prepare"]),
        "",
    ]

    if config.mode == "offline-cache":
        lines.extend(
            [
                "echo '==> Starting hidden-state extraction server'",
                f"CUDA_VISIBLE_DEVICES={config.vllm_gpus} "
                + _quote_args(commands["hidden-server"])
                + f" > {hidden_server_log_arg} 2>&1 &",
                "HIDDEN_SERVER_PID=$!",
                "cleanup_hidden_server() {",
                '  kill "$HIDDEN_SERVER_PID" 2>/dev/null || true',
                '  wait "$HIDDEN_SERVER_PID" 2>/dev/null || true',
                "}",
                "trap cleanup_hidden_server EXIT",
                *health_wait_block,
                "",
                "echo '==> Caching hidden states'",
                _quote_args(commands["cache"]),
                "",
            ]
        )
    elif config.mode in {"online-cache", "online-delete"}:
        lines.extend(
            [
                "echo '==> Starting hidden-state extraction server'",
                f"CUDA_VISIBLE_DEVICES={config.vllm_gpus} "
                + _quote_args(commands["hidden-server"])
                + f" > {hidden_server_log_arg} 2>&1 &",
                "HIDDEN_SERVER_PID=$!",
                "cleanup_hidden_server() {",
                '  kill "$HIDDEN_SERVER_PID" 2>/dev/null || true',
                '  wait "$HIDDEN_SERVER_PID" 2>/dev/null || true',
                "}",
                "trap cleanup_hidden_server EXIT",
                *health_wait_block,
                "",
            ]
        )

    lines.extend(
        [
            "echo '==> Training official DFlash speculator'",
            f"CUDA_VISIBLE_DEVICES={config.train_gpus} " + _quote_args(commands["train"]),
            "",
            "echo '==> Checkpoint format validation'",
            _quote_args(
                [
                    config.python_bin,
                    "-m",
                    "dflasher.cli",
                    "official",
                    "inspect-checkpoint",
                    str(config.best_checkpoint),
                    "--source-model",
                    config.source_model,
                    "--expected-block-size",
                    str(config.block_size),
                    "--expected-draft-vocab-size",
                    str(config.draft_vocab_size),
                    *[
                        item
                        for layer_id in config.resolved_target_layer_ids()
                        for item in ("--expected-layer-id", str(layer_id))
                    ],
                ]
            ),
            "",
            "echo '==> Done'",
            f"echo 'Best checkpoint: {config.best_checkpoint}'",
            "",
        ]
    )
    return "\n".join(lines)


def write_pipeline_script(config: OfficialPipelineConfig) -> Path:
    config.workspace.mkdir(parents=True, exist_ok=True)
    config.logs_dir.mkdir(parents=True, exist_ok=True)
    script = render_pipeline_script(config)
    config.script_path.write_text(script)
    config.script_path.chmod(0o755)
    write_manifest(config)
    return config.script_path


def load_manifest(manifest_path: Path) -> OfficialPipelineConfig:
    payload = json.loads(manifest_path.read_text())
    payload.pop("commands", None)
    payload["workspace"] = Path(payload["workspace"])
    payload["speculators_repo"] = Path(payload["speculators_repo"])
    if payload.get("benchmark_prompts"):
        payload["benchmark_prompts"] = Path(payload["benchmark_prompts"])
    if payload.get("target_layer_ids") is not None:
        payload["target_layer_ids"] = tuple(payload["target_layer_ids"])
    for key in ("vllm_args", "serve_args", "train_args", "prepare_args"):
        if payload.get(key) is not None:
            payload[key] = tuple(payload[key])
    return OfficialPipelineConfig(**payload)


def run_command(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
) -> None:
    process = subprocess.run(args, cwd=cwd, env=env, check=False)
    if process.returncode != 0:
        raise RuntimeError(
            f"Command failed with exit code {process.returncode}: {_quote_args(args)}"
        )


def inspect_speculators_checkpoint(
    checkpoint_dir: Path,
    *,
    source_model: str | None = None,
    expected_layer_ids: tuple[int, ...] | None = None,
    expected_block_size: int | None = None,
    expected_draft_vocab_size: int | None = None,
) -> dict[str, object]:
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing config.json in {checkpoint_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Missing model.safetensors in {checkpoint_dir}")

    payload = json.loads(config_path.read_text())
    algorithm = payload.get("speculators_config", {}).get("algorithm")
    model_type = payload.get("speculators_model_type")
    architectures = payload.get("architectures") or []
    block_size = payload.get("block_size")
    proposal_methods = payload.get("speculators_config", {}).get("proposal_methods", [])
    speculative_tokens = None
    if proposal_methods:
        speculative_tokens = proposal_methods[0].get("speculative_tokens")
    if algorithm != "dflash" or model_type != "dflash":
        raise ValueError(
            "Checkpoint is not a vLLM Speculators DFlash checkpoint: "
            f"speculators_model_type={model_type!r}, algorithm={algorithm!r}"
        )
    if architectures != ["DFlashDraftModel"]:
        raise ValueError(f"Unexpected architectures for DFlash checkpoint: {architectures!r}")
    if payload.get("mask_token_id") is None:
        raise ValueError("DFlash checkpoint config is missing mask_token_id.")
    if not payload.get("aux_hidden_state_layer_ids"):
        raise ValueError("DFlash checkpoint config is missing aux_hidden_state_layer_ids.")
    if block_size is None:
        raise ValueError("DFlash checkpoint config is missing block_size.")
    if speculative_tokens != block_size - 1:
        raise ValueError(
            "DFlash greedy proposal speculative_tokens must equal block_size - 1: "
            f"speculative_tokens={speculative_tokens}, block_size={block_size}"
        )
    verifier = payload.get("speculators_config", {}).get("verifier", {})
    if not verifier.get("name_or_path"):
        raise ValueError("DFlash checkpoint config is missing verifier.name_or_path.")
    verifier_name = verifier.get("name_or_path")
    if source_model is not None and verifier_name != source_model:
        raise ValueError(
            "DFlash checkpoint verifier does not match the requested source model: "
            f"{verifier_name!r} != {source_model!r}"
        )
    if expected_layer_ids is not None:
        actual_layer_ids = tuple(payload.get("aux_hidden_state_layer_ids") or ())
        if actual_layer_ids != expected_layer_ids:
            raise ValueError(
                "DFlash checkpoint target layers do not match the expected layers: "
                f"{actual_layer_ids!r} != {expected_layer_ids!r}"
            )
    if expected_block_size is not None and block_size != expected_block_size:
        raise ValueError(
            "DFlash checkpoint block_size does not match the expected value: "
            f"{block_size!r} != {expected_block_size!r}"
        )
    if (
        expected_draft_vocab_size is not None
        and payload.get("draft_vocab_size") != expected_draft_vocab_size
    ):
        raise ValueError(
            "DFlash checkpoint draft_vocab_size does not match the expected value: "
            f"{payload.get('draft_vocab_size')!r} != {expected_draft_vocab_size!r}"
        )
    with safe_open(weights_path, framework="pt") as weights_file:
        weight_keys = set(weights_file.keys())
    required_weight_keys = {"fc.weight", "hidden_norm.weight", "norm.weight"}
    missing_weight_keys = required_weight_keys - weight_keys
    if missing_weight_keys:
        raise ValueError(
            "DFlash checkpoint weights are missing required keys: "
            + ", ".join(sorted(missing_weight_keys))
        )
    if not any(key.startswith("layers.") for key in weight_keys):
        raise ValueError("DFlash checkpoint weights are missing draft layer weights.")
    if not any(key.startswith("lm_head.") for key in weight_keys):
        raise ValueError("DFlash checkpoint weights are missing lm_head weights.")

    return {
        "checkpoint": str(checkpoint_dir),
        "speculators_model_type": model_type,
        "architectures": architectures,
        "algorithm": algorithm,
        "block_size": payload.get("block_size"),
        "mask_token_id": payload.get("mask_token_id"),
        "draft_vocab_size": payload.get("draft_vocab_size"),
        "aux_hidden_state_layer_ids": payload.get("aux_hidden_state_layer_ids"),
        "proposal_speculative_tokens": speculative_tokens,
        "verifier": verifier,
        "weight_key_count": len(weight_keys),
    }


def read_prompts(prompts_file: Path | None) -> list[str]:
    if prompts_file is None:
        return [
            "Explain speculative decoding in one sentence.",
            "Write a Python function that returns the square of a number.",
            "Why does a DFlash drafter still need target verification?",
        ]
    prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


def benchmark_vllm(
    base_url: str,
    model: str,
    prompts_file: Path | None,
    max_tokens: int,
    concurrency: int,
    output_json: Path | None = None,
    log_file: Path | None = None,
) -> dict[str, object]:
    prompts = read_prompts(prompts_file)
    started = time.perf_counter()
    responses = asyncio.run(
        _benchmark_vllm_async(
            base_url=base_url,
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            concurrency=concurrency,
        )
    )
    completion_tokens = sum(int(item["completion_tokens"]) for item in responses)

    elapsed_total = time.perf_counter() - started
    report = {
        "model": model,
        "base_url": base_url,
        "prompt_count": len(prompts),
        "completion_tokens": completion_tokens,
        "elapsed_seconds": elapsed_total,
        "tokens_per_second": completion_tokens / elapsed_total if elapsed_total else 0.0,
        "responses": responses,
        "prometheus_metrics": fetch_vllm_spec_decode_metrics(base_url),
        "log_metrics": parse_vllm_log_metrics(log_file) if log_file else [],
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


async def _benchmark_vllm_async(
    base_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> list[dict[str, object]]:
    semaphore = asyncio.Semaphore(max(1, concurrency))
    limits = httpx.Limits(max_connections=max(1, concurrency))

    async def request_one(client: httpx.AsyncClient, prompt: str) -> dict[str, object]:
        async with semaphore:
            request_started = time.perf_counter()
            response = await client.post(
                f"{base_url.rstrip('/')}/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                },
            )
            response.raise_for_status()
            payload = response.json()
            elapsed = time.perf_counter() - request_started
            usage = payload.get("usage", {})
            token_count = int(usage.get("completion_tokens") or 0)
            return {
                "prompt": prompt,
                "elapsed_seconds": elapsed,
                "completion_tokens": token_count,
                "text": payload.get("choices", [{}])[0].get("text", ""),
            }

    async with httpx.AsyncClient(timeout=300, limits=limits) as client:
        return await asyncio.gather(*(request_one(client, prompt) for prompt in prompts))


def validate_vllm_equivalence(
    target_base_url: str,
    draft_base_url: str,
    target_model: str,
    draft_model: str,
    prompts_file: Path | None,
    max_tokens: int,
    output_json: Path | None = None,
) -> dict[str, object]:
    prompts = read_prompts(prompts_file)
    responses = asyncio.run(
        _validate_vllm_equivalence_async(
            target_base_url=target_base_url,
            draft_base_url=draft_base_url,
            target_model=target_model,
            draft_model=draft_model,
            prompts=prompts,
            max_tokens=max_tokens,
        )
    )
    exact_matches = sum(1 for item in responses if item["match"])
    report: dict[str, object] = {
        "prompt_count": len(prompts),
        "exact_matches": exact_matches,
        "all_match": exact_matches == len(prompts),
        "responses": responses,
    }
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(report, indent=2) + "\n")
    return report


async def _validate_vllm_equivalence_async(
    *,
    target_base_url: str,
    draft_base_url: str,
    target_model: str,
    draft_model: str,
    prompts: list[str],
    max_tokens: int,
) -> list[dict[str, object]]:
    async with httpx.AsyncClient(timeout=300) as client:
        results = []
        for prompt in prompts:
            target_text = await _completion_text(
                client,
                target_base_url,
                target_model,
                prompt,
                max_tokens,
            )
            draft_text = await _completion_text(
                client,
                draft_base_url,
                draft_model,
                prompt,
                max_tokens,
            )
            results.append(
                {
                    "prompt": prompt,
                    "match": target_text == draft_text,
                    "target_text": target_text,
                    "draft_text": draft_text,
                }
            )
        return results


async def _completion_text(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> str:
    response = await client.post(
        f"{base_url.rstrip('/')}/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        },
    )
    response.raise_for_status()
    return response.json().get("choices", [{}])[0].get("text", "")


def fetch_vllm_spec_decode_metrics(base_url: str) -> dict[str, float]:
    metrics_url = base_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    metrics_url = metrics_url.rstrip("/") + "/metrics"
    try:
        response = httpx.get(metrics_url, timeout=10)
        response.raise_for_status()
    except Exception:
        return {}

    wanted_prefixes = (
        "vllm:spec_decode_num_accepted_tokens",
        "vllm:spec_decode_num_draft_tokens",
        "vllm:spec_decode_num_drafts",
    )
    metrics: dict[str, float] = {}
    for line in response.text.splitlines():
        if line.startswith("#"):
            continue
        for prefix in wanted_prefixes:
            if line.startswith(prefix):
                parts = line.split()
                if len(parts) >= 2:
                    metrics[parts[0]] = float(parts[-1])
    return metrics


def parse_vllm_log_metrics(log_file: Path | None) -> list[str]:
    if log_file is None or not log_file.exists():
        return []
    needles = ("accept", "spec", "draft", "throughput", "tokens/s")
    matches = []
    for line in log_file.read_text(errors="ignore").splitlines():
        lowered = line.lower()
        if any(needle in lowered for needle in needles):
            matches.append(line[-500:])
    return matches[-50:]
