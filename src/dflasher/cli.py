from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.table import Table

from dflasher.evaluation import evaluate
from dflasher.generation import dflash_generate, greedy_generate, load_runtime
from dflasher.hf import get_target_shape
from dflasher.mac import (
    MacPipelineConfig,
    load_mac_manifest,
    mac_command_plan,
    mac_required_preflight_ok,
    run_mac_preflight,
    run_mac_stage,
    write_mac_pipeline_script,
)
from dflasher.model_profile import resolve_model_profile
from dflasher.official import (
    OfficialPipelineConfig,
    benchmark_vllm,
    command_plan,
    inspect_speculators_checkpoint,
    install_speculators_reference,
    load_manifest,
    run_command,
    run_preflight,
    validate_vllm_equivalence,
    write_pipeline_script,
)
from dflasher.omlx import (
    OMLX_MODEL_ROOT,
    OMLX_MODEL_SETTINGS_PATH,
    OmlxBuildOptions,
    OmlxCacheMetadata,
    build_omlx_draft,
    evaluate_omlx_dflash,
    extract_omlx_hidden_cache,
    generate_omlx_dflash,
    init_omlx_draft,
    install_omlx_draft_for_app,
    make_omlx_draft_config,
    native_omlx_dflash_compatibility,
    patch_omlx_app_for_minimax,
    read_model_config,
    train_omlx_draft_from_cache,
)
from dflasher.speculators_bridge import SpeculatorsRecipe, render_speculators_script
from dflasher.training import TrainOptions
from dflasher.training import train as train_draft
from dflasher.zlab_mlx import write_zlab_mlx_script, zlab_mlx_benchmark_command

app = typer.Typer(help="Train and test DFlash-style draft models for Hugging Face causal LMs.")
official_app = typer.Typer(help="Run the official vLLM Speculators DFlash pipeline.")
mac_app = typer.Typer(help="Run Mac-friendly DFlash-lite and z-lab MLX workflows.")
zlab_app = typer.Typer(help="Inspect and serve known z-lab DFlash draft models.")
omlx_app = typer.Typer(help="Build and run local OMLX/MLX DFlash drafts.")
app.add_typer(official_app, name="official")
app.add_typer(mac_app, name="mac")
app.add_typer(zlab_app, name="zlab")
app.add_typer(omlx_app, name="omlx")
console = Console()


def _shell_join(args: list[str]) -> str:
    return shlex.join(args)


def _exit_invalid_config(exc: ValueError) -> NoReturn:
    console.print(f"[red]Invalid configuration:[/red] {exc}")
    raise typer.Exit(code=2) from exc


def _fail_preflight_items(items) -> NoReturn:
    table = Table(title="Preflight failed")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    for item in items:
        table.add_row(item.name, "PASS" if item.ok else "FAIL", item.detail)
    console.print(table)
    raise typer.Exit(code=1)


def _ensure_output_available(out: Path, overwrite: bool) -> None:
    if not out.exists():
        return
    if overwrite:
        if out.is_dir():
            shutil.rmtree(out)
        else:
            out.unlink()
        return
    raise ValueError(f"Output path already exists. Pass --overwrite to replace it: {out}")


@app.command()
def inspect(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config code.")
    ] = False,
):
    """Print source model dimensions used by dflasher."""
    shape = get_target_shape(source_model, trust_remote_code=trust_remote_code)
    table = Table(title=source_model)
    table.add_column("field")
    table.add_column("value")
    table.add_row("hidden_size", str(shape.hidden_size))
    table.add_row("num_hidden_layers", str(shape.num_hidden_layers))
    table.add_row("vocab_size", str(shape.vocab_size))
    table.add_row("max_position_embeddings", str(shape.max_position_embeddings))
    console.print(table)


@app.command()
def train(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output draft model directory.")],
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file.")
    ] = None,
    dataset: Annotated[str | None, typer.Option(help="Optional Hugging Face dataset name.")] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column.")] = "text",
    data_limit: Annotated[
        int | None, typer.Option(help="Maximum number of text rows to load.")
    ] = None,
    allow_builtin_data: Annotated[
        bool,
        typer.Option(help="Allow tiny bundled text data for smoke/debug training."),
    ] = False,
    max_length: Annotated[int, typer.Option(help="Token truncation length.")] = 256,
    block_size: Annotated[int, typer.Option(help="Draft block size including anchor token.")] = 4,
    draft_hidden_size: Annotated[int, typer.Option(help="Internal draft hidden size.")] = 128,
    draft_layers: Annotated[int, typer.Option(help="Number of draft decoder layers.")] = 2,
    heads: Annotated[int, typer.Option(help="Number of draft attention heads.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 4,
    max_steps: Annotated[int, typer.Option(help="Training optimizer steps.")] = 100,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 5e-4,
    loss_fn: Annotated[str, typer.Option(help="kl_div or ce target-logit alignment.")] = "kl_div",
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "auto",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
):
    """Train a generic DFlash-lite draft model."""
    try:
        options = TrainOptions(
            source_model=source_model,
            output_dir=out,
            texts_file=texts_file,
            dataset_name=dataset,
            dataset_split=dataset_split,
            text_column=text_column,
            data_limit=data_limit,
            allow_builtin_data=allow_builtin_data,
            max_length=max_length,
            block_size=block_size,
            draft_hidden_size=draft_hidden_size,
            num_draft_layers=draft_layers,
            num_attention_heads=heads,
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    train_draft(options)


def _resolve_build_backend(backend: str, device: str) -> str:
    normalized = backend.lower()
    if normalized == "official":
        return "cuda"
    if normalized in {"lite", "mac", "mac-lite", "cuda", "mlx", "omlx"}:
        return "mac-lite" if normalized == "mac" else normalized
    if normalized != "auto":
        return normalized
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if device == "mps" or (
        device == "auto"
        and platform.system() == "Darwin"
        and platform.machine() in {"arm64", "aarch64"}
    ):
        return "mac-lite"
    return "lite"


def _require_build_training_data(
    texts_file: str | None,
    dataset: str | None,
    allow_builtin_data: bool,
) -> None:
    if texts_file or dataset or allow_builtin_data:
        return
    raise ValueError(
        "build requires --texts-file or --dataset for a useful draft. "
        "Pass --allow-builtin-data only for smoke/debug builds."
    )


@app.command("build")
def build(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Final draft model output directory.")],
    backend: Annotated[
        str,
        typer.Option(
            help="auto, lite, mac-lite, mac, cuda, or official. cuda creates vLLM DFlash."
        ),
    ] = "auto",
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing output directory."),
    ] = False,
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file for lite builds.")
    ] = None,
    dataset: Annotated[
        str | None,
        typer.Option(help="Hugging Face dataset name for lite builds; official uses --data."),
    ] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split for lite builds.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column for lite builds.")] = "text",
    data_limit: Annotated[int | None, typer.Option(help="Lite build row limit.")] = None,
    allow_builtin_data: Annotated[
        bool,
        typer.Option(help="Allow tiny bundled text data for smoke/debug builds only."),
    ] = False,
    prompts_file: Annotated[
        str | None,
        typer.Option(help="Prompts file used to verify exact greedy equivalence for lite builds."),
    ] = None,
    max_length: Annotated[int, typer.Option(help="Lite token truncation length.")] = 256,
    block_size: Annotated[int, typer.Option(help="Draft block size.")] = 8,
    draft_hidden_size: Annotated[int, typer.Option(help="Lite internal draft hidden size.")] = 128,
    draft_layers: Annotated[int, typer.Option(help="Draft decoder layers.")] = 5,
    heads: Annotated[int, typer.Option(help="Lite draft attention heads.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Lite training batch size.")] = 4,
    max_steps: Annotated[int, typer.Option(help="Lite training optimizer steps.")] = 100,
    learning_rate: Annotated[float, typer.Option(help="Training learning rate.")] = 3e-4,
    loss_fn: Annotated[str, typer.Option(help="kl_div or ce for lite builds.")] = "kl_div",
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto for lite builds.")] = "auto",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, fp32, fp16, bf16, or auto.")
    ] = "float32",
    verify_max_new_tokens: Annotated[int, typer.Option(help="Lite verification token count.")] = 12,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer/config code.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
    workspace: Annotated[
        Path | None,
        typer.Option(help="CUDA official pipeline workspace. Defaults beside --out."),
    ] = None,
    speculators_repo: Annotated[Path, typer.Option(help="Local vLLM Speculators repo.")] = Path(
        "/tmp/vllm-speculators-reference"
    ),
    data: Annotated[
        str, typer.Option(help="Official Speculators data source: sharegpt, ultrachat, or path.")
    ] = "sharegpt",
    max_samples: Annotated[int, typer.Option(help="Official training sample count.")] = 5000,
    seq_length: Annotated[int, typer.Option(help="Official total sequence length.")] = 8192,
    epochs: Annotated[int, typer.Option(help="Official training epochs.")] = 5,
    max_anchors: Annotated[int, typer.Option(help="Official maximum sampled anchors.")] = 3072,
    draft_vocab_size: Annotated[
        int,
        typer.Option(help="Official reduced draft vocab size."),
    ] = 8192,
    mode: Annotated[
        str, typer.Option(help="Official mode: online-delete, online-cache, or offline-cache.")
    ] = "offline-cache",
    layer_policy: Annotated[
        str,
        typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*."),
    ] = "auto",
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit official target layer id."),
    ] = None,
    allow_experimental: Annotated[
        bool,
        typer.Option(help="Allow experimental/preview Speculators verifier families for CUDA."),
    ] = False,
    install_reference: Annotated[
        bool,
        typer.Option(
            "--install-reference/--no-install-reference",
            help="Clone the Speculators reference repo automatically for CUDA builds.",
        ),
    ] = True,
    plan_only: Annotated[
        bool,
        typer.Option(help="For CUDA, write the runnable plan but do not execute training."),
    ] = False,
    skip_preflight: Annotated[
        bool,
        typer.Option(
            help="Skip CUDA preflight checks. Intended only for offline script generation."
        ),
    ] = False,
    python_bin: Annotated[
        str,
        typer.Option(help="Python executable used in CUDA scripts."),
    ] = "python",
    vllm_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for vLLM.")] = "0",
    train_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for training.")] = "0",
    vllm_arg: Annotated[
        list[str] | None,
        typer.Option("--vllm-arg", help="Repeatable extra arg passed to hidden-server vLLM."),
    ] = None,
    serve_arg: Annotated[
        list[str] | None,
        typer.Option("--serve-arg", help="Repeatable extra arg passed to final vLLM serve."),
    ] = None,
    train_arg: Annotated[
        list[str] | None,
        typer.Option("--train-arg", help="Repeatable extra arg passed to Speculators train.py."),
    ] = None,
    prepare_arg: Annotated[
        list[str] | None,
        typer.Option("--prepare-arg", help="Repeatable extra arg passed to prepare_data.py."),
    ] = None,
    omlx_cache_dir: Annotated[
        Path | None,
        typer.Option(help="OMLX hidden-state cache directory for --backend omlx/mlx."),
    ] = None,
    omlx_max_samples: Annotated[
        int,
        typer.Option(help="OMLX cache sample count for --backend omlx/mlx."),
    ] = 8,
    omlx_mask_token_id: Annotated[
        int | None,
        typer.Option(
            help="OMLX draft mask token id. Defaults to source tokenizer <fim_pad> when available."
        ),
    ] = None,
    omlx_intermediate_size: Annotated[
        int | None,
        typer.Option(help="OMLX draft FFN intermediate size. Defaults to source config."),
    ] = None,
    omlx_loss_fn: Annotated[
        str,
        typer.Option(
            help=(
                "OMLX training loss: hidden-mse, ce, ce-hidden, topk-kl, "
                "ce-topk-kl, or ce-hidden-topk-kl."
            )
        ),
    ] = "ce-hidden",
    omlx_hidden_loss_weight: Annotated[
        float,
        typer.Option(help="Hidden-state MSE weight when --omlx-loss-fn=ce-hidden."),
    ] = 0.01,
    omlx_hidden_target: Annotated[
        str,
        typer.Option(help="OMLX hidden MSE target: selected or final."),
    ] = "selected",
    omlx_target_top_k: Annotated[
        int,
        typer.Option(help="Store target top-k logits per next-token position for KL distillation."),
    ] = 0,
    omlx_topk_loss_weight: Annotated[
        float,
        typer.Option(help="Top-k KL loss weight for topk-kl blended objectives."),
    ] = 1.0,
    omlx_topk_temperature: Annotated[
        float,
        typer.Option(help="Teacher/student temperature for top-k KL loss."),
    ] = 1.0,
    omlx_anchor_span_tokens: Annotated[
        int,
        typer.Option(help="Limit generated-token anchor sampling to the first N tokens; 0 disables."),
    ] = 0,
    omlx_first_anchor_probability: Annotated[
        float,
        typer.Option(help="Probability of sampling the first generated DFlash anchor."),
    ] = 0.0,
    omlx_anchor_margin_min: Annotated[
        float,
        typer.Option(help="Minimum target top1-top2 logit margin for margin-aware anchors."),
    ] = 0.0,
    omlx_anchor_margin_top_fraction: Annotated[
        float,
        typer.Option(help="Sample only from the top fraction of target margin anchors; 0 disables."),
    ] = 0.0,
    omlx_label_source: Annotated[
        str,
        typer.Option(help="OMLX CE label source: raw-next-token or target-greedy."),
    ] = "raw-next-token",
    omlx_generated_continuation_tokens: Annotated[
        int,
        typer.Option(help="OMLX cache target-greedy continuation tokens appended per prompt."),
    ] = 0,
    omlx_use_chat_template: Annotated[
        bool,
        typer.Option(help="Tokenize OMLX training texts through the model chat template."),
    ] = False,
    omlx_include_prefill_anchors: Annotated[
        bool,
        typer.Option(
            help="Include prompt/prefill token anchors when generated continuations are used."
        ),
    ] = False,
):
    """Build a draft model from a source model.

    `lite` and `mac-lite` produce a generic local dflasher draft. `cuda` produces
    a vLLM Speculators DFlash checkpoint and requires a CUDA/vLLM environment.
    """
    resolved_backend = _resolve_build_backend(backend, device)
    if resolved_backend not in {"lite", "mac-lite", "cuda", "mlx", "omlx"}:
        raise typer.BadParameter(
            "backend must be auto, lite, mac-lite, mac, cuda, official, mlx, or omlx"
        )
    try:
        if not plan_only:
            _ensure_output_available(out, overwrite=overwrite)
    except ValueError as exc:
        _exit_invalid_config(exc)

    if resolved_backend in {"lite", "mac-lite"}:
        try:
            _require_build_training_data(texts_file, dataset, allow_builtin_data)
            options = TrainOptions(
                source_model=source_model,
                output_dir=out,
                texts_file=texts_file,
                dataset_name=dataset,
                dataset_split=dataset_split,
                text_column=text_column,
                data_limit=data_limit,
                max_length=max_length,
                block_size=block_size,
                draft_hidden_size=draft_hidden_size,
                num_draft_layers=draft_layers,
                num_attention_heads=heads,
                batch_size=batch_size,
                max_steps=max_steps,
                learning_rate=learning_rate,
                loss_fn=loss_fn,
                device="mps" if resolved_backend == "mac-lite" and device == "auto" else device,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                seed=seed,
                allow_builtin_data=allow_builtin_data,
            )
        except ValueError as exc:
            _exit_invalid_config(exc)
        draft_dir = train_draft(options)
        result = evaluate(
            source_model,
            draft_dir,
            prompts_file,
            verify_max_new_tokens,
            options.device,
            torch_dtype,
            trust_remote_code=trust_remote_code,
        )
        manifest = {
            "source_model": source_model,
            "backend": resolved_backend,
            "output": str(draft_dir),
            "format": "dflasher.dflash-lite",
            "verified_exact_matches": result.exact_matches,
            "verified_prompts": result.prompts,
            "mean_acceptance": result.mean_acceptance,
        }
        (draft_dir / "dflasher_build_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        console.print(
            f"[green]Built local DFlash-lite draft[/green] {draft_dir} "
            f"(exact={result.exact_matches}/{result.prompts}, "
            f"mean acceptance={result.mean_acceptance:.2f})"
        )
        if result.exact_matches != result.prompts:
            raise typer.Exit(code=1)
        return

    if resolved_backend in {"mlx", "omlx"}:
        try:
            options = OmlxBuildOptions(
                source_model=source_model,
                output_dir=out,
                texts_file=texts_file,
                dataset_name=dataset,
                dataset_split=dataset_split,
                text_column=text_column,
                data_limit=data_limit,
                allow_builtin_data=allow_builtin_data,
                cache_dir=omlx_cache_dir,
                max_samples=omlx_max_samples,
                max_length=max_length,
                block_size=block_size,
                draft_layers=draft_layers,
                intermediate_size=omlx_intermediate_size,
                layer_policy=layer_policy,
                target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
                mask_token_id=omlx_mask_token_id,
                max_steps=max_steps,
                learning_rate=learning_rate,
                loss_fn=omlx_loss_fn,
                hidden_loss_weight=omlx_hidden_loss_weight,
                hidden_target=omlx_hidden_target,
                label_source=omlx_label_source,
                generated_continuation_tokens=omlx_generated_continuation_tokens,
                use_chat_template=omlx_use_chat_template,
                include_prefill_anchors=omlx_include_prefill_anchors,
                target_top_k=omlx_target_top_k,
                topk_loss_weight=omlx_topk_loss_weight,
                topk_temperature=omlx_topk_temperature,
                anchor_span_tokens=omlx_anchor_span_tokens,
                first_anchor_probability=omlx_first_anchor_probability,
                anchor_margin_min=omlx_anchor_margin_min,
                anchor_margin_top_fraction=omlx_anchor_margin_top_fraction,
                seed=seed,
                overwrite=overwrite,
                train=not plan_only,
            )
            draft_dir = build_omlx_draft(options)
        except (RuntimeError, ValueError) as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
        console.print(f"[green]Built OMLX DFlash draft[/green] {draft_dir}")
        return

    official_workspace = workspace or out.parent / f"{out.name}.official-workspace"
    if install_reference:
        try:
            install_speculators_reference(speculators_repo)
        except RuntimeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1) from exc
    try:
        config = OfficialPipelineConfig(
            source_model=source_model,
            workspace=official_workspace,
            speculators_repo=speculators_repo,
            data=data,
            max_samples=max_samples,
            seq_length=seq_length,
            epochs=epochs,
            learning_rate=learning_rate,
            block_size=block_size,
            max_anchors=max_anchors,
            draft_layers=draft_layers,
            draft_vocab_size=draft_vocab_size,
            mode=mode,  # type: ignore[arg-type]
            layer_policy=layer_policy,  # type: ignore[arg-type]
            target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
            trust_remote_code=trust_remote_code,
            allow_experimental=allow_experimental,
            python_bin=python_bin,
            vllm_gpus=vllm_gpus,
            train_gpus=train_gpus,
            vllm_args=tuple(vllm_arg or ()),
            serve_args=tuple(serve_arg or ()),
            train_args=tuple(train_arg or ()),
            prepare_args=tuple(prepare_arg or ()),
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    if not skip_preflight:
        items = run_preflight(config, include_environment=not plan_only)
        if not all(item.ok for item in items):
            _fail_preflight_items(items)
    script_path = write_pipeline_script(config)
    console.print(f"[green]Wrote CUDA DFlash plan:[/green] {script_path}")
    if plan_only:
        console.print(
            f"[yellow]Plan only:[/yellow] run `bash {script_path}` in a CUDA environment, "
            f"then copy {config.best_checkpoint} to {out}."
        )
        return
    try:
        run_command(["bash", str(script_path)])
        inspect_speculators_checkpoint(
            config.best_checkpoint,
            source_model=source_model,
            expected_layer_ids=config.resolved_target_layer_ids(),
            expected_block_size=block_size,
            expected_draft_vocab_size=draft_vocab_size,
        )
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(config.best_checkpoint, out)
    except (RuntimeError, OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    (out / "dflasher_build_manifest.json").write_text(
        json.dumps(
            {
                "source_model": source_model,
                "backend": "cuda",
                "output": str(out),
                "format": "vllm.speculators.dflash",
                "workspace": str(official_workspace),
                "source_checkpoint": str(config.best_checkpoint),
            },
            indent=2,
        )
        + "\n"
    )
    console.print(f"[green]Built vLLM Speculators DFlash draft[/green] {out}")


@app.command()
def generate(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    draft_dir: Annotated[Path, typer.Argument(help="Path to a dflasher draft model directory.")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt text.")],
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 32,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "auto",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
):
    """Generate with target-verified DFlash-lite greedy speculative decoding."""
    tokenizer, target, draft, runtime_device = load_runtime(
        source_model,
        draft_dir,
        device,
        torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime_device)
    output_ids, stats = dflash_generate(
        draft, target, input_ids, max_new_tokens, tokenizer.eos_token_id
    )
    console.print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
    console.print(
        f"[dim]mean acceptance={stats.mean_acceptance:.2f}, "
        f"target steps={stats.target_steps}, drafted tokens={stats.drafted_tokens}[/dim]"
    )


@app.command()
def eval(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    draft_dir: Annotated[Path, typer.Argument(help="Path to a dflasher draft model directory.")],
    prompts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited prompts file.")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 24,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "auto",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
):
    """Verify speculative output exactly matches target greedy output."""
    result = evaluate(
        source_model,
        draft_dir,
        prompts_file,
        max_new_tokens,
        device,
        torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    console.print(
        f"Exact matches: {result.exact_matches}/{result.prompts}; "
        f"mean acceptance: {result.mean_acceptance:.2f}"
    )
    if result.exact_matches != result.prompts:
        raise typer.Exit(code=1)


@app.command()
def smoke(
    source_model: Annotated[
        str,
        typer.Option(
            "--source", "--source-model", help="Small Hugging Face model for smoke testing."
        ),
    ] = "sshleifer/tiny-gpt2",
    workdir: Annotated[Path | None, typer.Option(help="Directory for smoke artifacts.")] = None,
    max_steps: Annotated[int, typer.Option(help="Tiny training step count.")] = 12,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "auto",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
):
    """Download a small model, train a tiny draft, and verify target-equivalent decoding."""
    root = workdir or Path(tempfile.mkdtemp(prefix="dflasher-smoke-"))
    draft_dir = root / "draft"
    console.print(f"[bold]Smoke workdir[/bold] {root}")
    try:
        options = TrainOptions(
            source_model=source_model,
            output_dir=draft_dir,
            block_size=4,
            draft_hidden_size=64,
            num_draft_layers=1,
            num_attention_heads=2,
            batch_size=2,
            max_steps=max_steps,
            max_length=96,
            device=device,
            trust_remote_code=trust_remote_code,
            allow_builtin_data=True,
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    train_draft(options)
    result = evaluate(
        source_model,
        draft_dir,
        max_new_tokens=12,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    console.print(
        f"[green]Smoke passed[/green] exact={result.exact_matches}/{result.prompts}, "
        f"mean acceptance={result.mean_acceptance:.2f}"
    )


@app.command()
def compare(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    draft_dir: Annotated[Path, typer.Argument(help="Path to a dflasher draft model directory.")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt text.")],
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 32,
    device: Annotated[str, typer.Option(help="cpu, cuda, mps, or auto.")] = "auto",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
):
    """Print baseline greedy and speculative outputs side by side."""
    tokenizer, target, draft, runtime_device = load_runtime(
        source_model,
        draft_dir,
        device,
        torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime_device)
    baseline = greedy_generate(target, input_ids, max_new_tokens, tokenizer.eos_token_id)
    speculative, stats = dflash_generate(
        draft, target, input_ids, max_new_tokens, tokenizer.eos_token_id
    )
    table = Table(title="Target equivalence check")
    table.add_column("decoder")
    table.add_column("text")
    table.add_row("target greedy", tokenizer.decode(baseline[0], skip_special_tokens=True))
    table.add_row("dflasher", tokenizer.decode(speculative[0], skip_special_tokens=True))
    console.print(table)
    console.print(f"token exact match: {baseline.tolist() == speculative.tolist()}")
    console.print(f"mean acceptance: {stats.mean_acceptance:.2f}")
    if baseline.tolist() != speculative.tolist():
        raise typer.Exit(code=1)


def _read_prompt_lines(prompts_file: Path | None, fallback: str) -> list[str]:
    if prompts_file is None:
        return [fallback]
    prompts = [line.strip() for line in prompts_file.read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError(f"No prompts found in {prompts_file}")
    return prompts


@omlx_app.command("inspect")
def omlx_inspect(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    layer_policy: Annotated[
        str, typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*.")
    ] = "auto",
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target hidden layer id."),
    ] = None,
):
    """Inspect a local OMLX/MLX source model without loading weights."""
    try:
        cfg = read_model_config(source_model)
        draft_cfg = make_omlx_draft_config(
            source_model=source_model,
            block_size=8,
            draft_layers=2,
            layer_policy=layer_policy,
            target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    table = Table(title=source_model)
    table.add_column("field")
    table.add_column("value")
    table.add_row("model_type", str(cfg.get("model_type")))
    table.add_row("architectures", ", ".join(cfg.get("architectures", [])))
    table.add_row("hidden_size", str(draft_cfg.hidden_size))
    table.add_row("num_target_layers", str(draft_cfg.num_target_layers))
    table.add_row("vocab_size", str(draft_cfg.vocab_size))
    table.add_row("selected_layer_ids", " ".join(map(str, draft_cfg.target_layer_ids)))
    table.add_row("quantization", "yes" if cfg.get("quantization") else "no")
    table.add_row("draft_format", draft_cfg.draft_format)
    console.print(table)


@omlx_app.command("extract-cache")
def omlx_extract_cache(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output hidden-state cache dir.")],
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file.")
    ] = None,
    dataset: Annotated[str | None, typer.Option(help="Optional Hugging Face dataset name.")] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column.")] = "text",
    data_limit: Annotated[int | None, typer.Option(help="Maximum rows to load.")] = None,
    allow_builtin_data: Annotated[
        bool, typer.Option(help="Allow tiny bundled data for smoke/debug extraction.")
    ] = False,
    max_samples: Annotated[int, typer.Option(help="Maximum cache samples.")] = 8,
    max_length: Annotated[int, typer.Option(help="Token truncation length.")] = 128,
    block_size: Annotated[int, typer.Option(help="Draft block size.")] = 8,
    layer_policy: Annotated[
        str, typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*.")
    ] = "auto",
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target hidden layer id."),
    ] = None,
    mask_token_id: Annotated[
        int | None,
        typer.Option(
            help="Draft mask token id. Defaults to source tokenizer <fim_pad> when available."
        ),
    ] = None,
    label_source: Annotated[
        str,
        typer.Option(help="CE label source saved in the cache: raw-next-token or target-greedy."),
    ] = "raw-next-token",
    generated_continuation_tokens: Annotated[
        int,
        typer.Option(help="Target-greedy continuation tokens appended per prompt."),
    ] = 0,
    use_chat_template: Annotated[
        bool,
        typer.Option(help="Tokenize each text through the model chat template."),
    ] = False,
    include_prefill_anchors: Annotated[
        bool,
        typer.Option(
            help="Include prompt/prefill token anchors when generated continuations are used."
        ),
    ] = False,
    target_top_k: Annotated[
        int,
        typer.Option(help="Store target top-k logits per next-token position for KL distillation."),
    ] = 0,
    hidden_target: Annotated[
        str,
        typer.Option(help="Hidden MSE target saved in the cache: selected or final."),
    ] = "selected",
    overwrite: Annotated[bool, typer.Option(help="Replace existing cache dir.")] = False,
):
    """Extract selected hidden states from a local MLX source model into a cache."""
    try:
        extract_omlx_hidden_cache(
            source_model=source_model,
            cache_dir=out,
            texts_file=texts_file,
            dataset_name=dataset,
            dataset_split=dataset_split,
            text_column=text_column,
            data_limit=data_limit,
            allow_builtin_data=allow_builtin_data,
            max_samples=max_samples,
            max_length=max_length,
            block_size=block_size,
            layer_policy=layer_policy,
            target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
            mask_token_id=mask_token_id,
            label_source=label_source,
            generated_continuation_tokens=generated_continuation_tokens,
            use_chat_template=use_chat_template,
            include_prefill_anchors=include_prefill_anchors,
            target_top_k=target_top_k,
            hidden_target=hidden_target,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@omlx_app.command("init-draft")
def omlx_init_draft(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output draft directory.")],
    block_size: Annotated[int, typer.Option(help="Draft block size.")] = 8,
    draft_layers: Annotated[int, typer.Option(help="Number of lightweight draft layers.")] = 2,
    intermediate_size: Annotated[
        int | None,
        typer.Option(help="Draft FFN intermediate size. Defaults to source config."),
    ] = None,
    layer_policy: Annotated[
        str, typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*.")
    ] = "auto",
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target hidden layer id."),
    ] = None,
    mask_token_id: Annotated[
        int | None,
        typer.Option(
            help="Draft mask token id. Defaults to source tokenizer <fim_pad> when available."
        ),
    ] = None,
    overwrite: Annotated[bool, typer.Option(help="Replace existing draft dir.")] = False,
):
    """Create a local MLX DFlash draft checkpoint skeleton for the source model."""
    try:
        init_omlx_draft(
            source_model=source_model,
            output_dir=out,
            block_size=block_size,
            draft_layers=draft_layers,
            intermediate_size=intermediate_size,
            layer_policy=layer_policy,
            target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
            mask_token_id=mask_token_id,
            overwrite=overwrite,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Initialized OMLX DFlash draft[/green] {out}")


@omlx_app.command("train")
def omlx_train(
    cache_dir: Annotated[Path, typer.Argument(help="Hidden-state cache directory.")],
    draft_dir: Annotated[Path, typer.Argument(help="OMLX DFlash draft directory.")],
    max_steps: Annotated[int, typer.Option(help="Training optimizer steps.")] = 20,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 1e-4,
    source_model: Annotated[
        str | None,
        typer.Option(help="Source model path/id for CE loss. Defaults to cache metadata."),
    ] = None,
    loss_fn: Annotated[
        str,
        typer.Option(help="OMLX training loss: hidden-mse, ce, or ce-hidden."),
    ] = "ce-hidden",
    hidden_loss_weight: Annotated[
        float,
        typer.Option(help="Hidden-state MSE weight when --loss-fn=ce-hidden."),
    ] = 0.01,
    topk_loss_weight: Annotated[
        float,
        typer.Option(help="Top-k KL loss weight for topk-kl blended objectives."),
    ] = 1.0,
    topk_temperature: Annotated[
        float,
        typer.Option(help="Teacher/student temperature for top-k KL loss."),
    ] = 1.0,
    anchor_span_tokens: Annotated[
        int,
        typer.Option(help="Limit generated-token anchor sampling to the first N tokens; 0 disables."),
    ] = 0,
    first_anchor_probability: Annotated[
        float,
        typer.Option(help="Probability of sampling the first generated DFlash anchor."),
    ] = 0.0,
    anchor_margin_min: Annotated[
        float,
        typer.Option(help="Minimum target top1-top2 logit margin for margin-aware anchors."),
    ] = 0.0,
    anchor_margin_top_fraction: Annotated[
        float,
        typer.Option(help="Sample only from the top fraction of target margin anchors; 0 disables."),
    ] = 0.0,
    label_source: Annotated[
        str | None,
        typer.Option(help="Expected cache label source; verifies metadata when set."),
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
):
    """Train the OMLX draft from an extracted hidden-state cache."""
    try:
        train_omlx_draft_from_cache(
            cache_dir=cache_dir,
            draft_dir=draft_dir,
            max_steps=max_steps,
            learning_rate=learning_rate,
            source_model=source_model,
            loss_fn=loss_fn,
            hidden_loss_weight=hidden_loss_weight,
            topk_loss_weight=topk_loss_weight,
            topk_temperature=topk_temperature,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=first_anchor_probability,
            anchor_margin_min=anchor_margin_min,
            anchor_margin_top_fraction=anchor_margin_top_fraction,
            expected_label_source=label_source,
            seed=seed,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@omlx_app.command("build")
def omlx_build(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output draft directory.")],
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file.")
    ] = None,
    dataset: Annotated[str | None, typer.Option(help="Optional Hugging Face dataset name.")] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column.")] = "text",
    data_limit: Annotated[int | None, typer.Option(help="Maximum rows to load.")] = None,
    allow_builtin_data: Annotated[
        bool, typer.Option(help="Allow tiny bundled data for smoke/debug builds.")
    ] = False,
    cache_dir: Annotated[Path | None, typer.Option(help="Hidden-state cache dir.")] = None,
    max_samples: Annotated[int, typer.Option(help="Maximum cache samples.")] = 8,
    max_length: Annotated[int, typer.Option(help="Token truncation length.")] = 128,
    block_size: Annotated[int, typer.Option(help="Draft block size.")] = 8,
    draft_layers: Annotated[int, typer.Option(help="Number of lightweight draft layers.")] = 2,
    intermediate_size: Annotated[
        int | None,
        typer.Option(help="Draft FFN intermediate size. Defaults to source config."),
    ] = None,
    layer_policy: Annotated[
        str, typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*.")
    ] = "auto",
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target hidden layer id."),
    ] = None,
    mask_token_id: Annotated[
        int | None,
        typer.Option(
            help="Draft mask token id. Defaults to source tokenizer <fim_pad> when available."
        ),
    ] = None,
    max_steps: Annotated[int, typer.Option(help="Training optimizer steps.")] = 20,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 1e-4,
    loss_fn: Annotated[
        str,
        typer.Option(help="OMLX training loss: hidden-mse, ce, or ce-hidden."),
    ] = "ce-hidden",
    hidden_loss_weight: Annotated[
        float,
        typer.Option(help="Hidden-state MSE weight when --loss-fn=ce-hidden."),
    ] = 0.01,
    hidden_target: Annotated[
        str,
        typer.Option(help="Hidden MSE target: selected or final."),
    ] = "selected",
    label_source: Annotated[
        str,
        typer.Option(help="CE label source: raw-next-token or target-greedy."),
    ] = "raw-next-token",
    generated_continuation_tokens: Annotated[
        int,
        typer.Option(help="Target-greedy continuation tokens appended per prompt."),
    ] = 0,
    use_chat_template: Annotated[
        bool,
        typer.Option(help="Tokenize each text through the model chat template."),
    ] = False,
    include_prefill_anchors: Annotated[
        bool,
        typer.Option(
            help="Include prompt/prefill token anchors when generated continuations are used."
        ),
    ] = False,
    target_top_k: Annotated[
        int,
        typer.Option(help="Store target top-k logits per next-token position for KL distillation."),
    ] = 0,
    topk_loss_weight: Annotated[
        float,
        typer.Option(help="Top-k KL loss weight for topk-kl blended objectives."),
    ] = 1.0,
    topk_temperature: Annotated[
        float,
        typer.Option(help="Teacher/student temperature for top-k KL loss."),
    ] = 1.0,
    anchor_span_tokens: Annotated[
        int,
        typer.Option(help="Limit generated-token anchor sampling to the first N tokens; 0 disables."),
    ] = 0,
    first_anchor_probability: Annotated[
        float,
        typer.Option(help="Probability of sampling the first generated DFlash anchor."),
    ] = 0.0,
    anchor_margin_min: Annotated[
        float,
        typer.Option(help="Minimum target top1-top2 logit margin for margin-aware anchors."),
    ] = 0.0,
    anchor_margin_top_fraction: Annotated[
        float,
        typer.Option(help="Sample only from the top fraction of target margin anchors; 0 disables."),
    ] = 0.0,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
    overwrite: Annotated[bool, typer.Option(help="Replace existing outputs.")] = False,
    skip_train: Annotated[
        bool,
        typer.Option(help="Initialize and cache but skip draft training."),
    ] = False,
):
    """Build a local MLX DFlash draft from an OMLX/MLX source model."""
    try:
        options = OmlxBuildOptions(
            source_model=source_model,
            output_dir=out,
            texts_file=texts_file,
            dataset_name=dataset,
            dataset_split=dataset_split,
            text_column=text_column,
            data_limit=data_limit,
            allow_builtin_data=allow_builtin_data,
            cache_dir=cache_dir,
            max_samples=max_samples,
            max_length=max_length,
            block_size=block_size,
            draft_layers=draft_layers,
            intermediate_size=intermediate_size,
            layer_policy=layer_policy,
            target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
            mask_token_id=mask_token_id,
            max_steps=max_steps,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            hidden_loss_weight=hidden_loss_weight,
            hidden_target=hidden_target,
            label_source=label_source,
            generated_continuation_tokens=generated_continuation_tokens,
            use_chat_template=use_chat_template,
            include_prefill_anchors=include_prefill_anchors,
            target_top_k=target_top_k,
            topk_loss_weight=topk_loss_weight,
            topk_temperature=topk_temperature,
            anchor_span_tokens=anchor_span_tokens,
            first_anchor_probability=first_anchor_probability,
            anchor_margin_min=anchor_margin_min,
            anchor_margin_top_fraction=anchor_margin_top_fraction,
            seed=seed,
            overwrite=overwrite,
            train=not skip_train,
        )
        draft_dir = build_omlx_draft(options)
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(f"[green]Built OMLX DFlash draft[/green] {draft_dir}")


@omlx_app.command("generate")
def omlx_generate(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    draft_dir: Annotated[Path, typer.Argument(help="OMLX DFlash draft directory.")],
    prompt: Annotated[str, typer.Option("--prompt", "-p", help="Prompt text.")],
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 64,
    verify_mode: Annotated[str, typer.Option(help="DFlash verify mode.")] = "dflash",
    block_tokens: Annotated[int | None, typer.Option(help="Requested DFlash block tokens.")] = None,
    verify_len_cap: Annotated[
        int | None,
        typer.Option(help="Max target tokens verified per DFlash cycle."),
    ] = None,
    draft_window_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft attention window size."),
    ] = None,
    draft_sink_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft attention sink size."),
    ] = None,
    target_fa_window: Annotated[
        int | None,
        typer.Option(help="Target flash-attention window; 0 disables offline FA windowing."),
    ] = None,
    prefill_step_size: Annotated[
        int | None,
        typer.Option(help="Offline DFlash prefill step size."),
    ] = None,
    use_chat_template: Annotated[
        bool,
        typer.Option(help="Tokenize the prompt through the model chat template."),
    ] = False,
):
    """Generate with an OMLX source model and local MLX DFlash draft."""
    try:
        text, _tokens, mean_acceptance = generate_omlx_dflash(
            source_model=source_model,
            draft_dir=draft_dir,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            verify_mode=verify_mode,
            block_tokens=block_tokens,
            verify_len_cap=verify_len_cap,
            draft_window_size=draft_window_size,
            draft_sink_size=draft_sink_size,
            target_fa_window=target_fa_window,
            prefill_step_size=prefill_step_size,
            use_chat_template=use_chat_template,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(text)
    console.print(f"[dim]mean acceptance={mean_acceptance:.2f}[/dim]")


@omlx_app.command("eval")
def omlx_eval(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    draft_dir: Annotated[Path, typer.Argument(help="OMLX DFlash draft directory.")],
    prompts_file: Annotated[Path | None, typer.Option(help="Newline-delimited prompts.")] = None,
    prompt: Annotated[str, typer.Option(help="Fallback prompt when --prompts-file is omitted.")] = (
        "Explain speculative decoding in one sentence."
    ),
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 32,
    verify_mode: Annotated[str, typer.Option(help="DFlash verify mode.")] = "dflash",
    block_tokens: Annotated[int | None, typer.Option(help="Requested DFlash block tokens.")] = None,
    verify_len_cap: Annotated[
        int | None,
        typer.Option(help="Max target tokens verified per DFlash cycle."),
    ] = None,
    draft_window_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft attention window size."),
    ] = None,
    draft_sink_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft attention sink size."),
    ] = None,
    target_fa_window: Annotated[
        int | None,
        typer.Option(help="Target flash-attention window; 0 disables offline FA windowing."),
    ] = None,
    prefill_step_size: Annotated[
        int | None,
        typer.Option(help="Offline DFlash prefill step size."),
    ] = None,
    use_chat_template: Annotated[
        bool,
        typer.Option(help="Tokenize prompts through the model chat template."),
    ] = False,
):
    """Check that OMLX DFlash output matches target greedy tokens at temperature 0."""
    try:
        prompts = _read_prompt_lines(prompts_file, prompt)
        result = evaluate_omlx_dflash(
            source_model=source_model,
            draft_dir=draft_dir,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
            verify_mode=verify_mode,
            block_tokens=block_tokens,
            verify_len_cap=verify_len_cap,
            draft_window_size=draft_window_size,
            draft_sink_size=draft_sink_size,
            target_fa_window=target_fa_window,
            prefill_step_size=prefill_step_size,
            use_chat_template=use_chat_template,
        )
    except (RuntimeError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print(
        f"Exact matches: {result.exact_matches}/{result.prompts}; "
        f"mean acceptance: {result.mean_acceptance:.2f}; "
        f"draft acceptance: {result.mean_draft_acceptance:.2f}; "
        f"max prompt tokens: {result.max_prompt_tokens}"
    )
    if result.exact_matches != result.prompts:
        raise typer.Exit(code=1)


@omlx_app.command("compat")
def omlx_compat(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
):
    """Check native oMLX DFlash compatibility for a source model."""
    try:
        compatible, reason = native_omlx_dflash_compatibility(source_model)
    except ValueError as exc:
        _exit_invalid_config(exc)
    if compatible:
        console.print("compatible" if not reason else f"compatible: {reason}")
    else:
        console.print(f"incompatible: {reason}")
    if not compatible:
        raise typer.Exit(code=1)


@omlx_app.command("patch-app")
def omlx_patch_app(
    app_path: Annotated[
        Path,
        typer.Option(help="Installed oMLX.app bundle path."),
    ] = Path("/Applications/oMLX.app"),
    overwrite_target_backend: Annotated[
        bool,
        typer.Option(help="Rewrite the bundled MiniMax-M2 target backend file."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Report files that would change without writing them."),
    ] = False,
):
    """Patch the installed oMLX app so MiniMax-M2 can use DFlashEngine."""
    try:
        result = patch_omlx_app_for_minimax(
            app_path=app_path,
            overwrite_target_backend=overwrite_target_backend,
            dry_run=dry_run,
        )
    except (OSError, ValueError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title="oMLX app MiniMax-M2 DFlash patch")
    table.add_column("field")
    table.add_column("value")
    table.add_row("app_path", str(result.app_path))
    table.add_row("target_ops", str(result.target_ops_path))
    table.add_row("target_backend", str(result.target_backend_path))
    table.add_row("spec_epoch", str(result.spec_epoch_path))
    table.add_row("dflash_engine", str(result.dflash_engine_path))
    table.add_row("model_settings", str(result.model_settings_path))
    table.add_row("dflash_lifecycle", str(result.dflash_lifecycle_path))
    table.add_row("changed", "none" if not result.changed_paths else str(len(result.changed_paths)))
    for path in result.changed_paths:
        table.add_row("changed_path", str(path))
    console.print(table)


@omlx_app.command("install-app")
def omlx_install_app(
    source_model: Annotated[str, typer.Argument(help="Local OMLX/MLX source model directory.")],
    draft_dir: Annotated[Path, typer.Argument(help="OMLX DFlash draft directory.")],
    installed_name: Annotated[
        str | None,
        typer.Option(help="Directory name under ~/.omlx/models for the installed draft."),
    ] = None,
    settings_path: Annotated[
        Path,
        typer.Option(help="Path to the local oMLX model_settings.json file."),
    ] = OMLX_MODEL_SETTINGS_PATH,
    model_root: Annotated[
        Path,
        typer.Option(help="oMLX model root used when copying the draft."),
    ] = OMLX_MODEL_ROOT,
    no_copy: Annotated[
        bool,
        typer.Option(
            help="Point model settings at draft_dir instead of copying to ~/.omlx/models."
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        typer.Option(help="Replace an existing installed draft directory."),
    ] = False,
    ssd_cache: Annotated[
        bool,
        typer.Option("--ssd-cache/--no-ssd-cache", help="Enable oMLX DFlash SSD cache."),
    ] = True,
    verify_mode: Annotated[
        str,
        typer.Option(help="DFlash verify mode passed to oMLX."),
    ] = "adaptive",
    draft_quant: Annotated[
        bool,
        typer.Option(
            "--draft-quant/--no-draft-quant",
            help="Quantize the draft at oMLX load time to reduce memory pressure.",
        ),
    ] = False,
    draft_quant_weight_bits: Annotated[
        int,
        typer.Option(help="Draft quantization weight bits passed to oMLX."),
    ] = 4,
    draft_quant_activation_bits: Annotated[
        int,
        typer.Option(help="Draft quantization activation bits passed to oMLX."),
    ] = 16,
    draft_quant_group_size: Annotated[
        int,
        typer.Option(help="Draft quantization group size passed to oMLX."),
    ] = 64,
    dflash_max_ctx: Annotated[
        int | None,
        typer.Option(help="Prompt token threshold where oMLX falls back from DFlash."),
    ] = None,
    dflash_draft_window_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft cache window size passed to oMLX."),
    ] = None,
    dflash_draft_sink_size: Annotated[
        int | None,
        typer.Option(help="DFlash draft sink size passed to oMLX."),
    ] = None,
    dflash_verify_len_cap: Annotated[
        int | None,
        typer.Option(help="Maximum verifier tokens per DFlash cycle."),
    ] = None,
    dflash_block_tokens: Annotated[
        int | None,
        typer.Option(help="Maximum DFlash block tokens requested at runtime."),
    ] = None,
):
    """Install a draft into local oMLX model settings."""
    try:
        result = install_omlx_draft_for_app(
            source_model=source_model,
            draft_dir=draft_dir,
            installed_name=installed_name,
            settings_path=settings_path,
            model_root=model_root,
            copy_draft=not no_copy,
            overwrite=overwrite,
            ssd_cache=ssd_cache,
            verify_mode=verify_mode,
            draft_quant_enabled=draft_quant,
            draft_quant_weight_bits=draft_quant_weight_bits,
            draft_quant_activation_bits=draft_quant_activation_bits,
            draft_quant_group_size=draft_quant_group_size,
            dflash_max_ctx=dflash_max_ctx,
            dflash_draft_window_size=dflash_draft_window_size,
            dflash_draft_sink_size=dflash_draft_sink_size,
            dflash_verify_len_cap=dflash_verify_len_cap,
            dflash_block_tokens=dflash_block_tokens,
        )
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    table = Table(title="oMLX DFlash settings")
    table.add_column("field")
    table.add_column("value")
    table.add_row("source_model", result.source_model)
    table.add_row("draft_model", result.draft_model)
    table.add_row("settings_path", str(result.settings_path))
    table.add_row("native_compatible", str(result.compatible_with_native_omlx))
    if result.compatibility_reason:
        table.add_row("compatibility_reason", result.compatibility_reason)
    console.print(table)


@omlx_app.command("cache-info")
def omlx_cache_info(
    cache_dir: Annotated[Path, typer.Argument(help="Hidden-state cache directory.")],
):
    """Print OMLX hidden-state cache metadata."""
    try:
        metadata = OmlxCacheMetadata.load(cache_dir)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc
    console.print_json(data=asdict(metadata))


@app.command("speculators-script")
def speculators_script(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Path to write the shell script.")],
    output_dir: Annotated[
        Path, typer.Option(help="Training output directory used inside the script.")
    ],
    data: Annotated[
        str, typer.Option(help="sharegpt, ultrachat, or custom data path for Speculators.")
    ] = "sharegpt",
    max_samples: Annotated[int, typer.Option(help="Maximum training samples.")] = 5000,
    seq_length: Annotated[int, typer.Option(help="Training sequence length.")] = 8192,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 5,
    learning_rate: Annotated[float, typer.Option(help="Training learning rate.")] = 3e-4,
    block_size: Annotated[int, typer.Option(help="DFlash block size.")] = 8,
    max_anchors: Annotated[int, typer.Option(help="Maximum sampled anchors.")] = 3072,
    draft_layers: Annotated[int, typer.Option(help="Number of draft layers.")] = 5,
    draft_vocab_size: Annotated[int, typer.Option(help="Reduced draft vocabulary size.")] = 8192,
    vllm_port: Annotated[int, typer.Option(help="vLLM hidden-state server port.")] = 8000,
    vllm_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for vLLM.")] = "0",
    train_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for training.")] = "0",
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Pass --trust-remote-code through prepare, hidden server, and train."),
    ] = False,
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target hidden layer id."),
    ] = None,
    layer_policy: Annotated[
        str,
        typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*."),
    ] = "auto",
    vllm_arg: Annotated[
        list[str] | None,
        typer.Option("--vllm-arg", help="Repeatable extra arg passed to hidden-server vLLM."),
    ] = None,
    prepare_arg: Annotated[
        list[str] | None,
        typer.Option("--prepare-arg", help="Repeatable extra arg passed to prepare_data.py."),
    ] = None,
):
    """Write an official-style vLLM Speculators DFlash training script."""
    recipe = SpeculatorsRecipe(
        source_model=source_model,
        output_dir=output_dir,
        data=data,
        max_samples=max_samples,
        seq_length=seq_length,
        epochs=epochs,
        learning_rate=learning_rate,
        block_size=block_size,
        max_anchors=max_anchors,
        num_layers=draft_layers,
        draft_vocab_size=draft_vocab_size,
        vllm_port=vllm_port,
        vllm_gpus=vllm_gpus,
        train_gpus=train_gpus,
        target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
        layer_policy=layer_policy,  # type: ignore[arg-type]
        trust_remote_code=trust_remote_code,
        vllm_args=tuple(vllm_arg or ()),
        prepare_args=tuple(prepare_arg or ()),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    script = render_speculators_script(recipe)
    out.write_text(script)
    out.chmod(0o755)
    console.print(f"[green]Wrote Speculators DFlash script to[/green] {out}")


def _official_config(
    source_model: str,
    workspace: Path,
    speculators_repo: Path,
    data: str,
    max_samples: int,
    seq_length: int,
    epochs: int,
    learning_rate: float,
    block_size: int,
    max_anchors: int,
    draft_layers: int,
    draft_arch: str | None,
    draft_vocab_size: int,
    mode: str,
    vllm_port: int,
    vllm_gpus: str,
    train_gpus: str,
    train_processes: int,
    layer_policy: str,
    trust_remote_code: bool,
    target_layer_id: list[int] | None,
    validate_hidden_states: bool = True,
    python_bin: str = "python",
    server_start_timeout: int = 900,
    allow_experimental: bool = False,
    prepare_trust_remote_code: bool = True,
    vllm_args: tuple[str, ...] = (),
    serve_args: tuple[str, ...] = (),
    train_args: tuple[str, ...] = (),
    prepare_args: tuple[str, ...] = (),
) -> OfficialPipelineConfig:
    return OfficialPipelineConfig(
        source_model=source_model,
        workspace=workspace,
        speculators_repo=speculators_repo,
        data=data,
        max_samples=max_samples,
        seq_length=seq_length,
        epochs=epochs,
        learning_rate=learning_rate,
        block_size=block_size,
        max_anchors=max_anchors,
        draft_layers=draft_layers,
        draft_arch=draft_arch,
        draft_vocab_size=draft_vocab_size,
        mode=mode,  # type: ignore[arg-type]
        vllm_port=vllm_port,
        vllm_gpus=vllm_gpus,
        train_gpus=train_gpus,
        train_processes=train_processes,
        layer_policy=layer_policy,  # type: ignore[arg-type]
        trust_remote_code=trust_remote_code,
        target_layer_ids=tuple(target_layer_id) if target_layer_id else None,
        validate_hidden_states=validate_hidden_states,
        python_bin=python_bin,
        server_start_timeout=server_start_timeout,
        allow_experimental=allow_experimental,
        prepare_trust_remote_code=prepare_trust_remote_code,
        vllm_args=vllm_args,
        serve_args=serve_args,
        train_args=train_args,
        prepare_args=prepare_args,
    )


@official_app.command("preflight")
def official_preflight(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    workspace: Annotated[Path, typer.Option(help="Pipeline workspace.")] = Path(
        "runs/qwen3-0.6b-official"
    ),
    speculators_repo: Annotated[Path, typer.Option(help="Local vLLM Speculators repo.")] = Path(
        "/tmp/vllm-speculators-reference"
    ),
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config/tokenizer code.")
    ] = False,
    allow_experimental: Annotated[
        bool,
        typer.Option(help="Allow experimental/preview Speculators verifier families."),
    ] = False,
    prepare_trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--prepare-trust-remote-code/--no-prepare-trust-remote-code",
            help="Pass --trust-remote-code to prepare_data.py when global trust is enabled.",
        ),
    ] = True,
    static_only: Annotated[
        bool,
        typer.Option(help="Skip CUDA/package runtime checks and validate only model/repo/scripts."),
    ] = False,
):
    """Check whether this machine can run the official DFlash path."""
    config = OfficialPipelineConfig(
        source_model=source_model,
        workspace=workspace,
        speculators_repo=speculators_repo,
        trust_remote_code=trust_remote_code,
        allow_experimental=allow_experimental,
        prepare_trust_remote_code=prepare_trust_remote_code,
    )
    table = Table(title="Official DFlash preflight")
    table.add_column("check")
    table.add_column("status")
    table.add_column("detail")
    items = run_preflight(config, include_environment=not static_only)
    for item in items:
        table.add_row(item.name, "PASS" if item.ok else "FAIL", item.detail)
    console.print(table)
    if not all(item.ok for item in items):
        raise typer.Exit(code=1)


@official_app.command("inspect-model")
def official_inspect_model(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config code.")
    ] = False,
):
    """Inspect model family, Speculators compatibility, and target-layer policies."""
    profile = resolve_model_profile(
        source_model,
        trust_remote_code=trust_remote_code,
        allow_name_fallback=True,
    )
    table = Table(title=source_model)
    table.add_column("field")
    table.add_column("value")
    table.add_row("family", profile.family)
    table.add_row("family_label", profile.family_label)
    table.add_row("model_type", profile.model_type)
    table.add_row("architectures", ", ".join(profile.architectures) or "-")
    table.add_row("hidden_size", str(profile.hidden_size))
    table.add_row("num_hidden_layers", str(profile.num_hidden_layers))
    table.add_row("vocab_size", str(profile.vocab_size))
    table.add_row("recommended_draft_arch", profile.recommended_draft_arch)
    table.add_row("speculators_layer_policy", profile.speculators_layer_policy)
    table.add_row("zlab_layer_policy", profile.zlab_layer_policy)
    table.add_row("zlab_draft_model", profile.known_zlab_draft_model or "-")
    table.add_row("zlab_num_speculative_tokens", str(profile.zlab_num_speculative_tokens))
    table.add_row(
        "speculators_fields",
        "PASS"
        if profile.can_train_with_speculators
        else "missing: " + ", ".join(profile.missing_speculators_fields),
    )
    for backend, support in profile.backend_support.items():
        table.add_row(f"backend:{backend}", f"{support.status} - {support.detail}")
    for policy in ("auto", "speculators", "dflash5", "zlab5", "zlab6"):
        table.add_row(
            f"layers:{policy}",
            " ".join(map(str, profile.target_layer_ids(policy))),  # type: ignore[arg-type]
        )
    table.add_row(
        "layers:zlab-auto",
        " ".join(map(str, profile.zlab_target_layer_ids())),
    )
    if profile.notes:
        table.add_row("notes", " | ".join(profile.notes))
    console.print(table)


@official_app.command("install-reference")
def official_install_reference(
    repo: Annotated[Path, typer.Option(help="Local Speculators repo path.")] = Path(
        "/tmp/vllm-speculators-reference"
    ),
    update: Annotated[
        bool, typer.Option(help="Fast-forward update if repo already exists.")
    ] = False,
):
    """Clone or update the vLLM Speculators reference repository."""
    path = install_speculators_reference(repo, update=update)
    console.print(f"[green]Speculators reference ready:[/green] {path}")


def _mac_config(
    source_model: str,
    workspace: Path,
    texts_file: str | None,
    dataset: str | None,
    dataset_split: str,
    text_column: str,
    data_limit: int | None,
    max_length: int,
    block_size: int,
    draft_hidden_size: int,
    draft_layers: int,
    heads: int,
    batch_size: int,
    max_steps: int,
    learning_rate: float,
    loss_fn: str,
    device: str,
    torch_dtype: str,
    trust_remote_code: bool,
    seed: int,
    eval_max_new_tokens: int,
    allow_builtin_data: bool = False,
    python_bin: str = "python",
) -> MacPipelineConfig:
    return MacPipelineConfig(
        source_model=source_model,
        workspace=workspace,
        texts_file=texts_file,
        dataset=dataset,
        dataset_split=dataset_split,
        text_column=text_column,
        data_limit=data_limit,
        max_length=max_length,
        block_size=block_size,
        draft_hidden_size=draft_hidden_size,
        draft_layers=draft_layers,
        heads=heads,
        batch_size=batch_size,
        max_steps=max_steps,
        learning_rate=learning_rate,
        loss_fn=loss_fn,
        device=device,
        torch_dtype=torch_dtype,
        trust_remote_code=trust_remote_code,
        seed=seed,
        eval_max_new_tokens=eval_max_new_tokens,
        allow_builtin_data=allow_builtin_data,
        python_bin=python_bin,
    )


@mac_app.command("preflight")
def mac_preflight(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    device: Annotated[str, typer.Option(help="mps, cpu, cuda, or auto.")] = "mps",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config/tokenizer code.")
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(help="Also fail when optional z-lab MLX packages/checkpoints are missing."),
    ] = False,
):
    """Check whether this machine can run the Mac DFlash paths."""
    items = run_mac_preflight(
        source_model,
        device=device,
        trust_remote_code=trust_remote_code,
    )
    table = Table(title="Mac DFlash preflight")
    table.add_column("check")
    table.add_column("status")
    table.add_column("required")
    table.add_column("detail")
    for item in items:
        table.add_row(
            item.name,
            "PASS" if item.ok else "FAIL",
            "yes" if item.required else "optional",
            item.detail,
        )
    console.print(table)
    if strict:
        failed = not all(item.ok for item in items)
    else:
        failed = not mac_required_preflight_ok(items)
    if failed:
        raise typer.Exit(code=1)


@mac_app.command("plan")
def mac_plan(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    workspace: Annotated[Path, typer.Option(help="Pipeline workspace.")] = Path(
        "runs/mac-dflash-lite"
    ),
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file.")
    ] = None,
    dataset: Annotated[str | None, typer.Option(help="Optional Hugging Face dataset name.")] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column.")] = "text",
    data_limit: Annotated[
        int | None, typer.Option(help="Maximum number of text rows to load.")
    ] = None,
    max_length: Annotated[int, typer.Option(help="Token truncation length.")] = 256,
    block_size: Annotated[int, typer.Option(help="Draft block size including anchor token.")] = 4,
    draft_hidden_size: Annotated[int, typer.Option(help="Internal draft hidden size.")] = 128,
    draft_layers: Annotated[int, typer.Option(help="Number of draft decoder layers.")] = 2,
    heads: Annotated[int, typer.Option(help="Number of draft attention heads.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 2,
    max_steps: Annotated[int, typer.Option(help="Training optimizer steps.")] = 100,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 5e-4,
    loss_fn: Annotated[str, typer.Option(help="kl_div or ce target-logit alignment.")] = "kl_div",
    device: Annotated[str, typer.Option(help="mps, cpu, cuda, or auto.")] = "mps",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
    eval_max_new_tokens: Annotated[
        int, typer.Option(help="Tokens used by the generated eval command.")
    ] = 12,
    allow_builtin_data: Annotated[
        bool,
        typer.Option(help="Use dflasher's tiny built-in smoke dataset instead of real data."),
    ] = False,
    python_bin: Annotated[
        str,
        typer.Option(help="Python executable used in the generated Mac script."),
    ] = "python",
):
    """Write a Mac-friendly local DFlash-lite train/eval script."""
    try:
        config = _mac_config(
            source_model,
            workspace,
            texts_file,
            dataset,
            dataset_split,
            text_column,
            data_limit,
            max_length,
            block_size,
            draft_hidden_size,
            draft_layers,
            heads,
            batch_size,
            max_steps,
            learning_rate,
            loss_fn,
            device,
            torch_dtype,
            trust_remote_code,
            seed,
            eval_max_new_tokens,
            allow_builtin_data,
            python_bin,
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    script_path = write_mac_pipeline_script(config)
    console.print(f"[green]Wrote Mac pipeline script:[/green] {script_path}")
    console.print(f"[green]Wrote manifest:[/green] {config.manifest_path}")
    table = Table(title="Mac command plan")
    table.add_column("stage")
    table.add_column("command")
    for stage, command in mac_command_plan(config).items():
        table.add_row(stage, " ".join(command))
    console.print(table)


@mac_app.command("train-lite")
def mac_train_lite(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Output draft model directory.")],
    texts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited training text file.")
    ] = None,
    dataset: Annotated[str | None, typer.Option(help="Optional Hugging Face dataset name.")] = None,
    dataset_split: Annotated[str, typer.Option(help="Dataset split.")] = "train",
    text_column: Annotated[str, typer.Option(help="Dataset text column.")] = "text",
    data_limit: Annotated[
        int | None, typer.Option(help="Maximum number of text rows to load.")
    ] = None,
    max_length: Annotated[int, typer.Option(help="Token truncation length.")] = 256,
    block_size: Annotated[int, typer.Option(help="Draft block size including anchor token.")] = 4,
    draft_hidden_size: Annotated[int, typer.Option(help="Internal draft hidden size.")] = 128,
    draft_layers: Annotated[int, typer.Option(help="Number of draft decoder layers.")] = 2,
    heads: Annotated[int, typer.Option(help="Number of draft attention heads.")] = 4,
    batch_size: Annotated[int, typer.Option(help="Training batch size.")] = 2,
    max_steps: Annotated[int, typer.Option(help="Training optimizer steps.")] = 100,
    learning_rate: Annotated[float, typer.Option(help="AdamW learning rate.")] = 5e-4,
    loss_fn: Annotated[str, typer.Option(help="kl_div or ce target-logit alignment.")] = "kl_div",
    device: Annotated[str, typer.Option(help="mps, cpu, cuda, or auto.")] = "mps",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 13,
    allow_builtin_data: Annotated[
        bool,
        typer.Option(help="Use dflasher's tiny built-in smoke dataset instead of real data."),
    ] = False,
):
    """Train a local DFlash-lite draft with Mac defaults."""
    try:
        options = TrainOptions(
            source_model=source_model,
            output_dir=out,
            texts_file=texts_file,
            dataset_name=dataset,
            dataset_split=dataset_split,
            text_column=text_column,
            data_limit=data_limit,
            max_length=max_length,
            block_size=block_size,
            draft_hidden_size=draft_hidden_size,
            num_draft_layers=draft_layers,
            num_attention_heads=heads,
            batch_size=batch_size,
            max_steps=max_steps,
            learning_rate=learning_rate,
            loss_fn=loss_fn,
            device=device,
            torch_dtype=torch_dtype,
            trust_remote_code=trust_remote_code,
            seed=seed,
            allow_builtin_data=allow_builtin_data,
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    train_draft(options)


@mac_app.command("eval-lite")
def mac_eval_lite(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")],
    draft_dir: Annotated[Path, typer.Argument(help="Path to a dflasher draft model directory.")],
    prompts_file: Annotated[
        str | None, typer.Option(help="Local newline-delimited prompts file.")
    ] = None,
    max_new_tokens: Annotated[int, typer.Option(help="Maximum new tokens.")] = 24,
    device: Annotated[str, typer.Option(help="mps, cpu, cuda, or auto.")] = "mps",
    torch_dtype: Annotated[
        str, typer.Option(help="float32, float16, bfloat16, or auto.")
    ] = "float32",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model/tokenizer code.")
    ] = False,
):
    """Verify a local DFlash-lite draft with Mac defaults."""
    result = evaluate(
        source_model,
        draft_dir,
        prompts_file,
        max_new_tokens,
        device,
        torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    console.print(
        f"Exact matches: {result.exact_matches}/{result.prompts}; "
        f"mean acceptance: {result.mean_acceptance:.2f}"
    )
    if result.exact_matches != result.prompts:
        raise typer.Exit(code=1)


@mac_app.command("run-stage")
def mac_run_stage(
    stage: Annotated[str, typer.Argument(help="train or eval.")],
    manifest: Annotated[
        Path, typer.Option(help="Manifest from `dflasher mac plan`.")
    ] = Path("runs/mac-dflash-lite/dflasher_mac_manifest.json"),
):
    """Run one Mac DFlash-lite stage from a generated manifest."""
    if stage not in {"train", "eval"}:
        raise typer.BadParameter("stage must be 'train' or 'eval'")
    try:
        config = load_mac_manifest(manifest)
        run_mac_stage(config, stage)  # type: ignore[arg-type]
    except ValueError as exc:
        _exit_invalid_config(exc)
    except RuntimeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1) from exc


@mac_app.command("zlab-mlx-script")
def mac_zlab_mlx_script(
    source_model: Annotated[str, typer.Argument(help="Hugging Face target/source model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Python script output path.")],
    draft_model: Annotated[
        str | None,
        typer.Option(help="Explicit z-lab DFlash draft model id. Defaults to known mapping."),
    ] = None,
    prompt: Annotated[str, typer.Option(help="Default prompt embedded in the script.")] = (
        "How many positive whole-number divisors does 196 have?"
    ),
    block_size: Annotated[int, typer.Option(help="DFlash MLX block size.")] = 16,
    max_tokens: Annotated[int, typer.Option(help="Maximum generated tokens.")] = 256,
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 0.0,
    enable_thinking: Annotated[
        bool, typer.Option(help="Pass enable_thinking=True when the tokenizer supports it.")
    ] = True,
    force: Annotated[
        bool,
        typer.Option(help="Write the script even when z-lab MLX support is not confirmed."),
    ] = False,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config code while resolving profile.")
    ] = False,
):
    """Alias for `dflasher zlab mlx-script`."""
    _write_zlab_mlx_script_command(
        source_model=source_model,
        out=out,
        draft_model=draft_model,
        prompt=prompt,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        force=force,
        trust_remote_code=trust_remote_code,
    )


def _write_zlab_mlx_script_command(
    *,
    source_model: str,
    out: Path,
    draft_model: str | None,
    prompt: str,
    block_size: int,
    max_tokens: int,
    temperature: float,
    enable_thinking: bool,
    force: bool,
    trust_remote_code: bool,
) -> None:
    profile = resolve_model_profile(
        source_model,
        trust_remote_code=trust_remote_code,
        allow_name_fallback=True,
    )
    selected_draft = draft_model or profile.known_zlab_draft_model
    if selected_draft is None:
        console.print(
            "[red]No bundled z-lab draft mapping for this source model.[/red] "
            "Pass --draft-model explicitly."
        )
        raise typer.Exit(code=1)
    support = profile.backend_support["zlab_mlx"]
    if support.status != "supported" and not force:
        console.print(
            f"[red]z-lab MLX support is not confirmed for this family:[/red] "
            f"{support.status}: {support.detail}. Pass --force to write the script anyway."
        )
        raise typer.Exit(code=1)
    write_zlab_mlx_script(
        out,
        source_model=source_model,
        draft_model=selected_draft,
        prompt=prompt,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
    )
    console.print(f"[green]Wrote z-lab MLX script:[/green] {out}")
    console.print(f"[bold]family[/bold] {profile.family_label} ({profile.family})")
    console.print(f"[bold]draft[/bold] {selected_draft}")
    console.print(f"[bold]support[/bold] {support.status}: {support.detail}")
    console.print(_shell_join(["python", str(out)]), soft_wrap=True)


@zlab_app.command("serve-command")
def zlab_serve_command(
    source_model: Annotated[str, typer.Argument(help="Hugging Face target/source model id.")],
    draft_model: Annotated[
        str | None,
        typer.Option(help="Explicit z-lab DFlash draft model id. Defaults to known mapping."),
    ] = None,
    backend: Annotated[str, typer.Option(help="vllm or sglang.")] = "vllm",
    port: Annotated[int, typer.Option(help="OpenAI-compatible server port.")] = 8000,
    num_speculative_tokens: Annotated[
        int | None,
        typer.Option(help="Override speculative token count."),
    ] = None,
    attention_backend: Annotated[
        str | None,
        typer.Option(help="Target attention backend. Defaults to z-lab's family recommendation."),
    ] = None,
    draft_attention_backend: Annotated[
        str | None,
        typer.Option(help="Draft/speculative attention backend, e.g. flash_attn or fa4."),
    ] = None,
    max_num_batched_tokens: Annotated[
        int,
        typer.Option(help="vLLM max batched tokens."),
    ] = 32768,
    tp_size: Annotated[int, typer.Option(help="SGLang tensor parallel size.")] = 1,
    mem_fraction_static: Annotated[
        float,
        typer.Option(help="SGLang --mem-fraction-static value."),
    ] = 0.75,
    mamba_scheduler_strategy: Annotated[
        str,
        typer.Option(help="SGLang mamba scheduler strategy."),
    ] = "extra_buffer",
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Pass --trust-remote-code to the serving backend."),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Print the command even when backend support is only preview/unknown."),
    ] = False,
):
    """Print a serving command for a known z-lab DFlash checkpoint."""
    profile = resolve_model_profile(
        source_model,
        trust_remote_code=trust_remote_code,
        allow_name_fallback=True,
    )
    selected_draft = draft_model or profile.known_zlab_draft_model
    if selected_draft is None:
        console.print(
            "[red]No bundled z-lab draft mapping for this source model.[/red] "
            "Pass --draft-model explicitly."
        )
        raise typer.Exit(code=1)

    if backend == "vllm":
        support = profile.backend_support["zlab_vllm"]
        tokens = num_speculative_tokens or profile.zlab_num_speculative_tokens
        resolved_attention_backend = (
            attention_backend
            if attention_backend is not None
            else ("triton_attn" if profile.family == "gemma4" else "flash_attn")
        )
        speculative_config = json.dumps(
            {
                "method": "dflash",
                "model": selected_draft,
                "num_speculative_tokens": tokens,
                **(
                    {"attention_backend": draft_attention_backend or "flash_attn"}
                    if profile.family == "gemma4"
                    else {}
                ),
            },
            separators=(",", ":"),
        )
        command = [
            "vllm",
            "serve",
            source_model,
            "--port",
            str(port),
            "--speculative-config",
            speculative_config,
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--attention-backend",
            resolved_attention_backend,
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
    elif backend == "sglang":
        support = profile.backend_support["zlab_sglang"]
        tokens = num_speculative_tokens or profile.default_block_size
        resolved_attention_backend = attention_backend or "trtllm_mha"
        resolved_draft_attention_backend = draft_attention_backend or "fa4"
        command = [
            "env",
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1",
            "python",
            "-m",
            "sglang.launch_server",
            "--model-path",
            source_model,
            "--speculative-algorithm",
            "DFLASH",
            "--speculative-draft-model-path",
            selected_draft,
            "--speculative-num-draft-tokens",
            str(tokens),
            "--tp-size",
            str(tp_size),
            "--attention-backend",
            resolved_attention_backend,
            "--speculative-draft-attention-backend",
            resolved_draft_attention_backend,
            "--mem-fraction-static",
            str(mem_fraction_static),
            "--mamba-scheduler-strategy",
            mamba_scheduler_strategy,
            "--port",
            str(port),
        ]
        if trust_remote_code:
            command.append("--trust-remote-code")
    else:
        raise typer.BadParameter("backend must be 'vllm' or 'sglang'")

    if support.status not in {"supported", "requires_custom_backend"} and not force:
        console.print(
            f"[red]z-lab {backend} support is not confirmed for this family:[/red] "
            f"{support.status}: {support.detail}. Pass --force to print the command anyway."
        )
        raise typer.Exit(code=1)

    console.print(f"[bold]family[/bold] {profile.family_label} ({profile.family})")
    console.print(f"[bold]draft[/bold] {selected_draft}")
    console.print(f"[bold]support[/bold] {support.status}: {support.detail}")
    if profile.family == "gemma4" and backend == "vllm":
        console.print(
            "[yellow]Gemma4 DFlash requires z-lab's Gemma4-capable vLLM build "
            "or Docker image.[/yellow]"
        )
    console.print(_shell_join(command), soft_wrap=True)


@zlab_app.command("mlx-script")
def zlab_mlx_script(
    source_model: Annotated[str, typer.Argument(help="Hugging Face target/source model id.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Python script output path.")],
    draft_model: Annotated[
        str | None,
        typer.Option(help="Explicit z-lab DFlash draft model id. Defaults to known mapping."),
    ] = None,
    prompt: Annotated[str, typer.Option(help="Default prompt embedded in the script.")] = (
        "How many positive whole-number divisors does 196 have?"
    ),
    block_size: Annotated[int, typer.Option(help="DFlash MLX block size.")] = 16,
    max_tokens: Annotated[int, typer.Option(help="Maximum generated tokens.")] = 256,
    temperature: Annotated[float, typer.Option(help="Sampling temperature.")] = 0.0,
    enable_thinking: Annotated[
        bool, typer.Option(help="Pass enable_thinking=True when the tokenizer supports it.")
    ] = True,
    force: Annotated[
        bool,
        typer.Option(help="Write the script even when z-lab MLX support is not confirmed."),
    ] = False,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config code while resolving profile.")
    ] = False,
):
    """Write a z-lab DFlash MLX inference script for Apple Silicon."""
    _write_zlab_mlx_script_command(
        source_model=source_model,
        out=out,
        draft_model=draft_model,
        prompt=prompt,
        block_size=block_size,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=enable_thinking,
        force=force,
        trust_remote_code=trust_remote_code,
    )


@zlab_app.command("mlx-benchmark-command")
def zlab_mlx_benchmark(
    source_model: Annotated[str, typer.Argument(help="Hugging Face target/source model id.")],
    draft_model: Annotated[
        str | None,
        typer.Option(help="Explicit z-lab DFlash draft model id. Defaults to known mapping."),
    ] = None,
    dataset: Annotated[str, typer.Option(help="z-lab benchmark dataset.")] = "gsm8k",
    max_samples: Annotated[int, typer.Option(help="Maximum benchmark samples.")] = 128,
    enable_thinking: Annotated[
        bool, typer.Option(help="Pass --enable-thinking to z-lab benchmark.")
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Print the command even when z-lab MLX support is not confirmed."),
    ] = False,
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config code while resolving profile.")
    ] = False,
):
    """Print the z-lab MLX benchmark command for a known DFlash checkpoint."""
    profile = resolve_model_profile(
        source_model,
        trust_remote_code=trust_remote_code,
        allow_name_fallback=True,
    )
    selected_draft = draft_model or profile.known_zlab_draft_model
    if selected_draft is None:
        console.print(
            "[red]No bundled z-lab draft mapping for this source model.[/red] "
            "Pass --draft-model explicitly."
        )
        raise typer.Exit(code=1)
    support = profile.backend_support["zlab_mlx"]
    if support.status != "supported" and not force:
        console.print(
            f"[red]z-lab MLX support is not confirmed for this family:[/red] "
            f"{support.status}: {support.detail}. Pass --force to print the command anyway."
        )
        raise typer.Exit(code=1)
    command = zlab_mlx_benchmark_command(
        source_model=source_model,
        draft_model=selected_draft,
        dataset=dataset,
        max_samples=max_samples,
        enable_thinking=enable_thinking,
    )
    console.print(f"[bold]family[/bold] {profile.family_label} ({profile.family})")
    console.print(f"[bold]draft[/bold] {selected_draft}")
    console.print(f"[bold]support[/bold] {support.status}: {support.detail}")
    console.print(_shell_join(command), soft_wrap=True)


@official_app.command("plan")
def official_plan(
    source_model: Annotated[str, typer.Argument(help="Hugging Face source/target model id.")] = (
        "Qwen/Qwen3-0.6B"
    ),
    workspace: Annotated[Path, typer.Option(help="Pipeline workspace.")] = Path(
        "runs/qwen3-0.6b-official"
    ),
    speculators_repo: Annotated[Path, typer.Option(help="Local vLLM Speculators repo.")] = Path(
        "/tmp/vllm-speculators-reference"
    ),
    data: Annotated[str, typer.Option(help="sharegpt, ultrachat, or custom jsonl path.")] = (
        "sharegpt"
    ),
    max_samples: Annotated[int, typer.Option(help="Training sample count.")] = 5000,
    seq_length: Annotated[int, typer.Option(help="Total sequence length.")] = 8192,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 5,
    learning_rate: Annotated[float, typer.Option(help="Training learning rate.")] = 3e-4,
    block_size: Annotated[int, typer.Option(help="DFlash block size.")] = 8,
    max_anchors: Annotated[int, typer.Option(help="Max sampled anchors.")] = 3072,
    draft_layers: Annotated[int, typer.Option(help="Official DFlash draft layers.")] = 5,
    draft_arch: Annotated[
        str | None,
        typer.Option(help="Draft architecture for Speculators train.py, e.g. qwen3 or llama."),
    ] = None,
    draft_vocab_size: Annotated[int, typer.Option(help="Reduced draft vocab size.")] = 8192,
    layer_policy: Annotated[
        str,
        typer.Option(help="auto, speculators, dflash5, zlab5, zlab6, or zlab-linspace*."),
    ] = "auto",
    trust_remote_code: Annotated[
        bool, typer.Option(help="Allow custom HF model config/tokenizer code.")
    ] = False,
    allow_experimental: Annotated[
        bool,
        typer.Option(help="Allow experimental/preview Speculators verifier families."),
    ] = False,
    prepare_trust_remote_code: Annotated[
        bool,
        typer.Option(
            "--prepare-trust-remote-code/--no-prepare-trust-remote-code",
            help="Pass --trust-remote-code to prepare_data.py when global trust is enabled.",
        ),
    ] = True,
    check_environment: Annotated[
        bool,
        typer.Option(help="Also require CUDA/package runtime preflight before writing the plan."),
    ] = False,
    skip_preflight: Annotated[
        bool,
        typer.Option(help="Skip static model/repo/script compatibility checks."),
    ] = False,
    mode: Annotated[
        str,
        typer.Option(help="online-delete, online-cache, or offline-cache."),
    ] = "offline-cache",
    vllm_port: Annotated[int, typer.Option(help="vLLM port.")] = 8000,
    vllm_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for vLLM.")] = "0",
    train_gpus: Annotated[str, typer.Option(help="CUDA_VISIBLE_DEVICES for training.")] = "0",
    train_processes: Annotated[int, typer.Option(help="torchrun process count.")] = 1,
    python_bin: Annotated[
        str,
        typer.Option(help="Python executable used inside generated CUDA scripts."),
    ] = "python",
    server_start_timeout: Annotated[
        int,
        typer.Option(help="Seconds to wait for the hidden-state server health check."),
    ] = 900,
    vllm_arg: Annotated[
        list[str] | None,
        typer.Option("--vllm-arg", help="Repeatable extra arg passed to hidden-server vLLM."),
    ] = None,
    serve_arg: Annotated[
        list[str] | None,
        typer.Option("--serve-arg", help="Repeatable extra arg passed to final vLLM serve."),
    ] = None,
    train_arg: Annotated[
        list[str] | None,
        typer.Option("--train-arg", help="Repeatable extra arg passed to Speculators train.py."),
    ] = None,
    prepare_arg: Annotated[
        list[str] | None,
        typer.Option("--prepare-arg", help="Repeatable extra arg passed to prepare_data.py."),
    ] = None,
    target_layer_id: Annotated[
        list[int] | None,
        typer.Option("--target-layer-id", help="Repeatable explicit target layer id."),
    ] = None,
):
    """Write manifest and runnable script for the official Speculators DFlash path."""
    try:
        config = _official_config(
            source_model,
            workspace,
            speculators_repo,
            data,
            max_samples,
            seq_length,
            epochs,
            learning_rate,
            block_size,
            max_anchors,
            draft_layers,
            draft_arch,
            draft_vocab_size,
            mode,
            vllm_port,
            vllm_gpus,
            train_gpus,
            train_processes,
            layer_policy,
            trust_remote_code,
            target_layer_id,
            python_bin=python_bin,
            server_start_timeout=server_start_timeout,
            allow_experimental=allow_experimental,
            prepare_trust_remote_code=prepare_trust_remote_code,
            vllm_args=tuple(vllm_arg or ()),
            serve_args=tuple(serve_arg or ()),
            train_args=tuple(train_arg or ()),
            prepare_args=tuple(prepare_arg or ()),
        )
    except ValueError as exc:
        _exit_invalid_config(exc)
    if not skip_preflight:
        items = run_preflight(config, include_environment=check_environment)
        if not all(item.ok for item in items):
            _fail_preflight_items(items)
    try:
        script_path = write_pipeline_script(config)
    except ValueError as exc:
        _exit_invalid_config(exc)
    manifest_path = config.manifest_path
    console.print(f"[green]Wrote pipeline script:[/green] {script_path}")
    console.print(f"[green]Wrote manifest:[/green] {manifest_path}")
    table = Table(title="Command plan")
    table.add_column("stage")
    table.add_column("command")
    for stage, command in command_plan(config).items():
        table.add_row(stage, " ".join(command))
    console.print(table)


@official_app.command("run-stage")
def official_run_stage(
    stage: Annotated[
        str,
        typer.Argument(help="prepare, cache, train, serve, or benchmark."),
    ],
    manifest: Annotated[
        Path | None,
        typer.Option(help="Manifest from `dflasher official plan`; overrides other options."),
    ] = None,
    source_model: Annotated[str, typer.Option(help="Hugging Face source/target model id.")] = (
        "Qwen/Qwen3-0.6B"
    ),
    workspace: Annotated[Path, typer.Option(help="Pipeline workspace.")] = Path(
        "runs/qwen3-0.6b-official"
    ),
    speculators_repo: Annotated[Path, typer.Option(help="Local vLLM Speculators repo.")] = Path(
        "/tmp/vllm-speculators-reference"
    ),
    data: Annotated[str, typer.Option(help="sharegpt, ultrachat, or custom jsonl path.")] = (
        "sharegpt"
    ),
    max_samples: Annotated[int, typer.Option(help="Training sample count.")] = 5000,
    seq_length: Annotated[int, typer.Option(help="Total sequence length.")] = 8192,
    epochs: Annotated[int, typer.Option(help="Training epochs.")] = 5,
    mode: Annotated[str, typer.Option(help="online-delete, online-cache, or offline-cache.")] = (
        "offline-cache"
    ),
):
    """Run one official pipeline stage with the current Python environment."""
    if manifest is not None:
        try:
            config = load_manifest(manifest)
        except ValueError as exc:
            _exit_invalid_config(exc)
    else:
        try:
            config = OfficialPipelineConfig(
                source_model=source_model,
                workspace=workspace,
                speculators_repo=speculators_repo,
                data=data,
                max_samples=max_samples,
                seq_length=seq_length,
                epochs=epochs,
                mode=mode,  # type: ignore[arg-type]
            )
        except ValueError as exc:
            _exit_invalid_config(exc)
    commands = command_plan(config)
    if stage not in commands:
        raise typer.BadParameter(f"Unknown stage: {stage}")
    env = os.environ.copy()
    repo_src = str(config.speculators_repo / "src")
    env["PYTHONPATH"] = f"{repo_src}:{env.get('PYTHONPATH', '')}"
    if stage in {"hidden-server", "serve"}:
        env["CUDA_VISIBLE_DEVICES"] = config.vllm_gpus
    if stage == "train":
        env["CUDA_VISIBLE_DEVICES"] = config.train_gpus
    run_command(commands[stage], cwd=config.speculators_repo, env=env)


@official_app.command("inspect-checkpoint")
def official_inspect_checkpoint(
    checkpoint_dir: Annotated[Path, typer.Argument(help="Speculators checkpoint directory.")],
    source_model: Annotated[
        str | None,
        typer.Option(help="Expected verifier/source model id for this draft."),
    ] = None,
    expected_layer_id: Annotated[
        list[int] | None,
        typer.Option("--expected-layer-id", help="Repeatable expected aux hidden layer id."),
    ] = None,
    expected_block_size: Annotated[
        int | None, typer.Option(help="Expected DFlash block_size.")
    ] = None,
    expected_draft_vocab_size: Annotated[
        int | None, typer.Option(help="Expected reduced draft vocabulary size.")
    ] = None,
):
    """Verify a checkpoint is a vLLM Speculators DFlash checkpoint."""
    result = inspect_speculators_checkpoint(
        checkpoint_dir,
        source_model=source_model,
        expected_layer_ids=tuple(expected_layer_id) if expected_layer_id else None,
        expected_block_size=expected_block_size,
        expected_draft_vocab_size=expected_draft_vocab_size,
    )
    table = Table(title="Speculators DFlash checkpoint")
    table.add_column("field")
    table.add_column("value")
    for key, value in result.items():
        table.add_row(key, str(value))
    console.print(table)


@official_app.command("serve-command")
def official_serve_command(
    checkpoint_dir: Annotated[Path, typer.Argument(help="Speculators checkpoint directory.")],
    port: Annotated[int, typer.Option(help="vLLM OpenAI server port.")] = 8000,
    trust_remote_code: Annotated[
        bool,
        typer.Option(help="Pass --trust-remote-code to vLLM serve."),
    ] = False,
    vllm_arg: Annotated[
        list[str] | None,
        typer.Option("--vllm-arg", help="Repeatable extra arg passed to vLLM serve."),
    ] = None,
):
    """Print the vLLM command that serves a trained DFlash checkpoint."""
    inspect_speculators_checkpoint(checkpoint_dir)
    command = ["vllm", "serve", str(checkpoint_dir), "--port", str(port)]
    if trust_remote_code:
        command.append("--trust-remote-code")
    command.extend(vllm_arg or ())
    console.print(_shell_join(command), soft_wrap=True)


@official_app.command("benchmark")
def official_benchmark(
    checkpoint_dir: Annotated[Path, typer.Argument(help="Speculators checkpoint directory.")],
    base_url: Annotated[str, typer.Option(help="OpenAI-compatible vLLM endpoint.")] = (
        "http://localhost:8000/v1"
    ),
    model: Annotated[str | None, typer.Option(help="Model name sent to vLLM.")] = None,
    prompts_file: Annotated[Path | None, typer.Option(help="Newline-delimited prompts.")] = None,
    max_tokens: Annotated[int, typer.Option(help="Max generated tokens per prompt.")] = 256,
    concurrency: Annotated[int, typer.Option(help="Reserved for future parallel requests.")] = 1,
    output_json: Annotated[Path | None, typer.Option(help="Benchmark report path.")] = None,
    log_file: Annotated[Path | None, typer.Option(help="Optional vLLM log to parse.")] = None,
):
    """Measure vLLM serving throughput and collect speculative log hints."""
    inspect_speculators_checkpoint(checkpoint_dir)
    resolved_model = model or str(checkpoint_dir)
    report = benchmark_vllm(
        base_url=base_url,
        model=resolved_model,
        prompts_file=prompts_file,
        max_tokens=max_tokens,
        concurrency=concurrency,
        output_json=output_json,
        log_file=log_file,
    )
    console.print_json(data=report)


@official_app.command("validate-equivalence")
def official_validate_equivalence(
    target_base_url: Annotated[
        str,
        typer.Option(help="OpenAI-compatible endpoint serving the source/target model."),
    ],
    draft_base_url: Annotated[
        str,
        typer.Option(help="OpenAI-compatible endpoint serving the DFlash checkpoint."),
    ],
    target_model: Annotated[str, typer.Option(help="Model name for the target endpoint.")],
    draft_model: Annotated[str, typer.Option(help="Model name for the DFlash endpoint.")],
    prompts_file: Annotated[Path | None, typer.Option(help="Newline-delimited prompts.")] = None,
    max_tokens: Annotated[int, typer.Option(help="Max generated tokens per prompt.")] = 128,
    output_json: Annotated[Path | None, typer.Option(help="Equivalence report path.")] = None,
):
    """Compare target-only vLLM output and DFlash-served output at temperature 0."""
    report = validate_vllm_equivalence(
        target_base_url=target_base_url,
        draft_base_url=draft_base_url,
        target_model=target_model,
        draft_model=draft_model,
        prompts_file=prompts_file,
        max_tokens=max_tokens,
        output_json=output_json,
    )
    console.print_json(data=report)
    if not report["all_match"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
