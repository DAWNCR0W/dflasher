from __future__ import annotations

import json
import subprocess

from dflasher.mac import (
    MacPipelineConfig,
    MacPreflightItem,
    load_mac_manifest,
    mac_command_plan,
    mac_required_preflight_ok,
    run_mac_stage,
    write_mac_pipeline_script,
)


def test_mac_command_plan_wraps_local_train_and_eval_with_device(tmp_path):
    config = MacPipelineConfig(
        source_model="sshleifer/tiny-gpt2",
        workspace=tmp_path,
        texts_file="examples/train_texts.txt",
        device="mps",
        max_steps=3,
        eval_max_new_tokens=5,
        allow_builtin_data=True,
    )

    commands = mac_command_plan(config)

    assert commands["train"][1:4] == ["-m", "dflasher.cli", "train"]
    assert "sshleifer/tiny-gpt2" in commands["train"]
    assert commands["train"][commands["train"].index("--device") + 1] == "mps"
    assert commands["train"][commands["train"].index("--texts-file") + 1] == (
        "examples/train_texts.txt"
    )
    assert commands["eval"][1:4] == ["-m", "dflasher.cli", "eval"]
    assert "sshleifer/tiny-gpt2" in commands["eval"]
    assert commands["eval"][commands["eval"].index("--max-new-tokens") + 1] == "5"


def test_mac_command_plan_includes_dataset_limit_dtype_seed_and_trust_remote_code(tmp_path):
    config = MacPipelineConfig(
        source_model="Qwen/Qwen3.5-4B",
        workspace=tmp_path,
        dataset="HuggingFaceTB/smollm-corpus",
        data_limit=7,
        device="mps",
        torch_dtype="float16",
        seed=99,
        trust_remote_code=True,
    )

    commands = mac_command_plan(config)

    assert commands["train"][commands["train"].index("--dataset") + 1] == (
        "HuggingFaceTB/smollm-corpus"
    )
    assert commands["train"][commands["train"].index("--data-limit") + 1] == "7"
    assert commands["train"][commands["train"].index("--torch-dtype") + 1] == "float16"
    assert commands["train"][commands["train"].index("--seed") + 1] == "99"
    assert "--trust-remote-code" in commands["train"]
    assert "--trust-remote-code" in commands["eval"]


def test_mac_pipeline_script_and_manifest_roundtrip(tmp_path):
    config = MacPipelineConfig(
        source_model="sshleifer/tiny-gpt2",
        workspace=tmp_path,
        device="cpu",
        max_steps=1,
        allow_builtin_data=True,
        python_bin="python3",
    )

    script_path = write_mac_pipeline_script(config)
    manifest = json.loads(config.manifest_path.read_text())
    loaded = load_mac_manifest(config.manifest_path)

    assert script_path.exists()
    assert "python3 -m dflasher.cli train sshleifer/tiny-gpt2" in script_path.read_text()
    assert "-m dflasher.cli train" in manifest["commands"]["train"]
    assert "--allow-builtin-data" in manifest["commands"]["train"]
    assert loaded.source_model == config.source_model
    assert loaded.workspace == tmp_path


def test_mac_preflight_required_checks_ignore_optional_mlx_failures():
    items = [
        MacPreflightItem("torch import", True, "ok", required=True),
        MacPreflightItem("mlx import", False, "optional", required=False),
    ]

    assert mac_required_preflight_ok(items)


def test_mac_run_stage_reports_subprocess_failure(monkeypatch, tmp_path):
    config = MacPipelineConfig(
        source_model="sshleifer/tiny-gpt2",
        workspace=tmp_path,
        allow_builtin_data=True,
    )

    class FakeProcess:
        returncode = 9

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: FakeProcess())

    try:
        run_mac_stage(config, "train")
    except RuntimeError as exc:
        assert "exit code 9" in str(exc)
    else:
        raise AssertionError("run_mac_stage accepted a failing subprocess")


def test_mac_pipeline_rejects_invalid_train_geometry(tmp_path):
    try:
        MacPipelineConfig(
            source_model="sshleifer/tiny-gpt2",
            workspace=tmp_path,
            draft_hidden_size=10,
            heads=3,
        )
    except ValueError as exc:
        assert "divisible" in str(exc)
    else:
        raise AssertionError("MacPipelineConfig accepted invalid head geometry")
