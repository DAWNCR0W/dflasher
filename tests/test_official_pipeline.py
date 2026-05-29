from __future__ import annotations

import json

import pytest
import torch
from safetensors.torch import save_file

from dflasher.official import (
    OfficialPipelineConfig,
    command_plan,
    inspect_speculators_checkpoint,
    load_manifest,
    require_speculators_repo,
    run_preflight,
    write_pipeline_script,
)


def test_command_plan_uses_torchrun_for_multi_process_training(tmp_path):
    config = OfficialPipelineConfig(
        source_model="Qwen/Qwen3-0.6B",
        workspace=tmp_path,
        target_layer_ids=(2, 14, 25),
        train_processes=2,
    )

    train_cmd = command_plan(config)["train"]

    assert train_cmd[:4] == ["torchrun", "--standalone", "--nproc_per_node", "2"]
    assert "--speculator-type" in train_cmd
    assert "dflash" in train_cmd


def test_official_config_rejects_invalid_mode_and_process_count(tmp_path):
    with pytest.raises(ValueError, match="mode"):
        OfficialPipelineConfig(workspace=tmp_path, mode="typo")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="train_processes"):
        OfficialPipelineConfig(workspace=tmp_path, train_processes=0)

    with pytest.raises(ValueError, match="layer_policy"):
        OfficialPipelineConfig(workspace=tmp_path, layer_policy="bad")  # type: ignore[arg-type]


def test_official_default_draft_arch_is_vllm_compatible_llama(tmp_path):
    config = OfficialPipelineConfig(workspace=tmp_path, target_layer_ids=(2, 14, 25))

    assert config.resolved_draft_arch() == "llama"
    assert command_plan(config)["train"][
        command_plan(config)["train"].index("--draft-arch") + 1
    ] == "llama"


def test_official_preflight_flags_non_llama_draft_arch(monkeypatch, tmp_path):
    class FakeProfile:
        family = "qwen3_dense"
        family_label = "Qwen3"
        model_type = "qwen3"
        hidden_size = 1024
        num_hidden_layers = 28
        vocab_size = 32000
        can_train_with_speculators = True
        missing_speculators_fields = ()
        recommended_draft_arch = "qwen3"
        backend_support = {
            "speculators": type(
                "Support",
                (),
                {"status": "supported", "detail": "ok"},
            )()
        }

    monkeypatch.setattr("dflasher.official.resolve_model_profile", lambda *args: FakeProfile())
    monkeypatch.setattr("dflasher.official.importlib.util.find_spec", lambda name: object())
    monkeypatch.setattr("dflasher.official.shutil.which", lambda name: "/usr/bin/curl")
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("prepare_data.py", "data_generation_offline.py", "train.py"):
        (scripts / name).write_text("")
    config = OfficialPipelineConfig(
        workspace=tmp_path / "work",
        speculators_repo=tmp_path,
        draft_arch="qwen3",
        target_layer_ids=(2, 14, 25),
    )

    items = run_preflight(config)

    arch_item = next(item for item in items if item.name == "draft arch vLLM serving")
    assert not arch_item.ok


def test_hidden_server_command_uses_dflasher_launcher_and_trust_remote_code(tmp_path):
    config = OfficialPipelineConfig(
        source_model="MiniMaxAI/MiniMax-M2.7",
        workspace=tmp_path,
        target_layer_ids=(1, 13, 24, 36, 47, 59),
        trust_remote_code=True,
    )

    hidden_cmd = command_plan(config)["hidden-server"]
    prepare_cmd = command_plan(config)["prepare"]
    train_cmd = command_plan(config)["train"]

    assert hidden_cmd[:3] == [hidden_cmd[0], "-m", "dflasher.hidden_server"]
    assert "--trust-remote-code" in hidden_cmd
    assert "--target-layer-ids" in hidden_cmd
    assert "--trust-remote-code" in prepare_cmd
    assert "--trust-remote-code" in train_cmd


def test_official_command_plan_preserves_runtime_passthrough_args(tmp_path):
    config = OfficialPipelineConfig(
        source_model="MiniMaxAI/MiniMax-M2.7",
        workspace=tmp_path,
        target_layer_ids=(1, 13, 24),
        trust_remote_code=True,
        vllm_args=("--tensor-parallel-size", "2"),
        serve_args=("--gpu-memory-utilization", "0.88"),
        train_args=("--gradient-checkpointing",),
        prepare_args=("--chat-template", "auto"),
    )

    commands = command_plan(config)

    assert "--chat-template" in commands["prepare"]
    assert "--tensor-parallel-size" in commands["hidden-server"]
    assert "--gpu-memory-utilization" in commands["serve"]
    assert "--gradient-checkpointing" in commands["train"]


def test_required_speculators_repo_does_not_require_legacy_launch_wrapper(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in ("prepare_data.py", "data_generation_offline.py", "train.py"):
        (scripts / name).write_text("")

    require_speculators_repo(tmp_path)


def test_official_command_plan_modes_and_python_module_commands(tmp_path):
    offline = OfficialPipelineConfig(
        source_model="Qwen/Qwen3-0.6B",
        workspace=tmp_path / "offline",
        target_layer_ids=(2, 14, 25),
        mode="offline-cache",
    )
    online_cache = OfficialPipelineConfig(
        source_model="Qwen/Qwen3-0.6B",
        workspace=tmp_path / "online-cache",
        target_layer_ids=(2, 14, 25),
        mode="online-cache",
    )
    online_delete = OfficialPipelineConfig(
        source_model="Qwen/Qwen3-0.6B",
        workspace=tmp_path / "online-delete",
        target_layer_ids=(2, 14, 25),
        mode="online-delete",
    )

    offline_train = command_plan(offline)["train"]
    cache_train = command_plan(online_cache)["train"]
    delete_train = command_plan(online_delete)["train"]
    benchmark_cmd = command_plan(offline)["benchmark"]

    assert "--hidden-states-path" in offline_train
    assert offline_train[offline_train.index("--on-missing") + 1] == "raise"
    assert cache_train[cache_train.index("--on-missing") + 1] == "generate"
    assert cache_train[cache_train.index("--on-generate") + 1] == "cache"
    assert "--hidden-states-path" not in delete_train
    assert delete_train[delete_train.index("--on-generate") + 1] == "delete"
    assert benchmark_cmd[1:4] == ["-m", "dflasher.cli", "official"]


def test_official_pipeline_script_and_manifest_roundtrip(tmp_path):
    config = OfficialPipelineConfig(
        source_model="Qwen/Qwen3-0.6B",
        workspace=tmp_path,
        target_layer_ids=(2, 14, 25),
        benchmark_prompts=tmp_path / "prompts.txt",
        python_bin="python3.11",
    )

    script_path = write_pipeline_script(config)
    loaded = load_manifest(config.manifest_path)
    script = script_path.read_text()

    assert "CUDA_VISIBLE_DEVICES=0" in script
    assert "trap cleanup_hidden_server EXIT" in script
    assert "SECONDS_WAITED" in script
    assert "kill -0 \"$HIDDEN_SERVER_PID\"" in script
    assert "-m dflasher.cli official inspect-checkpoint" in script
    assert "python3.11 -m dflasher.hidden_server" in script
    assert "--source-model Qwen/Qwen3-0.6B" in script
    assert "--expected-block-size 8" in script
    assert loaded.workspace == tmp_path
    assert loaded.python_bin == "python3.11"
    assert loaded.target_layer_ids == (2, 14, 25)
    assert loaded.benchmark_prompts == tmp_path / "prompts.txt"


def test_inspect_speculators_checkpoint_validates_dflash_contract(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    config = {
        "architectures": ["DFlashDraftModel"],
        "speculators_model_type": "dflash",
        "block_size": 8,
        "mask_token_id": 151669,
        "draft_vocab_size": 8192,
        "aux_hidden_state_layer_ids": [2, 14, 25],
        "speculators_config": {
            "algorithm": "dflash",
            "default_proposal_method": "greedy",
            "proposal_methods": [
                {
                    "proposal_type": "greedy",
                    "speculative_tokens": 7,
                    "verifier_accept_k": 1,
                    "accept_tolerance": 0.0,
                }
            ],
            "verifier": {"name_or_path": "Qwen/Qwen3-0.6B", "architectures": []},
        },
    }
    (checkpoint / "config.json").write_text(json.dumps(config))
    save_file(
        {
            "fc.weight": torch.zeros((1, 1)),
            "hidden_norm.weight": torch.zeros((1,)),
            "norm.weight": torch.zeros((1,)),
            "layers.0.self_attn.q_proj.weight": torch.zeros((1, 1)),
            "lm_head.weight": torch.zeros((1, 1)),
        },
        checkpoint / "model.safetensors",
    )

    result = inspect_speculators_checkpoint(checkpoint)

    assert result["algorithm"] == "dflash"
    assert result["proposal_speculative_tokens"] == 7
