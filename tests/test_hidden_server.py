from __future__ import annotations

import argparse
import json

from dflasher import hidden_server


class FakeConfig:
    num_hidden_layers = 62


def test_hidden_server_builds_trust_remote_code_vllm_command(monkeypatch, tmp_path):
    calls = {}

    def fake_from_pretrained(model, trust_remote_code=False):
        calls["model"] = model
        calls["trust_remote_code"] = trust_remote_code
        return FakeConfig()

    monkeypatch.setattr(hidden_server.AutoConfig, "from_pretrained", fake_from_pretrained)
    args = argparse.Namespace(
        model="MiniMaxAI/MiniMax-M2.7",
        hidden_states_path=str(tmp_path / "hidden"),
        target_layer_ids=[1, 13, 24],
        include_last_layer=True,
        trust_remote_code=True,
        dry_run=False,
    )

    command = hidden_server.build_vllm_command(args, ["--", "--port", "8000"])

    assert calls == {"model": "MiniMaxAI/MiniMax-M2.7", "trust_remote_code": True}
    assert command[:3] == [command[0], "-m", "vllm.entrypoints.cli.main"]
    speculative_config = json.loads(command[command.index("--speculative_config") + 1])
    layers = speculative_config["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ]
    assert layers == [1, 13, 24, 62]
    assert "--trust-remote-code" in command


def test_hidden_server_default_layers_are_valid_for_small_models(monkeypatch, tmp_path):
    class SmallConfig:
        num_hidden_layers = 2

    monkeypatch.setattr(
        hidden_server.AutoConfig,
        "from_pretrained",
        lambda model, trust_remote_code=False: SmallConfig(),
    )
    args = argparse.Namespace(
        model="tiny",
        hidden_states_path=str(tmp_path / "hidden"),
        target_layer_ids=None,
        include_last_layer=True,
        trust_remote_code=False,
        dry_run=False,
    )

    command = hidden_server.build_vllm_command(args, ["--", "--port", "8000"])
    speculative_config = json.loads(command[command.index("--speculative_config") + 1])
    layers = speculative_config["draft_model_config"]["hf_config"][
        "eagle_aux_hidden_state_layer_ids"
    ]

    assert layers == [0, 1, 2]
    assert "--" not in command


def test_hidden_server_rejects_out_of_range_layers(monkeypatch, tmp_path):
    monkeypatch.setattr(
        hidden_server.AutoConfig,
        "from_pretrained",
        lambda model, trust_remote_code=False: FakeConfig(),
    )
    args = argparse.Namespace(
        model="tiny",
        hidden_states_path=str(tmp_path / "hidden"),
        target_layer_ids=[-1, 100],
        include_last_layer=False,
        trust_remote_code=False,
        dry_run=False,
    )

    try:
        hidden_server.build_vllm_command(args, [])
    except ValueError as exc:
        assert "target_layer_ids" in str(exc)
    else:
        raise AssertionError("hidden_server accepted invalid target layer IDs")
