from __future__ import annotations

import json

from typer.testing import CliRunner

from dflasher.cli import app
from dflasher.omlx import (
    OMLX_CACHE_FORMAT,
    OmlxBuildOptions,
    OmlxCacheMetadata,
    make_omlx_draft_config,
    resolve_omlx_layer_ids,
)


def write_minimax_config(path):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["MiniMaxM2ForCausalLM"],
                "model_type": "minimax_m2",
                "hidden_size": 3072,
                "intermediate_size": 1536,
                "num_hidden_layers": 62,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 200064,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5_000_000,
                "max_position_embeddings": 204800,
                "quantization": {"group_size": 64, "bits": 4, "mode": "affine"},
            }
        )
        + "\n"
    )
    return path


def test_omlx_minimax_config_resolves_zlab_layers(tmp_path):
    model_dir = write_minimax_config(tmp_path / "MiniMax-M2.7-oQ4")

    config = make_omlx_draft_config(
        str(model_dir),
        block_size=8,
        draft_layers=2,
    )

    assert config.target_layer_ids == (1, 13, 24, 36, 47, 59)
    assert config.hidden_size == 3072
    assert config.num_key_value_heads == 8
    assert config.num_hidden_layers == 2
    assert config.draft_format == "dflasher.omlx-dflash"


def test_omlx_explicit_layer_ids_are_validated(tmp_path):
    model_dir = write_minimax_config(tmp_path / "model")

    assert resolve_omlx_layer_ids(str(model_dir), "auto", (2, 31, 59)) == (2, 31, 59)

    try:
        resolve_omlx_layer_ids(str(model_dir), "auto", (62,))
    except ValueError as exc:
        assert "out of range" in str(exc)
    else:
        raise AssertionError("accepted an out-of-range OMLX layer id")


def test_omlx_cache_metadata_roundtrip(tmp_path):
    metadata = OmlxCacheMetadata(
        source_model="/models/minimax",
        cache_format=OMLX_CACHE_FORMAT,
        selected_layer_ids=(1, 13),
        hidden_size=16,
        context_width=32,
        vocab_size=128,
        block_size=4,
        mask_token_id=0,
        max_length=32,
        samples=1,
        files=("sample_00000.npz",),
        dtype="float16",
    )

    metadata.save(tmp_path)

    assert OmlxCacheMetadata.load(tmp_path) == metadata


def test_cli_omlx_inspect_reads_local_config(tmp_path):
    model_dir = write_minimax_config(tmp_path / "model")
    runner = CliRunner()

    result = runner.invoke(app, ["omlx", "inspect", str(model_dir)])

    assert result.exit_code == 0
    assert "minimax_m2" in result.output
    assert "1 13 24 36 47 59" in result.output
    assert "dflasher.omlx-dflash" in result.output


def test_build_backend_omlx_delegates_to_omlx_builder(monkeypatch, tmp_path):
    calls = {}

    def fake_build(options):
        calls["options"] = options
        options.output_dir.mkdir(parents=True)
        return options.output_dir

    monkeypatch.setattr("dflasher.cli.build_omlx_draft", fake_build)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "build",
            "/models/minimax-oQ4",
            "--backend",
            "omlx",
            "--out",
            str(tmp_path / "draft"),
            "--texts-file",
            "examples/train_texts.txt",
            "--omlx-cache-dir",
            str(tmp_path / "cache"),
            "--omlx-max-samples",
            "2",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0
    assert isinstance(calls["options"], OmlxBuildOptions)
    assert calls["options"].source_model == "/models/minimax-oQ4"
    assert calls["options"].max_samples == 2
    assert calls["options"].train is False
    assert "OMLX DFlash draft" in result.output
