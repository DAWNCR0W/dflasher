from __future__ import annotations

import json

from typer.testing import CliRunner

from dflasher.cli import app
from dflasher.omlx import (
    OMLX_CACHE_FORMAT,
    OmlxBuildOptions,
    OmlxCacheMetadata,
    install_omlx_draft_for_app,
    make_omlx_draft_config,
    native_omlx_dflash_compatibility,
    patch_omlx_app_for_minimax,
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


def write_minimax_draft_config(path, source_model):
    path.mkdir()
    (path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DFlashDraftModel"],
                "model_type": "minimax_m2",
                "hidden_size": 3072,
                "num_hidden_layers": 2,
                "num_attention_heads": 48,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "intermediate_size": 1536,
                "vocab_size": 200064,
                "rms_norm_eps": 1e-6,
                "rope_theta": 5_000_000,
                "max_position_embeddings": 204800,
                "block_size": 8,
                "num_target_layers": 62,
                "mask_token_id": 0,
                "layer_types": ["full_attention", "full_attention"],
                "dflash_config": {
                    "target_layer_ids": [1, 13, 24, 36, 47, 59],
                    "mask_token_id": 0,
                },
                "dflasher": {
                    "source_model": str(source_model),
                    "training_objective": "ce-hidden",
                },
            }
        )
        + "\n"
    )
    (path / "model.safetensors").write_text("placeholder")
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


def test_omlx_minimax_is_native_compatible(tmp_path):
    model_dir = write_minimax_config(tmp_path / "MiniMax-M2.7-oQ4")

    compatible, reason = native_omlx_dflash_compatibility(str(model_dir))

    assert compatible is True
    assert "patch-app" in reason


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
            "--omlx-loss-fn",
            "hidden-mse",
            "--omlx-hidden-loss-weight",
            "0.0",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0
    assert isinstance(calls["options"], OmlxBuildOptions)
    assert calls["options"].source_model == "/models/minimax-oQ4"
    assert calls["options"].max_samples == 2
    assert calls["options"].loss_fn == "hidden-mse"
    assert calls["options"].hidden_loss_weight == 0.0
    assert calls["options"].train is False
    assert "OMLX DFlash draft" in result.output


def test_install_omlx_draft_updates_model_settings(monkeypatch, tmp_path):
    model_dir = write_minimax_config(tmp_path / "MiniMax-M2.7-oQ4")
    draft_dir = write_minimax_draft_config(tmp_path / "draft", model_dir)
    settings_path = tmp_path / "model_settings.json"
    model_root = tmp_path / "omlx-models"
    monkeypatch.setattr("dflasher.omlx.OMLX_MODEL_ROOT", model_root)

    result = install_omlx_draft_for_app(
        source_model=str(model_dir),
        draft_dir=draft_dir,
        settings_path=settings_path,
        overwrite=True,
    )

    payload = json.loads(settings_path.read_text())
    settings = payload["models"][model_dir.name]
    assert settings["dflash_enabled"] is True
    assert settings["dflash_draft_model"] == str(model_root / f"{model_dir.name}-DFlash-dflasher")
    assert settings["dflash_verify_mode"] == "adaptive"
    assert result.compatible_with_native_omlx is True


def test_install_omlx_draft_rejects_installed_name_path_escape(tmp_path):
    model_dir = write_minimax_config(tmp_path / "MiniMax-M2.7-oQ4")
    draft_dir = write_minimax_draft_config(tmp_path / "draft", model_dir)

    try:
        install_omlx_draft_for_app(
            source_model=str(model_dir),
            draft_dir=draft_dir,
            settings_path=tmp_path / "model_settings.json",
            model_root=tmp_path / "models",
            installed_name="../escape",
        )
    except ValueError as exc:
        assert "single directory name" in str(exc)
    else:
        raise AssertionError("accepted an installed_name that escapes the model root")


def test_install_omlx_draft_rejects_mismatched_source(tmp_path):
    model_dir = write_minimax_config(tmp_path / "MiniMax-M2.7-oQ4")
    other_model_dir = write_minimax_config(tmp_path / "Other-MiniMax")
    draft_dir = write_minimax_draft_config(tmp_path / "draft", other_model_dir)

    try:
        install_omlx_draft_for_app(
            source_model=str(model_dir),
            draft_dir=draft_dir,
            settings_path=tmp_path / "model_settings.json",
            model_root=tmp_path / "models",
        )
    except ValueError as exc:
        assert "Draft source_model does not match" in str(exc)
    else:
        raise AssertionError("accepted a draft for a different source model")


def write_fake_omlx_app(path):
    engine_dir = (
        path
        / "Contents"
        / "Python"
        / "framework-mlx-framework"
        / "lib"
        / "python3.11"
        / "site-packages"
        / "dflash_mlx"
        / "engine"
    )
    engine_dir.mkdir(parents=True)
    (engine_dir / "target_ops.py").write_text(
        "\n".join(
            [
                "TARGET_BACKENDS = [",
                '    "dflash_mlx.engine.target_qwen_gdn:QwenGdnTargetOps",',
                '    "dflash_mlx.engine.target_gemma4:Gemma4TargetOps",',
                "]",
                "",
            ]
        )
    )
    omlx_engine_dir = path / "Contents" / "Resources" / "omlx" / "engine"
    omlx_engine_dir.mkdir(parents=True)
    (omlx_engine_dir / "dflash.py").write_text(
        "\n".join(
            [
                "def is_dflash_compatible(model_path):",
                '    model_type = "minimax_m2"',
                '    is_qwen = "qwen" in model_type',
                '    is_gemma4 = model_type in ("gemma4", "gemma4_text")',
                "    if not (is_qwen or is_gemma4):",
                '        return False, "DFlash supports only Qwen and Gemma4 models"',
                '    return True, ""',
                "",
            ]
        )
    )
    return path


def test_patch_omlx_app_for_minimax_updates_engine_files(tmp_path):
    app_dir = write_fake_omlx_app(tmp_path / "oMLX.app")

    result = patch_omlx_app_for_minimax(app_dir)

    assert len(result.changed_paths) == 3
    assert result.target_backend_path.exists()
    assert "MiniMaxM2TargetOps" in result.target_backend_path.read_text()
    assert "target_minimax_m2:MiniMaxM2TargetOps" in result.target_ops_path.read_text()
    dflash_text = result.dflash_engine_path.read_text()
    assert 'is_minimax_m2 = model_type == "minimax_m2"' in dflash_text
    assert "if not (is_qwen or is_gemma4 or is_minimax_m2):" in dflash_text
    assert result.target_ops_path.with_suffix(".py.dflasher.bak").exists()
    assert result.dflash_engine_path.with_suffix(".py.dflasher.bak").exists()


def test_cli_omlx_patch_app_is_idempotent(tmp_path):
    app_dir = write_fake_omlx_app(tmp_path / "oMLX.app")
    runner = CliRunner()

    first = runner.invoke(app, ["omlx", "patch-app", "--app-path", str(app_dir)])
    second = runner.invoke(app, ["omlx", "patch-app", "--app-path", str(app_dir)])

    assert first.exit_code == 0
    assert "changed" in first.output
    assert second.exit_code == 0
    assert "none" in second.output
