from __future__ import annotations

from typer.testing import CliRunner

from dflasher.cli import app
from dflasher.model_profile import BackendSupport


class FakeProfile:
    family = "gemma4"
    family_label = "Gemma 4"
    known_zlab_draft_model = "z-lab/gemma-4-26B-A4B-it-DFlash"
    zlab_num_speculative_tokens = 15
    default_block_size = 16
    backend_support = {
        "zlab_vllm": BackendSupport(
            "requires_custom_backend",
            "needs custom vLLM",
        ),
        "zlab_sglang": BackendSupport("supported", "supported"),
    }


class FakeMlxProfile:
    family = "qwen3_5_dense"
    family_label = "Qwen3.5/Qwen3.6"
    known_zlab_draft_model = "z-lab/Qwen3.5-4B-DFlash"
    zlab_num_speculative_tokens = 15
    default_block_size = 16
    backend_support = {
        "zlab_mlx": BackendSupport("supported", "tested on Qwen3.5 models"),
    }


class FakeNoDraftProfile:
    family = "unknown"
    family_label = "unknown"
    known_zlab_draft_model = None
    zlab_num_speculative_tokens = 15
    default_block_size = 16
    backend_support = {
        "zlab_vllm": BackendSupport("unknown", "unknown"),
        "zlab_sglang": BackendSupport("unknown", "unknown"),
        "zlab_mlx": BackendSupport("unknown", "unknown"),
    }


def test_zlab_vllm_command_includes_gemma_attention_backends(monkeypatch):
    monkeypatch.setattr("dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeProfile())
    runner = CliRunner()

    result = runner.invoke(app, ["zlab", "serve-command", "google/gemma-4-26B-A4B-it"])

    assert result.exit_code == 0
    assert "--attention-backend triton_attn" in result.output
    assert '"attention_backend":"flash_attn"' in result.output
    assert "custom vLLM" in result.output


def test_zlab_sglang_command_includes_required_runtime_flags(monkeypatch):
    monkeypatch.setattr("dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeProfile())
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["zlab", "serve-command", "google/gemma-4-26B-A4B-it", "--backend", "sglang"],
    )

    assert result.exit_code == 0
    assert "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1" in result.output
    assert "--speculative-draft-attention-backend fa4" in result.output
    assert "--mamba-scheduler-strategy extra_buffer" in result.output


def test_official_plan_invalid_mode_exits_without_traceback(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile",
        lambda *args, **kwargs: FakeProfile(),
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "official",
            "plan",
            "Qwen/Qwen3-0.6B",
            "--mode",
            "typo",
            "--workspace",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output
    assert "Traceback" not in result.output


def test_zlab_mlx_script_writes_official_mlx_runtime_import(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeMlxProfile()
    )
    runner = CliRunner()
    out = tmp_path / "run_zlab_mlx.py"

    result = runner.invoke(
        app,
        [
            "zlab",
            "mlx-script",
            "Qwen/Qwen3.5-4B",
            "--out",
            str(out),
            "--prompt",
            "hello",
        ],
    )

    assert result.exit_code == 0
    script = out.read_text()
    assert "from dflash.model_mlx import load, load_draft, stream_generate" in script
    assert "z-lab/Qwen3.5-4B-DFlash" in script
    assert "python" in result.output


def test_mac_zlab_mlx_alias_writes_same_script(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeMlxProfile()
    )
    runner = CliRunner()
    out = tmp_path / "run_zlab_mlx.py"

    result = runner.invoke(
        app,
        ["mac", "zlab-mlx-script", "Qwen/Qwen3.5-4B", "--out", str(out)],
    )

    assert result.exit_code == 0
    assert out.exists()


def test_zlab_mlx_benchmark_command_uses_public_backend(monkeypatch):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeMlxProfile()
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        ["zlab", "mlx-benchmark-command", "Qwen/Qwen3.5-4B", "--enable-thinking"],
    )

    assert result.exit_code == 0
    assert "python -m dflash.benchmark --backend mlx" in result.output
    assert "--draft-model z-lab/Qwen3.5-4B-DFlash" in result.output


def test_zlab_commands_fail_without_known_or_explicit_draft(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeNoDraftProfile()
    )
    runner = CliRunner()

    serve = runner.invoke(app, ["zlab", "serve-command", "unknown/model"])
    mlx = runner.invoke(
        app,
        ["zlab", "mlx-script", "unknown/model", "--out", str(tmp_path / "mlx.py")],
    )
    benchmark = runner.invoke(app, ["zlab", "mlx-benchmark-command", "unknown/model"])

    assert serve.exit_code == 1
    assert mlx.exit_code == 1
    assert benchmark.exit_code == 1


def test_zlab_mlx_script_requires_force_for_unconfirmed_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeNoDraftProfile()
    )
    runner = CliRunner()
    out = tmp_path / "mlx.py"

    blocked = runner.invoke(
        app,
        [
            "zlab",
            "mlx-script",
            "unknown/model",
            "--draft-model",
            "z-lab/unknown-DFlash",
            "--out",
            str(out),
        ],
    )
    forced = runner.invoke(
        app,
        [
            "zlab",
            "mlx-script",
            "unknown/model",
            "--draft-model",
            "z-lab/unknown-DFlash",
            "--out",
            str(out),
            "--force",
        ],
    )

    assert blocked.exit_code == 1
    assert forced.exit_code == 0


def test_zlab_mlx_benchmark_requires_force_for_unconfirmed_backend(monkeypatch):
    monkeypatch.setattr(
        "dflasher.cli.resolve_model_profile", lambda *args, **kwargs: FakeNoDraftProfile()
    )
    runner = CliRunner()

    blocked = runner.invoke(
        app,
        [
            "zlab",
            "mlx-benchmark-command",
            "unknown/model",
            "--draft-model",
            "z-lab/unknown-DFlash",
        ],
    )
    forced = runner.invoke(
        app,
        [
            "zlab",
            "mlx-benchmark-command",
            "unknown/model",
            "--draft-model",
            "z-lab/unknown-DFlash",
            "--force",
        ],
    )

    assert blocked.exit_code == 1
    assert forced.exit_code == 0
