from __future__ import annotations

from typer.testing import CliRunner

from dflasher.cli import app
from dflasher.mac import MacPreflightItem


def test_mac_plan_writes_script_and_manifest(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "mac",
            "plan",
            "sshleifer/tiny-gpt2",
            "--workspace",
            str(tmp_path),
            "--device",
            "cpu",
            "--max-steps",
            "1",
            "--eval-max-new-tokens",
            "2",
            "--allow-builtin-data",
        ],
    )

    assert result.exit_code == 0
    assert (tmp_path / "run_mac_dflash_lite.sh").exists()
    assert (tmp_path / "dflasher_mac_manifest.json").exists()
    assert "--device cpu" in result.output


def test_mac_plan_invalid_config_exits_without_traceback(tmp_path):
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "mac",
            "plan",
            "sshleifer/tiny-gpt2",
            "--workspace",
            str(tmp_path),
            "--block-size",
            "1",
        ],
    )

    assert result.exit_code == 2
    assert "Invalid configuration" in result.output
    assert "Traceback" not in result.output


def test_mac_preflight_strict_fails_on_optional_mlx(monkeypatch):
    monkeypatch.setattr(
        "dflasher.cli.run_mac_preflight",
        lambda *args, **kwargs: [
            MacPreflightItem("torch import", True, "ok", required=True),
            MacPreflightItem("mlx import", False, "optional", required=False),
        ],
    )
    runner = CliRunner()

    non_strict = runner.invoke(app, ["mac", "preflight", "tiny", "--device", "cpu"])
    strict = runner.invoke(
        app,
        ["mac", "preflight", "tiny", "--device", "cpu", "--strict"],
    )

    assert non_strict.exit_code == 0
    assert strict.exit_code == 1
