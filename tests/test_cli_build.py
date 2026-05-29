from __future__ import annotations

from dataclasses import dataclass

from typer.testing import CliRunner

from dflasher.cli import app


@dataclass(frozen=True)
class FakeEvalResult:
    exact_matches: int = 1
    prompts: int = 1
    mean_acceptance: float = 1.0


def test_build_mac_lite_outputs_local_draft(monkeypatch, tmp_path):
    calls = {}

    def fake_train(options):
        calls["source_model"] = options.source_model
        calls["output_dir"] = options.output_dir
        calls["allow_builtin_data"] = options.allow_builtin_data
        options.output_dir.mkdir(parents=True)
        return options.output_dir

    monkeypatch.setattr("dflasher.cli.train_draft", fake_train)
    monkeypatch.setattr("dflasher.cli.evaluate", lambda *args, **kwargs: FakeEvalResult())
    runner = CliRunner()
    out = tmp_path / "draft"

    result = runner.invoke(
        app,
        [
            "build",
            "sshleifer/tiny-gpt2",
            "--backend",
            "mac-lite",
            "--out",
            str(out),
            "--allow-builtin-data",
        ],
    )

    assert result.exit_code == 0
    assert calls["source_model"] == "sshleifer/tiny-gpt2"
    assert calls["output_dir"] == out
    assert calls["allow_builtin_data"] is True
    assert "DFlash-lite" in result.output


def test_build_cuda_plan_only_writes_official_script(tmp_path):
    runner = CliRunner()
    spec_repo = tmp_path / "speculators"
    scripts = spec_repo / "scripts"
    scripts.mkdir(parents=True)
    for name in ("prepare_data.py", "data_generation_offline.py", "train.py"):
        (scripts / name).write_text("--trust-remote-code --speculator-type dflash --draft-arch")
    work = tmp_path / "work"

    result = runner.invoke(
        app,
        [
            "build",
            "Qwen/Qwen3-0.6B",
            "--backend",
            "cuda",
            "--out",
            str(tmp_path / "draft"),
            "--workspace",
            str(work),
            "--speculators-repo",
            str(spec_repo),
            "--skip-preflight",
            "--plan-only",
        ],
    )

    assert result.exit_code == 0
    assert (work / "run_official_dflash.sh").exists()
    assert "CUDA DFlash plan" in result.output
