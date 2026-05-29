from __future__ import annotations

from dflasher.speculators_bridge import SpeculatorsRecipe, render_speculators_script


def test_legacy_speculators_script_uses_dflasher_hidden_server(tmp_path):
    script = render_speculators_script(
        SpeculatorsRecipe(
            source_model="MiniMaxAI/MiniMax-M2.7",
            output_dir=tmp_path,
            target_layer_ids=(1, 13, 24),
            trust_remote_code=True,
        )
    )

    assert "python -m dflasher.hidden_server" in script
    assert "scripts/launch_vllm.py" not in script
    assert "--hidden-states-path \"$HIDDEN_STATES_DIR\"" in script
    assert "TRUST_REMOTE_CODE_ARG=--trust-remote-code" in script
    assert "scripts/prepare_data.py" in script
    assert "--seq-length 8192 \\\n  $TRUST_REMOTE_CODE_ARG" in script
    assert "--draft-vocab-size 8192" in script
