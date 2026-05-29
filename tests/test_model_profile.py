from __future__ import annotations

from dflasher.model_profile import (
    fallback_model_profile_from_name,
    infer_family,
    target_layer_ids_for_policy,
)


def test_speculators_default_layers_match_public_training_example():
    assert target_layer_ids_for_policy(36, "qwen3_dense", "speculators") == (2, 18, 33)


def test_zlab_linspace_layers_match_public_checkpoint_patterns():
    assert target_layer_ids_for_policy(36, "qwen3_dense", "zlab5") == (1, 9, 17, 25, 33)
    assert target_layer_ids_for_policy(64, "qwen3_5_dense", "zlab5") == (
        1,
        16,
        31,
        46,
        61,
    )
    assert target_layer_ids_for_policy(61, "kimi", "zlab6") == (1, 12, 24, 35, 47, 58)


def test_family_detection_handles_requested_non_qwen_models():
    assert infer_family("minimaxai/minimax-m2.7", "minimax_m2") == "minimax"
    assert infer_family("deepseek-ai/deepseek-v3.2", "deepseek_v32") == "deepseek"
    assert infer_family("qwen/qwen3.6-27b", "qwen3_5_text") == "qwen3_6_dense"


def test_name_fallback_keeps_known_zlab_mapping_when_config_is_gated():
    profile = fallback_model_profile_from_name(
        "MiniMaxAI/MiniMax-M2.7",
        RuntimeError("gated"),
    )

    assert profile.family == "minimax"
    assert profile.known_zlab_draft_model == "z-lab/MiniMax-M2.7-DFlash"
    assert profile.backend_support["zlab_vllm"].status == "preview"
