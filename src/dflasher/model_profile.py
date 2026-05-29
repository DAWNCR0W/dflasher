from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from transformers import AutoConfig

LayerPolicy = Literal[
    "auto",
    "speculators",
    "dflash5",
    "zlab5",
    "zlab6",
    "zlab-linspace5",
    "zlab-linspace6",
]
BackendName = Literal["speculators", "zlab_vllm", "zlab_sglang", "zlab_transformers", "zlab_mlx"]
SupportStatus = Literal[
    "supported",
    "experimental",
    "preview",
    "requires_custom_backend",
    "coming_soon",
    "unsupported",
    "unknown",
]


REQUIRED_SPECULATORS_FIELDS = (
    "vocab_size",
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "initializer_range",
    "rms_norm_eps",
)

SPECULATORS_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "hidden_size": ("n_embd", "d_model"),
    "intermediate_size": ("n_inner", "ffn_hidden_size"),
    "num_hidden_layers": ("n_layer", "num_layers"),
    "num_attention_heads": ("n_head", "n_heads"),
    "num_key_value_heads": ("num_attention_heads", "n_head", "n_heads"),
    "max_position_embeddings": ("n_positions", "n_ctx", "seq_length"),
    "hidden_act": ("hidden_activation", "activation_function"),
}


@dataclass(frozen=True)
class BackendSupport:
    status: SupportStatus
    detail: str


@dataclass(frozen=True)
class FamilyDefaults:
    family: str
    label: str
    speculators_status: SupportStatus
    speculators_detail: str
    zlab_vllm_status: SupportStatus
    zlab_vllm_detail: str
    zlab_sglang_status: SupportStatus
    zlab_sglang_detail: str
    zlab_transformers_status: SupportStatus
    zlab_transformers_detail: str
    zlab_mlx_status: SupportStatus
    zlab_mlx_detail: str
    recommended_draft_arch: str
    speculators_layer_policy: LayerPolicy
    zlab_layer_policy: LayerPolicy
    zlab_context_layer_count: int
    zlab_num_speculative_tokens: int
    default_block_size: int
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelProfile:
    source_model: str
    family: str
    family_label: str
    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int | None
    num_hidden_layers: int | None
    vocab_size: int | None
    missing_speculators_fields: tuple[str, ...]
    recommended_draft_arch: str
    speculators_layer_policy: LayerPolicy
    zlab_layer_policy: LayerPolicy
    zlab_context_layer_count: int
    zlab_num_speculative_tokens: int
    default_block_size: int
    known_zlab_draft_model: str | None
    backend_support: dict[BackendName, BackendSupport]
    notes: tuple[str, ...]

    @property
    def can_train_with_speculators(self) -> bool:
        return not self.missing_speculators_fields and self.backend_support[
            "speculators"
        ].status in {
            "supported",
            "experimental",
            "preview",
        }

    def target_layer_ids(self, policy: LayerPolicy = "auto") -> tuple[int, ...]:
        if self.num_hidden_layers is None:
            return ()
        resolved_policy = self.speculators_layer_policy if policy == "auto" else policy
        return target_layer_ids_for_policy(self.num_hidden_layers, self.family, resolved_policy)

    def zlab_target_layer_ids(self, policy: LayerPolicy = "auto") -> tuple[int, ...]:
        if self.num_hidden_layers is None:
            return ()
        resolved_policy = self.zlab_layer_policy if policy == "auto" else policy
        return target_layer_ids_for_policy(self.num_hidden_layers, self.family, resolved_policy)


KNOWN_ZLAB_DRAFTS: tuple[tuple[str, str], ...] = (
    ("gemma-4-31b-it", "z-lab/gemma-4-31B-it-DFlash"),
    ("gemma-4-26b-a4b-it", "z-lab/gemma-4-26B-A4B-it-DFlash"),
    ("minimax-m2.7", "z-lab/MiniMax-M2.7-DFlash"),
    ("minimax-m2.5", "z-lab/MiniMax-M2.5-DFlash"),
    ("kimi-k2.6", "z-lab/Kimi-K2.6-DFlash"),
    ("kimi-k2.5", "z-lab/Kimi-K2.5-DFlash"),
    ("qwen3.6-27b", "z-lab/Qwen3.6-27B-DFlash"),
    ("qwen3.6-35b-a3b", "z-lab/Qwen3.6-35B-A3B-DFlash"),
    ("qwen3.5-4b", "z-lab/Qwen3.5-4B-DFlash"),
    ("qwen3.5-9b", "z-lab/Qwen3.5-9B-DFlash"),
    ("qwen3.5-27b", "z-lab/Qwen3.5-27B-DFlash"),
    ("qwen3.5-35b-a3b", "z-lab/Qwen3.5-35B-A3B-DFlash"),
    ("qwen3.5-122b-a10b", "z-lab/Qwen3.5-122B-A10B-DFlash"),
    ("gpt-oss-20b", "z-lab/gpt-oss-20b-DFlash"),
    ("gpt-oss-120b", "z-lab/gpt-oss-120b-DFlash"),
    ("qwen3-coder-next", "z-lab/Qwen3-Coder-Next-DFlash"),
    ("qwen3-coder-30b-a3b", "z-lab/Qwen3-Coder-30B-A3B-DFlash"),
    ("qwen3-4b", "z-lab/Qwen3-4B-DFlash-b16"),
    ("qwen3-8b", "z-lab/Qwen3-8B-DFlash-b16"),
    ("llama-3.1-8b-instruct", "z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat"),
)


def resolve_model_profile(
    source_model: str,
    trust_remote_code: bool = False,
    *,
    allow_name_fallback: bool = False,
) -> ModelProfile:
    try:
        config = AutoConfig.from_pretrained(source_model, trust_remote_code=trust_remote_code)
    except Exception as exc:
        if allow_name_fallback:
            return fallback_model_profile_from_name(source_model, exc)
        raise

    if hasattr(config, "text_config"):
        config = config.text_config

    model_type = getattr(config, "model_type", "unknown") or "unknown"
    source_lower = source_model.lower()
    family = infer_family(source_lower, model_type)
    defaults = family_defaults(family)
    missing_fields = tuple(
        field for field in REQUIRED_SPECULATORS_FIELDS if config_field(config, field) is None
    )
    if config_field(config, "hidden_act") is None:
        missing_fields = (*missing_fields, "hidden_act|hidden_activation")

    notes = [*defaults.notes, *profile_notes(family, model_type)]
    return ModelProfile(
        source_model=source_model,
        family=family,
        family_label=defaults.label,
        model_type=model_type,
        architectures=tuple(getattr(config, "architectures", None) or ()),
        hidden_size=config_field(config, "hidden_size"),
        num_hidden_layers=config_field(config, "num_hidden_layers"),
        vocab_size=config_field(config, "vocab_size"),
        missing_speculators_fields=missing_fields,
        recommended_draft_arch=defaults.recommended_draft_arch,
        speculators_layer_policy=defaults.speculators_layer_policy,
        zlab_layer_policy=defaults.zlab_layer_policy,
        zlab_context_layer_count=defaults.zlab_context_layer_count,
        zlab_num_speculative_tokens=defaults.zlab_num_speculative_tokens,
        default_block_size=defaults.default_block_size,
        known_zlab_draft_model=known_zlab_draft_model(source_model),
        backend_support=backend_support_from_defaults(defaults),
        notes=tuple(dict.fromkeys(notes)),
    )


def fallback_model_profile_from_name(source_model: str, exc: Exception) -> ModelProfile:
    family = infer_family(source_model.lower(), "unknown")
    defaults = family_defaults(family)
    notes = (
        *defaults.notes,
        "Could not load Hugging Face config; set HF_TOKEN/login, pass --trust-remote-code, "
        "or use a local config path before training.",
        f"Config error: {type(exc).__name__}: {compact_exception_message(exc)}",
    )
    return ModelProfile(
        source_model=source_model,
        family=family,
        family_label=defaults.label,
        model_type="unknown",
        architectures=(),
        hidden_size=None,
        num_hidden_layers=None,
        vocab_size=None,
        missing_speculators_fields=REQUIRED_SPECULATORS_FIELDS,
        recommended_draft_arch=defaults.recommended_draft_arch,
        speculators_layer_policy=defaults.speculators_layer_policy,
        zlab_layer_policy=defaults.zlab_layer_policy,
        zlab_context_layer_count=defaults.zlab_context_layer_count,
        zlab_num_speculative_tokens=defaults.zlab_num_speculative_tokens,
        default_block_size=defaults.default_block_size,
        known_zlab_draft_model=known_zlab_draft_model(source_model),
        backend_support=backend_support_from_defaults(defaults),
        notes=notes,
    )


def compact_exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return repr(exc)
    return message.split("\n\n", 1)[0].replace("\n", " ")


def config_field(config, field: str):
    value = getattr(config, field, None)
    if value is not None:
        return value
    for alias in SPECULATORS_FIELD_ALIASES.get(field, ()):
        value = getattr(config, alias, None)
        if value is not None:
            return value
    return None


def infer_family(source_lower: str, model_type: str) -> str:
    model_type_lower = model_type.lower()
    haystack = f"{source_lower} {model_type_lower}"
    if "qwen3.6" in haystack:
        if any(marker in haystack for marker in ("a3b", "a10b", "moe")):
            return "qwen3_6_moe"
        return "qwen3_6_dense"
    if "qwen3.5" in haystack or "qwen3_5" in haystack:
        if any(marker in haystack for marker in ("a3b", "a10b", "moe")):
            return "qwen3_5_moe"
        return "qwen3_5_dense"
    if "qwen3-coder-next" in haystack or "qwen3_next" in haystack:
        return "qwen3_next"
    if "qwen3-coder" in haystack:
        return "qwen3_moe"
    if model_type_lower == "qwen3" or "qwen3-" in haystack or "/qwen3" in haystack:
        if any(marker in haystack for marker in ("a3b", "a22b", "moe")):
            return "qwen3_moe"
        return "qwen3_dense"
    candidates = (
        ("minimax", ("minimax", "mini-max")),
        ("deepseek", ("deepseek",)),
        ("kimi", ("kimi", "moonshot")),
        ("gemma4", ("gemma-4", "gemma4")),
        ("gemma", ("gemma",)),
        ("gpt_oss", ("gpt-oss", "gpt_oss")),
        ("mistral", ("mistral",)),
        ("mixtral", ("mixtral",)),
        ("phi", ("phi-", "/phi", "phi3", "phi4")),
        ("falcon", ("falcon",)),
        ("gpt_neox", ("gpt-neox", "gpt_neox", "pythia")),
        ("gpt2", ("gpt2", "gpt-2")),
        ("starcoder", ("starcoder", "starcoder2")),
        ("baichuan", ("baichuan",)),
        ("yi", ("01-ai", "/yi-", "yi-")),
        ("llama", ("llama",)),
        ("glm", ("glm",)),
    )
    for family, needles in candidates:
        if any(needle in haystack for needle in needles):
            return family
    return model_type_lower


def family_defaults(family: str) -> FamilyDefaults:
    if family in {"qwen3_dense", "qwen3_moe", "qwen3_next"}:
        return FamilyDefaults(
            family=family,
            label="Qwen3",
            speculators_status="supported",
            speculators_detail="Speculators has a published Qwen3 DFlash training example.",
            zlab_vllm_status="supported",
            zlab_vllm_detail="z-lab publishes Qwen3 DFlash checkpoints.",
            zlab_sglang_status="supported",
            zlab_sglang_detail="z-lab documents SGLang DFlash serving for Qwen-family models.",
            zlab_transformers_status="supported",
            zlab_transformers_detail="z-lab Transformers backend supports Qwen3 and LLaMA only.",
            zlab_mlx_status="supported",
            zlab_mlx_detail="z-lab MLX backend is tested on Qwen3/Qwen3.5/Gemma-4 models.",
            recommended_draft_arch="qwen3",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab5",
            zlab_context_layer_count=5,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
        )
    if family in {"qwen3_5_dense", "qwen3_5_moe", "qwen3_6_dense", "qwen3_6_moe"}:
        return FamilyDefaults(
            family=family,
            label="Qwen3.5/Qwen3.6",
            speculators_status="experimental",
            speculators_detail=(
                "Use only if the installed transformers/vLLM stack can load the verifier."
            ),
            zlab_vllm_status="supported",
            zlab_vllm_detail="z-lab publishes Qwen3.5/Qwen3.6 DFlash checkpoints.",
            zlab_sglang_status="supported",
            zlab_sglang_detail="z-lab documents SGLang DFlash serving for Qwen-family models.",
            zlab_transformers_status="unsupported",
            zlab_transformers_detail="z-lab limits the Transformers backend to Qwen3 and LLaMA.",
            zlab_mlx_status="supported",
            zlab_mlx_detail="z-lab MLX backend is tested on Qwen3.5 models.",
            recommended_draft_arch="qwen3",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab5",
            zlab_context_layer_count=5,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
        )
    if family in {"minimax", "kimi"}:
        return FamilyDefaults(
            family=family,
            label="MiniMax/Kimi",
            speculators_status="experimental",
            speculators_detail=(
                "Requires verifier support in vLLM plus custom/gated HF config access."
            ),
            zlab_vllm_status="preview",
            zlab_vllm_detail="z-lab publishes preview DFlash checkpoints for this family.",
            zlab_sglang_status="preview",
            zlab_sglang_detail=(
                "Use z-lab/SGLang commands when the target backend supports the model."
            ),
            zlab_transformers_status="unsupported",
            zlab_transformers_detail="No z-lab Transformers backend support is documented.",
            zlab_mlx_status="unknown",
            zlab_mlx_detail="No public z-lab MLX support statement for this family.",
            recommended_draft_arch="llama",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab6",
            zlab_context_layer_count=6,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
            notes=("These models are often gated and may require --trust-remote-code.",),
        )
    if family == "deepseek":
        return FamilyDefaults(
            family=family,
            label="DeepSeek",
            speculators_status="experimental",
            speculators_detail=(
                "Only possible when vLLM and transformers fully support the verifier."
            ),
            zlab_vllm_status="coming_soon",
            zlab_vllm_detail="z-lab lists DeepSeek DFlash as coming soon.",
            zlab_sglang_status="coming_soon",
            zlab_sglang_detail="z-lab lists DeepSeek DFlash as coming soon.",
            zlab_transformers_status="unsupported",
            zlab_transformers_detail="No z-lab Transformers backend support is documented.",
            zlab_mlx_status="unknown",
            zlab_mlx_detail="No public z-lab MLX support statement for DeepSeek DFlash.",
            recommended_draft_arch="llama",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab6",
            zlab_context_layer_count=6,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
            notes=("No public z-lab DeepSeek DFlash checkpoint was available in the README.",),
        )
    if family == "gemma4":
        return FamilyDefaults(
            family=family,
            label="Gemma 4",
            speculators_status="experimental",
            speculators_detail="Speculators documents a RedHatAI Gemma DFlash checkpoint.",
            zlab_vllm_status="requires_custom_backend",
            zlab_vllm_detail="z-lab Gemma4 DFlash currently needs a temporary/custom vLLM build.",
            zlab_sglang_status="supported",
            zlab_sglang_detail=(
                "Use the z-lab SGLang path if the model/backend combination is supported."
            ),
            zlab_transformers_status="unsupported",
            zlab_transformers_detail="z-lab limits the Transformers backend to Qwen3 and LLaMA.",
            zlab_mlx_status="supported",
            zlab_mlx_detail="z-lab MLX backend is tested on Gemma-4 models.",
            recommended_draft_arch="llama",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab6",
            zlab_context_layer_count=6,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
        )
    if family == "gpt_oss":
        return FamilyDefaults(
            family=family,
            label="gpt-oss",
            speculators_status="experimental",
            speculators_detail=(
                "Requires gpt-oss verifier support and target-specific mask/vocab handling."
            ),
            zlab_vllm_status="supported",
            zlab_vllm_detail="z-lab publishes gpt-oss DFlash checkpoints.",
            zlab_sglang_status="unknown",
            zlab_sglang_detail="No specific z-lab SGLang path is documented for gpt-oss.",
            zlab_transformers_status="unsupported",
            zlab_transformers_detail="No z-lab Transformers backend support is documented.",
            zlab_mlx_status="unknown",
            zlab_mlx_detail="No public z-lab MLX support statement for gpt-oss.",
            recommended_draft_arch="llama",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab5",
            zlab_context_layer_count=5,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
        )
    if family == "llama":
        return FamilyDefaults(
            family=family,
            label="LLaMA",
            speculators_status="supported",
            speculators_detail="Llama-style verifier configs are the Speculators fallback path.",
            zlab_vllm_status="supported",
            zlab_vllm_detail="z-lab publishes a LLaMA 3.1 DFlash checkpoint.",
            zlab_sglang_status="supported",
            zlab_sglang_detail="Use z-lab/SGLang when the target backend supports the model.",
            zlab_transformers_status="supported",
            zlab_transformers_detail="z-lab Transformers backend supports Qwen3 and LLaMA only.",
            zlab_mlx_status="unknown",
            zlab_mlx_detail="No public z-lab MLX support statement for LLaMA DFlash.",
            recommended_draft_arch="llama",
            speculators_layer_policy="speculators",
            zlab_layer_policy="zlab5",
            zlab_context_layer_count=5,
            zlab_num_speculative_tokens=15,
            default_block_size=16,
        )
    return FamilyDefaults(
        family=family,
        label=family,
        speculators_status="experimental",
        speculators_detail=(
            "Best-effort generic path: may train only if transformers, vLLM, and "
            "Speculators can load this verifier."
        ),
        zlab_vllm_status="unknown",
        zlab_vllm_detail="No known z-lab DFlash checkpoint mapping is bundled.",
        zlab_sglang_status="unknown",
        zlab_sglang_detail="No known z-lab DFlash checkpoint mapping is bundled.",
        zlab_transformers_status="unsupported",
        zlab_transformers_detail="No z-lab Transformers backend support is documented.",
        zlab_mlx_status="unknown",
        zlab_mlx_detail="No public z-lab MLX support statement for this family.",
        recommended_draft_arch="llama",
        speculators_layer_policy="speculators",
        zlab_layer_policy="zlab5",
        zlab_context_layer_count=5,
        zlab_num_speculative_tokens=15,
        default_block_size=16,
    )


def backend_support_from_defaults(defaults: FamilyDefaults) -> dict[BackendName, BackendSupport]:
    return {
        "speculators": BackendSupport(defaults.speculators_status, defaults.speculators_detail),
        "zlab_vllm": BackendSupport(defaults.zlab_vllm_status, defaults.zlab_vllm_detail),
        "zlab_sglang": BackendSupport(defaults.zlab_sglang_status, defaults.zlab_sglang_detail),
        "zlab_transformers": BackendSupport(
            defaults.zlab_transformers_status,
            defaults.zlab_transformers_detail,
        ),
        "zlab_mlx": BackendSupport(defaults.zlab_mlx_status, defaults.zlab_mlx_detail),
    }


def known_zlab_draft_model(source_model: str) -> str | None:
    normalized = normalize_model_name(source_model)
    for needle, draft_model in KNOWN_ZLAB_DRAFTS:
        if needle in normalized:
            return draft_model
    return None


def normalize_model_name(source_model: str) -> str:
    return (
        source_model.lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace("--", "-")
        .strip()
    )


def profile_notes(family: str, model_type: str) -> tuple[str, ...]:
    notes = []
    if family in {"minimax", "kimi", "deepseek", "glm"}:
        notes.append(
            "This family may require trust_remote_code, gated weights, or a vLLM backend "
            "that already supports the target model."
        )
    if model_type not in {
        "qwen3",
        "qwen3_5_text",
        "qwen3_5_moe_text",
        "qwen3_moe",
        "qwen3_next",
        "llama",
        "gemma",
        "gemma3",
        "gemma4",
        "gemma4_text",
        "gpt_oss",
        "minimax_m2",
        "kimi_k2",
        "deepseek_v3",
        "deepseek_v32",
        "deepseek_v4",
    }:
        notes.append(
            "Speculators may still train if the config exposes the required decoder fields."
        )
    return tuple(notes)


def target_layer_ids_for_policy(
    num_hidden_layers: int,
    family: str,
    policy: LayerPolicy = "auto",
) -> tuple[int, ...]:
    if num_hidden_layers <= 0:
        return ()
    if policy == "auto":
        policy = family_defaults(family).speculators_layer_policy
    policy = normalize_layer_policy(policy)
    if policy == "speculators":
        return _speculators_default(num_hidden_layers)
    if policy == "dflash5":
        return _even_layers(num_hidden_layers, count=5, start=2, end_offset=2)
    if policy == "zlab5":
        return _even_layers(num_hidden_layers, count=5, start=1, end_offset=3)
    if policy == "zlab6":
        return _even_layers(num_hidden_layers, count=6, start=1, end_offset=3)
    raise ValueError(f"Unsupported layer policy: {policy}")


def normalize_layer_policy(policy: LayerPolicy) -> LayerPolicy:
    if policy == "zlab-linspace5":
        return "zlab5"
    if policy == "zlab-linspace6":
        return "zlab6"
    return policy


def _speculators_default(num_hidden_layers: int) -> tuple[int, ...]:
    if num_hidden_layers < 5:
        return _even_layers(
            num_hidden_layers,
            count=min(3, num_hidden_layers),
            start=0,
            end_offset=1,
        )
    return (2, num_hidden_layers // 2, num_hidden_layers - 3)


def _even_layers(
    num_hidden_layers: int,
    count: int,
    start: int,
    end_offset: int,
) -> tuple[int, ...]:
    if count <= 1:
        return (max(0, min(num_hidden_layers - 1, num_hidden_layers // 2)),)
    start = max(0, min(start, num_hidden_layers - 1))
    end = max(start, num_hidden_layers - end_offset)
    span = end - start
    return tuple(int(round(start + (index * span) / (count - 1))) for index in range(count))
