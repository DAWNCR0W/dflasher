from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from dflasher.data import BUILTIN_TEXTS
from dflasher.generation import dflash_generate, greedy_generate, load_runtime


@dataclass(frozen=True)
class EvalResult:
    prompts: int
    exact_matches: int
    mean_acceptance: float


def load_prompts(prompts_file: str | None) -> list[str]:
    if prompts_file is None:
        return BUILTIN_TEXTS[:4]
    prompts = [line.strip() for line in Path(prompts_file).read_text().splitlines() if line.strip()]
    if not prompts:
        raise ValueError("No prompts were loaded.")
    return prompts


def evaluate(
    source_model: str,
    draft_dir: str | Path,
    prompts_file: str | None = None,
    max_new_tokens: int = 24,
    device: str = "auto",
    torch_dtype: str = "float32",
    trust_remote_code: bool = False,
) -> EvalResult:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative.")
    tokenizer, target, draft, runtime_device = load_runtime(
        source_model,
        draft_dir,
        device,
        torch_dtype,
        trust_remote_code=trust_remote_code,
    )
    prompts = load_prompts(prompts_file)
    exact_matches = 0
    acceptance_values = []

    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(runtime_device)
        with torch.no_grad():
            baseline = greedy_generate(target, input_ids, max_new_tokens, tokenizer.eos_token_id)
            speculative, stats = dflash_generate(
                draft,
                target,
                input_ids,
                max_new_tokens,
                tokenizer.eos_token_id,
            )
        if torch.equal(baseline, speculative):
            exact_matches += 1
        acceptance_values.append(stats.mean_acceptance)

    mean_acceptance = sum(acceptance_values) / len(acceptance_values)
    return EvalResult(
        prompts=len(prompts),
        exact_matches=exact_matches,
        mean_acceptance=mean_acceptance,
    )
