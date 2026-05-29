from __future__ import annotations

import textwrap
from pathlib import Path


def render_zlab_mlx_script(
    *,
    source_model: str,
    draft_model: str,
    prompt: str,
    block_size: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> str:
    return textwrap.dedent(
        f"""\
        #!/usr/bin/env python3
        from __future__ import annotations

        import argparse
        import sys

        DEFAULT_MODEL = {source_model!r}
        DEFAULT_DRAFT_MODEL = {draft_model!r}
        DEFAULT_PROMPT = {prompt!r}


        def render_prompt(tokenizer, text: str, enable_thinking: bool) -> str:
            messages = [{{"role": "user", "content": text}}]
            if not hasattr(tokenizer, "apply_chat_template"):
                return text
            try:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking,
                )
            except TypeError:
                return tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )


        def main() -> int:
            parser = argparse.ArgumentParser(
                description="Run z-lab DFlash MLX speculative decoding on Apple Silicon."
            )
            parser.add_argument("--model", default=DEFAULT_MODEL)
            parser.add_argument("--draft-model", default=DEFAULT_DRAFT_MODEL)
            parser.add_argument("--prompt", default=DEFAULT_PROMPT)
            parser.add_argument("--block-size", type=int, default={block_size})
            parser.add_argument("--max-tokens", type=int, default={max_tokens})
            parser.add_argument("--temperature", type=float, default={temperature})
            parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction,
                                default={enable_thinking!r})
            args = parser.parse_args()

            try:
                from dflash.model_mlx import load, load_draft, stream_generate
            except ImportError as exc:
                print(
                    "Missing z-lab DFlash MLX runtime. Install it with: "
                    "pip install -e '.[zlab-mlx]' from dflasher, or "
                    "pip install 'dflash[mlx] @ git+https://github.com/z-lab/dflash.git'",
                    file=sys.stderr,
                )
                print(f"Import error: {{exc}}", file=sys.stderr)
                return 2

            model, tokenizer = load(args.model)
            draft = load_draft(args.draft_model)
            rendered_prompt = render_prompt(tokenizer, args.prompt, args.enable_thinking)

            last_tps = 0.0
            for result in stream_generate(
                model,
                draft,
                tokenizer,
                rendered_prompt,
                block_size=args.block_size,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            ):
                print(result.text, end="", flush=True)
                last_tps = float(getattr(result, "generation_tps", last_tps))

            print(f"\\nThroughput: {{last_tps:.2f}} tok/s")
            return 0


        if __name__ == "__main__":
            raise SystemExit(main())
        """
    )


def write_zlab_mlx_script(
    out: Path,
    *,
    source_model: str,
    draft_model: str,
    prompt: str,
    block_size: int = 16,
    max_tokens: int = 256,
    temperature: float = 0.0,
    enable_thinking: bool = True,
) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        render_zlab_mlx_script(
            source_model=source_model,
            draft_model=draft_model,
            prompt=prompt,
            block_size=block_size,
            max_tokens=max_tokens,
            temperature=temperature,
            enable_thinking=enable_thinking,
        )
    )
    out.chmod(0o755)
    return out


def zlab_mlx_benchmark_command(
    *,
    source_model: str,
    draft_model: str,
    dataset: str,
    max_samples: int,
    enable_thinking: bool,
) -> list[str]:
    command = [
        "python",
        "-m",
        "dflash.benchmark",
        "--backend",
        "mlx",
        "--model",
        source_model,
        "--draft-model",
        draft_model,
        "--dataset",
        dataset,
        "--max-samples",
        str(max_samples),
    ]
    if enable_thinking:
        command.append("--enable-thinking")
    return command
