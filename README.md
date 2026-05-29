# dflasher

> [!WARNING]
> This repository is incomplete and has not been fully tested yet. Treat it as
> experimental research/developer tooling, not production-ready infrastructure.

[![CI](https://github.com/dawncr0w/dflasher/actions/workflows/ci.yml/badge.svg)](https://github.com/dawncr0w/dflasher/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-experimental-orange.svg)](README.md)

`dflasher` trains and tests DFlash-style speculative draft models for Hugging Face
causal language models.

The local implementation is a practical **DFlash-lite generic drafter**:

- the source model is frozen and acts as the target/verifier;
- selected target hidden states condition a small draft Transformer;
- the draft model predicts a block of future tokens from one clean anchor token;
- generation is target-verified greedy speculative decoding, so accepted output is
  checked against the target model and should match target greedy tokens exactly.

The project has four practical paths:

- `dflasher build`: the product-style entry point. It accepts a source model and
  writes a draft directory to `--out`. On Linux/CUDA it builds an official
  vLLM/Speculators DFlash checkpoint; on Mac it builds a local DFlash-lite draft.
- `dflasher train`: a local DFlash-lite trainer that runs on CPU/MPS/CUDA and is
  useful for correctness experiments.
- `dflasher mac`: Mac-friendly wrappers around the local PyTorch/MPS trainer,
  plus z-lab MLX script generation for Apple Silicon inference.
- `dflasher official`: an orchestration layer for the public vLLM Speculators
  DFlash pipeline. This is the path that produces vLLM/Speculators checkpoints
  with `config.json` and `model.safetensors`.

## Install

```bash
git clone https://github.com/dawncr0w/dflasher.git
cd dflasher
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

For the official Speculators client wrappers:

```bash
pip install -e ".[dev,official]"
```

For Mac local DFlash-lite training on Apple Silicon:

```bash
pip install -e ".[dev,mac]"
```

For the z-lab MLX runtime, use a separate Apple Silicon environment and install
the z-lab package:

```bash
pip install -e ".[zlab-mlx]"
```

For actual vLLM serving/training on Linux/CUDA, install the vLLM extra in that
CUDA environment:

```bash
pip install -e ".[dev,official,vllm]"
```

The `vllm` extra targets vLLM `0.20.1+`, which is the version range z-lab
documents for core DFlash serving support. Some z-lab public checkpoints still
need custom builds, such as the Gemma4 Docker image or SWA support branch called
out by the z-lab README.

## Quick smoke test

```bash
dflasher smoke --source sshleifer/tiny-gpt2 --workdir ./runs/smoke --device cpu
```

## Build from a source model

Use `build` when you want the CLI contract directly: source model in, draft
model directory out.

```bash
dflasher build sshleifer/tiny-gpt2 \
  --backend lite \
  --texts-file examples/train_texts.txt \
  --out ./runs/tiny-gpt2-draft \
  --max-steps 30 \
  --device cpu
```

For CUDA/vLLM official DFlash output:

```bash
dflasher build Qwen/Qwen3-0.6B \
  --backend cuda \
  --out ./runs/qwen3-0.6b-dflash-draft \
  --workspace ./runs/qwen3-0.6b-official \
  --speculators-repo /path/to/speculators \
  --max-samples 5000 \
  --seq-length 8192
```

Use `--plan-only` to write the CUDA script/manifest without executing training.

## Train a draft model

The normal trainer requires real data. Use `--texts-file` or `--dataset`; the
tiny built-in dataset is only available behind `--allow-builtin-data` for smoke
or debug runs.

```bash
dflasher train sshleifer/tiny-gpt2 \
  --texts-file examples/train_texts.txt \
  --out ./runs/tiny-gpt2-draft \
  --block-size 4 \
  --draft-layers 1 \
  --draft-hidden-size 64 \
  --heads 2 \
  --batch-size 2 \
  --max-steps 30 \
  --device cpu
```

This local trainer uses target logits as the teacher by default:

- `--loss-fn kl_div` aligns the draft distribution to the frozen target distribution.
- `--loss-fn ce` trains against the target argmax token.

## Mac DFlash paths

Mac support is split into two explicit paths so the hardware expectations stay
clear:

- local DFlash-lite training uses this project and PyTorch MPS (`dflasher mac`);
- z-lab DFlash MLX uses z-lab's public draft checkpoints and MLX runtime
  (`dflasher zlab mlx-script` or the `dflasher mac zlab-mlx-script` alias).

Run the Mac preflight first:

```bash
dflasher mac preflight Qwen/Qwen3.5-4B --device mps
```

Create a local DFlash-lite train/eval script for MPS:

```bash
dflasher mac plan sshleifer/tiny-gpt2 \
  --workspace ./runs/mac-tiny-gpt2 \
  --texts-file examples/train_texts.txt \
  --device mps \
  --max-steps 30

bash ./runs/mac-tiny-gpt2/run_mac_dflash_lite.sh
```

The generated script simply wraps the already-tested local trainer/evaluator:

```text
python -m dflasher.cli train ... --device mps
python -m dflasher.cli eval ... --device mps
```

Or build a local Mac draft directly:

```bash
dflasher build sshleifer/tiny-gpt2 \
  --backend mac-lite \
  --texts-file examples/train_texts.txt \
  --out ./runs/tiny-gpt2-mac-draft \
  --device mps \
  --max-steps 30
```

For z-lab public DFlash checkpoints on Apple Silicon, generate a small MLX script:

```bash
dflasher zlab mlx-script Qwen/Qwen3.5-4B \
  --out ./runs/qwen35_mlx.py \
  --prompt "How many positive whole-number divisors does 196 have?"

python ./runs/qwen35_mlx.py
```

Or print the z-lab benchmark command:

```bash
dflasher zlab mlx-benchmark-command Qwen/Qwen3.5-4B \
  --dataset gsm8k \
  --max-samples 128 \
  --enable-thinking
```

For families where dflasher cannot confirm public z-lab MLX support, the MLX
script and benchmark commands require an explicit `--force`.

The Mac DFlash-lite path creates `dflasher` draft directories. The z-lab MLX path
does not train a new draft; it runs an existing z-lab DFlash checkpoint via
`from dflash.model_mlx import load, load_draft, stream_generate`.

## Official vLLM Speculators DFlash path

The official path follows the public Speculators flow:

```text
prepare_data.py
python -m dflasher.hidden_server
data_generation_offline.py hidden-state cache
train.py --speculator-type dflash
vllm serve checkpoint_best
dflasher official benchmark
```

This requires a Linux/CUDA environment with `speculators` and `vllm` installed.
On a Mac CPU/MPS environment, `preflight` will intentionally fail the CUDA/vLLM
checks.

The generated script starts `python -m dflasher.hidden_server` instead of calling
the upstream `launch_vllm.py` directly, so `--trust-remote-code` also reaches the
hidden-state server's `AutoConfig.from_pretrained(...)` call for custom-config
families such as MiniMax/Kimi. Current Speculators `prepare_data.py` also
supports `--trust-remote-code`, so dflasher passes it through there when the
global flag is enabled.

```bash
dflasher official preflight Qwen/Qwen3-0.6B \
  --speculators-repo /path/to/speculators \
  --static-only
```

For MiniMax, DeepSeek, Qwen3.5/3.6, Gemma4, gpt-oss, and unknown future
families, dflasher only proceeds by default when the family is marked supported.
Use `--allow-experimental` when your local Transformers/vLLM/Speculators stack
can actually load that verifier. This keeps the CLI generic without pretending
that every model is verified on every backend.

Inspect any source model before planning:

```bash
dflasher official inspect-model Qwen/Qwen3.6-27B --trust-remote-code
dflasher official inspect-model MiniMaxAI/MiniMax-M2.7 --trust-remote-code
dflasher official inspect-model deepseek-ai/DeepSeek-V3.2 --trust-remote-code
```

`inspect-model` does not assume Qwen3. It builds a family profile from the
Hugging Face config when available, then reports:

- Speculators compatibility and required decoder fields;
- z-lab vLLM/SGLang/Transformers/MLX support status;
- default Speculators target layers, for example `2, N//2, N-3`;
- default z-lab public-checkpoint layers, for example
  `round(linspace(1, N-3, k))`;
- known z-lab draft model IDs when dflasher can map them from the source model.

Current family handling:

| Family | Examples | dflasher default |
| --- | --- | --- |
| Qwen3 dense/MoE/Coder/Next | `qwen3`, `qwen3_moe`, `qwen3_next` | Speculators supported; z-lab style supported |
| Qwen3.5/Qwen3.6 | `qwen3_5_text`, Qwen3.6 model IDs | z-lab style first; Speculators experimental |
| MiniMax/Kimi | `minimax_m2`, `kimi_k2` | z-lab preview; Speculators experimental |
| DeepSeek | `deepseek_v3`, `deepseek_v32`, `deepseek_v4` | marked coming soon/experimental until public DFlash checkpoints are available |
| Gemma 4 | `gemma4_text` | z-lab custom-vLLM/SGLang/MLX path; Speculators experimental |
| gpt-oss | `gpt_oss` | z-lab supported; Speculators experimental |
| LLaMA | `llama` | Speculators and z-lab style supported |

For known z-lab checkpoints, dflasher can print a serving command directly:

```bash
dflasher zlab serve-command Qwen/Qwen3.6-27B --backend vllm --trust-remote-code
dflasher zlab serve-command MiniMaxAI/MiniMax-M2.7 --backend sglang --trust-remote-code --force
```

The z-lab command helper follows the current z-lab README conventions: vLLM
commands include an attention backend by default, Gemma4 commands include the
draft-side `flash_attn` setting and warn that a Gemma4-capable vLLM build/Docker
image is required, and SGLang commands include the long-context env flag plus the
draft attention/backend scheduler flags from the public examples.

If a z-lab serving backend is only preview/unknown for a family, pass `--force`
to print the command anyway.

Generate the full 5k-sample Qwen3-0.6B plan:

```bash
dflasher official plan Qwen/Qwen3-0.6B \
  --workspace ./runs/qwen3-0.6b-official \
  --speculators-repo /path/to/speculators \
  --max-samples 5000 \
  --seq-length 8192 \
  --epochs 5 \
  --mode offline-cache \
  --block-size 8 \
  --max-anchors 3072 \
  --draft-layers 5 \
  --draft-vocab-size 8192 \
  --draft-arch llama \
  --python-bin python \
  --vllm-gpus 0 \
  --train-gpus 0
```

For large or custom verifier models, pass vLLM options through explicitly:

```bash
dflasher official plan MiniMaxAI/MiniMax-M2.7 \
  --trust-remote-code \
  --allow-experimental \
  --vllm-arg=--tensor-parallel-size \
  --vllm-arg=4 \
  --serve-arg=--tensor-parallel-size \
  --serve-arg=4
```

`--draft-arch` defaults to `llama` for vLLM serving compatibility with current
Speculators docs. Pass `--draft-arch qwen3` only when your installed Speculators
and vLLM stack explicitly supports that draft architecture end to end.
`official plan` runs static model/repo/script checks by default. Add
`--check-environment` to require CUDA/package checks before writing the script,
or `--skip-preflight` only for offline script generation.

Run the generated script in a CUDA environment:

```bash
bash ./runs/qwen3-0.6b-official/run_official_dflash.sh
```

Or build directly to a final official draft output directory:

```bash
dflasher build Qwen/Qwen3-0.6B \
  --backend cuda \
  --out ./runs/qwen3-0.6b-dflash-draft \
  --workspace ./runs/qwen3-0.6b-official \
  --speculators-repo /path/to/speculators \
  --max-samples 5000 \
  --seq-length 8192
```

Or run stages from the manifest:

```bash
dflasher official run-stage prepare \
  --manifest ./runs/qwen3-0.6b-official/dflasher_official_manifest.json
```

After training:

```bash
dflasher official inspect-checkpoint \
  ./runs/qwen3-0.6b-official/checkpoints/checkpoint_best

dflasher official serve-command \
  ./runs/qwen3-0.6b-official/checkpoints/checkpoint_best

vllm serve ./runs/qwen3-0.6b-official/checkpoints/checkpoint_best --port 8000

dflasher official benchmark \
  ./runs/qwen3-0.6b-official/checkpoints/checkpoint_best \
  --base-url http://localhost:8000/v1 \
  --output-json ./runs/qwen3-0.6b-official/benchmark.json

dflasher official validate-equivalence \
  --target-base-url http://localhost:8001/v1 \
  --draft-base-url http://localhost:8000/v1 \
  --target-model Qwen/Qwen3-0.6B \
  --draft-model ./runs/qwen3-0.6b-official/checkpoints/checkpoint_best
```

## Verify exact target equivalence

```bash
dflasher eval sshleifer/tiny-gpt2 ./runs/tiny-gpt2-draft \
  --prompts-file examples/prompts.txt \
  --max-new-tokens 12 \
  --device cpu
```

## Generate

```bash
dflasher generate sshleifer/tiny-gpt2 ./runs/tiny-gpt2-draft \
  --prompt "Speculative decoding" \
  --max-new-tokens 24 \
  --device cpu
```

## Research basis

DFlash trains a lightweight block diffusion drafter for speculative decoding. The
paper describes extracting selected target hidden states, conditioning the drafter
with those context features, sampling masked response blocks around anchor tokens,
using a position-decayed cross entropy loss, and sharing the target embedding/LM
head while keeping the target frozen.

The local `dflasher train` path follows those implementation ideas where they are
architecture-neutral, but keeps the first version simple:

- generic PyTorch cross-attention instead of z-lab's Qwen3-specific KV injection;
- greedy-only target-equivalent verification;
- no FlexAttention sparse block training yet.

The `dflasher official` path delegates DFlash KV-injection architecture,
hidden-state extraction, sparse/block training, reduced vocab mapping, and
checkpoint format to vLLM Speculators. For non-Qwen families, the resolver keeps
the workflow explicit: it will plan when the model config exposes the fields that
Speculators needs, and it will surface gated/custom-backend requirements instead
of pretending every model can be trained on every machine.

Useful references:

- DFlash paper: https://arxiv.org/abs/2602.06036
- z-lab DFlash repository: https://github.com/z-lab/dflash
- z-lab model card example: https://huggingface.co/z-lab/Qwen3-8B-DFlash-b16
