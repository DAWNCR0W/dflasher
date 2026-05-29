# Contributing

Thanks for taking a look at dflasher. The project is experimental, so small,
well-tested changes are the easiest to review.

## Development Setup

```bash
git clone https://github.com/dawncr0w/dflasher.git
cd dflasher
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality Checks

Run these before opening a pull request:

```bash
ruff check .
pytest -q
pip check
```

## Pull Requests

- Keep pull requests focused on one behavior or cleanup.
- Add or update tests when behavior changes.
- Document hardware assumptions for CUDA, MPS, MLX, or vLLM changes.
- Do not commit generated checkpoints, hidden-state caches, local runs, or model
  artifacts.

## Project Scope

dflasher aims to provide a practical CLI for building and testing
DFlash-style draft models. The local DFlash-lite path is meant for correctness
experiments. The official CUDA path delegates real vLLM/Speculators checkpoint
training to the upstream toolchain.
