# AGENTS.md

This file provides guidance to agents when working with code in this repository.

## Project Structure

Multi-project Python workspace managed with `uv`:

| Directory | Purpose | Python | Package Manager |
|-----------|---------|--------|-----------------|
| `agent/` | CLI coding agent (smolagents) | >=3.11 | uv + hatchling |
| `llm-fine-tuning/` | LLM fine-tuning pipeline | >=3.13 | uv |
| `local-llm/` | Local LLM experiments | >=3.13 | uv |
| `serving/` | vLLM serving scripts | - | - |

## Commands

### agent/
```bash
cd agent && uv sync                          # Install dependencies
cd agent && uv run pytest                    # Run tests
cd agent && uv run pytest tests/test_config.py -v  # Single test
cd agent && uv run agent "your task"         # Run coding agent
cd agent && uv run agent config set project-dir /path/to/project  # Configure
```

### llm-fine-tuning/
```bash
cd llm-fine-tuning && uv sync                # Install dependencies (includes PyTorch CUDA 13.0)
cd llm-fine-tuning && uv run python main.py  # Run main script
```

## Critical Gotchas

- **CUDA 13.0 index**: `llm-fine-tuning/pyproject.toml` has custom PyTorch CUDA 13.0 index (`https://download.pytorch.org/whl/cu130`). Verify this URL is valid before installing.
- **Config path**: Agent config is at `~/.config/coding-agent/config.toml`, NOT in the project directory.
- **No shared venv**: Each subproject has its own `.venv`. Run commands from the project directory.
- **`.envrc` files**: `llm-fine-tuning/` and `serving/` use direnv-style `.envrc` for virtualenv activation.
