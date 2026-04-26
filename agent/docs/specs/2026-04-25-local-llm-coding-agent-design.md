# Local LLM Coding Agent — Design Spec

Date: 2026-04-25

## Overview

A CLI chat loop agent that uses a local LLM (vllm or llama.cpp) to autonomously write code and invoke tools in order to build software projects. Built on smolagents `CodeAgent`. The agent operates inside a configured project directory and requires explicit user approval before any action that crosses the boundary (external network, paths outside project dir, destructive operations).

## Goals

- Write code and invoke tools autonomously to complete open-ended project-building tasks
- Support vllm and llama.cpp backends interchangeably via OpenAI-compatible API
- Sandboxed by default: operates inside a configured project directory
- Confirm before destructive or out-of-boundary actions
- Persistent configuration — set project dir once, not per session

## Architecture

```
user input → CLI chat loop → CodeAgent (smolagents)
                                    ↓
                             Safety Layer
                            /             \
                   in-boundary           out-of-boundary
                   (auto-allow)          (prompt user y/n)
                        ↓                      ↓
                     Tools               proceed or abort
               (file, shell, git,
                web search)
```

The `CodeAgent` writes Python code to invoke tools rather than making JSON function calls. This gives it the flexibility to compose tool calls, loop, and handle errors — well-suited for open-ended project-building tasks.

## Configuration

Stored at `~/.config/coding-agent/config.toml`. Set once via `agent config set <key> <value>`.

```toml
[workspace]
project_dir = "/path/to/your/project"

[backend]
base_url = "http://localhost:8000/v1"
model = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"
```

- Switching between vllm and llama.cpp requires only changing `base_url` and `model`.
- No per-session configuration needed.

## Tools

All tool calls are routed through the safety layer before execution.

| Tool | Purpose | Safety check |
|---|---|---|
| `FileReadTool` | Read files, list directories | Path must be inside project dir |
| `FileWriteTool` | Create, edit, delete files | Delete → confirm; outside project dir → confirm |
| `ShellTool` | Execute bash commands | Destructive patterns (rm -rf, kill, etc.) → confirm |
| `GitTool` | git add/commit/push/branch/reset | push/force-push/reset --hard → confirm |
| `WebSearchTool` | DuckDuckGo search (no API key) | Always prompts once per session (external network) |

### Safety Layer Rules

1. **Path boundary:** Any file path argument is resolved to an absolute path and checked against `project_dir`. If outside, prompt user.
2. **Destructive pattern:** Shell commands are scanned for known-dangerous patterns. Git operations are classified by risk level.
3. **External network:** Web search prompts once per session for approval; subsequent searches in the same session are auto-allowed.

## CLI Interface

```
$ agent "Build a FastAPI app with a /health endpoint and a /users CRUD"

[Agent] Planning task...
[Tool] FileWriteTool: create main.py
[Tool] ShellTool: pip install fastapi uvicorn
[Tool] ShellTool: uvicorn main:app --reload
...
[Agent] Done. Created main.py and requirements.txt. Server runs on port 8000.

> (follow-up input or Ctrl+C to exit)
```

- Streaming output: reasoning and tool calls are printed as they happen.
- Confirmation prompts pause execution and wait for `y/n`.
- Follow-up messages continue in the same conversation context.

## Project Structure

```
agent/
├── main.py          # CLI entry point, chat loop, streaming
├── config.py        # Load/save ~/.config/coding-agent/config.toml
├── agent.py         # smolagents CodeAgent setup, model connection
├── safety.py        # Boundary checks, confirmation prompts
├── tools/
│   ├── file.py      # FileReadTool, FileWriteTool
│   ├── shell.py     # ShellTool
│   ├── git.py       # GitTool
│   └── search.py    # WebSearchTool (DuckDuckGo)
├── docs/
│   └── specs/
│       └── 2026-04-25-local-llm-coding-agent-design.md
└── pyproject.toml
```

## Dependencies

- `smolagents` — agent framework and CodeAgent
- `openai` — OpenAI-compatible client for vllm/llama.cpp
- `duckduckgo-search` — web search (no API key)
- `tomli` / `tomllib` — config file parsing
- `rich` — CLI output formatting

## Error Handling

- If the model backend is unreachable on startup, print a clear error with the configured `base_url` and exit.
- If a tool call fails, the agent receives the error as output and can retry or take a different approach — smolagents handles this automatically.
- If the user declines a confirmation prompt, the tool call is aborted and the agent is notified so it can adjust its plan.

## Out of Scope

- Multi-agent collaboration
- Persistent memory across sessions (agent starts fresh each session)
- Authentication or multi-user support
- Web UI
