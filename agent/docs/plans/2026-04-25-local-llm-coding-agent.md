# Local LLM Coding Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a CLI chat-loop coding agent that uses a local LLM (vllm or llama.cpp) to autonomously write code and invoke tools, sandboxed to a configured project directory.

**Architecture:** smolagents `CodeAgent` backed by an OpenAI-compatible local API. A safety layer wraps every tool call to enforce path boundaries and confirm destructive operations. Config is stored at `~/.config/coding-agent/config.toml` and persists across sessions.

**Tech Stack:** Python 3.11+, smolagents, openai, duckduckgo-search, rich, pytest, uv

---

## File Map

| File | Responsibility |
|---|---|
| `agent/pyproject.toml` | Package metadata, deps, CLI entry point |
| `agent/coding_agent/__init__.py` | Package marker |
| `agent/coding_agent/config.py` | Load/save `~/.config/coding-agent/config.toml` |
| `agent/coding_agent/safety.py` | Path boundary checks, destructive pattern detection, confirmation prompts |
| `agent/coding_agent/tools/__init__.py` | Package marker |
| `agent/coding_agent/tools/file.py` | `FileReadTool`, `FileWriteTool`, `FileDeleteTool` |
| `agent/coding_agent/tools/shell.py` | `ShellTool` — runs bash, checks destructive patterns |
| `agent/coding_agent/tools/git.py` | `GitTool` — runs git commands, confirms risky ops |
| `agent/coding_agent/tools/search.py` | `WebSearchTool` — DuckDuckGo, prompts once per session |
| `agent/coding_agent/agent.py` | Assembles `CodeAgent` with all tools and model |
| `agent/coding_agent/main.py` | CLI entry point: `agent "task"` and `agent config set` |
| `agent/tests/test_config.py` | Config load/save/set tests |
| `agent/tests/test_safety.py` | Boundary check and pattern detection tests |
| `agent/tests/tools/test_file.py` | FileReadTool, FileWriteTool, FileDeleteTool tests |
| `agent/tests/tools/test_shell.py` | ShellTool tests |
| `agent/tests/tools/test_git.py` | GitTool tests |

---

## Task 1: Project Setup

**Files:**
- Create: `agent/pyproject.toml`
- Create: `agent/coding_agent/__init__.py`
- Create: `agent/coding_agent/tools/__init__.py`
- Create: `agent/tests/__init__.py`
- Create: `agent/tests/tools/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "coding-agent"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = [
    "smolagents>=1.13.0",
    "openai>=1.0.0",
    "duckduckgo-search>=7.0.0",
    "rich>=13.0.0",
]

[project.scripts]
agent = "coding_agent.main:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[dependency-groups]
dev = [
    "pytest>=8.0.0",
    "pytest-mock>=3.14.0",
]
```

- [ ] **Step 2: Create package skeleton**

Create these empty files (each contains only a newline):
- `agent/coding_agent/__init__.py`
- `agent/coding_agent/tools/__init__.py`
- `agent/tests/__init__.py`
- `agent/tests/tools/__init__.py`

- [ ] **Step 3: Install dependencies**

Run from `agent/`:
```bash
cd agent && uv sync
```
Expected: lockfile created, all packages installed without errors.

---

## Task 2: Config Module

**Files:**
- Create: `agent/coding_agent/config.py`
- Create: `agent/tests/test_config.py`

- [ ] **Step 1: Write failing tests**

`agent/tests/test_config.py`:
```python
from pathlib import Path
import pytest
from coding_agent.config import Config, load, save, set_value


def test_load_reads_toml(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[workspace]\nproject_dir = "/home/user/proj"\n\n[backend]\nbase_url = "http://localhost:8000/v1"\nmodel = "qwen"\n'
    )
    monkeypatch.setattr("coding_agent.config.CONFIG_PATH", config_file)
    c = load()
    assert c.project_dir == Path("/home/user/proj")
    assert c.base_url == "http://localhost:8000/v1"
    assert c.model == "qwen"


def test_load_missing_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("coding_agent.config.CONFIG_PATH", tmp_path / "missing.toml")
    with pytest.raises(FileNotFoundError, match="agent config set"):
        load()


def test_save_and_reload(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    monkeypatch.setattr("coding_agent.config.CONFIG_PATH", config_file)
    c = Config(project_dir=Path("/tmp/myproj"), base_url="http://localhost:9000/v1", model="llama")
    save(c)
    reloaded = load()
    assert reloaded.project_dir == Path("/tmp/myproj")
    assert reloaded.base_url == "http://localhost:9000/v1"
    assert reloaded.model == "llama"


def test_set_value_project_dir(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[workspace]\nproject_dir = "/old"\n\n[backend]\nbase_url = "http://localhost:8000/v1"\nmodel = "q"\n'
    )
    monkeypatch.setattr("coding_agent.config.CONFIG_PATH", config_file)
    set_value("project-dir", "/new/path")
    c = load()
    assert c.project_dir == Path("/new/path")


def test_set_value_invalid_key(tmp_path, monkeypatch):
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        '[workspace]\nproject_dir = "/p"\n\n[backend]\nbase_url = "u"\nmodel = "m"\n'
    )
    monkeypatch.setattr("coding_agent.config.CONFIG_PATH", config_file)
    with pytest.raises(ValueError, match="Unknown config key"):
        set_value("invalid-key", "value")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd agent && uv run pytest tests/test_config.py -v
```
Expected: ImportError or ModuleNotFoundError — `coding_agent.config` doesn't exist yet.

- [ ] **Step 3: Implement config.py**

`agent/coding_agent/config.py`:
```python
from __future__ import annotations
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_PATH = Path.home() / ".config" / "coding-agent" / "config.toml"


@dataclass
class Config:
    project_dir: Path
    base_url: str = "http://localhost:8000/v1"
    model: str = "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"


def load() -> Config:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Config not found at {CONFIG_PATH}.\n"
            "Run: agent config set project-dir /path/to/your/project"
        )
    with open(CONFIG_PATH, "rb") as f:
        data = tomllib.load(f)
    backend = data.get("backend", {})
    return Config(
        project_dir=Path(data["workspace"]["project_dir"]),
        base_url=backend.get("base_url", "http://localhost:8000/v1"),
        model=backend.get("model", "cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit"),
    )


def save(config: Config) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        f'[workspace]\nproject_dir = "{config.project_dir}"\n\n'
        f'[backend]\nbase_url = "{config.base_url}"\nmodel = "{config.model}"\n'
    )


def set_value(key: str, value: str) -> None:
    try:
        config = load()
    except FileNotFoundError:
        config = Config(project_dir=Path("."))
    match key:
        case "project-dir":
            config.project_dir = Path(value).expanduser().resolve()
        case "base-url":
            config.base_url = value
        case "model":
            config.model = value
        case _:
            raise ValueError(f"Unknown config key: {key!r}. Valid: project-dir, base-url, model")
    save(config)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd agent && uv run pytest tests/test_config.py -v
```
Expected: 5 tests PASSED.

---

## Task 3: Safety Layer

**Files:**
- Create: `agent/coding_agent/safety.py`
- Create: `agent/tests/test_safety.py`

- [ ] **Step 1: Write failing tests**

`agent/tests/test_safety.py`:
```python
from pathlib import Path
import pytest
from coding_agent.safety import (
    is_within_project,
    is_destructive_shell,
    is_destructive_git,
    approve_web_search,
    reset_session_state,
)


def test_path_inside_project():
    project = Path("/home/user/project")
    assert is_within_project("/home/user/project/src/main.py", project) is True
    assert is_within_project("src/main.py", project) is True  # relative path resolved against cwd, but still safe


def test_path_outside_project():
    project = Path("/home/user/project")
    assert is_within_project("/home/user/other/file.py", project) is False


def test_path_traversal_blocked():
    project = Path("/home/user/project")
    assert is_within_project("/home/user/project/../other/secret.py", project) is False


def test_destructive_shell_patterns():
    assert is_destructive_shell("rm -rf /tmp/test") is True
    assert is_destructive_shell("rm -r ./build") is True
    assert is_destructive_shell("kill 1234") is True
    assert is_destructive_shell("killall python") is True


def test_safe_shell_commands():
    assert is_destructive_shell("ls -la") is False
    assert is_destructive_shell("python main.py") is False
    assert is_destructive_shell("pip install fastapi") is False
    assert is_destructive_shell("git status") is False


def test_destructive_git_patterns():
    assert is_destructive_git("push --force origin main") is True
    assert is_destructive_git("push -f origin main") is True
    assert is_destructive_git("reset --hard HEAD~1") is True
    assert is_destructive_git("clean -f") is True
    assert is_destructive_git("branch -D old-branch") is True


def test_safe_git_commands():
    assert is_destructive_git("add -A") is False
    assert is_destructive_git('commit -m "fix bug"') is False
    assert is_destructive_git("status") is False
    assert is_destructive_git("log --oneline") is False


def test_web_search_approved_once(monkeypatch):
    reset_session_state()
    monkeypatch.setattr("builtins.input", lambda _: "y")
    assert approve_web_search() is True
    # Second call should not prompt again
    monkeypatch.setattr("builtins.input", lambda _: (_ for _ in ()).throw(AssertionError("should not prompt")))
    assert approve_web_search() is True


def test_web_search_denied(monkeypatch):
    reset_session_state()
    monkeypatch.setattr("builtins.input", lambda _: "n")
    assert approve_web_search() is False
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd agent && uv run pytest tests/test_safety.py -v
```
Expected: ImportError — `coding_agent.safety` doesn't exist yet.

- [ ] **Step 3: Implement safety.py**

`agent/coding_agent/safety.py`:
```python
from __future__ import annotations
from pathlib import Path

DESTRUCTIVE_SHELL_PATTERNS = [
    "rm -rf", "rm -r ", "rm -f ", "rmdir",
    "kill ", "killall", "pkill",
    "mkfs", "dd if=", "> /dev/", "truncate",
]

DESTRUCTIVE_GIT_PATTERNS = [
    "--force", "-f ", "reset --hard", "clean -f",
    "branch -D", "push origin :",
]

_web_search_approved: bool = False


def is_within_project(path: str | Path, project_dir: Path) -> bool:
    resolved = Path(path).expanduser().resolve()
    try:
        resolved.relative_to(project_dir.resolve())
        return True
    except ValueError:
        return False


def is_destructive_shell(command: str) -> bool:
    lower = command.lower()
    return any(pat in lower for pat in DESTRUCTIVE_SHELL_PATTERNS)


def is_destructive_git(args: str) -> bool:
    lower = args.lower()
    return any(pat in lower for pat in DESTRUCTIVE_GIT_PATTERNS)


def confirm(prompt: str) -> bool:
    try:
        response = input(f"\n[Safety] {prompt} [y/N] ").strip().lower()
        return response == "y"
    except (EOFError, KeyboardInterrupt):
        return False


def approve_web_search() -> bool:
    global _web_search_approved
    if _web_search_approved:
        return True
    _web_search_approved = confirm("Allow web search (external network access) for this session?")
    return _web_search_approved


def reset_session_state() -> None:
    global _web_search_approved
    _web_search_approved = False
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd agent && uv run pytest tests/test_safety.py -v
```
Expected: 10 tests PASSED.

---

## Task 4: File Tools

**Files:**
- Create: `agent/coding_agent/tools/file.py`
- Create: `agent/tests/tools/test_file.py`

- [ ] **Step 1: Write failing tests**

`agent/tests/tools/test_file.py`:
```python
from pathlib import Path
import pytest
from coding_agent.tools.file import FileReadTool, FileWriteTool, FileDeleteTool


@pytest.fixture
def project(tmp_path):
    return tmp_path


def test_read_file(project):
    (project / "hello.txt").write_text("hello world")
    tool = FileReadTool(project)
    result = tool.forward("hello.txt")
    assert result == "hello world"


def test_read_directory(project):
    (project / "src").mkdir()
    (project / "src" / "main.py").write_text("")
    (project / "README.md").write_text("")
    tool = FileReadTool(project)
    result = tool.forward(str(project))
    assert "README.md" in result
    assert "src" in result


def test_read_outside_project_denied(project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    tool = FileReadTool(project)
    result = tool.forward("/etc/passwd")
    assert "Aborted" in result


def test_write_file(project):
    tool = FileWriteTool(project)
    result = tool.forward("src/main.py", "print('hello')")
    assert "Written" in result
    assert (project / "src" / "main.py").read_text() == "print('hello')"


def test_write_creates_parents(project):
    tool = FileWriteTool(project)
    tool.forward("a/b/c/deep.py", "x = 1")
    assert (project / "a" / "b" / "c" / "deep.py").exists()


def test_write_outside_project_denied(project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    tool = FileWriteTool(project)
    result = tool.forward("/tmp/outside.py", "x = 1")
    assert "Aborted" in result


def test_delete_file_confirmed(project, monkeypatch):
    f = project / "delete_me.txt"
    f.write_text("bye")
    monkeypatch.setattr("builtins.input", lambda _: "y")
    tool = FileDeleteTool(project)
    result = tool.forward("delete_me.txt")
    assert "Deleted" in result
    assert not f.exists()


def test_delete_file_denied(project, monkeypatch):
    f = project / "keep_me.txt"
    f.write_text("stay")
    monkeypatch.setattr("builtins.input", lambda _: "n")
    tool = FileDeleteTool(project)
    result = tool.forward("keep_me.txt")
    assert "Aborted" in result
    assert f.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd agent && uv run pytest tests/tools/test_file.py -v
```
Expected: ImportError — `coding_agent.tools.file` doesn't exist yet.

- [ ] **Step 3: Implement tools/file.py**

`agent/coding_agent/tools/file.py`:
```python
from __future__ import annotations
from pathlib import Path
from smolagents import Tool
from coding_agent import safety


class FileReadTool(Tool):
    name = "file_read"
    description = "Read a file's contents or list a directory. Path is relative to project dir."
    inputs = {"path": {"type": "string", "description": "File or directory path"}}
    output_type = "string"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir

    def forward(self, path: str) -> str:
        p = Path(path) if Path(path).is_absolute() else self.project_dir / path
        if not safety.is_within_project(p, self.project_dir):
            if not safety.confirm(f"Read path outside project: {path}. Allow?"):
                return "Aborted: path is outside project directory."
        if p.is_dir():
            entries = sorted(p.iterdir())
            return "\n".join(str(e.relative_to(self.project_dir)) for e in entries)
        return p.read_text(encoding="utf-8")


class FileWriteTool(Tool):
    name = "file_write"
    description = "Write content to a file (creates parent dirs automatically)."
    inputs = {
        "path": {"type": "string", "description": "File path (relative to project dir)"},
        "content": {"type": "string", "description": "Content to write"},
    }
    output_type = "string"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir

    def forward(self, path: str, content: str) -> str:
        p = Path(path) if Path(path).is_absolute() else self.project_dir / path
        if not safety.is_within_project(p, self.project_dir):
            if not safety.confirm(f"Write outside project: {path}. Allow?"):
                return "Aborted: path is outside project directory."
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Written: {p}"


class FileDeleteTool(Tool):
    name = "file_delete"
    description = "Delete a file. Always asks for confirmation."
    inputs = {"path": {"type": "string", "description": "File path to delete"}}
    output_type = "string"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir

    def forward(self, path: str) -> str:
        if not safety.confirm(f"Delete file: {path}?"):
            return "Aborted: deletion cancelled."
        p = Path(path) if Path(path).is_absolute() else self.project_dir / path
        if not p.exists():
            return f"File not found: {p}"
        p.unlink()
        return f"Deleted: {p}"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd agent && uv run pytest tests/tools/test_file.py -v
```
Expected: 8 tests PASSED.

---

## Task 5: Shell Tool

**Files:**
- Create: `agent/coding_agent/tools/shell.py`
- Create: `agent/tests/tools/test_shell.py`

- [ ] **Step 1: Write failing tests**

`agent/tests/tools/test_shell.py`:
```python
from pathlib import Path
import pytest
from coding_agent.tools.shell import ShellTool


@pytest.fixture
def project(tmp_path):
    return tmp_path


def test_run_safe_command(project):
    tool = ShellTool(project)
    result = tool.forward("echo hello")
    assert "hello" in result


def test_run_in_project_dir(project):
    (project / "marker.txt").write_text("found")
    tool = ShellTool(project)
    result = tool.forward("ls")
    assert "marker.txt" in result


def test_destructive_command_denied(project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    tool = ShellTool(project)
    result = tool.forward("rm -rf .")
    assert "Aborted" in result


def test_destructive_command_allowed(project, monkeypatch, tmp_path):
    target = project / "deletable"
    target.mkdir()
    monkeypatch.setattr("builtins.input", lambda _: "y")
    tool = ShellTool(project)
    result = tool.forward(f"rm -rf {target}")
    assert "Aborted" not in result
    assert not target.exists()


def test_failed_command_includes_exit_code(project):
    tool = ShellTool(project)
    result = tool.forward("exit 1")
    assert "exit code: 1" in result


def test_stderr_included_in_output(project):
    tool = ShellTool(project)
    result = tool.forward("python3 -c \"import sys; sys.stderr.write('err\\n')\"")
    assert "err" in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd agent && uv run pytest tests/tools/test_shell.py -v
```
Expected: ImportError — `coding_agent.tools.shell` doesn't exist yet.

- [ ] **Step 3: Implement tools/shell.py**

`agent/coding_agent/tools/shell.py`:
```python
from __future__ import annotations
import subprocess
from pathlib import Path
from smolagents import Tool
from coding_agent import safety


class ShellTool(Tool):
    name = "shell"
    description = "Run a bash command in the project directory. Returns stdout, stderr, and exit code."
    inputs = {"command": {"type": "string", "description": "Bash command to execute"}}
    output_type = "string"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir

    def forward(self, command: str) -> str:
        if safety.is_destructive_shell(command):
            if not safety.confirm(f"Run potentially destructive command:\n  {command}\nAllow?"):
                return "Aborted: command cancelled."
        result = subprocess.run(
            command,
            shell=True,
            cwd=self.project_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        parts = [result.stdout]
        if result.stderr:
            parts.append(f"STDERR:\n{result.stderr}")
        if result.returncode != 0:
            parts.append(f"(exit code: {result.returncode})")
        return "\n".join(p for p in parts if p).strip() or "(no output)"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd agent && uv run pytest tests/tools/test_shell.py -v
```
Expected: 6 tests PASSED.

---

## Task 6: Git Tool

**Files:**
- Create: `agent/coding_agent/tools/git.py`
- Create: `agent/tests/tools/test_git.py`

- [ ] **Step 1: Write failing tests**

`agent/tests/tools/test_git.py`:
```python
from pathlib import Path
import subprocess
import pytest
from coding_agent.tools.git import GitTool


@pytest.fixture
def git_project(tmp_path):
    subprocess.run("git init", shell=True, cwd=tmp_path, capture_output=True)
    subprocess.run('git config user.email "test@test.com"', shell=True, cwd=tmp_path, capture_output=True)
    subprocess.run('git config user.name "Test"', shell=True, cwd=tmp_path, capture_output=True)
    return tmp_path


def test_safe_git_command(git_project):
    tool = GitTool(git_project)
    result = tool.forward("status")
    assert "branch" in result.lower() or "nothing" in result.lower()


def test_git_add_and_commit(git_project):
    (git_project / "file.txt").write_text("content")
    tool = GitTool(git_project)
    tool.forward("add file.txt")
    result = tool.forward('commit -m "add file"')
    assert "file" in result.lower() or "main" in result.lower() or "master" in result.lower()


def test_destructive_git_denied(git_project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "n")
    tool = GitTool(git_project)
    result = tool.forward("reset --hard HEAD")
    assert "Aborted" in result


def test_destructive_git_allowed(git_project, monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _: "y")
    tool = GitTool(git_project)
    result = tool.forward("reset --hard HEAD")
    assert "Aborted" not in result
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd agent && uv run pytest tests/tools/test_git.py -v
```
Expected: ImportError — `coding_agent.tools.git` doesn't exist yet.

- [ ] **Step 3: Implement tools/git.py**

`agent/coding_agent/tools/git.py`:
```python
from __future__ import annotations
import subprocess
from pathlib import Path
from smolagents import Tool
from coding_agent import safety


class GitTool(Tool):
    name = "git"
    description = "Run a git command in the project directory. E.g. args='add -A' or 'commit -m \"msg\"'."
    inputs = {"args": {"type": "string", "description": "Git subcommand and arguments"}}
    output_type = "string"

    def __init__(self, project_dir: Path) -> None:
        super().__init__()
        self.project_dir = project_dir

    def forward(self, args: str) -> str:
        if safety.is_destructive_git(args):
            if not safety.confirm(f"Run destructive git command:\n  git {args}\nAllow?"):
                return "Aborted: git command cancelled."
        result = subprocess.run(
            f"git {args}",
            shell=True,
            cwd=self.project_dir,
            capture_output=True,
            text=True,
        )
        output = (result.stdout + result.stderr).strip()
        return output or "(no output)"
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd agent && uv run pytest tests/tools/test_git.py -v
```
Expected: 4 tests PASSED.

---

## Task 7: Web Search Tool

**Files:**
- Create: `agent/coding_agent/tools/search.py`

No unit tests for this tool — it requires a live network call. Manual verification in Task 9.

- [ ] **Step 1: Implement tools/search.py**

`agent/coding_agent/tools/search.py`:
```python
from __future__ import annotations
from smolagents import Tool
from coding_agent import safety


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web with DuckDuckGo. Requires one-time approval per session."
    inputs = {"query": {"type": "string", "description": "Search query string"}}
    output_type = "string"

    def forward(self, query: str) -> str:
        if not safety.approve_web_search():
            return "Aborted: web search not approved for this session."
        from duckduckgo_search import DDGS
        results = DDGS().text(query, max_results=5)
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )
```

---

## Task 8: Agent Assembly

**Files:**
- Create: `agent/coding_agent/agent.py`

- [ ] **Step 1: Implement agent.py**

`agent/coding_agent/agent.py`:
```python
from __future__ import annotations
from smolagents import CodeAgent, OpenAIServerModel
from coding_agent.config import Config
from coding_agent.tools.file import FileReadTool, FileWriteTool, FileDeleteTool
from coding_agent.tools.shell import ShellTool
from coding_agent.tools.git import GitTool
from coding_agent.tools.search import WebSearchTool


def build_agent(config: Config) -> CodeAgent:
    model = OpenAIServerModel(
        model_id=config.model,
        api_base=config.base_url,
        api_key="not-needed",
    )
    tools = [
        FileReadTool(config.project_dir),
        FileWriteTool(config.project_dir),
        FileDeleteTool(config.project_dir),
        ShellTool(config.project_dir),
        GitTool(config.project_dir),
        WebSearchTool(),
    ]
    return CodeAgent(tools=tools, model=model, verbosity_level=1)
```

- [ ] **Step 2: Verify import chain is clean**

```bash
cd agent && uv run python -c "from coding_agent.agent import build_agent; print('OK')"
```
Expected: `OK` (no import errors).

---

## Task 9: CLI Entry Point

**Files:**
- Create: `agent/coding_agent/main.py`

- [ ] **Step 1: Implement main.py**

`agent/coding_agent/main.py`:
```python
from __future__ import annotations
import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from coding_agent import config as cfg
from coding_agent import safety

console = Console()


def run_chat(agent, initial_task: str | None = None) -> None:
    safety.reset_session_state()
    if initial_task:
        task = initial_task
    else:
        try:
            task = console.input("\n[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            return

    while task.strip():
        try:
            agent.run(task)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted.[/yellow]")
        try:
            task = console.input("\n[bold green]> [/bold green]")
        except (EOFError, KeyboardInterrupt):
            break


def cmd_config_set(args: argparse.Namespace) -> None:
    cfg.set_value(args.key, args.value)
    console.print(f"[green]Set {args.key} = {args.value}[/green]")


def cmd_run(args: argparse.Namespace) -> None:
    try:
        config = cfg.load()
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        sys.exit(1)

    console.print(Panel(
        f"Project: [bold]{config.project_dir}[/bold]\n"
        f"Model:   {config.model}\n"
        f"Backend: {config.base_url}",
        title="Coding Agent",
    ))

    try:
        from coding_agent.agent import build_agent
        agent = build_agent(config)
    except Exception as e:
        console.print(f"[red]Failed to connect to model backend at {config.base_url}:\n{e}[/red]")
        sys.exit(1)

    initial_task = " ".join(args.task) if args.task else None
    run_chat(agent, initial_task)


def main() -> None:
    parser = argparse.ArgumentParser(prog="agent", description="Local LLM coding agent")
    sub = parser.add_subparsers(dest="command")

    config_parser = sub.add_parser("config", help="Manage configuration")
    config_sub = config_parser.add_subparsers(dest="config_command")
    set_parser = config_sub.add_parser("set", help="Set a config value")
    set_parser.add_argument("key", choices=["project-dir", "base-url", "model"])
    set_parser.add_argument("value")

    parser.add_argument("task", nargs="*", help="Task to run (optional; enters chat loop if omitted)")

    args = parser.parse_args()

    if args.command == "config":
        if args.config_command == "set":
            cmd_config_set(args)
        else:
            parser.print_help()
    else:
        cmd_run(args)
```

- [ ] **Step 2: Verify CLI wiring**

```bash
cd agent && uv run agent --help
```
Expected: help text showing `agent [task]` and `agent config set`.

```bash
cd agent && uv run agent config set project-dir /tmp/test-project
```
Expected: `Set project-dir = /tmp/test-project`

---

## Task 10: Full Test Suite + Smoke Test

- [ ] **Step 1: Run all tests**

```bash
cd agent && uv run pytest -v
```
Expected: All tests PASSED (no failures, no errors).

- [ ] **Step 2: Smoke test with live backend**

Ensure vllm is running (see `serving/run.sh`), then:

```bash
cd agent && uv run agent config set project-dir /tmp/smoke-test && mkdir -p /tmp/smoke-test
cd agent && uv run agent "Create a hello.py file that prints Hello, World!"
```

Expected:
- Agent panel shows project dir and model
- Agent reasons, calls `file_write`, creates `/tmp/smoke-test/hello.py`
- Running `python /tmp/smoke-test/hello.py` prints `Hello, World!`

- [ ] **Step 3: Verify backend switching**

```bash
cd agent && uv run agent config set base-url http://localhost:8080/v1
cd agent && uv run agent config set model my-llama-model
cat ~/.config/coding-agent/config.toml
```
Expected: config file shows updated values. Switching back:
```bash
cd agent && uv run agent config set base-url http://localhost:8000/v1
cd agent && uv run agent config set model cyankiwi/Qwen3.6-35B-A3B-AWQ-4bit
```
