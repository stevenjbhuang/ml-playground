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
