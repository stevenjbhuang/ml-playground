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
