from pathlib import Path
import yaml

from core.configuration import ConfigurationService, ConfigurationContext
from core.config.settings import settings

def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)

def test_root_yaml_override(tmp_path: Path, monkeypatch):
    # --- create fake project structure
    root_cfg = {"settings": {"MODEL": "gpt-test"}}
    _write_yaml(tmp_path / "config.yaml", root_cfg)

    # --- service pointing at tmp project
    service = ConfigurationService(project_root=tmp_path)
    cfg = service.get_config()

    assert cfg["MODEL"] == "gpt-test"

def test_customer_yaml_merging(tmp_path: Path):
    # root config
    _write_yaml(tmp_path / "config.yaml", {"settings": {"TOP_K": 5}})

    # customer config
    cust_yaml = tmp_path / "customers" / "acme" / "config" / "acme.yaml"
    _write_yaml(cust_yaml, {"settings": {"TOP_K": 9, "MODEL": "acme-model"}})

    service = ConfigurationService(project_root=tmp_path)
    ctx = ConfigurationContext(customer_id="acme")
    cfg = service.get_config(ctx)

    # customer overrides root
    assert cfg["TOP_K"] == 9
    assert cfg["MODEL"] == "acme-model"

def test_runtime_overrides(tmp_path: Path):
    service = ConfigurationService(project_root=tmp_path, user_overrides={"NEW_KEY": 123})
    cfg = service.get_config()
    assert cfg["NEW_KEY"] == 123