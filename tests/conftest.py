# tests/conftest.py
import pytest
import shutil
from pathlib import Path

@pytest.fixture
def sample_row():
    """A sample row for use in loader/cleaner/chunker tests."""
    return {"id": 1, "text": "Hello world!"}

@pytest.fixture(autouse=True)
def isolate_fs(tmp_path: Path, monkeypatch):
    """Prevent tests from accidentally touching real project files."""
    monkeypatch.chdir(tmp_path)
    yield
    # cleanup
    shutil.rmtree(tmp_path, ignore_errors=True)
