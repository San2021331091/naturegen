import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    from app import storage
    monkeypatch.setattr(storage, 'DATA', tmp_path)
    return storage
