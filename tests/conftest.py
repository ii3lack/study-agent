# tests/conftest.py
import pytest


@pytest.fixture
def tmp_sessions_dir(tmp_path):
    p = tmp_path / "storage/sessions"
    p.mkdir(parents=True)
    return p
