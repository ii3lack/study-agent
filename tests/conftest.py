# tests/conftest.py
"""全局 pytest fixture。"""

import pytest
from pathlib import Path


@pytest.fixture
def tmp_sessions_dir(tmp_path: Path) -> Path:
    """每个测试一个独立的 sessions 目录。"""
    p = tmp_path / "sessions"
    p.mkdir()
    return p
