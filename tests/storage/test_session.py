"""SessionStore 测试 —— 每个测试一个事实,用 tmp_sessions_dir 隔离。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from src.agent.message import SystemMessage, UserMessage
from src.storage.session import (
    SessionError,
    SessionInfo,
    SessionNotFound,
    SessionStore,
)


@pytest.fixture
def store(tmp_sessions_dir: Path) -> SessionStore:
    return SessionStore(sessions_dir=tmp_sessions_dir)


def _sys(content: str = "你是助手") -> SystemMessage:
    return SystemMessage(content=content)


def _user(content: str) -> UserMessage:
    return UserMessage(content=content)


# ---- create ----
def test_create_returns_session_id(store: SessionStore) -> None:
    """create 应返回非空 session_id。"""
    sid = store.create_session("夏日重现", "glm-4.7", [_sys()])
    assert isinstance(sid, str) and sid


def test_create_writes_index_file(store: SessionStore, tmp_sessions_dir: Path) -> None:
    """create 应在 <sessions_dir>/<session_id>/index.json 写文件。"""
    sid = store.create_session("test", "glm-4.7", [_sys()])
    assert (tmp_sessions_dir / sid / "index.json").exists()


def test_create_rejects_empty_messages(store: SessionStore) -> None:
    with pytest.raises(SessionError, match="不能为空"):
        store.create_session("x", "glm-4.7", [])


def test_create_rejects_non_system_first(store: SessionStore) -> None:
    with pytest.raises(SessionError, match="第一条必须是 system"):
        store.create_session("x", "glm-4.7", [_user("hi")])


# ---- list ----
def test_list_empty_returns_empty_list(store: SessionStore) -> None:
    assert store.list_sessions() == []


def test_list_returns_all_created(store: SessionStore) -> None:
    store.create_session("a", "glm-4.7", [_sys()])
    store.create_session("b", "glm-4.7", [_sys()])
    store.create_session("c", "glm-4.7", [_sys()])
    result = store.list_sessions()
    assert len(result) == 3
    assert all(isinstance(item, SessionInfo) for item in result)


def test_list_ordered_by_updated_at_desc(store: SessionStore) -> None:
    """最新创建的排在最前。"""
    sid_a = store.create_session("first", "glm-4.7", [_sys()])
    # 时间戳精确到秒,人为 sleep 保证 mtime 递增
    import time

    time.sleep(1.05)
    sid_c = store.create_session("third", "glm-4.7", [_sys()])
    result = store.list_sessions()
    assert result[0].session_id == sid_c  # 最新的最前
    assert result[-1].session_id == sid_a  # 最旧的最后


def test_list_ignores_dirs_without_index(
    store: SessionStore, tmp_sessions_dir: Path
) -> None:
    """没有 index.json 的目录应该被忽略(不崩)。"""
    store.create_session("a", "glm-4.7", [_sys()])  # 合法
    (tmp_sessions_dir / "garbage_dir").mkdir()  # 非法
    result = store.list_sessions()
    assert len(result) == 1


# ---- load ----
def test_load_returns_full_data(store: SessionStore) -> None:
    sid = store.create_session("夏日重现", "glm-4.7", [_sys(), _user("hi")])
    data = store.load_session(sid)
    assert data["session_name"] == "夏日重现"
    assert data["model"] == "glm-4.7"
    assert data["messages"] == [_sys(), _user("hi")]


def test_load_missing_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound):
        store.load_session("nonexistent-id")


# ---- delete ----
def test_delete_removes_directory(store: SessionStore, tmp_sessions_dir: Path) -> None:
    sid = store.create_session("a", "glm-4.7", [_sys()])
    target = tmp_sessions_dir / sid
    assert target.exists()
    store.delete_session(sid)
    assert not target.exists()


def test_delete_missing_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound):
        store.delete_session("nonexistent-id")


# ---- save ----
def test_save_updates_messages_and_updated_at(store: SessionStore) -> None:
    """save_session 应回写 messages 并刷新 updated_at。"""
    import time

    sid = store.create_session("a", "glm-4.7", [_sys()])
    time.sleep(1.05)  # 时间戳精确到秒,人为 sleep 保证可比较
    new_messages = [_sys(), _user("hi")]
    store.save_session(sid, new_messages)
    data = store.load_session(sid)
    assert data["messages"] == new_messages
    assert data["updated_at"] > data["created_at"]


def test_save_missing_raises(store: SessionStore) -> None:
    with pytest.raises(SessionNotFound):
        store.save_session("nonexistent-id", [_sys()])


def test_save_rejects_non_system_first(store: SessionStore) -> None:
    sid = store.create_session("a", "glm-4.7", [_sys()])
    with pytest.raises(SessionError, match="第一条必须是 system"):
        store.save_session(sid, [_user("hi")])
