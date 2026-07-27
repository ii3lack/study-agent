"""会话存储 —— 一个会话一个目录,目录里放 index.json。

设计:
- session_id = uuid4 字符串,同时是目录名
- 每个 session 是 sessions_dir/<session_id>/index.json
- 目录结构给将来留口子(messages/、images/、trace.json 等)
- 全部用 Path 操作,不用 os.path
"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


class SessionError(Exception):
    """会话错误。"""


class SessionNotFound(FileNotFoundError):
    """会话不存在。"""


@dataclass(frozen=True)
class SessionInfo:
    """会话元数据 —— list() 的返回元素。"""
    session_id: str
    session_name: str
    created_at: str
    updated_at: str
    model: str


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class SessionStore:
    """JSON 文件型会话存储。"""

    def __init__(self, sessions_dir: Path) -> None:
        self.sessions_dir = sessions_dir
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

    def create_session(
        self,
        name: str,
        model: str,
        messages: list[dict],
    ) -> str:
        """创建会话,返回 session_id。

        约束:messages 至少 1 条,且第一条 role 必须是 system。
        """
        if not messages:
            raise SessionError("messages 不能为空")
        if messages[0].get("role") != "system":
            raise SessionError("messages 第一条必须是 system")

        session_id = uuid.uuid4().hex
        path = self.sessions_dir / session_id
        path.mkdir(parents=True, exist_ok=False)

        data = {
            "session_id": session_id,
            "session_name": name,
            "created_at": _now(),
            "updated_at": _now(),
            "model": model,
            "messages": messages,
        }
        (path / "index.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return session_id

    def list_sessions(self) -> list[SessionInfo]:
        """列出所有会话,按 updated_at 倒序(最新在前)。"""
        items: list[SessionInfo] = []
        for d in self.sessions_dir.iterdir():
            index = d / "index.json"
            if not (d.is_dir() and index.exists()):
                continue
            data = json.loads(index.read_text(encoding="utf-8"))
            items.append(
                SessionInfo(
                    session_id=data["session_id"],
                    session_name=data["session_name"],
                    created_at=data["created_at"],
                    updated_at=data["updated_at"],
                    model=data["model"],
                )
            )
        items.sort(key=lambda x: x.updated_at, reverse=True)
        return items

    def load_session(self, session_id: str) -> dict:
        """加载会话原始数据(messages、name、model 等)。"""
        index = self.sessions_dir / session_id / "index.json"
        if not index.exists():
            raise SessionNotFound(f"Session {session_id} not found")
        return json.loads(index.read_text(encoding="utf-8"))

    def delete_session(self, session_id: str) -> None:
        """删除整个会话目录。"""
        path = self.sessions_dir / session_id
        if not path.exists():
            raise SessionNotFound(f"Session {session_id} not found")
        shutil.rmtree(path)
