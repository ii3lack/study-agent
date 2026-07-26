from pathlib import Path
import os
import uuid


def uuid4() -> str:
    """生成一个唯一的会话 ID。"""

    return str(uuid.uuid4())


def list_sessions(sessions_dir: Path | None = None) -> list[Path]:
    """列出指定目录下所有会话 JSON 文件，按修改时间倒序。

    Args:
        sessions_dir: 会话目录；为 None 时使用 ${SESSIONS_DIR}
                      或当前工作目录下的 "sessions" 子目录。
    """
    if sessions_dir is None:
        env = os.getenv("SESSIONS_DIR", "sessions")
        sessions_dir = (Path.cwd() / env).resolve()

    if not sessions_dir.exists():
        return []
    return sorted(
        sessions_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True
    )


def create_session(session_name: str, sessions_dir: Path | None = None) -> Path:
    """创建一个新的会话 JSON 文件。

    Args:
        session_name: 会话名称（不含扩展名）。
        sessions_dir: 会话目录；为 None 时使用 ${SESSIONS_DIR}
                      或当前工作目录下的 "sessions" 子目录。
    """
    if sessions_dir is None:
        env = os.getenv("SESSIONS_DIR", "sessions")
        sessions_dir = (Path.cwd() / env).resolve()

    sessions_dir.mkdir(parents=True, exist_ok=True)
    session_file = sessions_dir / f"{uuid4()}.json"
    if not session_file.exists():
        session_file.write_text("[]", encoding="utf-8")
    return session_file


if __name__ == "__main__":
    print(uuid4())
