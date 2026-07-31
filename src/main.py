# main.py
"""主程序入口"""

import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from src.ai import Client
from src.storage.session import SessionStore
from src.agent.runner import Runner
from src.cli.tui import AgentCLI
from src.agent.state import (
    AgentState,
    DEFAULT_SYSTEM_PROMPT,
)
from src.tools.file_tools import (
    edit_file,
    edit_file_tool,
    list_files,
    list_files_tool,
    read_file,
    read_file_tool,
    write_file,
    write_file_tool,
)
from src.agent.message import SystemMessage

TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "edit_file": edit_file,
}


def main():
    console = Console()
    cli = AgentCLI(console=console)
    console.print("Start Study AI Agent")

    # 长寿命组件在启动时创建一次。
    client = Client()
    store = SessionStore(sessions_dir=Path.cwd() / "storage" / "sessions")
    model: str | None = os.getenv("MODEL", "")
    print(f"Using model: {model}")

    # 让用户挑一个旧会话恢复，或新建。恢复旧会话才拿得到大上下文，
    # 用来压测上下文管理（度量 / 压缩 / 缓存权衡）。
    state = _resume_or_create(store, cli, model)
    runner = Runner(
        client=client,
        tools=[
            read_file_tool,
            write_file_tool,
            list_files_tool,
            edit_file_tool,
        ],
        tool_fns=TOOL_FUNCTIONS,
        max_turns=3,
    )

    try:
        while True:
            user_input = input("USER: ")
            for event in runner.run(state, user_input):
                cli.render(event)
            # Runner 只改内存里的 messages，这里回写磁盘持久化
            store.save_session(state.session_id, state.messages)
    except (KeyboardInterrupt, EOFError):
        console.print("Study AI Agent is stopping...")


def _resume_or_create(store: SessionStore, cli: AgentCLI, model: str) -> AgentState:
    """启动时：列出历史会话让用户挑一个恢复，或新建。"""
    chosen_id = cli.choose_session(store.list_sessions())

    if chosen_id is None:  # 新建
        name = f"会话 {datetime.now():%Y-%m-%d %H:%M:%S}"
        chosen_id = store.create_session(
            name=name,
            model=model,
            messages=[SystemMessage(content=DEFAULT_SYSTEM_PROMPT)],
        )

    # load_session 已把 messages 反序列化成 Message 对象，直接拿来构造 state。
    info = store.load_session(chosen_id)
    return AgentState(
        session_id=info["session_id"],
        created_at=info["created_at"],
        updated_at=info["updated_at"],
        model=model,
        system_prompt=info["messages"][0].content,  # 取实际存着的 system 内容
        messages=info["messages"],
    )


if __name__ == "__main__":
    main()
