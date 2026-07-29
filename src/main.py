# main.py
"""主程序入口"""

import os
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
    # 此前在循环内每轮重建：对话记忆每轮被丢弃，且每轮向磁盘
    # 泄漏一个只含 system 消息的孤儿会话目录。
    client = Client()
    store = SessionStore(sessions_dir=Path.cwd() / "storage" / "sessions")

    s_id = store.create_session(
        name="Study AI Agent",
        model=os.getenv("MODEL", "glm-5.2"),
        messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
    )
    s_info = store.load_session(s_id)

    # messages 只含 system；用户消息由 runner.run(state, user_input)
    # 负责追加一次 —— 不要在初始列表里再放一份，否则会重复。
    state = AgentState(
        session_id=s_info["session_id"],
        created_at=s_info["created_at"],
        updated_at=s_info["updated_at"],
        model=s_info["model"],
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
        metadata={},
    )
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


if __name__ == "__main__":
    main()
