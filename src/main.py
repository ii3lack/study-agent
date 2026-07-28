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
from src.agent.state import (
    AgentState,
    DEFAULT_SYSTEM_PROMPT,
    StateInvariantError,
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
    console.print("Start Study AI Agent")
    try:
        while True:
            user_input = input("USER: ")
            client = Client()
            work_space = os.getcwd()
            sessions_dir = Path(work_space + "/storage/sessions")
            state = SessionStore(sessions_dir=sessions_dir)
            s_id = state.create_session(
                name="Study AI Agent",
                model="glm-4.7",
                messages=[
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    }
                ],
            )
            s_info = state.load_session(s_id)
            state = AgentState(
                session_id=s_info["session_id"],
                created_at=s_info["created_at"],
                updated_at=s_info["updated_at"],
                model=s_info["model"],
                system_prompt=DEFAULT_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "system",
                        "content": DEFAULT_SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": "请写一个关于机器学习的程序",
                    },
                ],
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
            events = runner.run(state, user_input)
            for event in events:
                console.print(event)
    except KeyboardInterrupt:
        console.print("Study AI Agent is stopping...")


if __name__ == "__main__":
    main()
