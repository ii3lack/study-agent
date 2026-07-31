"""harness —— 把一个 Task 喂给 Runner 跑一遍，收集成 RunResult。

依赖注入：client 从外部传入。测试塞 FakeClient（免费、确定），
正式评估塞真 Client（src.ai.Client）。harness 本体一行不改。

副作用说明：真跑会用到项目的 work_space/（agent 的 sandbox 区）。
harness 跑前清空 task.clean_files 声明的文件、跑后做快照、再清理。
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

from src.agent.runner import Runner, ToolStart
from src.agent.message import AssistantMessage, Message, SystemMessage
from src.agent.state import AgentState, DEFAULT_SYSTEM_PROMPT
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

from evals.types import RunResult, Task

# 模型优先从环境变量 MODEL 读（在 .env 里配置），没设就用这个兜底
_FALLBACK_MODEL = "glm-4.7"

_TOOLS = [read_file_tool, write_file_tool, list_files_tool, edit_file_tool]
_TOOL_FNS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "edit_file": edit_file,
}


def run_task(
    task: Task,
    client,
    *,
    model: str | None = None,
    max_turns: int = 8,
) -> RunResult:
    """跑一个 task，返回 RunResult。"""
    model = model or os.getenv("MODEL", _FALLBACK_MODEL)
    work_space = Path.cwd() / "work_space"
    work_space.mkdir(parents=True, exist_ok=True)

    # ---- setup：清空声明的文件，保证干净起点 ----
    for name in task.clean_files:
        (work_space / name).unlink(missing_ok=True)

    try:
        state = _fresh_state(model)
        runner = Runner(
            client=client, tools=_TOOLS, tool_fns=_TOOL_FNS, max_turns=max_turns
        )
        events = list(runner.run(state, task.user_input))

        tool_calls = [
            {"name": e.name, "arguments": e.arguments}
            for e in events
            if isinstance(e, ToolStart)
        ]
        final_answer = _last_assistant_content(state.messages)
        files = _snapshot(work_space)
    finally:
        # ---- teardown：清理 eval 产生的文件，保持 work_space 整洁 ----
        for name in task.clean_files:
            (work_space / name).unlink(missing_ok=True)

    return RunResult(
        final_answer=final_answer,
        tool_calls=tool_calls,
        files=files,
        events=events,
    )


def _fresh_state(model: str) -> AgentState:
    now = datetime.now().isoformat(timespec="seconds")
    return AgentState(
        session_id="eval",
        created_at=now,
        updated_at=now,
        model=model,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        messages=[SystemMessage(content=DEFAULT_SYSTEM_PROMPT)],
    )


def _last_assistant_content(messages: list[Message]) -> str:
    """取最后一条有正文的 assistant 消息作为最终回答。"""
    for msg in reversed(messages):
        if isinstance(msg, AssistantMessage) and msg.content:
            return msg.content
    return ""


def _snapshot(work_space: Path) -> dict[str, str]:
    """把 work_space 下的文件读成 {相对路径: 内容}。"""
    files: dict[str, str] = {}
    for path in sorted(work_space.rglob("*")):
        if path.is_file():
            rel = path.relative_to(work_space).as_posix()
            files[rel] = path.read_text(encoding="utf-8", errors="replace")
    return files
