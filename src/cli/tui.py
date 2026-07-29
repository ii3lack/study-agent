"""Agent 命令行交互界面 —— 订阅 Runner 的事件流并渲染。

设计沿用 Runner 的解耦思路：本模块只做「Event → 屏幕」，
不知道 Runner / Client / SessionStore 的存在。给它一个事件迭代器即可。

用法：
    cli = AgentCLI()
    for event in runner.run(state, user_input):
        cli.render(event)
"""

from __future__ import annotations

import json
from typing import Iterable

from rich.console import Console
from rich.panel import Panel

from src.agent.runner import (
    Error,
    Event,
    RunEnd,
    ToolResult,
    ToolStart,
    UserToken,
)

# 工具结果超过这个长度就截断，避免刷屏
_MAX_RESULT_CHARS = 800


class AgentCLI:
    """把 Runner 的事件流渲染到终端。

    - reasoning token → 暗色斜体，前缀 💭，流式追加
    - content token   → 正文，流式追加
    - ToolStart       → 一行 🔧 工具名(参数)
    - ToolResult      → 绿色 Panel（过长自动截断）
    - Error           → 红色 Panel
    - RunEnd          → 收尾换行
    """

    def __init__(
        self,
        console: Console | None = None,
        *,
        show_reasoning: bool = True,
    ) -> None:
        self.console = console or Console()
        self.show_reasoning = show_reasoning
        self._mid_line = False  # 是否正处在一行流式输出的中间
        self._last_kind: str | None = None  # 上一个 token 的类型

    # ---- 对外入口 ----

    def render(self, event: Event) -> None:
        """渲染单个事件，按类型分发。"""
        if isinstance(event, UserToken):
            self._render_token(event)
        elif isinstance(event, ToolStart):
            self._render_tool_start(event)
        elif isinstance(event, ToolResult):
            self._render_tool_result(event)
        elif isinstance(event, Error):
            self._end_line()
            self.console.print(
                Panel(event.message, title="❌ 错误", border_style="red")
            )
            self._mid_line = False
            self._last_kind = None
        elif isinstance(event, RunEnd):
            self._end_line()
        # TurnStart / TurnEnd：暂不可视化，留作扩展（如轮次进度条）

    def render_all(self, events: Iterable[Event]) -> None:
        """消费整个事件流。"""
        for event in events:
            self.render(event)

    # ---- 内部 ----

    def _end_line(self) -> None:
        """若正处在流式行中间，先补一个换行，避免块级内容覆盖它。"""
        if self._mid_line:
            self.console.print()
            self._mid_line = False

    def _render_token(self, token: UserToken) -> None:
        if token.kind == "reasoning":
            if not self.show_reasoning:
                return
            if self._last_kind != "reasoning":
                # 新一段思考：另起一行 + 打前缀
                self._end_line()
                self.console.print("💭 ", style="dim italic", end="")
                self._mid_line = True
            self.console.print(token.text, style="dim italic", end="")
            self._last_kind = "reasoning"
        else:  # content
            if self._last_kind == "reasoning":
                self._end_line()  # 思考结束，正文另起一行
            self.console.print(token.text, end="")
            self._mid_line = True
            self._last_kind = "content"

    def _render_tool_start(self, event: ToolStart) -> None:
        self._end_line()
        try:
            args = json.dumps(event.arguments, ensure_ascii=False)
        except (TypeError, ValueError):
            args = str(event.arguments)
        self.console.print(f"🔧 {event.name}({args})", style="cyan")
        self._mid_line = False
        self._last_kind = None

    def _render_tool_result(self, event: ToolResult) -> None:
        content = event.content
        if len(content) > _MAX_RESULT_CHARS:
            content = (
                content[:_MAX_RESULT_CHARS]
                + f"\n…（已截断，原长 {len(event.content)} 字符）"
            )
        self.console.print(
            Panel(
                content,
                title=f"📄 {event.name} 结果",
                border_style="green",
                expand=False,
            )
        )
        self._mid_line = False
        self._last_kind = None
