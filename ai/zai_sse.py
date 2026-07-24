import os
import json
import time
from typing import cast

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from .client import Client

from tools.markdown_tools import (
    read_markdown,
    write_markdown,
    list_files,
)  # 把函数也导入
from tools.markdown_tools import (
    read_markdown_tool,
    write_markdown_tool,
    list_files_tool,
)

TOOL_FUNCTIONS = {
    "read_markdown": read_markdown,
    "write_markdown": write_markdown,
    "list_files": list_files,
}

console = Console()


work_space_dir = os.path.join(os.getcwd(), "work_space")

tools = [read_markdown_tool, write_markdown_tool, list_files_tool]

agent_init = {
    "role": "system",
    "content": f"你是专业视觉创作团队，请使用中文回答用户的问题, 你工作的文件路径是 {work_space_dir}， 不要有任何指令想要操作这个区域外的任何文件",
}

messages = [agent_init]

ai_client = Client()


def _thinking_panel(reasoning: str) -> Panel:
    """思考进行中：只显示最新一段，避免刷屏。"""
    return Panel(
        Text(reasoning[-300:], style="dim"),
        title="💭 思考中…",
        border_style="dim",
    )


def _collapsed_thinking(elapsed: float) -> Text:
    """思考结束：折叠成一行摘要。"""
    return Text(f"💭 已深度思考 {elapsed:.1f} 秒", style="dim italic")


def stream_chat(response) -> tuple[str, str, dict]:
    """流式渲染一次回复：思考时显示思考框，正文来时折叠思考并显示正文。

    返回 (正文, 思考, 重组好的 tool_calls)。
    """
    reasoning = ""
    answer = ""
    tool_calls_acc = {}  # index -> {"id", "name", "arguments"}  ← 关键：累积碎片
    start = time.perf_counter()

    def final_view():
        """最终定型：折叠思考 + 正文；若正文为空，则把思考当正文显示。"""
        if answer:
            parts = []
            if reasoning:
                parts.append(_collapsed_thinking(time.perf_counter() - start))
            parts.append(Markdown(answer))
            return Group(*parts)
        if reasoning:  # 模型把答案放进了思考（content 为空）
            return Markdown(reasoning)
        return Markdown("_(无回复)_")

    with Live(
        console=console, refresh_per_second=12, vertical_overflow="ellipsis"
    ) as live:
        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # 思考碎片
            piece = getattr(delta, "reasoning_content", None)
            if piece:
                reasoning += piece
                live.update(_thinking_panel(reasoning))

            # 正文碎片
            if delta.content:
                answer += delta.content
                live.update(final_view())  # 正文一来就折叠思考、显示正文

            # 工具调用碎片：按 index 累积，把 name 和 arguments 拼回完整调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = tool_calls_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments  # ← 参数一片片拼接

        live.update(final_view())  # 流结束：最终定型（兜底，确保一定折叠/显示）

    return answer, reasoning, tool_calls_acc


def chat_with_agent(messages, max_turns=10):
    """单个用户问题内的 agent 循环：请求 →（调工具 → 再请求）* → 最终答案。

    max_turns 限制最多向模型请求几轮，防止模型陷入无限调工具的死循环。
    """
    for turn in range(max_turns):
        response = ai_client.chat(
            model="glm-4.7",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=True,  # ← 让"回答"流式，靠的就是它
            thinking={"type": "enabled"},
            max_tokens=65536,
            temperature=1.0,
        )
        answer, reasoning, tool_calls_acc = stream_chat(response)

        if not tool_calls_acc:  # 无工具 = 最终答案
            messages.append(
                {"role": "assistant", "content": answer, "reasoning_content": reasoning}
            )
            return

        # 有工具：① 存这轮带 tool_calls 的 assistant 消息
        messages.append(
            {
                "role": "assistant",
                "content": answer,  # 调工具时通常为空
                "tool_calls": [
                    {
                        "id": s["id"],
                        "type": "function",
                        "function": {
                            "name": s["name"],
                            "arguments": s["arguments"],
                        },
                    }
                    for s in tool_calls_acc.values()
                ],
            }
        )
        # ② 逐个执行工具，存结果
        for s in tool_calls_acc.values():
            try:
                args = json.loads(s["arguments"])
                result = TOOL_FUNCTIONS[s["name"]](**args)
            except Exception as e:
                result = f"工具 {s['name']} 执行错误：{e}"
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": s["id"],
                    "content": str(result),
                }
            )
        # ③ 回到 for 顶部，带着工具结果再问模型（turn+1）

    # for 跑满 max_turns 还没 return = 陷入工具死循环
    print(f"⚠️ 达到最大步数 {max_turns}，强制结束，避免死循环")


def main() -> None:
    """命令行多轮对话主循环。"""
    console.print("您好！我是 dry-light，为视觉创作者打造的 AI Agent 应用")
    console.print("按 Ctrl+C 退出")
    while True:
        ask = input("USER:  ")
        messages.append({"role": "user", "content": ask})
        chat_with_agent(messages)
