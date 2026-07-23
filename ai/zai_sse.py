import os
from typing import cast

from dotenv import load_dotenv
from rich.console import Console, Group
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text
from zai import ZhipuAiClient
from zai.core._streaming import StreamResponse
from zai.types.chat import ChatCompletionChunk

console = Console()

# 把 .env 里的变量加载进 os.environ（返回 True/False，不是字典！）
load_dotenv()

# 用 os.getenv 从环境变量读取 API Key
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API_KEY，请在 .env 文件中配置")

client = ZhipuAiClient(api_key=api_key)  # 请填写您自己的 API Key

agent_init = {
    "role": "system",
    "content": "你是专业视觉创作团队，请使用中文回答用户的问题",
}

messages = [agent_init]


import time

console = Console()


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


def stream_chat(response) -> str:
    """流式渲染：折叠的思考过程 + Markdown 正文，返回完整回答。"""
    reasoning = ""
    answer = ""
    folded = False
    start = time.perf_counter()

    with Live(
        console=console, refresh_per_second=12, vertical_overflow="ellipsis"
    ) as live:
        for chunk in response:
            delta = chunk.choices[0].delta

            # 1) 思考内容（GLM 思考模式放在 reasoning_content）
            piece = getattr(delta, "reasoning_content", None)
            if piece:
                reasoning += piece
                live.update(_thinking_panel(reasoning))
                continue

            # 2) 正式回答
            if delta.content:
                if reasoning and not folded:
                    folded = True  # 第一帧回答到来 → 折叠思考
                answer += delta.content
                parts = []
                if folded:
                    parts.append(_collapsed_thinking(time.perf_counter() - start))
                parts.append(Markdown(answer))
                live.update(Group(*parts))

    return answer


def chat_with_agent(messages: list[dict[str, str]]) -> None:
    response = cast(
        "StreamResponse[ChatCompletionChunk]",
        client.chat.completions.create(
            model="glm-4.7",
            messages=messages,
            thinking={
                "type": "enabled",  # 启用深度思考模式
            },
            stream=True,  # 启用流式输出
            max_tokens=65536,  # 最大输出 tokens
            temperature=1.0,  # 控制输出的随机性
        ),
    )
    answer = stream_chat(response)  # ← 折叠思考 + 渲染 Markdown
    messages.append({"role": "assistant", "content": answer})


def main() -> None:
    """命令行多轮对话主循环。"""
    console.print("您好！我是 dry-light，为视觉创作者打造的 AI Agent 应用")
    while True:
        ask = input("user input:  ")
        messages.append({"role": "user", "content": ask})
        chat_with_agent(messages)


if __name__ == "__main__":
    main()
