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
from zai import ZhipuAiClient
from zai.core._streaming import StreamResponse
from zai.types.chat import ChatCompletionChunk

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

# 把 .env 里的变量加载进 os.environ（返回 True/False，不是字典！）
load_dotenv()

# 用 os.getenv 从环境变量读取 API Key
api_key = os.getenv("API_KEY")
if not api_key:
    raise RuntimeError("未找到 API_KEY，请在 .env 文件中配置")

client = ZhipuAiClient(api_key=api_key)  # 请填写您自己的 API Key

work_space_dir = os.path.join(os.getcwd(), "work_space")

tools = [read_markdown_tool, write_markdown_tool, list_files_tool]

agent_init = {
    "role": "system",
    "content": f"你是专业视觉创作团队，请使用中文回答用户的问题, 你工作的文件路径是 {work_space_dir}， 不要有任何指令想要操作这个区域外的任何文件",
}

messages = [agent_init]


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


def stream_chat(response) -> tuple[str, str]:
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

    return answer, reasoning


def chat_with_agent(messages):
    while True:
        response = client.chat.completions.create(
            model="glm-4.7",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            stream=False,
            thinking={"type": "enabled"},
            max_tokens=65536,
            temperature=1.0,
        )
        message = response.choices[0].message  # 非流式:直接拿 message

        if message.tool_calls:  # 模型要调工具
            messages.append(message)  # ① 先存这条带 tool_calls 的 assistant 消息
            for tool_call in message.tool_calls:  # ② 逐个执行(别写死 get_weather!)
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)
                try:
                    result = TOOL_FUNCTIONS[name](**args)
                except Exception as e:
                    result = f"工具 {name} 执行错误：{e}"
                messages.append(
                    {  # ③ 存工具结果
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": str(result),
                    }
                )
            continue  # ④ 回去再问模型
        else:  # 无工具调用 = 最终答案
            console.print(Markdown(message.content))
            messages.append({"role": "assistant", "content": message.content})
            break


def main() -> None:
    """命令行多轮对话主循环。"""
    console.print("您好！我是 dry-light，为视觉创作者打造的 AI Agent 应用")
    console.print("按 Ctrl+C 退出")
    while True:
        ask = input("USER:  ")
        messages.append({"role": "user", "content": ask})
        chat_with_agent(messages)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        console.print(e)
        console.print("\n👋 再见！")
