import os
import json
import time
import pprint
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


def stream_chat(response) -> tuple[str, str, dict]:
    """流式渲染一次回复，返回 (正文, 思考, 重组好的 tool_calls)。"""
    reasoning = ""
    answer = ""
    tool_calls_acc = {}  # index -> {"id", "name", "arguments"}  ← 关键：累积碎片
    folded = False
    start = time.perf_counter()

    with Live(
        console=console, refresh_per_second=12, vertical_overflow="ellipsis"
    ) as live:
        for chunk in response:
            pprint.pprint(chunk)
            pprint.pprint("$" * 50)
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            pprint.pprint(delta)
            pprint.pprint("*" * 50)
            # 思考碎片
            piece = getattr(delta, "reasoning_content", None)
            if piece:
                reasoning += piece
                # live.update(_thinking_panel(reasoning))
                continue

            # 正文碎片
            if delta.content:
                if reasoning and not folded:
                    folded = True
                answer += delta.content
                parts = []
                if folded:
                    parts.append(_collapsed_thinking(time.perf_counter() - start))
                parts.append(Markdown(answer))
                # live.update(Group(*parts))

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
                        slot[
                            "arguments"
                        ] += tc.function.arguments  # ← 参数是一片片拼接的

    return answer, reasoning, tool_calls_acc


def chat_with_agent(messages):
    while True:
        pprint.pprint(messages)
        response = cast(
            StreamResponse,
            client.chat.completions.create(
                model="glm-4.7",
                messages=messages,
                tools=tools,
                tool_choice="auto",
                stream=True,  # ← 让"回答"流式，靠的就是它
                thinking={"type": "enabled"},
                max_tokens=65536,
                temperature=1.0,
            ),
        )
        answer, reasoning, tool_calls_acc = stream_chat(response)

        if tool_calls_acc:  # 模型要调工具
            print("模型要调工具：")
            print(tool_calls_acc)
            # ① 存这轮带 tool_calls 的 assistant 消息
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
            continue  # ③ 带着工具结果回去再问模型
        else:  # 无工具 = 最终答案
            messages.append(
                {"role": "assistant", "content": answer, "reasoning_content": reasoning}
            )
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
    except KeyboardInterrupt:
        # Ctrl+C：KeyboardInterrupt 继承自 BaseException 而非 Exception，
        # 必须单独捕获，否则会打印原始 traceback
        console.print("\n👋 再见！")
    except Exception as e:
        console.print(e)
        console.print("\n👋 再见！")
