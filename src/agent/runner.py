"""Runner — 事件流驱动的 agent loop（ReAct 循环）。

从 zai_sse.py 的 chat_with_agent() + stream_chat() 拆分而来：
- 循环调度 + 消息管理 → 本文件（Runner）
- rich 渲染 → zai_sse.py（未来改为 render_events()）

设计三不原则：
- 不 import 具体 Client —— client 从构造参数注入
- 不 save —— 调用方决定何时持久化
- 不渲染 —— 只 yield Event，谁订阅谁处理
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Iterator

from src.agent.state import AgentState


# ============================================================
# Event 类型 —— Runner 和外部世界的唯一通信方式
# ============================================================
# 订阅者（CLI / Web / Trace）拿到这些事件后自己决定怎么渲染/记录。
# Runner 不关心谁在听。


@dataclass
class Event:
    """所有事件的基类。"""


@dataclass
class TurnStart(Event):
    """第 N 轮循环开始。"""

    turn: int


@dataclass
class UserToken(Event):
    """模型吐出了一个 token（最频繁的事件）。

    kind="reasoning" → 思考碎片（对应 zai_sse.py 的 💭 思考面板）
    kind="content"   → 正文碎片（对应 zai_sse.py 的 Markdown 渲染）
    """

    text: str
    kind: str  # "reasoning" | "content"


@dataclass
class ToolStart(Event):
    """即将调用一个工具。"""

    name: str
    arguments: dict


@dataclass
class ToolResult(Event):
    """工具执行完毕，这是它的返回内容。"""

    name: str
    content: str


@dataclass
class TurnEnd(Event):
    """第 N 轮循环结束（模型给出了最终回答，没有 tool_calls）。"""

    turn: int


@dataclass
class RunEnd(Event):
    """整个 run() 结束。无论正常退出还是 Error，最后一定有它。"""


@dataclass
class Error(Event):
    """出了可恢复的错误（如 max_turns 超限）。"""

    message: str


# ============================================================
# Runner —— ReAct 循环的最小实现
# ============================================================


class Runner:
    """事件流驱动的 agent loop。

    对照 zai_sse.py：
    - chat_with_agent() 的 for 循环  → run()
    - stream_chat() 的流式累积       → _stream_one_turn()
    - rich Live 渲染                 → 删掉，改为 yield Event

    参数全部从外部注入（依赖注入），测试时可以换成 FakeClient。
    """

    def __init__(
        self,
        client,  # 任何有 .chat(**kwargs) 方法的对象
        tools: list[dict],  # OpenAI 格式的工具 schema 列表
        tool_fns: dict,  # {"read_file": read_file, ...}
        *,
        max_turns: int = 5,
    ) -> None:
        self.client = client
        self.tools = tools
        self.tool_fns = tool_fns
        self.max_turns = max_turns

    # ---- 核心方法 ----

    def run(
        self, state: AgentState, user_input: str | None = None
    ) -> Iterator[Event]:
        """执行 agent 循环，yield 事件流。

        对应 zai_sse.py 第 140-199 行的 chat_with_agent()，
        但把 messages.append（渲染）换成了 yield Event。

        调用方用法：
            for event in runner.run(state, user_input="你好"):
                # 自己决定怎么处理每个 event
        """
        # ---- 对应 zai_sse.py 第 208 行：用户消息入列 ----
        if user_input is not None:
            state.messages.append({"role": "user", "content": user_input})

        # ---- 对应 zai_sse.py 第 145 行：for _turn in range(max_turns) ----
        for turn in range(self.max_turns):
            yield TurnStart(turn=turn)

            # 调一次 LLM，流式拿 token，同时累积 tool_calls
            for token in self._stream_one_turn(state):
                yield token  # 透传给订阅者

            # 流结束后，从 self 上读累积结果
            answer = self._answer
            reasoning = self._reasoning
            tool_calls_acc = self._tool_calls_acc

            # ---- 对应 zai_sse.py 第 131-136 行：只有思考没有正文 ----
            if not answer and reasoning and not tool_calls_acc:
                answer = reasoning
                reasoning = ""

            # ---- 对应 zai_sse.py 第 158-162 行：无工具 = 最终答案 ----
            if not tool_calls_acc:
                state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "reasoning_content": reasoning,
                    }
                )
                yield TurnEnd(turn=turn)
                break  # ← 自然退出

            # ---- 对应 zai_sse.py 第 165-181 行：存 assistant + tool_calls ----
            state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
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

            # ---- 对应 zai_sse.py 第 183-195 行：逐个执行工具 ----
            for s in tool_calls_acc.values():
                name = s["name"]
                try:
                    args = json.loads(s["arguments"])
                except json.JSONDecodeError as e:
                    result = f"参数解析失败: {e}"
                else:
                    yield ToolStart(name=name, arguments=args)
                    try:
                        result = self.tool_fns[name](**args)
                    except Exception as e:
                        # ★ 关键设计：tool 异常当消息，不当中断
                        # 模型会看到错误内容，有机会自我纠正
                        result = f"工具 {name} 执行错误: {e}"

                yield ToolResult(name=name, content=str(result))
                state.messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": s["id"],
                        "content": str(result),
                    }
                )
            # 回到 for 顶部 → 带着工具结果再问 LLM

        else:
            # for 跑满 max_turns 没 break → 死循环保护
            # 对应 zai_sse.py 第 199 行
            yield Error(
                message=f"达到最大步数 {self.max_turns}，强制结束"
            )

        yield RunEnd()  # 无论如何，最后一定有 RunEnd

    # ---- 内部方法 ----

    def _stream_one_turn(self, state: AgentState) -> Iterator[UserToken]:
        """调一次 chat() 流式响应，yield UserToken，同时在 self 上累积。

        对应 zai_sse.py 第 71-137 行的 stream_chat()，
        但删掉了所有 rich 渲染，只保留累积逻辑 + yield。

        为什么挂在 self 上而不是 return？
        因为 Python generator 不能同时 yield 和 return 复杂数据。
        run() 消费完这个 generator 后，从 self._answer 等读取结果。
        """
        # 初始化本轮累积器
        self._answer = ""
        self._reasoning = ""
        self._tool_calls_acc: dict[int, dict] = {}  # 和 zai_sse.py 第 79 行一样

        response = self.client.chat(
            model=state.model,  # ← 从 state 读，不写死 "glm-4.7"
            messages=state.messages,
            tools=self.tools,
            tool_choice="auto",
            stream=True,
            thinking={"type": "enabled"},
            max_tokens=65536,
            temperature=1.0,
        )

        for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            # ---- 思考碎片（对应 zai_sse.py 第 91-95 行）----
            piece = getattr(delta, "reasoning_content", None)
            if piece:
                self._reasoning += piece
                yield UserToken(text=piece, kind="reasoning")

            # ---- 正文碎片（对应 zai_sse.py 第 98-106 行）----
            if delta.content:
                self._answer += delta.content
                yield UserToken(text=delta.content, kind="content")

            # ---- 工具调用碎片（对应 zai_sse.py 第 109-128 行）----
            # 流式协议下 tool_calls 是分片到达的：
            #   第 1 片带 id + name，后续片带 arguments 碎片
            # 用 index 做 key 累积成完整调用
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    slot = self._tool_calls_acc.setdefault(
                        tc.index, {"id": "", "name": "", "arguments": ""}
                    )
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["arguments"] += tc.function.arguments
