"""边界 mapper：线上 dict（供应商 wire 格式）↔ 领域 Message。

为什么单独一个模块、而不是 Message 上的方法？
转换认识的是"某个供应商的 JSON 形状"——这是边界的事，不是领域的事。
把它隔离在这里：wire 格式漂移（比如 GLM 的 reasoning_content）只污染这一个文件，
message.py 和 runner.py 始终干净。多厂商时，这里会裂成"每厂商一个 mapper"。
"""

from __future__ import annotations

from src.agent.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)


def message_from_api(data: dict) -> Message:
    """线上 dict → 领域 Message，按 data["role"] 分派。

    wire 形状（对照 tests/agent/test_state.py 和 storage/*/index.json）：
      system:    {"role": "system", "content": str}
      user:      {"role": "user", "content": str}
      assistant: {"role": "assistant", "content": str,
                  "tool_calls"?: [{"id", "type": "function",
                                   "function": {"name", "arguments"}}],
                  "reasoning_content"?: str}
      tool:      {"role": "tool", "tool_call_id": str, "content": str}
    """
    role = data["role"]
    if role == "system":
        return SystemMessage(content=data["content"])
    if role == "user":
        return UserMessage(content=data["content"])
    if role == "assistant":
        raw_calls = data.get("tool_calls")
        return AssistantMessage(
            content=data.get("content"),
            # 把嵌套的 wire 形状拍平成 ToolCall，构造一个全新的 tuple——不动入参。
            tool_calls=(
                tuple(
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=tc["function"]["arguments"],
                    )
                    for tc in raw_calls
                )
                if raw_calls
                else None
            ),
            reasoning_content=data.get("reasoning_content"),
        )
    if role == "tool":
        return ToolMessage(tool_call_id=data["tool_call_id"], content=data["content"])
    raise ValueError(f"Unknown role: {role}")


def message_to_api(msg: Message) -> dict:
    """领域 Message → 线上 dict，用 isinstance 分派，还原成 wire 形状。

    关键：assistant 的 tool_calls / reasoning_content 是"可选"的——
    没有就别放这个 key，要还原出 runner.py 里那种干净的 wire 形状。
    """
    if isinstance(msg, SystemMessage):
        return {"role": "system", "content": msg.content}
    if isinstance(msg, UserMessage):
        return {"role": "user", "content": msg.content}
    if isinstance(msg, ToolMessage):
        return {"role": "tool", "tool_call_id": msg.tool_call_id, "content": msg.content}
    if isinstance(msg, AssistantMessage):
        data: dict = {"role": "assistant", "content": msg.content}
        if msg.tool_calls:  # None / 空 tuple 都不放这个 key
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in msg.tool_calls
            ]
        if msg.reasoning_content is not None:
            data["reasoning_content"] = msg.reasoning_content
        return data
    raise ValueError(f"Unknown message type: {type(msg)}")


# ---- 批量版本，纯机械活 ----


def messages_from_api(items: list[dict]) -> list[Message]:
    return [message_from_api(d) for d in items]


def messages_to_api(msgs: list[Message]) -> list[dict]:
    return [message_to_api(m) for m in msgs]
