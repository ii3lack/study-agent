"""AgentState — Agent 唯一可变状态容器。

消息 schema 与 OpenAI Chat Completions 对齐。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

DEFAULT_SYSTEM_PROMPT = "你是专业视觉创作助手，请使用中文回答用户的问题"


class StateInvariantError(Exception):
    """AgentState 不变量被违反。"""


@dataclass
class AgentState:
    session_id: str
    created_at: str
    updated_at: str
    model: str
    system_prompt: str
    messages: list[dict]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_invariants(self.messages)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AgentState":
        data = json.loads(text)
        return cls(**data)


def _validate_invariants(messages: list[dict]) -> None:
    """校验消息列表的关键不变量。"""
    if not messages:
        raise StateInvariantError("messages 不能为空")
    if messages[0].get("role") != "system":
        raise StateInvariantError("messages 第一条必须是 system")

    # tool_calls / tool 消息配对
    pending: list[str] = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                pending.append(tc["id"])
        elif msg.get("role") == "tool":
            tool_call_id = msg.get("tool_call_id")
            if tool_call_id not in pending:
                raise StateInvariantError(
                    f"tool_call_id={tool_call_id} 没有对应的 assistant tool_calls"
                )
            pending.remove(tool_call_id)

    if pending:
        raise StateInvariantError(f"未配对的 tool_call_id: {pending}")
