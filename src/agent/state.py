"""AgentState — Agent 唯一可变状态容器。

消息用领域类型 Message 建模（见 message.py），不再是一堆松散 dict。
持久化 / 发给 API 时，经由 serialization.py 的 mapper 还原成 wire 形状。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from src.agent.message import (
    AssistantMessage,
    Message,
    SystemMessage,
    ToolMessage,
)
from src.agent.serialization import messages_from_api, messages_to_api

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
    messages: list[Message]
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        _validate_invariants(self.messages)

    def to_json(self) -> str:
        # 不能无脑 asdict：那会吐出领域形状（扁平、带一堆 None），
        # 不是 GLM / 存储要的 wire 形状。messages 必须走 mapper 还原。
        data = {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "model": self.model,
            "system_prompt": self.system_prompt,
            "messages": messages_to_api(self.messages),
            "metadata": self.metadata,
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, text: str) -> "AgentState":
        data = json.loads(text)
        # 存储里是 wire dict，读回来要还原成 Message 对象，
        # 否则 __post_init__ 的 isinstance 校验会当场翻车。
        data["messages"] = messages_from_api(data["messages"])
        return cls(**data)


def _validate_invariants(messages: list[Message]) -> None:
    """校验消息列表的关键不变量。

    类型化之后，下面三条**一条都没删掉**——因为它们全是"列表层 / 关系层"
    性质，单条消息的类型根本管不着。类型只买到了"单条消息的形状"
    （比如 ToolMessage 必带 tool_call_id），仅此而已。
    """
    # ① 基数：list[Message] 不保证非空，这条删不掉。
    if not messages:
        raise StateInvariantError("messages 不能为空")

    # ② 位置：类型不保证"第一条"是 system，只是把字符串判断换成 isinstance。
    #    注意这里必须有 not——"不是 system 才报错"，和你原来 `!= "system"` 的否定一致。
    if not isinstance(messages[0], SystemMessage):
        raise StateInvariantError("messages 第一条必须是 system")

    # ③ 关系：每个 assistant tool_call 都必须有对应的 ToolMessage 回应。
    #    这是跨消息性质，类型永远表达不了——逻辑一行不少。
    pending: list[str] = []
    for msg in messages:
        # 注意 `and msg.tool_calls`：最终回答的 assistant 它的 tool_calls 是 None，
        # 直接 for tc in None 会炸。你原来 `and msg.get("tool_calls")` 的守卫不能丢。
        if isinstance(msg, AssistantMessage) and msg.tool_calls:
            pending.extend(tc.id for tc in msg.tool_calls)
        elif isinstance(msg, ToolMessage):
            if msg.tool_call_id not in pending:
                raise StateInvariantError(
                    f"tool_call_id={msg.tool_call_id} 没有对应的 assistant tool_calls"
                )
            pending.remove(msg.tool_call_id)

    if pending:
        raise StateInvariantError(f"未配对的 tool_call_id: {pending}")
