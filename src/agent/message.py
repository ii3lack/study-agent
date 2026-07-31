"""领域消息类型 —— Agent 内部唯一认可的消息模型。

设计原则（Q2 的落地）：
- 这里只建模"消息是什么形状"，不认识任何供应商的 JSON 格式。
- 所以故意没有 to_api_dict() / from_api_dict()：一旦领域类型认识了
  智谱/OpenAI 的 wire 形状，换模型或 API 改字段时，核心模型就得跟着改。
- 线上 dict ↔ Message 的转换隔离在边界 serialization.py 里。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    """assistant 发起的一次工具调用（领域内的干净形状）。

    线上的嵌套形状 {"id","type":"function","function":{"name","arguments"}}
    由 mapper 拍平/还原；领域里只留 id / name / arguments。
    arguments 保持 JSON 字符串——解析成 dict 是工具执行方的事。
    """

    id: str
    name: str
    arguments: str


@dataclass(frozen=True)
class SystemMessage:
    content: str


@dataclass(frozen=True)
class UserMessage:
    content: str


@dataclass(frozen=True)
class AssistantMessage:
    """只有 assistant 会携带 tool_calls —— 这是类型唯一能"自动保证"的形状。"""

    content: str | None = None
    tool_calls: tuple[ToolCall, ...] | None = None
    reasoning_content: str | None = None


@dataclass(frozen=True)
class ToolMessage:
    """一条 tool 消息必须回应某个 call —— tool_call_id 是必填字段。"""

    tool_call_id: str
    content: str


# 联合类型：一条消息必是且仅是这四者之一。
# dispatch 用 isinstance（见 state.py / serialization.py），不靠 "role" 字符串。
Message = SystemMessage | UserMessage | AssistantMessage | ToolMessage
