"""AgentState dataclass 与消息 schema 不变量测试。"""

from datetime import datetime

import pytest

from src.agent.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from src.agent.state import (
    DEFAULT_SYSTEM_PROMPT,
    AgentState,
    StateInvariantError,
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def test_create_minimal_state():
    state = AgentState(
        session_id="abc",
        created_at=_now(),
        updated_at=_now(),
        model="glm-4.7",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        messages=[SystemMessage(content=DEFAULT_SYSTEM_PROMPT)],
        metadata={},
    )
    assert state.session_id == "abc"
    assert state.model == "glm-4.7"
    assert len(state.messages) == 1


def test_first_message_must_be_system():
    """类型化之后这条照样 raise —— "第一条是 system"是位置性质，类型管不着。"""
    with pytest.raises(StateInvariantError, match="第一条必须是 system"):
        AgentState(
            session_id="x",
            created_at=_now(),
            updated_at=_now(),
            model="glm-4.7",
            system_prompt="x",
            messages=[UserMessage(content="hi")],  # ← 第一条不是 system
            metadata={},
        )


def test_tool_calls_must_have_matching_tool_results():
    """类型化之后这条照样 raise —— 配对是关系性质，类型管不着。"""
    with pytest.raises(StateInvariantError, match="tool_call_id"):
        AgentState(
            session_id="x",
            created_at=_now(),
            updated_at=_now(),
            model="glm-4.7",
            system_prompt="x",
            messages=[
                SystemMessage(content="x"),
                UserMessage(content="hi"),
                AssistantMessage(
                    content="",
                    tool_calls=(ToolCall(id="tc1", name="read_file", arguments="{}"),),
                ),
                # 缺 ToolMessage
            ],
            metadata={},
        )


def test_tool_calls_match_succeeds():
    state = AgentState(
        session_id="x",
        created_at=_now(),
        updated_at=_now(),
        model="glm-4.7",
        system_prompt="x",
        messages=[
            SystemMessage(content="x"),
            UserMessage(content="hi"),
            AssistantMessage(
                content="",
                tool_calls=(ToolCall(id="tc1", name="read_file", arguments="{}"),),
            ),
            ToolMessage(tool_call_id="tc1", content="ok"),
            AssistantMessage(content="done"),
        ],
        metadata={},
    )
    assert len(state.messages) == 5


def test_json_roundtrip():
    """AgentState 应能 JSON 序列化/反序列化不丢字段（经 mapper 走 wire 形状）。"""
    state = AgentState(
        session_id="abc",
        created_at=_now(),
        updated_at=_now(),
        model="glm-4.7",
        system_prompt="x",
        messages=[SystemMessage(content="x")],
        metadata={"trace_id": "t-1"},
    )
    j = state.to_json()
    loaded = AgentState.from_json(j)
    assert loaded.session_id == state.session_id
    assert loaded.metadata == state.metadata
    assert loaded.messages == state.messages


def test_metadata_default_is_empty_dict():
    state = AgentState(
        session_id="x",
        created_at=_now(),
        updated_at=_now(),
        model="glm-4.7",
        system_prompt="x",
        messages=[SystemMessage(content="x")],
    )
    assert state.metadata == {}
