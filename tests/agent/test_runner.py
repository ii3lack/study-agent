"""Runner 测试：事件流、工具执行、错误降级、max_turns。

TDD 顺序（难度递增）：
1. 无 tool_calls → 一轮自然退出
2. 有 tool_calls → 执行工具并续接
3. tool 抛异常 → 当消息不当中断
4. 跑满 max_turns → Error 退出

每个测试的结构都一样：
  编造 FakeClient 的返回 → 喂给 Runner → 检查事件流 + messages
"""

from datetime import datetime
from typing import Iterator

from src.agent.runner import (
    Error,
    RunEnd,
    Runner,
    ToolResult,
    ToolStart,
    TurnEnd,
    TurnStart,
    UserToken,
)
from src.agent.state import AgentState, DEFAULT_SYSTEM_PROMPT


# ============================================================
# Fake 对象 —— 模拟 zai client 的流式返回
# ============================================================
# Runner 只调 client.chat()，拿到一个可迭代的 chunks。
# 每个 chunk 需要有 .choices[0].delta，delta 上有
# .content / .reasoning_content / .tool_calls。
# 下面的 Fake 类就是模拟这个结构。


class FakeFunction:
    """模拟 tool_call 里的 function 字段。"""

    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    """模拟 delta.tool_calls 里的一个元素。"""

    def __init__(self, index: int, id: str, name: str, arguments: str):
        self.index = index
        self.id = id
        self.function = FakeFunction(name, arguments)


class FakeDelta:
    """模拟 chunk.choices[0].delta。"""

    def __init__(self, content=None, reasoning_content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = reasoning_content
        self.tool_calls = tool_calls


class FakeChoice:
    def __init__(self, delta):
        self.delta = delta


class FakeChunk:
    """模拟一个 SSE chunk。"""

    def __init__(self, delta):
        self.choices = [FakeChoice(delta)]


class FakeClient:
    """模拟 zai Client。

    参数 responses 是一个列表的列表：
      - 外层：每次调 chat() 取一组
      - 内层：这组里的每个 FakeChunk 依次返回

    例：
      FakeClient([
          [FakeChunk(FakeDelta(content="第一轮"))],   # 第 1 次 chat()
          [FakeChunk(FakeDelta(content="第二轮"))],   # 第 2 次 chat()
      ])
    """

    def __init__(self, responses: list[list[FakeChunk]]):
        self._responses = list(responses)
        self._call_count = 0

    def chat(self, **kwargs) -> Iterator[FakeChunk]:
        resp = self._responses[self._call_count]
        self._call_count += 1
        return iter(resp)


# ============================================================
# 工具函数
# ============================================================


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _state() -> AgentState:
    """造一个最小的合法 AgentState。"""
    return AgentState(
        session_id="test",
        created_at=_now(),
        updated_at=_now(),
        model="glm-4.7",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        messages=[{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}],
    )


# ============================================================
# 测试 1：无 tool_calls → 一轮自然退出
# ============================================================
# 最简单的场景：用户问问题，模型直接回答，不调工具。
# 事件流应该是：TurnStart → UserToken(s) → TurnEnd → RunEnd


def test_no_tool_calls_exits_after_one_turn():
    # 编造：模型先思考，再回答
    chunks = [
        FakeChunk(FakeDelta(reasoning_content="让我想想...")),
        FakeChunk(FakeDelta(content="你好！")),
    ]
    client = FakeClient([chunks])
    runner = Runner(client=client, tools=[], tool_fns={})

    state = _state()
    events = list(runner.run(state, user_input="hi"))

    # 检查事件流的组成
    kinds = [type(e).__name__ for e in events]
    assert "TurnStart" in kinds
    assert "UserToken" in kinds
    assert "TurnEnd" in kinds
    assert "RunEnd" in kinds
    assert "Error" not in kinds  # 不应该有错误

    # 检查 UserToken 的内容
    tokens = [e for e in events if isinstance(e, UserToken)]
    assert any(t.text == "让我想想..." and t.kind == "reasoning" for t in tokens)
    assert any(t.text == "你好！" and t.kind == "content" for t in tokens)

    # 检查 messages：system + user + assistant = 3 条
    assert len(state.messages) == 3
    assert state.messages[1] == {"role": "user", "content": "hi"}
    assert state.messages[2]["role"] == "assistant"
    assert state.messages[2]["content"] == "你好！"

    # chat() 只被调了 1 次（1 轮就退出了）
    assert client._call_count == 1


# ============================================================
# 测试 2：有 tool_calls → 执行工具，带结果再问 LLM
# ============================================================
# 模型第一轮说"我要调 echo 工具"，我们执行后把结果塞回 messages，
# 模型第二轮看到工具结果，给出最终回答。


def test_tool_calls_executes_and_continues():
    # 第一轮：模型返回 tool_call
    tool_call = FakeToolCall(
        index=0, id="tc-1", name="echo", arguments='{"text": "hello"}'
    )
    round1 = [
        FakeChunk(FakeDelta(tool_calls=[tool_call])),
    ]
    # 第二轮：模型看到工具结果后，给出最终回答
    round2 = [
        FakeChunk(FakeDelta(content="工具说了 hello")),
    ]
    client = FakeClient([round1, round2])
    runner = Runner(
        client=client,
        tools=[{"type": "function", "function": {"name": "echo"}}],
        tool_fns={"echo": lambda text: f"echoed:{text}"},
    )

    state = _state()
    events = list(runner.run(state, user_input="调工具"))

    # 检查事件：应该有 ToolStart 和 ToolResult
    kinds = [type(e).__name__ for e in events]
    assert "ToolStart" in kinds
    assert "ToolResult" in kinds
    assert "RunEnd" in kinds
    assert "Error" not in kinds

    # 检查 ToolResult 的内容
    tool_results = [e for e in events if isinstance(e, ToolResult)]
    assert len(tool_results) == 1
    assert tool_results[0].name == "echo"
    assert tool_results[0].content == "echoed:hello"

    # 检查 messages：
    #   system + user + assistant(带tool_calls) + tool + assistant(最终) = 5 条
    assert len(state.messages) == 5
    # assistant 带 tool_calls
    assert state.messages[2]["role"] == "assistant"
    assert "tool_calls" in state.messages[2]
    # tool 消息
    assert state.messages[3]["role"] == "tool"
    assert state.messages[3]["tool_call_id"] == "tc-1"
    assert state.messages[3]["content"] == "echoed:hello"
    # 最终 assistant
    assert state.messages[4]["role"] == "assistant"
    assert state.messages[4]["content"] == "工具说了 hello"

    # chat() 被调了 2 次（2 轮）
    assert client._call_count == 2


# ============================================================
# 测试 3：tool 抛异常 → 错误变成消息，循环不中断
# ============================================================
# 这是 ReAct 的关键设计：工具报错不是"程序崩了"，
# 而是"模型收到了一条错误消息"，模型有机会自我纠正。


def test_tool_exception_becomes_message():
    def bad_tool(**kwargs):
        raise RuntimeError("boom")

    tool_call = FakeToolCall(index=0, id="tc-err", name="bad", arguments="{}")
    round1 = [FakeChunk(FakeDelta(tool_calls=[tool_call]))]
    round2 = [FakeChunk(FakeDelta(content="抱歉，工具出错了"))]
    client = FakeClient([round1, round2])
    runner = Runner(
        client=client,
        tools=[],
        tool_fns={"bad": bad_tool},
    )

    state = _state()
    events = list(runner.run(state, user_input="go"))

    # ★ 不应该有 Error 事件（异常被吞成消息了）
    assert not any(isinstance(e, Error) for e in events)
    # 应该有 RunEnd（正常结束）
    assert any(isinstance(e, RunEnd) for e in events)

    # tool 消息里包含错误信息
    tool_msgs = [m for m in state.messages if m.get("role") == "tool"]
    assert len(tool_msgs) == 1
    assert "boom" in tool_msgs[0]["content"]
    assert "执行错误" in tool_msgs[0]["content"]

    # 循环继续了：模型看到了错误，给出了最终回答
    assert state.messages[-1]["role"] == "assistant"
    assert state.messages[-1]["content"] == "抱歉，工具出错了"


# ============================================================
# 测试 4：跑满 max_turns → Error 退出
# ============================================================
# 模型每轮都要调工具，永远不停。
# max_turns 是死循环保护，到了就强制退出并发 Error。


def test_max_turns_exceeded_emits_error():
    # 每轮都返回同一个 tool_call（模型"卡住了"）
    tool_call = FakeToolCall(index=0, id="tc-loop", name="loop", arguments="{}")
    round_chunks = [FakeChunk(FakeDelta(tool_calls=[tool_call]))]

    # 准备足够多的轮次（比 max_turns 多，确保不是"不够用"）
    client = FakeClient([round_chunks] * 10)
    runner = Runner(
        client=client,
        tools=[],
        tool_fns={"loop": lambda: "ok"},
        max_turns=2,  # ★ 只允许 2 轮
    )

    state = _state()
    events = list(runner.run(state, user_input="go"))

    # 应该有 Error 事件
    errors = [e for e in events if isinstance(e, Error)]
    assert len(errors) == 1
    assert "最大步数" in errors[0].message

    # Error 之后仍然有 RunEnd
    assert any(isinstance(e, RunEnd) for e in events)

    # chat() 刚好被调了 max_turns 次
    assert client._call_count == 2
