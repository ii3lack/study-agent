"""estimate_context_tokens 测试 —— 纯函数，重点守住"别漏消息类型"。"""

from src.agent.context import estimate_context_tokens
from src.agent.message import (
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)

SYS = SystemMessage(content="你是专业视觉创作助手，请使用中文回答用户的问题")


def test_empty_context_is_zero():
    assert estimate_context_tokens([]) == 0


def test_system_message_is_counted():
    """回归测试：system 每一轮都在上下文里、每一轮都计费，绝不能算成 0。

    这条正是为了抓住"手写分支漏掉 SystemMessage"那个 bug 而写的。
    """
    assert estimate_context_tokens([SYS]) > 0


def test_more_messages_means_larger_estimate():
    small = estimate_context_tokens([SYS])
    bigger = estimate_context_tokens([SYS, UserMessage(content="你好，请帮我做个方案")])
    assert bigger > small


def test_tool_calls_add_to_context():
    plain = estimate_context_tokens([SYS, AssistantMessage(content="好的")])
    with_call = estimate_context_tokens(
        [
            SYS,
            AssistantMessage(
                content="好的",
                tool_calls=(ToolCall(id="tc1", name="read_file", arguments='{"p": "x"}'),),
            ),
        ]
    )
    assert with_call > plain


def test_big_tool_result_spikes_context():
    """工具灌回一大段内容，上下文应明显暴涨——这就是聊天缓坡上的"工具尖峰"。"""
    before = estimate_context_tokens([SYS, UserMessage(content="读文件")])
    after = estimate_context_tokens(
        [
            SYS,
            UserMessage(content="读文件"),
            ToolMessage(tool_call_id="tc1", content="x" * 5000),
        ]
    )
    assert after > before + 4000
