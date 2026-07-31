"""上下文管理 —— 度量，以及未来的压缩 / 截断 / 预算。

这是"上下文工程"这块的落脚点：先能量出上下文有多大，
后面才谈得上在它撑爆窗口之前去管理它。
"""

from __future__ import annotations

import json

from src.agent.message import Message, UserMessage
from src.agent.serialization import messages_to_api


def estimate_context_tokens(messages: list[Message]) -> int:
    """估算这一整段上下文的 token 数。

    做法：把消息还原成"即将发给模型的 wire JSON"，数字符。
    - 数字符 ≈ 数 token（看增长趋势足够；真要精确，日后换 GLM tokenizer）。
    - 关键是它量的是"真正发出去的那一整坨"——system / reasoning / tool id /
      JSON 结构开销全在内，不靠逐类型分支，所以不会漏掉某种消息
      （这正是"手写分支版"踩的坑：循环路过了 system，却没有分支接住它）。
    """
    if not messages:
        return 0  # 空上下文就是 0；别让空列表的 "[]" 外壳算成 2 个 token
    return len(json.dumps(messages_to_api(messages), ensure_ascii=False))


def window_for_api(messages: list[Message], *, max_tokens: int) -> list[Message]:
    """非破坏性滑动窗口：返回 [system + 最近若干完整轮]，总量 ≤ max_tokens。
    以 UserMessage 为轮边界整段切 → 配对自动安全。"""
    if not messages:
        return []
    tokens = estimate_context_tokens(messages)
    if tokens <= max_tokens:
        return messages
    system_message = messages[0]
    # ① 找出每一轮的起点：所有 UserMessage 的下标
    starts = [i for i, m in enumerate(messages) if isinstance(m, UserMessage)]
    if not starts:
        return [system_message]
    for i in starts:
        suffix = messages[i:]
        new_messages = [system_message, *suffix]
        if estimate_context_tokens(new_messages) <= max_tokens:
            return new_messages

    return [system_message, *messages[starts[-1] :]]
