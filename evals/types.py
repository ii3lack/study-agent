"""eval 的核心数据结构：任务、运行结果、评分。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from src.agent.runner import Event


@dataclass(frozen=True)
class Task:
    """一个评估任务。"""

    id: str
    user_input: str  # 喂给 agent 的话
    # 跑前需要清空的文件（相对 work_space），保证每次从干净状态开始
    clean_files: tuple[str, ...] = ()


@dataclass
class RunResult:
    """harness 跑完一个 task 的产出。grader 只读它，不碰真实文件系统。"""

    final_answer: str  # 模型最后的正文回答
    tool_calls: list[dict]  # [{"name": ..., "arguments": {...}}, ...]
    files: dict[str, str]  # 跑完后 work_space 快照 {相对路径: 内容}
    events: list[Event] = field(default_factory=list)  # 全量事件，调试用


@dataclass(frozen=True)
class Grade:
    """一个任务的评分。"""

    passed: bool
    score: float  # 0.0 ~ 1.0
    reason: str


# grader 的签名：吃 (task, 结果)，吐 评分
Grader = Callable[[Task, RunResult], Grade]
