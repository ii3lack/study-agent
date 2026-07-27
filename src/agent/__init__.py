# src/agent/__init__.py
"""agent 包: 状态、运行器、可观测。"""

from .state import AgentState, DEFAULT_SYSTEM_PROMPT, StateInvariantError

__all__ = ["AgentState", "DEFAULT_SYSTEM_PROMPT", "StateInvariantError"]
