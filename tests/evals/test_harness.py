"""harness 测试：用 FakeClient 跑，不花 token，验证流水线本身。

关键：用一个会发出 write_file tool_call 的假 client，让真工具真写文件，
验证 harness 能正确收集 tool_calls + 文件快照 + 最终回答，并且跑完清理。
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

from evals.harness import run_task
from evals.types import Task


# ---- 最小 fake：模拟流式返回（结构对齐 zai/OpenAI 的 chunk）----
class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TC:
    def __init__(self, index, id, name, arguments):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.reasoning_content = None
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, delta):
        self.delta = delta


class _Chunk:
    def __init__(self, delta):
        self.choices = [_Choice(delta)]


class _FakeClient:
    def __init__(self, turns):
        self._turns = list(turns)
        self._i = 0

    def chat(self, **kwargs) -> Iterator[_Chunk]:
        turn = self._turns[self._i]
        self._i += 1
        return iter(turn)


def _write_then_answer_client() -> _FakeClient:
    """第 1 轮发一个 write_file 调用，第 2 轮给最终回答。"""
    turn1 = [
        _Chunk(
            _Delta(
                tool_calls=[
                    _TC(
                        0,
                        "call_1",
                        "write_file",
                        '{"file_path": "work_space/note.txt",'
                        ' "content": "hello eval"}',
                    )
                ]
            )
        )
    ]
    turn2 = [_Chunk(_Delta(content="已经写好了。"))]
    return _FakeClient([turn1, turn2])


def test_harness_collects_tool_calls_and_file_snapshot():
    task = Task(id="write_file", user_input="写文件", clean_files=("note.txt",))
    result = run_task(task, _write_then_answer_client())

    # 收集到一次 write_file 调用
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0]["name"] == "write_file"

    # 文件快照里真的有 note.txt，内容正确（世界状态判据）
    assert "note.txt" in result.files
    assert "hello eval" in result.files["note.txt"]

    # 最终回答被捕获
    assert "已经写好了" in result.final_answer


def test_harness_cleans_up_after_run():
    """跑完应清掉声明的文件，不在 work_space 留垃圾。"""
    task = Task(id="write_file", user_input="写", clean_files=("note.txt",))
    run_task(task, _write_then_answer_client())
    assert not (Path.cwd() / "work_space" / "note.txt").exists()
