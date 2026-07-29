"""grader 的单元测试：手工构造 RunResult，零成本、完全不碰模型。

grader 是 RunResult 的纯函数，所以可以这样直接喂假数据来测——
这正是它该有的测试方式，也顺便成了你 grader 的验收规格。
"""

from evals.graders import grade_write_file
from evals.types import RunResult, Task

_TASK = Task(id="write_file", user_input="写文件", clean_files=("note.txt",))


def _result(files, tool_calls=None, final_answer="done") -> RunResult:
    return RunResult(final_answer=final_answer, tool_calls=tool_calls or [], files=files)


def test_perfect_file_passes_full_score():
    """文件在、内容对 → 通过、满分。"""
    g = grade_write_file(_TASK, _result(files={"note.txt": "hello eval"}))
    assert g.passed is True
    assert g.score == 1.0


def test_wrong_content_not_full_score():
    """文件在但内容不对 → 不能给满分，且要说清原因。"""
    g = grade_write_file(_TASK, _result(files={"note.txt": "完全无关的内容"}))
    assert g.score < 1.0
    assert g.reason  # 理由不能空


def test_missing_file_fails():
    """压根没写文件 → 不通过、零分。"""
    g = grade_write_file(_TASK, _result(files={}))
    assert g.passed is False
    assert g.score == 0.0
