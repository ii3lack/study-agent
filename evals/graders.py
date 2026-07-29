"""打分器（grader）——eval 的概念核心。

每个 grader: (Task, RunResult) -> Grade。
判真模型输出的心法：判「世界状态」（文件写对没），别判模型怎么措辞。
"""

from evals.types import Grade, Grader, RunResult, Task


def grade_write_file(task: Task, result: RunResult) -> Grade:
    content = result.files.get("note.txt")  # ← 世界状态，清理前的冻结快照
    if content is None:
        return Grade(False, 0.0, "没生成 note.txt")
    if "hello eval" not in content:
        return Grade(False, 0.5, f"文件在但内容不对: {content!r}")
    # tool_calls 只当佐证（加分项），且是 dict，用 ["name"] 取
    used_tool = any(tc["name"] == "write_file" for tc in result.tool_calls)
    tail = "" if used_tool else "（注意：未见 write_file 调用）"
    return Grade(True, 1.0, "文件内容正确" + tail)


# task.id -> grader
GRADERS: dict[str, Grader] = {
    "write_file": grade_write_file,
}
