"""打分器（grader）——eval 的概念核心。

每个 grader: (Task, RunResult) -> Grade。
判真模型输出的心法：判「世界状态」（文件写对没），别判模型怎么措辞。
"""

from evals.types import Grade, Grader, RunResult, Task


def grade_write_file(task: Task, result: RunResult) -> Grade:
    """判定 write_file 任务是否成功。

    ★★★ 这个函数轮到你（学习者）实现 ★★★

    规格：
      - 金标准：result.files 里 "note.txt" 存在，且内容包含 "hello eval"
          → passed=True, score=1.0
      - 文件在但内容不对 → 部分分，例如 score=0.5
      - 文件压根没有 → passed=False, score=0.0
      - reason 写清楚为什么（看报告时好定位）
    加分：顺手检查 result.tool_calls，确认确实是通过 write_file 做到的，
          而不是模型嘴上说写了、其实没调工具。
    """
    raise NotImplementedError("轮到你了：实现 grade_write_file")


# task.id -> grader
GRADERS: dict[str, Grader] = {
    "write_file": grade_write_file,
}
