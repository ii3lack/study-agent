"""eval 入口：跑所有任务 → 打分 → 出报告。

用法：
    uv run python -m evals.run_eval
"""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from rich.table import Table

from src.ai import Client

from evals.graders import GRADERS
from evals.harness import run_task
from evals.tasks import TASKS
from evals.types import Grade


def run_all(client=None) -> list[tuple[str, Grade]]:
    """跑全部任务并打印报告，返回 [(task_id, Grade)]。"""
    client = client or Client()
    console = Console()
    rows: list[tuple[str, Grade]] = []

    for task in TASKS:
        console.print(f"▶ 跑任务 [bold]{task.id}[/] ...")
        result = run_task(task, client)
        grader = GRADERS[task.id]
        try:
            grade = grader(task, result)
        except Exception as e:  # grader 没实现/出错也别让整个报告崩
            grade = Grade(passed=False, score=0.0, reason=f"grader 异常: {e}")
        rows.append((task.id, grade))

    _print_report(rows)
    return rows


def _print_report(rows: list[tuple[str, Grade]]) -> None:
    console = Console()
    table = Table(title="Eval 报告")
    table.add_column("任务")
    table.add_column("通过")
    table.add_column("分数")
    table.add_column("理由")

    for task_id, grade in rows:
        table.add_row(
            task_id,
            "✅" if grade.passed else "❌",
            f"{grade.score:.2f}",
            grade.reason,
        )
    console.print(table)

    if rows:
        avg = sum(g.score for _, g in rows) / len(rows)
        console.print(f"\n总平均分：[bold]{avg:.2f}[/]（{len(rows)} 个任务）")


if __name__ == "__main__":
    run_all()
