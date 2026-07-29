"""评估任务集。1a 先放一个最简单的：让 agent 写文件。"""

from evals.types import Task

TASKS: list[Task] = [
    Task(
        id="write_file",
        user_input="请把文本 hello eval 写入文件 work_space/note.txt",
        clean_files=("note.txt",),
    ),
]
