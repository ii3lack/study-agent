import os
from pathlib import Path

# ============ 路径沙箱 ============
# 定义允许访问的根目录（可添加多个）
_PROJECT_ROOT = Path(os.getcwd()).resolve()
ALLOWED_ROOTS = [
    _PROJECT_ROOT / "work_space",
    _PROJECT_ROOT / "work_space" / "*",  # 允许子目录
    _PROJECT_ROOT / "storage",
    _PROJECT_ROOT / "storage" / "*",  # 允许子目录
]


def _check_sandbox(file_path: str) -> Path | str:
    """沙箱校验：返回解析后的 Path，或错误字符串。

    检查逻辑：
    1. 解析成绝对路径（处理 ../ 和符号链接）
    2. 检查是否在允许的根目录下
    3. 越界 → 返回错误字符串，模型会看到并调整行为
    """
    try:
        resolved = Path(file_path).resolve()
    except (OSError, ValueError) as e:
        return f"路径解析失败：{e}"

    # 检查是否在任一允许的根目录下
    for root in ALLOWED_ROOTS:
        try:
            resolved.relative_to(root)
            return resolved  # 在范围内，返回解析后的路径
        except ValueError:
            continue

    # 所有根目录都不匹配 → 越界
    return (
        f"❌ 路径越界：{file_path} 不在允许的目录范围内。"
        f"\n允许访问的目录：{', '.join(str(r) for r in ALLOWED_ROOTS)}"
        f"\n请使用 work_space/ 下的路径。"
    )


read_file_tool = {
    "type": "function",  # 固定:表示这是个函数工具
    "function": {
        "name": "read_file",  # 工具名(模型调用时会返回它,你靠它判断执行哪个函数)
        "description": "读取指定路径的文件,返回其文本内容",  # 干什么用
        "parameters": {  # ↓ 这部分就是 JSON Schema,描述参数
            "type": "object",  # 参数整体是一个对象
            "properties": {  # 每个参数的定义
                "file_path": {
                    "type": "string",
                    "description": "要读取的文件路径,例如 workspace/draft.md",
                }
            },
            "required": ["file_path"],  # 哪些参数必填
        },
    },
}


def read_file(file_path: str) -> str:
    """阅读文件并返回内容。"""
    result = _check_sandbox(file_path)
    if isinstance(result, str):
        return result
    return result.read_text(encoding="utf-8")


write_file_tool = {
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "将内容写入指定路径的文件",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要写入的文件路径,例如 workspace/draft.md",
                },
                "content": {
                    "type": "string",
                    "description": "要写入的 markdown 文本内容",
                },
            },
            "required": ["file_path", "content"],  # 两个参数都必填
        },
    },
}


def write_file(file_path: str, content: str) -> str:
    """将内容写入文件，返回成功提示。"""
    result = _check_sandbox(file_path)
    if isinstance(result, str):
        return result
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(content, encoding="utf-8")
    return f"已将内容写入 {file_path}"


list_files_tool = {
    "type": "function",
    "function": {
        "name": "list_files",
        "description": "列出指定目录下的所有文件",
        "parameters": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "description": "要列出的文件夹路径,例如 workspace",
                }
            },
        },
    },
}


def list_files(directory: str) -> list[str] | str:
    """列出目录下的所有文件。"""
    result = _check_sandbox(directory)
    if isinstance(result, str):
        return result
    return [str(file) for file in result.glob("*")]


edit_file_tool = {
    "type": "function",
    "function": {
        "name": "edit_file",
        "description": "编辑指定路径的文件,替换指定行范围的内容",
        "parameters": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "要编辑的文件路径,例如 workspace/draft.md",
                },
                "start_line": {
                    "type": "number",
                    "description": "要修改开始的行数(从1开始)",
                },
                "end_line": {
                    "type": "number",
                    "description": "要修改结束的行数(不包含该行)",
                },
                "new_content": {
                    "type": "string",
                    "description": "替换后的新内容",
                },
            },
            "required": ["file_path", "start_line", "end_line", "new_content"],
        },
    },
}


def edit_file(file_path: str, start_line: int, end_line: int, new_content: str) -> str:
    """把 [start_line, end_line) 这个行范围替换成 new_content。
    行号从 1 开始，end_line 不包含（左闭右开）。

    例：start_line=2, end_line=4  → 替换第 2、3 行
    """
    result = _check_sandbox(file_path)
    if isinstance(result, str):
        return result
    text = result.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)

    start_idx = start_line - 1
    end_idx = end_line - 1

    if not new_content.endswith("\n"):
        new_content += "\n"

    new_text = "".join(lines[:start_idx]) + new_content + "".join(lines[end_idx:])

    result.write_text(new_text, encoding="utf-8")
    return f"已将 {file_path} 的第 {start_line}-{end_line - 1} 行替换"
