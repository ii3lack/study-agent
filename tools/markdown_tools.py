from pathlib import Path

read_markdown_tool = {
    "type": "function",  # 固定:表示这是个函数工具
    "function": {
        "name": "read_markdown",  # 工具名(模型调用时会返回它,你靠它判断执行哪个函数)
        "description": "读取指定路径的 markdown 文件,返回其文本内容",  # 干什么用
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


def read_markdown(file_path: str) -> str:
    """阅读Markdown 文件并返回内容。"""
    return Path(file_path).read_text(encoding="utf-8")


write_markdown_tool = {
    "type": "function",
    "function": {
        "name": "write_markdown",
        "description": "将内容写入指定路径的 markdown 文件",
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


def write_markdown(file_path: str, content: str) -> str:
    """将内容写入Markdown 文件,返回成功提示。"""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
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


def list_files(directory: str) -> list[str]:
    """列出目录下的所有文件。"""
    return [str(file) for file in Path(directory).glob("*")]
