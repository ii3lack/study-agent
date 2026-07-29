"""AgentCLI 渲染测试：把输出写进 StringIO 的 Console，再对纯文本断言。

不依赖真实终端、不花 token——给 AgentCLI 喂构造好的 Event，检查渲染结果。
"""

import io

from rich.console import Console

from src.agent.runner import (
    Error,
    RunEnd,
    ToolResult,
    ToolStart,
    UserToken,
)
from src.cli.tui import AgentCLI


def _cli(**kwargs) -> tuple[AgentCLI, io.StringIO]:
    buf = io.StringIO()
    # force_terminal=False → 输出无 ANSI 颜色码，方便对纯文本断言
    console = Console(file=buf, width=100, force_terminal=False)
    return AgentCLI(console=console, **kwargs), buf


def test_content_token_rendered():
    cli, buf = _cli()
    cli.render(UserToken(text="hello world", kind="content"))
    assert "hello world" in buf.getvalue()


def test_reasoning_shown_by_default_with_marker():
    cli, buf = _cli()
    cli.render(UserToken(text="让我想想", kind="reasoning"))
    out = buf.getvalue()
    assert "让我想想" in out
    assert "💭" in out


def test_reasoning_hidden_when_disabled():
    cli, buf = _cli(show_reasoning=False)
    cli.render(UserToken(text="秘密思考", kind="reasoning"))
    assert "秘密思考" not in buf.getvalue()


def test_tool_start_shows_name_and_args():
    cli, buf = _cli()
    cli.render(ToolStart(name="read_file", arguments={"file_path": "a.txt"}))
    out = buf.getvalue()
    assert "read_file" in out
    assert "a.txt" in out


def test_tool_result_shows_content():
    cli, buf = _cli()
    cli.render(ToolResult(name="read_file", content="FILE CONTENT HERE"))
    assert "FILE CONTENT HERE" in buf.getvalue()


def test_long_result_truncated():
    cli, buf = _cli()
    cli.render(ToolResult(name="read_file", content="x" * 2000))
    out = buf.getvalue()
    assert "截断" in out
    assert "x" * 2000 not in out  # 超长原文被截掉了


def test_error_message_rendered():
    cli, buf = _cli()
    cli.render(Error(message="boom happened"))
    assert "boom happened" in buf.getvalue()


def test_full_stream_renders_and_separates_reasoning_from_content():
    cli, buf = _cli()
    events = [
        UserToken(text="思考中", kind="reasoning"),
        UserToken(text="你好", kind="content"),
        ToolStart(
            name="write_file", arguments={"file_path": "n.txt", "content": "hi"}
        ),
        ToolResult(name="write_file", content="ok"),
        Error(message="warn"),
        RunEnd(),
    ]
    cli.render_all(events)

    lines = buf.getvalue().splitlines()
    assert any("思考中" in ln for ln in lines)
    assert any("你好" in ln for ln in lines)
    # 思考与正文应分行：正文不紧跟在思考同一行
    reasoning_line = next(ln for ln in lines if "思考中" in ln)
    assert "你好" not in reasoning_line
