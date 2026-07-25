# main.py
"""主程序入口"""

from dotenv import load_dotenv

load_dotenv()

from rich.console import Console
from src.ai import zai_sse


def main():
    console = Console()
    console.print("Start Study AI Agent")
    try:
        zai_sse.main()
    except KeyboardInterrupt:
        console.print("Study AI Agent is stopping...")


if __name__ == "__main__":
    main()
