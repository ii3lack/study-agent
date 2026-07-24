# main.py
"""主程序入口"""

from dotenv import load_dotenv

load_dotenv()

from ai import zai_sse


def main():
    print("Study-Agent is starting...")
    print("Hello from study-agent!")
    try:
        zai_sse.main()
    except KeyboardInterrupt:
        print("Study-Agent is stopping...")


if __name__ == "__main__":
    main()
