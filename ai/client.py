# ai/client.py
"""ai 客户端"""

import os
from zai import ZhipuAiClient


class Client:
    """Langfuse 的 wrapper，自动采集 trace。"""

    def __init__(self):
        self.client = ZhipuAiClient(
            api_key=os.getenv("API_KEY"),
        )

    def chat(self, *args, **kwargs):
        """调用 Langfuse 的 ChatCompletion.create 方法。"""
        return self.client.chat.completions.create(*args, **kwargs)
