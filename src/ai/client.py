# ai/client.py
"""ai 客户端"""

import os
from zai import ZhipuAiClient


class Client:
    """ai provider。"""

    def __init__(self):
        self.client = ZhipuAiClient(
            api_key=os.getenv("API_KEY"),
        )

    def chat(self, *args, **kwargs):
        """调用 zai 的 ChatCompletion.create 方法。"""
        return self.client.chat.completions.create(*args, **kwargs)
