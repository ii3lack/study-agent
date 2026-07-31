# ai/client.py
"""ai 客户端"""

import os
from openai import OpenAI


class Client:
    """ai provider。"""

    def __init__(self):
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url=os.getenv("BASE_URL"),
        )

    def chat(self, *args, **kwargs):
        """调用 zai 的 ChatCompletion.create 方法。"""
        return self.client.chat.completions.create(*args, **kwargs)
