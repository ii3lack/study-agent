import os

from typing import cast

from dotenv import load_dotenv
from zai import ZhipuAiClient
from zai.core._streaming import StreamResponse
from zai.types.chat import ChatCompletionChunk


load_dotenv()  # 把 .env 里的变量加载进 os.environ（返回 True/False，不是字典！）
api_key = os.getenv("API_KEY")  # 用 os.getenv 从环境变量读取
client = ZhipuAiClient(api_key=api_key)  # 请填写您自己的 API Key

response = cast(
    "StreamResponse[ChatCompletionChunk]",
    client.chat.completions.create(
        model="glm-4.7",
        messages=[
            {"role": "user", "content": "作为一名营销专家，请为我的产品创作一个吸引人的口号"},
            {"role": "assistant", "content": "当然，要创作一个吸引人的口号，请告诉我一些关于您产品的信息"},
            {"role": "user", "content": "智谱开放平台"}
        ],
        thinking={
            "type": "enabled",    # 启用深度思考模式
        },
        stream=True,              # 启用流式输出
        max_tokens=65536,          # 最大输出tokens
        temperature=1.0           # 控制输出的随机性
    ),
)

# 流式获取回复
for chunk in response:
    if chunk.choices[0].delta.reasoning_content:
        print(chunk.choices[0].delta.reasoning_content, end='', flush=True)

    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end='', flush=True)