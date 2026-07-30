# Study Agent

从零学习 AI Agent 应用开发的实战项目。基于智谱 GLM 模型，用 Python 手写一个具备工具调用能力的 ReAct Agent，理解 Agent 框架的核心设计思想。

## 这个项目是什么

不是又一个 "LangChain 套壳 Demo"。这是一个**学习用 Agent**——每个核心模块都亲手写，目标是理解：

- **ReAct 循环**：模型思考 → 调工具 → 观察结果 → 再思考，循环是怎么跑起来的
- **事件流驱动**：Agent 循环只 emit 事件，不关心谁在听（CLI / Web / Trace 各自订阅）
- **工具沙箱**：模型能读写文件，但不能逃出工作目录
- **Eval 驱动迭代**：改了 prompt 或模型后，用量化指标判断变好还是变坏

## 项目结构

```
src/
├── main.py                 # 入口：组装 Client + Runner + CLI
├── agent/
│   ├── runner.py           # ReAct 循环核心（事件流驱动，yield Event）
│   └── state.py            # AgentState dataclass + 消息不变式校验
├── ai/
│   └── client.py           # 智谱 API 封装
├── cli/
│   └── tui.py              # Rich 终端渲染（订阅 Runner 事件）
├── storage/
│   └── session.py          # JSON 文件会话持久化
└── tools/
    └── file_tools.py       # 文件工具（read/write/list/edit）+ 沙箱

evals/                      # Eval 框架
├── types.py                # Task / RunResult / Grade 数据结构
├── tasks.py                # 任务定义
├── graders.py              # 评分器（检查世界状态，不检查措辞）
├── harness.py              # run_task()：跑任务 + 收集结果
└── run_eval.py             # CLI 入口：跑全部任务，输出报告

tests/                      # 测试（pytest）
├── agent/test_runner.py    # Runner 生命周期测试（6 个）
├── agent/test_state.py     # AgentState 不变量测试（6 个）
├── storage/test_session.py # SessionStore CRUD 测试（11 个）
└── evals/                  # Eval 零成本测试（FakeClient）

docs/
└── review/                 # 每周复盘总结
```

## 快速开始

```bash
# 安装依赖（推荐 uv）
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 ZHIPUAI_API_KEY

# 运行 Agent
python -m src.main

# 跑测试
pytest

# 跑 Eval
python -m evals.run_eval
```

## 核心设计

### ReAct 循环（runner.py）

```
用户输入 → [Runner.run()]
  ├─ 调 LLM（流式）
  │   ├─ 有 tool_calls → 执行工具 → 结果喂回 LLM → 继续循环
  │   └─ 无 tool_calls → 最终回答 → 结束
  ├─ 工具异常 → 转成消息喂回模型（不中断循环）
  └─ max_turns 超限 → 强制结束
```

Runner 只 yield Event，不渲染、不持久化、不 import 具体 Client。

### 事件类型

| Event | 含义 |
|---|---|
| `TurnStart` | 第 N 轮开始 |
| `UserToken` | 模型吐了一个 token（reasoning / content） |
| `ToolStart` | 即将调用工具 |
| `ToolResult` | 工具返回结果 |
| `TurnEnd` | 本轮结束（模型给出最终回答） |
| `RunEnd` | 整个 run 结束（保证最后一定有它） |
| `Error` | 可恢复错误 |

### 工具沙箱

所有文件操作限制在 `work_space/` 和 `storage/` 内。通过 `resolve()` + `relative_to()` 阻止 `../` 路径穿越。

## 学习路线

| 阶段 | 内容 | 状态 |
|---|---|---|
| 阶段 1 | 核心 Agent 循环 + 工具 + 会话存储 + Eval 骨架 | ✅ 基本完成 |
| 阶段 2 | SessionStorage Protocol + AgentMessage 类型 + 并发工具执行 | 🔲 待做 |
| 阶段 3 | 多 Agent 协作 + Trace 可观测 + Web 前端 | 🔲 未来 |

## Tech Stack

- **Python 3.11+**
- **智谱 GLM**（glm-5.2）— 通过 zai-sdk 调用
- **Rich** — 终端 UI 渲染
- **pytest** — 测试框架
- **Langfuse** — 可观测性（已引入，待集成）
