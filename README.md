# Study Agent

从零学习 AI Agent 应用开发的实战项目。基于智谱 GLM 模型，用 Python 手写一个具备工具调用能力的 ReAct Agent，理解 Agent 框架的核心设计思想。

## 这个项目是什么

不是又一个 "LangChain 套壳 Demo"。这是一个**学习用 Agent**——每个核心模块都亲手写，目标是理解：

- **ReAct 循环**：模型思考 → 调工具 → 观察结果 → 再思考，循环是怎么跑起来的
- **事件流驱动**：Agent 循环只 emit 事件，不关心谁在听（CLI / Web / Trace 各自订阅）
- **领域建模 + 边界转换**：消息在内部是类型（`Message`），只在边界（API / 存储）转成 wire 格式，理解"反腐蚀层"怎么把供应商差异挡在门外
- **工具沙箱**：模型能读写文件，但不能逃出工作目录
- **Eval 驱动迭代**：改了 prompt 或模型后，用量化指标判断变好还是变坏

## 项目结构

```
src/
├── main.py                 # 入口：组装 Client + Runner + CLI
├── agent/
│   ├── runner.py           # ReAct 循环核心（事件流驱动，yield Event）
│   ├── state.py            # AgentState + 消息不变式校验（非空 / 首位 / 配对）
│   ├── message.py          # 领域消息类型（Message 联合，纯领域，不认识 wire）
│   └── serialization.py    # 边界 mapper：wire dict ↔ Message（唯一认识 wire 的地方）
├── ai/
│   └── client.py           # 智谱 API 封装
├── cli/
│   └── tui.py              # Rich 终端渲染（订阅 Runner 事件）
├── storage/
│   └── session.py          # JSON 文件会话持久化（落盘走 messages_to_api）
└── tools/
    └── file_tools.py       # 文件工具（read/write/list/edit）+ 沙箱

evals/                      # Eval 框架
├── types.py                # Task / RunResult / Grade 数据结构
├── tasks.py                # 任务定义
├── graders.py              # 评分器（检查世界状态，不检查措辞）
├── harness.py              # run_task()：跑任务 + 收集结果
└── run_eval.py             # CLI 入口：跑全部任务，输出报告

tests/                      # 测试（pytest，共 40 个）
├── agent/test_runner.py    # Runner 生命周期测试（6 个）
├── agent/test_state.py     # AgentState 不变量测试（6 个）
├── storage/test_session.py # SessionStore CRUD 测试（11 个）
└── evals/                  # Eval 零成本测试（FakeClient）

work_space/                 # Agent 文件沙箱（工具读写的唯一区域，运行时产生）
storage/sessions/           # 会话持久化目录（一会话一目录，运行时产生）
```

## 快速开始

```bash
# 安装依赖（推荐 uv）
uv sync

# 配置 API Key
cp .env.example .env
# 编辑 .env，填入 API_KEY（可选 MODEL，默认 glm-5.2）

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

### 领域消息模型 + 边界序列化（message.py / serialization.py）

消息在领域内部是类型化的 `Message`（`SystemMessage / UserMessage / AssistantMessage / ToolMessage` 的联合），不再是松散 dict：

- **类型只保证"单条消息的形状"**——如 `ToolMessage` 必带 `tool_call_id`、只有 `AssistantMessage` 携带 `tool_calls`。
- **列表层 / 关系层不变量类型管不着**——"非空""第一条是 system""每个 tool_call 必须有对应回应"这三条，仍由 `state.py` 的 `_validate_invariants` 在运行时校验，类型化之后一条都没少。
- **wire 格式（供应商的 JSON 形状）被锁死在 `serialization.py` 这一个边界**：`message_from_api`（dict→Message）/ `message_to_api`（Message→dict）。GLM 的 `reasoning_content`、嵌套的 `tool_calls.function` 这些噪音，`runner.py` 和 `message.py` 一概不认识——将来换厂商，只改这一个文件。

由此，dict 只被允许出现在三个地方：**调 `client.chat` 的出口**、**会话落盘 / 读盘**、以及 **runner 内部的流式累积器**（临时态，不跨边界、不带不变量，故刻意不类型化）。

### 工具沙箱

所有文件操作限制在 `work_space/` 和 `storage/` 内。通过 `resolve()` + `relative_to()` 阻止 `../` 路径穿越。

## 学习路线

> **组织原则**：按 **Agent 专属知识**划分，每项标注它**解决什么问题** + 当前状态。
> 通用工程素养（Protocol 抽象、依赖倒置、YAGNI、边界/反腐蚀层）不在此列——它们换个项目照样用，已在"消息类型化 + 序列化边界"那一役练过。

### 已完成的基础

- **ReAct 循环**（runner.py，事件流驱动）
- **工具调用 + 工具沙箱**（file_tools.py）+ max_turns 死循环保护
- **会话存储**（session.py，JSON 持久化）
- **领域消息类型 `Message` + 序列化边界**（message.py / serialization.py，dict↔Message）
- **Eval 骨架**（evals/，世界状态判据）+ 流式输出

### Agent 知识地图（✅ 已做 / 🟡 部分 / 🔲 未做）

| 关注点 | 知识点 | 解决的问题 | 状态 |
|---|---|---|---|
| **决策核心** | ReAct 循环 | 模型不能直接行动，需"想→做→看"交替 | ✅ |
| | 规划 / 任务分解（plan-and-execute、Reflexion） | 复杂任务拆解 + 碰壁后反思重规划 | 🔲 |
| **行动能力** | 工具调用 + **工具描述设计** | 接地到真实动作；模型读着描述选工具 | 🟡 |
| | 并行工具调用 | 一轮多个工具，串行等 = 白拉长延迟 | 🔲 |
| | MCP（Model Context Protocol） | 工具接入标准化 | 🔲 |
| **上下文 / 记忆** | **上下文工程**（窗口管理 / caching / token 成本） | 窗口有限 + 每次全量重发 = 第一约束 | 🔲 |
| | 压缩 / 摘要（compaction） | 对话一长就爆窗口、烧钱 | 🔲 |
| | 长期记忆 / RAG 检索 | 私有/最新数据 + 跨会话记忆 | 🔲 |
| **可靠性** | 结构化输出 + 重试恢复 | 模型会吐坏 JSON，要兜住 | 🟡 |
| | 护栏 / **prompt injection 防护** | 工具结果是不可信输入，能劫持模型 | 🟡 |
| **可观测 / 评估** | Tracing（Langfuse） | 黑盒看不见轨迹就没法调试 | 🔲 |
| | Eval 深化（任务覆盖 / LLM-as-judge / 稳定性） | 输出不确定 → 量化判好坏 | 🟡 |
| | 成本 / 延迟工程（缓存 / 模型路由） | N 次调用成本延迟叠加 | 🔲 |
| **规模化** | 多 Agent 协作 / 编排 | 单 agent 管不过来，需分工 / 专精 / 监督 | 🔲 |

### 主线（推荐学习顺序）

1. **上下文工程** ← 当前。先**看见问题**：给 agent 加 token 计数，亲眼看上下文怎么失控；再做 prompt caching → 截断策略 → 摘要压缩。
2. **长期记忆 / RAG**：接上空着的 `storage/memory/`，让 agent 跨会话记事、按需检索（第一站的自然延伸：装不下的外置 + 检索）。
3. **Langfuse trace**（穿插）：调试之眼，顺带验证"事件流驱动"这个架构赌注。
4. **往后**：并行工具调用、prompt injection 防护、MCP、多 Agent。

> **为什么主线是上下文 / 记忆**：普通程序的难点在逻辑，agent 的难点在**上下文**——模型只知道窗口里的东西，而你每次调用都在重新组装它的整个世界观。循环、工具、存储都是骨架；怎么管理"装进窗口的东西"，才是 agent 工程区别于普通开发的核心，也是目前最大的缺口（消息现在无限增长）。

## Tech Stack

- **Python 3.11+**
- **智谱 GLM**（默认 glm-5.2，经 `MODEL` 环境变量配置）— 通过 zai-sdk 调用
- **Rich** — 终端 UI 渲染
- **pytest** — 测试框架
- **Langfuse** — 可观测性（已引入，待集成）
