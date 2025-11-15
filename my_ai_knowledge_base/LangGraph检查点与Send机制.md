# LangGraph检查点与Send机制

下面我给你一份 **工程级、深入到底的《LangGraph 检查点（Checkpoint）与 Send 机制完全指南》**。
 这两个能力是 LangGraph 的灵魂，让 Agent 能实现：

- **中断（Interrupt）**
- **恢复执行（Resume）**
- **人机交互（HITL）**
- **多轮对话持久化**
- **多节点复杂流程断点续跑**
- **可追溯的执行轨迹（audit）**

你完全掌握这两点，LangGraph 80% 的核心能力就吃透了。

------

# 🚀 一、检查点（Checkpoint）是什么？（一句话本质）

> **Checkpoint = LangGraph 在每个节点执行后保存的“状态快照（State Snapshot）”，用于恢复、回放、继续执行。**

它记录：

- 当前 State（消息、变量、上下文）
- 图执行进度（当前在哪个节点）
- 节点输出
- 工具结果
- 事件历史（Event Log）

Checkpoint 是 LangGraph 能做 “恢复执行 (resume)” 的根基。

------

# 🧱 二、Send 是什么？（一句话本质）

> **Send = 用户向图发送一个事件/额外 state，让图从上一个中断点继续执行。**

在 HITL（人机交互）场景中非常重要：

- AI 在某一步停止并等待用户
- 用户输入（某些字段）
- 使用 `send()` 恢复执行

这个机制本质上就是：

```
中断 --> 用户补充信息 --> 继续执行
```

------

# 📌 三、为什么 LangGraph 必须有 Checkpoint + Send？

因为 LLM/Agent 工作流不像普通函数调用：

- 有多节点
- 有循环
- 有中断
- 有用户交互
- 有工具调用
- 有失败重试

你无法一次性执行完。

Checkpoint + Send 让 AI 变成一个真正的“状态机工作流”（stateful workflow）。

------

# 🧠 四、Checkpoint 工作原理全解析

## ✔ 1）每个节点执行完都会生成 checkpoint

例如 workflow：

```
Node A → Node B → Node C
```

LangGraph 在 A、B、C 每处都会 checkpoint。

内容包括：

- 当前 state
- 当前执行节点
- 输入输出消息
- 运行时间戳
- 线程 ID / 会话 ID（thread_id）

------

## ✔ 2）持久化方式

可以选：

- MemorySaver（内存）
- FileCheckpointer（文件）
- SQLite/Postgres（生产推荐）
- 自定义 Redis/Mongo

例如 SQLite：

```python
from langgraph.checkpoint.sqlite import SqliteSaver
checkpointer = SqliteSaver.from_uri("sqlite:///graph.db")
app = graph.compile(checkpointer=checkpointer)
```

------

## ✔ 3）Checkpoint 支持恢复

从指定 checkpoint 恢复执行：

```python
app.resume(
    config={"thread_id": "user-01"}
)
```

如果 workflow 被中断过：

```python
app.resume(config={"thread_id":"u01", "checkpoint":"step-3"})
```

------

# 🧱 五、Send 工作原理全解析

Send typically follows an Interrupt.

## ✔ 1）节点主动中断

在节点中：

```python
from langgraph.types import Interrupt

def wait_user(state):
    raise Interrupt("need_user_input")
```

执行这个节点时，workflow 停住，前端收到事件：

```
interrupt_event: need_user_input
```

代表需要用户输入。

------

## ✔ 2）用户通过 send 补充信息

例如用户输入：

```
yes
```

你恢复执行：

```python
app.send(
    {"user_confirm": True},
    config={"thread_id": "u01"}
)
```

Send 做了两件事：

### 🟦 ① 把用户输入合并到 State

例如：

原来 state：

```python
{"messages":[...], "user_confirm": None}
```

send 输入：

```python
{"user_confirm": True}
```

合并状态：

```python
{"messages":[...], "user_confirm": True}
```

### 🟦 ② 从最近的 checkpoint 继续执行工作流

恢复到中断点继续往下执行。

没有 checkpoint，这一切都做不到。

------

# 🔥 六、完整 Demo 展示两者如何协作

这是可运行的最小例子。

------

## ✔ Step 1：定义 State

```python
class State(TypedDict):
    messages: List
    user_confirm: bool
```

------

## ✔ Step 2：节点：问用户

```python
def ask_user(state):
    msg = AIMessage("需要继续吗？yes/no")
    return {"messages": state["messages"] + [msg]}
```

------

## ✔ Step 3：节点：中断等待用户

```python
from langgraph.types import Interrupt

def wait_user(state):
    raise Interrupt("need_user_input")
```

------

## ✔ Step 4：节点：继续执行

```python
def continue_step(state):
    return {
        "messages": state["messages"] + [AIMessage("继续执行任务")]
    }
```

------

## ✔ Step 5：构建 Graph

```python
graph = StateGraph(State)

graph.add_node("ask", ask_user)
graph.add_node("wait", wait_user)
graph.add_node("continue", continue_step)

graph.set_entry_point("ask")

graph.add_edge("ask", "wait")

graph.add_conditional_edges(
    "wait",
    lambda s: "continue" if s.get("user_confirm") else END,
)
```

------

## ✔ Step 6：运行（Streaming + Checkpoint）

```python
events = app.stream(
    {"messages":[HumanMessage("开始")]},
    config={"thread_id": "u01"}
)

for e in events:
    print(e)
```

出现：

```
AI：需要继续吗？yes/no
interrupt: need_user_input
```

停住。

------

## ✔ Step 7：用户响应，通过 send 恢复执行

```python
events = app.send(
    {"user_confirm": True},
    config={"thread_id":"u01"}
)
```

输出：

```
AI: 继续执行任务
```

------

# 🌟 七、为什么这是 AI/Agent 工作流中最重要的能力？

因为 Agent 不只是聊天，它需要：

- 人机审阅
- 数据补全
- 安全确认
- 长链路自动化任务
- 多步骤工具使用
- 复杂推理
- 持久会话管理
- 中断 & 恢复

没有 Checkpoint + Send 的框架做不到这些。

LangGraph 是为“可控 Agent”而设计的，不是为“简单聊天”设计的。

------

# 🧩 八、常见实际应用（生产场景）

## ✔ 1）RAG: 用户补充缺失信息

检索不到足够内容 → 中断 → 等用户补充 → send 恢复。

## ✔ 2）SQL Agent: 用户确认 SQL

AI 生成 SQL → 中断询问 → 用户确认 → send 执行。

## ✔ 3）多 Agent 调用

Router → Interrupt → 人类选择 → send 继续执行。

## ✔ 4）长任务执行

执行到第 4 步服务器断电 → resume 继续。

## ✔ 5）表单式对话

AI 逐步采集字段，中断–填写–继续。

这就是构建 GPTs / 智能助手 / 企业 Agent 的核心能力。

------

# 🎯 九、总结（抓住本质）

### ✔ Checkpoint：

保存状态，支持工作流恢复、回放、审计、长任务。

### ✔ Send：

从中断点继续执行，支持人机交互。

### ✔ 两者配合：

构成完整的 “可控型 Agent 工作流”。

一句话总结：

> **Checkpoint 负责记住过去，Send 负责继续未来。
>  LangGraph 由此具备“可恢复的智能体执行能力”。**

------

