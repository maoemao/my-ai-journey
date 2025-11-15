# LangGraph ReAct应用

下面给你一份 **工程级、可直接用于生产的《LangGraph ReAct（Reason+Act）应用指南》**。
 重点是：**如何在 LangGraph 中落地一个可控、可观测、可恢复的 ReAct Agent**，并适配 DeepSeek / GPT / 自定义工具链。

我不会讲学术概念，只讲“怎么写、怎么跑、为什么这么设计”。

------

# 🚀 一、什么是 ReAct，在 LangGraph 里怎么落地？

ReAct = **Reason（推理） + Act（行动）循环**

传统 ReAct 结构：

```
Thought → Action → Observation → Thought → Action → ...
```

在 LangChain 里，是 LLM 自由写链式逻辑。
 在 LangGraph 里，更适合把它变成：

### ✔ 明确的节点

### ✔ 共享状态

### ✔ 条件路由

### ✔ 工具调用节点

### ✔ 中断恢复

也就是一个 **可控的有限状态机版本的 ReAct**。

这比“让 LLM 自己乱跳”强太多，更适合真实生产场景。

------

# 🧩 二、LangGraph ReAct 的最佳拆解结构

推荐你把 ReAct 切成 4 个节点：

### 1）`reason_node`

LLM 分析当前状态，决定下一步行动。

### 2）`router`

解析 reason 的 output：

- 如果要用工具 → 跳到 `tool_node`
- 如果可以直接回答 → 跳到 `final_answer_node`

### 3）`tool_node`

执行工具（搜索、SQL、RAG、API 等），写入 observation。

### 4）`final_answer_node`

LLM 根据 observation 汇总答复。

完整流程：

```
reason → router → (tool → reason → ...) OR final_answer
```

这个就是 **可控版 ReAct**。

------

# 🧱 三、定义 State（ReAct 必须有的字段）

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: List[BaseMessage]   # 历史对话
    thought: str                  # LLM 推理内容
    action: str                   # 工具动作
    action_input: str             # 工具参数
    observation: str              # 工具返回
```

LangGraph 会自动合并这些字段。

------

# 🧠 四、构建 ReAct Agent 的核心节点

## ⭐ 1）Reason 节点（推理）

这是 ReAct 的 “Thought” 部分。

```python
def reason_node(state: State):
    prompt = """
你是 ReAct agent，请按以下格式输出：
Thought: ...
Action: <tool_name> 或 "none"
Action Input: <参数，没有则为空>
"""
    msg = llm.invoke([
        *state["messages"],
        HumanMessage(prompt)
    ])
    
    # 假设你解析出了如下格式：
    thought, action, action_input = parse(msg.content)
    
    return {
        "thought": thought,
        "action": action,
        "action_input": action_input,
        "messages": state["messages"] + [msg]
    }
```

解析 output 格式是关键，你可以严格控制格式。

------

## ⭐ 2）Router 节点（决定走工具还是结束）

```python
def router(state: State):
    if state["action"] == "none":
        return "final_answer"
    else:
        return "tool"
```

------

## ⭐ 3）Tool 节点（执行工具动作）

例如搜索工具：

```python
def tool_node(state: State):
    action = state["action"]
    param = state["action_input"]

    if action == "search":
        result = search_api(param)
    elif action == "sql":
        result = sql_executor(param)
    else:
        result = f"Unknown tool: {action}"

    return {
        "observation": result,
        "messages": state["messages"] + [
            AIMessage(f"Observation: {result}")
        ]
    }
```

工具结果写入 observation。

------

## ⭐ 4）最终回答节点

```python
def final_answer(state: State):
    msg = llm.invoke(
        state["messages"] + [
            HumanMessage("基于上面的 observation 给出最终答案")
        ]
    )
    return {
        "messages": state["messages"] + [msg]
    }
```

------

# 🧱 五、构建 LangGraph 图（最核心部分）

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)

graph.add_node("reason", reason_node)
graph.add_node("tool", tool_node)
graph.add_node("final_answer", final_answer)

graph.set_entry_point("reason")

graph.add_conditional_edges("reason", router, {
    "tool": "tool",
    "final_answer": "final_answer"
})

graph.add_edge("tool", "reason")
graph.add_edge("final_answer", END)

app = graph.compile()
```

这个图就是通用 ReAct agent 的标准结构。

- reason → tool → reason 循环
- reason → final → END

完全符合 ReAct 的设计。

------

# 🧪 六、实际运行（流式）

```python
for event in app.stream(
    {"messages": [HumanMessage("地球到月球有多远？")]},
    config={"thread_id": "u01"}
):
    print(event)
```

输出会包含：

- 推理（Thought）
- 工具选择（Action）
- 工具结果（Observation）
- 最终回答（Answer）

你能看到完整 ReAct 过程。

------

# 🔥 七、加入 RAG = ReAct + Retrieval（企业最常用）

加一个 retriever 工具即可：

```
reason -> 
router -> 
(tool: retriever) -> 
reason -> 
final_answer
```

这比 “纯 RAG” 强，因为 LLM 能自主选择：

- 是否要检索？
- 检索几次？
- 是否需要进一步处理？

------

# 🧨 八、更高级功能：Interrupt + 人类确认（可控模式）

当 ReAct 生成高风险 action：

```
Action: run_sql
Action Input: "DROP TABLE user"
```

你可以在 router 阶段加一个中断：

```python
from langgraph.types import Interrupt

def router(state):
    if state["action"] == "run_sql" and "DROP" in state["action_input"]:
        raise Interrupt("dangerous_sql")
```

前端捕获后，让用户确认：

```python
app.send({"confirm": True}, config={"thread_id": "u01"})
```

这就是 **可控 Agent**，比 LangChain 的原始 ReAct 安全太多。

------

# 🧩 九、为什么 LangGraph 是 ReAct 的最佳落地框架？

| 能力                 | LangChain ReAct | LangGraph ReAct |
| -------------------- | --------------- | --------------- |
| 状态可控性           | ❌               | ✔               |
| 可中断               | ❌               | ✔               |
| 可恢复执行           | ❌               | ✔               |
| 流式输出             | 有限            | 全事件流        |
| 多节点控制           | 弱              | 强              |
| ReAct 工具调用稳定性 | 弱              | 强              |
| 可观测性             | 弱              | 极强            |

LangGraph 让 ReAct 从“实验算法”变成“工程级 agent 框架”。

------

# 🎯 十、总结（抓住本质）

### ✔ ReAct = 推理 + 行动循环

### ✔ LangGraph = 最适合实现可控 ReAct 的框架

（Reason → Router → Tool → Reason → … → Final Answer）

### ✔ Checkpoint + Send + Streaming

让 ReAct 成为：

- 可恢复
- 可观察
- 可控
- 可交互
- 可审计

的企业级智能体。

一句话：

> **LangChain 能 demo ReAct；LangGraph 才能部署 ReAct。**

------

