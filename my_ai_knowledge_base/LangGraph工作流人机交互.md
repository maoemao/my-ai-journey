# LangGraph工作流人机交互

下面我给你一份 **实用、工程级的《LangGraph 工作流人机交互（Human-in-the-Loop, HITL）指南》**。
 侧重点是：**让复杂 Agent / RAG 工作流真正做到“能停下来等人类输入，再继续执行”的能力。**

这在生产版 AI 系统中非常关键，尤其是：

- 工具执行需要用户确认
- RAG 需要用户补充上下文
- Agent 规划有多种选项需人为选择
- 高风险操作需人工审批
- 多阶段任务需中途人工介入

LangGraph 是目前最适合做 HITL 的框架，它的 **状态机 + checkpoint + 中断机制** 天然支持这一能力。

------

# 🚀 一、人机交互在 LangGraph 中的本质

一句话：

> **LangGraph 通过将节点设为“等待事件（wait for user event）”的阻塞点，实现工作流暂停 + 用户交互 + 恢复执行。**

它依赖三个能力：

### ✔ 1）StateGraph —— 将每一步拆成可控节点

### ✔ 2）checkpoint —— 记录状态，方便恢复

### ✔ 3）.send() —— 用户发送事件，恢复执行

这三个组合起来，就可以构建：

```
AI → 等待用户 → AI → 等待用户 → AI
```

的完整链路。

------

# 🧱 二、实现人机交互的三种方式

LangGraph 支持三种 HITL 模式：

## **方式 1：等待用户事件（ChatGPT Agent 交互模型）**

最常见，也是官方推荐。

关键方法是：

### **app.stream()**

用于执行图并在“等待人类事件（Human event）”时暂停。

### **app.send()**

用于继续执行图。

这两个方法是 HITL 的核心。

------

# 🧪 三、最小可运行的人机交互 Demo

下面这个例子你可以直接用来构建“用户确认 → AI 继续执行”的流程。

### ✔ State 定义

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class State(TypedDict):
    messages: List[BaseMessage]
    user_confirm: bool
```

### ✔ 节点：让 AI 提问“是否继续？”

```python
def ask_user(state: State):
    return {
        "messages": state["messages"] + [
            AIMessage("我找到了方案，你要继续吗？输入 yes/no")
        ]
    }
```

### ✔ 阻塞节点：等待用户输入事件

```python
from langgraph.types import Interrupt

def wait_user(state: State):
    raise Interrupt("need_user_input")
```

### ✔ 用户确认后的执行节点

```python
def continue_work(state: State):
    return {
        "messages": state["messages"] + [
            AIMessage("好的，我继续执行任务")
        ]
    }
```

### ✔ Router：判断是否继续

```python
def router(state: State):
    if state.get("user_confirm") is True:
        return "continue"
    else:
        return "end"
```

------

# 🧠 构建完整 LangGraph 工作流

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(State)

graph.add_node("ask", ask_user)
graph.add_node("wait", wait_user)
graph.add_node("continue", continue_work)

graph.add_edge("ask", "wait")
graph.add_conditional_edges("wait", router, 
    {"continue": "continue", "end": END})

graph.set_entry_point("ask")

app = graph.compile()
```

------

# 🎮 四、运行：人机交互完整流程

## ✔ 第 1 步：运行 app.stream()

```python
events = app.stream(
    {"messages": [HumanMessage("开始任务")]},
    config={"thread_id": "123"}
)

for e in events:
    print(e)
```

输出会类似：

```
AI: 我找到了方案，你要继续吗？输入 yes/no
等待事件：need_user_input
```

工作流暂停了。

------

## ✔ 第 2 步：用户回应（send）

```python
events = app.send(
    {"user_confirm": True},
    config={"thread_id": "123"}
)

for e in events:
    print(e)
```

输出：

```
AI: 好的，我继续执行任务
```

流程继续运行到结束。

------

# 🧠 五、常见的人机交互模式（真实产品场景）

## **模式 A：用户确认（Approve/Reject）**

RAG 检索到内容，问用户：

- 是否使用该文档？
- 是否执行这个 SQL？
- 是否打开这个 URL？
- 是否同意工具操作？

代码：等待事件 → 用户 .send() → 路由继续。

------

## **模式 B：补充缺失信息**

Agent 发现用户输入缺少信息时自动问：

```
你要查询哪个客户？
```

等待写入：

```python
app.send({"customer_id": "U48291"})
```

然后继续执行。

------

## **模式 C：多选操作（用户选择分支）**

AI 给出三个计划：

- 方案 A：快速执行
- 方案 B：安全优先
- 方案 C：成本最低

用户选择：

```
选 B
```

Graph 路由到 B 节点执行。

------

## **模式 D：人类审查 + AI 继续**

比如：

- 财务审批
- 法律文书校对
- 敏感操作确认

AI 停下 → 人 review → AI 继续。

------

## **模式 E：任务拆解 + 用户决策**

AI 规划出 5 个子任务：

```
是否执行子任务3？
```

你可以决定执行哪些。

------

# 🧰 六、和 LangChain 相比的优势？

LangChain 只能：

- Memory 存消息
- Tool 调用
- 无法暂停执行
- 无法恢复执行
- 工程能力不足

LangGraph：

- Node = 独立步骤
- 状态可持久化
- 可暂停
- 可恢复
- 全局状态可控
- 人机交互天然支持

LangGraph = 企业级 Agent 生命周期管理器。

------

# 🧨 七、工程级最佳实践

| 使用场景      | 建议                            |
| ------------- | ------------------------------- |
| SaaS、Chatbot | SQLCheckpointer（多用户持久化） |
| Agent 工作流  | 把所有用户分支都做成 Interrupt  |
| 表单补充      | 使用事件字段 user_event:{…}     |
| 工具执行确认  | wait → user_confirm → route     |
| 审计回放      | 启用 run history + checkpoint   |

------

# 🎯 八、总结（本质理解）

LangGraph 的人机交互能力来自：

### ✔ 有状态（state）

### ✔ 可中断（Interrupt）

### ✔ 可恢复（send/resume）

### ✔ 可持久化（checkpoint）

### ✔ 可路由（conditional edges）

这五个能力组合，让 LangGraph 成为目前最适合做 “HITL Agent” 的框架。

一句话总结：

> **LangGraph 不是简单的 LLM Pipeline，而是可以暂停、等待、恢复、分支的状态机。**

------

