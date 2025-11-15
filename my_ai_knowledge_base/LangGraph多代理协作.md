# LangGraph多代理协作

下面我给你一份 **工程级、可直接落地的《LangGraph 多代理（Multi-Agent）协作指南》**，重点讲 **如何在 LangGraph 中实现多智能体协作、任务拆解、ReAct + RAG + 工具链整合**，并保证可控、可恢复、可审计。

------

# 🚀 一、LangGraph 多代理协作本质

LangGraph 的多代理协作，核心思想是：

> **每个代理（Agent）作为独立节点或子图（SubGraph）存在，共享 State 与 checkpoint，通过 Router/Coordinator 控制任务流和协作顺序。**

典型场景：

- Planner Agent：拆解任务 → 生成子任务列表
- Executor Agent：执行每个子任务（可调用工具、检索、SQL）
- Critic Agent：评估结果 → 提出优化建议
- Human Agent：在必要节点插入 HITL（中断 + send）

优势：

- 可控性高
- 可观察每个代理的动作
- 可恢复、可回放
- 支持复杂任务和长链路

------

# 🧱 二、核心组件关系

```
Planner Agent → Executor Agent → Critic Agent
        ↓                 ↑
       Router / Coordinator
        ↓
      State / Checkpoint（共享）
```

- **State**：多代理共享状态
- **Checkpointer**：每个代理状态快照
- **Router / Coordinator**：控制代理执行顺序
- **Node / SubGraph**：每个代理的执行逻辑
- **Interrupt + Send**：支持 HITL 或用户干预

------

# 🧠 三、定义共享 State（Multi-Agent 必备）

```python
from typing import TypedDict, List
from langchain_core.messages import BaseMessage

class MultiAgentState(TypedDict):
    messages: List[BaseMessage]          # 历史消息
    tasks: List[str]                      # 待办子任务列表
    current_task: str                     # 当前执行任务
    observations: dict                    # 各代理执行结果
    agent_logs: dict                      # 每个代理执行记录
```

每个代理读取、更新共享 State。

------

# 🧩 四、构建多代理节点（最小示例）

### 1）Planner Agent

负责拆解任务：

```python
def planner_node(state: MultiAgentState):
    task_desc = state["messages"][-1].content
    subtasks = llm.invoke([HumanMessage(f"拆解任务: {task_desc}")])
    state["tasks"] = subtasks.split("\n")
    state["agent_logs"]["planner"] = subtasks
    return state
```

------

### 2）Executor Agent

逐个执行子任务：

```python
def executor_node(state: MultiAgentState):
    current = state["tasks"].pop(0)
    obs = tool_or_llm_execute(current)
    state["current_task"] = current
    state["observations"][current] = obs
    state["agent_logs"].setdefault("executor", []).append((current, obs))
    return state
```

------

### 3）Critic Agent

评估所有执行结果：

```python
def critic_node(state: MultiAgentState):
    obs_summary = "\n".join([f"{k}: {v}" for k, v in state["observations"].items()])
    critique = llm.invoke([HumanMessage(f"评估结果:\n{obs_summary}")])
    state["agent_logs"]["critic"] = critique
    return state
```

------

# 🔀 五、构建多代理工作流 Graph

```python
from langgraph.graph import StateGraph, END

graph = StateGraph(MultiAgentState)

graph.add_node("planner", planner_node)
graph.add_node("executor", executor_node)
graph.add_node("critic", critic_node)

graph.set_entry_point("planner")

# 条件流：Planner → Executor → Critic → END
graph.add_edge("planner", "executor")
graph.add_edge("executor", "executor", condition=lambda s: len(s["tasks"])>0)
graph.add_edge("executor", "critic", condition=lambda s: len(s["tasks"])==0)
graph.add_edge("critic", END)

app = graph.compile()
```

说明：

- Executor 节点支持循环 → 处理所有子任务
- Critic 节点在任务完成后执行
- Planner 只执行一次，拆解任务

------

# 🧪 六、运行示例（多代理协作）

```python
events = app.stream(
    {"messages": [HumanMessage("整理2024销售分析报告")]},
    config={"thread_id": "u01"}
)

for e in events:
    print(e)
```

- 每个事件可观察：Planner 生成的任务、Executor 执行结果、Critic 评估
- 可结合 Checkpointer 保存状态，支持中断恢复
- 可结合 HITL 让用户对某些子任务确认或修改

------

# 🔥 七、工程级优化建议

1. **多代理子图拆分**
   - 每个代理可以是独立 SubGraph，便于调试和复用
2. **共享 State + agent_logs**
   - 记录每个代理的历史，方便审计、回放和异常追踪
3. **Checkpoint + Streaming**
   - 每个节点执行都可 checkpoint
   - 流式输出 Executor 的中间结果
   - Critic 可在流式过程中持续评估
4. **Router / Coordinator**
   - 可动态调整代理顺序
   - 支持分支策略，例如不同任务由不同 Executor 处理
5. **HITL / Send**
   - 让用户在 Planner 拆解任务或 Executor 执行前中断确认
   - 支持安全审查、敏感操作确认

------

# 🧩 八、多代理 ReAct + RAG 扩展（生产最常用）

1. Planner Agent → 拆任务
2. Executor Agent → 对每个子任务使用 RAG 检索 + LLM 生成
3. Critic Agent → 评估执行结果是否完整或合理
4. Router Agent → 根据 Critic 输出调整下一步任务
5. Human Agent → HITL 处理异常或敏感操作

整个流程可持续、可恢复、可审计。

------

# 🎯 九、总结

**LangGraph 多代理协作 = SubGraph + 多节点 + Router + State + Checkpoint + HITL + Streaming**

特点：

- 可控多任务、多代理协作
- 中断恢复 + 人机交互
- 可追溯执行轨迹
- 可扩展为 RAG + ReAct + 工具链 Agent
- 企业级复杂 workflow 理想实现

一句话：

> **LangChain 是工具调用框架，LangGraph 是企业级多代理协作引擎。**

------

