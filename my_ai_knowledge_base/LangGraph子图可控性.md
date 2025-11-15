# LangGraph子图可控性

下面我给你一份 **工程级、实用的《LangGraph 子图（SubGraph）可控性指南》**，帮你理解和落地如何在复杂工作流中管理子图执行、路由控制、状态共享和中断恢复。

------

# 🚀 一、LangGraph 子图（SubGraph）是什么？

在 LangGraph 中：

- **子图（SubGraph） = 独立可执行的 Graph，可以嵌入到主 Graph 中作为节点执行**
- 支持自己的 State、节点、Edge
- 可以和主图共享 State 或部分 State
- 可以独立 checkpoint、流式执行、HITL 中断

作用：

- 拆分复杂任务 → 可复用模块化
- 复杂 ReAct / RAG 工作流 → 每个子流程独立可控
- 支持多代理协作 → 每个代理可以是一个子图
- 支持断点续跑 → 子图可单独 resume

------

# 🧱 二、子图可控性核心概念

1. **Entry / Exit**
   - 子图可以定义入口节点和出口节点
   - 主图调用子图时，只关心入口与出口
2. **State 管理**
   - 子图可以共享主图 State
   - 或使用自己的局部 State（可复用子图）
3. **Checkpoint 支持**
   - 子图执行可单独 checkpoint
   - 主图 checkpoint 会包含子图状态快照
4. **Interrupt / Send**
   - 子图内可 HITL 中断，外部通过 send 继续执行
   - 子图中断不会影响主图未完成节点，可恢复
5. **Conditional Edge / Router**
   - 子图出口可以根据内部状态返回不同结果
   - 主图根据返回值路由到不同节点

------

# 🧠 三、子图可控性的典型使用模式

### 1）模块化 ReAct

- 主图：Planner → 子图 Executor → Critic
- Executor 子图内部执行 RAG + 工具
- 子图可单独流式执行、HITL 中断、checkpoint

### 2）多代理协作

- 每个代理对应一个子图
- 主图只负责路由和状态共享
- 子图内独立执行 workflow、可复用

### 3）任务拆解与复用

- 主图调用子图执行子任务
- 子图返回子任务结果
- 主图根据子图结果动态路由到下一任务

------

# 🧩 四、子图实现示例

## 1）定义子图

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict

class SubState(TypedDict):
    sub_input: str
    sub_result: str

def sub_node1(state: SubState):
    # 模拟处理
    return {"sub_result": f"处理 {state['sub_input']}"}

subgraph = StateGraph(SubState)
subgraph.add_node("sub_node1", sub_node1)
subgraph.set_entry_point("sub_node1")
subgraph.add_edge("sub_node1", END)
```

------

## 2）主图调用子图

```python
from typing import TypedDict

class MainState(TypedDict):
    main_input: str
    subgraph_output: str

def call_subgraph(state: MainState):
    # 将主图状态传给子图
    sub_state = {"sub_input": state["main_input"]}
    # 执行子图
    sub_result = subgraph.invoke(sub_state)
    return {"subgraph_output": sub_result["sub_result"]}

main_graph = StateGraph(MainState)
main_graph.add_node("call_subgraph", call_subgraph)
main_graph.set_entry_point("call_subgraph")
main_graph.add_edge("call_subgraph", END)

app = main_graph.compile()
```

------

## 3）运行并可控

```python
out = app.invoke({"main_input": "子任务A"})
print(out)
# 输出: {'subgraph_output': '处理 子任务A'}
```

- 可在子图内部加 checkpoint
- 可在子图内部中断等待用户输入
- 子图可独立流式输出 token
- 子图返回结果后主图继续下一节点

------

# 🔧 五、子图可控性强化技巧

1. **局部 State + 全局 State**
   - 子图独立管理复杂流程状态
   - 主图只管理全局变量和路由
   - 避免状态污染
2. **Checkpoint**
   - 子图独立 checkpoint
   - 主图 checkpoint 可包含子图 snapshot
   - 支持断点续跑
3. **Interrupt + Send**
   - 子图内任意节点可 HITL 中断
   - send 可从中断点继续执行子图
   - 主图不受影响
4. **Conditional Exit / Router**
   - 子图出口可根据状态返回不同结果
   - 主图可动态选择下一节点
   - 支持复杂多代理协作
5. **流式输出**
   - 子图节点可流式输出 token / 事件
   - 主图可汇总多子图输出到前端或日志系统

------

# 🎯 六、总结

**LangGraph 子图可控性 = 可独立执行 + 独立 checkpoint + HITL 中断 + 状态可共享 + 条件路由**。

- **优势**：复杂工作流拆解、任务复用、多代理协作、可恢复执行、可流式输出
- **核心**：子图是主图的可控模块，支持独立观察、独立中断、独立流式
- **典型场景**：ReAct workflow、RAG + Tool workflow、多代理任务协作

一句话：

> **子图 = 可控、可复用、可中断、可恢复的模块化智能体执行单元。**

------

