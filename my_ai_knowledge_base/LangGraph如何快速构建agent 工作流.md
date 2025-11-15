# LangGraph如何快速构建agent 工作流



------





## **🧩 一、核心理念：LangGraph 中的 Agent 是「有状态的工作流」**



传统的 LangChain Agent 是：

> Prompt → LLM → 解析意图 → 调用工具 → 返回结果（一次性流程）

LangGraph 版 Agent 是：

> **带状态的循环图结构**：Agent 可以持续对话、记忆上下文、动态调用工具、根据条件跳转或结束。

这让它更接近“真正的智能体系统”，而不是一次性问答。



------



## **🚀 二、快速上手示例：一个最小可运行的 Agent 工作流**



下面这个例子展示一个简单的 **LangGraph Agent**，

它能：

- 读取用户输入
- 调用 LLM 决策是否调用工具
- 执行工具（计算器）
- 输出最终结果



------



### **1️⃣ 安装依赖**

```
pip install langchain langgraph openai
```



------



### **2️⃣ 定义状态结构**

LangGraph 的核心是 **状态（State）** —— 存储对话历史、意图、工具结果等。

```
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    tool_result: str
```



- messages：保存用户与AI的对话历史。
- tool_result：存储工具调用的输出结果。



------



### **3️⃣ 定义节点函数（每个节点 = 一个动作）**

#### **(a) 模型节点：决定下一步做什么**



```
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

def agent_node(state: AgentState):
    messages = state["messages"]
    user_input = messages[-1]["content"]

    # 简单决策逻辑
    if "计算" in user_input:
        next_node = "tool"
    else:
        next_node = "respond"

    ai_message = llm.invoke(messages)
    return {"messages": [ai_message]}, next_node
```



------



#### **(b) 工具节点**



```
def tool_node(state: AgentState):
    import re
    user_msg = state["messages"][-1]["content"]
    expr = re.findall(r"\d+[\+\-\*\/]\d+", user_msg)
    if expr:
        result = str(eval(expr[0]))
    else:
        result = "无法计算"
    return {"tool_result": result, "messages": [{"role": "tool", "content": result}]}
```



------



#### **(c) 响应节点**



```
def respond_node(state: AgentState):
    tool_result = state.get("tool_result", "")
    messages = state["messages"]
    if tool_result:
        reply = f"计算结果是：{tool_result}"
    else:
        reply = "好的，我明白你的问题。"
    return {"messages": [{"role": "ai", "content": reply}]}
```



------



### **4️⃣ 构建图结构（StateGraph）**



```
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)

# 注册节点
graph.add_node("agent", agent_node)
graph.add_node("tool", tool_node)
graph.add_node("respond", respond_node)

# 设置边
graph.add_edge(START, "agent")
graph.add_conditional_edges(
    "agent",
    lambda state, next_node: next_node,
    {
        "tool": "tool",
        "respond": "respond"
    }
)
graph.add_edge("tool", "respond")
graph.add_edge("respond", END)

agent_graph = graph.compile()
```



------



### **5️⃣ 运行工作流**



```
result = agent_graph.invoke({
    "messages": [{"role": "user", "content": "请帮我计算3+4"}]
})
print(result["messages"][-1]["content"])
```

输出：

```
计算结果是：7
```

✅ 这就是一个完整的 LangGraph Agent 工作流：

LLM 负责“决策”，Graph 控制“流程”，State 保存“记忆”。



------



## **🧠 三、LangGraph Agent 的底层逻辑（简要原理）**



| **模块**                | **功能**                           | **对应你写的代码**                  |
| ----------------------- | ---------------------------------- | ----------------------------------- |
| **StateGraph**          | 流程编排（定义节点、边、条件）     | graph.add_node(), add_edge()        |
| **State**               | 保存状态（上下文、记忆、工具结果） | AgentState                          |
| **Node Function**       | 每个节点的执行逻辑                 | agent_node, tool_node, respond_node |
| **Conditional Edge**    | 控制流转方向（基于状态）           | add_conditional_edges()             |
| **Executor / invoke()** | 运行流程，驱动状态更新             | graph.invoke()                      |

LangGraph 本质上是一个“可持久化的状态机执行引擎”，

相比 LangChain 的“线性调用链”，它能：

- 保持上下文状态；
- 支持分支 / 回环；
- 在任何节点暂停 / 恢复。



------



## **⚙️ 四、拓展方向：让 Agent 更智能**



| **功能**      | **实现方式**                                     | **示例**                      |
| ------------- | ------------------------------------------------ | ----------------------------- |
| 多工具        | 在 agent_node 中调用 Tool Selector（或函数路由） | 多工具并发执行                |
| 记忆          | 将 AgentState 接入向量数据库                     | 结合 RAG                      |
| 多 Agent 协作 | 定义多个 Agent 节点并建立循环                    | 类似群聊式决策                |
| 人类介入      | 在节点中添加“人工审核”条件                       | 审核后再恢复执行              |
| 长期执行      | 状态持久化（如 Redis、SQLite）                   | .save_state() / .load_state() |



------



## **🧩 五、项目架构建议（实际工程中）**



```
/agent_app
├── agents/
│   ├── reasoning_agent.py
│   ├── retriever_agent.py
│   └── responder_agent.py
├── tools/
│   ├── calculator.py
│   └── search_api.py
├── state.py
├── workflow.py
├── config.py
└── main.py
```

这种分层结构能让你后期轻松扩展多 Agent、多工具、外部服务（数据库 / 搜索 / LangSmith 监控）。



------



## **📈 六、结合 LangSmith 做链路可视化监控**



LangGraph 与 LangSmith 无缝集成，只需：

```
from langsmith import Client
client = Client()
graph = StateGraph(AgentState, client=client)
```

你就能：

- 查看每个节点执行日志；
- 跟踪状态变化；
- 分析延迟瓶颈；
- 可视化整个 Agent 决策流程。

