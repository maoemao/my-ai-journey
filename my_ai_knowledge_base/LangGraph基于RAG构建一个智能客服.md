# LangGraph基于RAG构建一个智能客服



------



## **一、系统架构设计**

```
用户输入 → LangGraph Agent → 判断意图 → 
   ├─ 检索知识库 (RAG) → 生成回答
   └─ 调用工具（如计算/查询） → 生成回答
→ 更新状态 → 返回给用户
```

组件说明：



- **状态 (State)**：保存对话历史、工具结果、上下文检索结果

- **节点 (Node)**：

  

  - 意图判断节点（LLM）
  - RAG 检索节点
  - 工具调用节点
  - 回答生成节点

  

- **边 (Edge)**：控制流程，可根据意图选择 RAG 或工具调用

- **执行器 (StateGraph)**：负责驱动工作流



------



## **二、准备依赖**



```
pip install langchain langgraph openai faiss-cpu
```



------



## **三、定义状态结构**



```
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class CustomerServiceState(TypedDict):
    messages: Annotated[list, add_messages]   # 对话历史
    intent: str                                # 用户意图
    rag_context: str                           # 检索到的文档上下文
    tool_result: str                           # 工具调用结果
```



------



## **四、构建节点函数**



### **1️⃣ 意图判断节点**



```
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def intent_node(state: CustomerServiceState):
    messages = state["messages"]
    user_input = messages[-1]["content"]
    
    prompt = f"""
    你是智能客服，判断用户意图属于以下类型：
    1. 查询知识库
    2. 调用工具
    请只返回 "knowledge" 或 "tool"
    用户消息: {user_input}
    """
    response = llm.invoke(messages + [{"role": "system", "content": prompt}])
    state["intent"] = "knowledge" if "knowledge" in response["content"].lower() else "tool"
    return state, state["intent"]
```



------



### **2️⃣ RAG 检索节点**



```
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

# 假设你已经构建好向量库
embedding = OpenAIEmbeddings()
vectorstore = FAISS.load_local("vector_db", embedding, allow_dangerous_deserialization=True)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever(), return_source_documents=False)

def rag_node(state: CustomerServiceState):
    query = state["messages"][-1]["content"]
    answer = qa_chain.run(query)
    state["rag_context"] = answer
    return state
```



------



### **3️⃣ 工具调用节点（示例：简单计算器）**



```
import re

def tool_node(state: CustomerServiceState):
    user_input = state["messages"][-1]["content"]
    expr = re.findall(r"\d+[\+\-\*\/]\d+", user_input)
    if expr:
        state["tool_result"] = str(eval(expr[0]))
    else:
        state["tool_result"] = "无法计算"
    return state
```



------



### **4️⃣ 回答生成节点**



```
def respond_node(state: CustomerServiceState):
    if state["intent"] == "knowledge":
        reply = state.get("rag_context", "抱歉，我没有找到相关信息。")
    elif state["intent"] == "tool":
        reply = f"计算结果是：{state.get('tool_result', '')}"
    else:
        reply = "抱歉，我不理解你的问题。"
    
    state["messages"].append({"role": "ai", "content": reply})
    return state
```



------



## **五、构建 LangGraph 工作流**



```
from langgraph.graph import StateGraph, START, END

graph = StateGraph(CustomerServiceState)

# 注册节点
graph.add_node("intent", intent_node)
graph.add_node("rag", rag_node)
graph.add_node("tool", tool_node)
graph.add_node("respond", respond_node)

# 设置边
graph.add_edge(START, "intent")
graph.add_conditional_edges(
    "intent",
    lambda state, next_node: state["intent"],
    {
        "knowledge": "rag",
        "tool": "tool"
    }
)
graph.add_edge("rag", "respond")
graph.add_edge("tool", "respond")
graph.add_edge("respond", END)

agent_graph = graph.compile()
```



------



## **六、运行示例**



```
initial_state = {
    "messages": [{"role": "user", "content": "请帮我查人工智能白皮书"}],
    "intent": "",
    "rag_context": "",
    "tool_result": ""
}

result = agent_graph.invoke(initial_state)
print(result["messages"][-1]["content"])
```

输出：

```
[智能客服回答内容，基于知识库检索]
```

如果用户输入类似 "请计算3+5"，Agent 会自动走工具节点，并返回：

```
计算结果是：8
```



------



## **七、拓展和优化**



1. **多轮对话支持**

   

   - 使用 state["messages"] 记录历史，结合 LangGraph + LangChain memory 可实现长期对话。

   

2. **多工具支持**

   

   - 在意图节点判断多种工具类型，通过条件边跳转不同工具节点。

   

3. **人工介入**

   

   - 可以在节点中添加“人工审核”，节点执行暂停，等待人工修改状态后继续。

   

4. **可视化与监控**

   

   - 使用 LangSmith 集成，观察节点执行顺序、状态变化、工具调用结果。

   

5. **多用户并发**

   

   - 每个用户单独实例化 CustomerServiceState，或者把状态存储在 Redis/数据库中，保证多用户并发安全。

   

------



总结：

- **LangGraph** 负责流程控制、状态管理、分支回环
- **RAG (LangChain)** 负责知识检索问答
- **工具节点** 负责计算、查询、API 调用等操作
- **回合节点** 生成最终回答



组合起来就能做一个 **智能客服 Agent**，支持多轮、工具调用、知识库检索、分支决策。



------



