# 如何用LangChain 开发agent智能体

这个问题问得非常好，也是你真正进入 LangChain 实战开发的核心：



> **如何用 LangChain 从零开发一个 Agent 智能体**。



我会带你从概念 → 架构 → 代码 → 调试 → 部署完整讲清楚，

看完这份指南，你就能自己写出一个可调用工具、有记忆、有检索能力的智能体。



------



## **🧩 一、Agent 是什么？**



在 LangChain 中，**Agent（智能体）= LLM + 工具 + 记忆 + 策略控制**。

通俗地说：



> 它不只是聊天，而是能“自己思考、决定要调用什么工具、执行后再总结结果”。



------





### **✅ Agent 架构图（逻辑）**

```
用户输入 → LLM 分析任务 → 调用工具（Tool/Chain）→ 得到结果 → LLM 整合 → 输出
```

例如：

> “帮我查一下广州天气并换算成美元旅游预算。”

LLM 可能执行：

1. 调用 get_weather("广州")
2. 调用 get_exchange_rate("CNY")
3. 总结计算，返回答案。

------



## **🧠 二、Agent 核心组件**



| **组件**            | **作用**                   |
| ------------------- | -------------------------- |
| **LLM**             | 智能体的“思考大脑”         |
| **Tools**           | 它可调用的“手脚”           |
| **Memory**          | 记忆系统，维持上下文       |
| **Prompt Template** | 指导智能体如何行动         |
| **AgentExecutor**   | 运行时引擎，管理链路与状态 |



------



## **⚙️ 三、快速创建一个 Agent（最小可运行示例）**



### **1️⃣ 安装依赖**

```
pip install langchain langchain-openai
```



### **2️⃣ 定义工具**

```
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询城市天气"""
    weather = {"广州": "晴 25°C", "北京": "阴 15°C"}
    return f"{city}天气：{weather.get(city, '未知')}"
```



### **3️⃣ 创建智能体**

```
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
tools = [get_weather]

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True  # 打印思考过程
)
```



### **4️⃣ 运行测试**

```
agent.invoke("请告诉我广州的天气。")
```

输出日志：

```
> Thought: 我需要查询广州天气
> Action: get_weather
> Action Input: 广州
> Observation: 广州天气：晴 25°C
> Final Answer: 广州今天晴，25°C。
```



------



## **🔧 四、Agent 类型选择（不同策略）**

| **AgentType**                        | **说明**                                    | **适用场景**           |
| ------------------------------------ | ------------------------------------------- | ---------------------- |
| **ZERO_SHOT_REACT_DESCRIPTION**      | 经典 ReAct 框架，自动思考-行动-总结         | 通用场景               |
| **CONVERSATIONAL_REACT_DESCRIPTION** | 支持多轮对话 + Memory                       | 聊天类                 |
| **OPENAI_FUNCTIONS**                 | 使用函数调用机制（OpenAI Function Calling） | 工具调用准确率高       |
| **STRUCTURED_CHAT_ZERO_SHOT_REACT**  | 支持结构化输入输出                          | 企业应用               |
| **PLAN_AND_EXECUTE**                 | 先规划再执行                                | 复杂任务（多步骤推理） |



------



## **💬 五、加入记忆（Memory）**



让 Agent 记得你之前说过什么：

```
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
```

执行：

```
agent.invoke("我叫猫哥。")
agent.invoke("我叫什么？")  # Agent 能回答“猫哥”
```



------



## **📚 六、加入知识库或检索工具（RAG）**



让 Agent 能查询外部知识：

```
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA

docs = ["LangChain 是一个用于构建 LLM 应用的框架", "它支持记忆、工具调用、Agent"]
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(docs, embeddings)

retriever = vectorstore.as_retriever()
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
```

把 QA 链注册成一个工具：

```
from langchain.tools import Tool

qa_tool = Tool(
    name="knowledge_base",
    func=qa_chain.run,
    description="从知识库中检索LangChain相关信息"
)

agent = initialize_agent(
    [qa_tool],
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

> 现在你就有了一个能查知识的智能体（带知识库的 RAG Agent）。



------



## **🔍 七、Agent 调试与监控**

### **1️⃣ 控制台调试**

verbose=True 可直接显示每步思考过程。

### **2️⃣ 自定义日志（CallbackHandler）**

可打印每次工具调用、输出等。



### **3️⃣ LangSmith 调试（推荐）**

```
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="你的LangSmith密钥"
```

👉 在 [smith.langchain.com](https://smith.langchain.com)![Attachment.tiff](Attachment.tiff) 上实时可视化查看：



- 每步 Action
- Prompt 内容
- Token 消耗
- 工具调用链路



------



## **🧠 八、Agent 的进阶架构**



你可以把 Agent 设计为以下几层：

| **层级**      | **说明**                           |
| ------------- | ---------------------------------- |
| **LLM 层**    | ChatOpenAI / Claude / Ollama       |
| **Memory 层** | 对话上下文记忆                     |
| **Tool 层**   | 外部能力（数据库、API、知识库等）  |
| **Logic 层**  | Agent 策略（ReAct / Plan&Execute） |
| **Server 层** | LangServe / FastAPI 部署           |



------



## **☁️ 九、部署你的 Agent（LangServe）**



LangServe 可以直接把 Agent 包成 API 服务。

```
from langserve import serve

serve(agent, port=8080)
```

现在访问：

```
POST http://localhost:8080/invoke
Body: {"input": "帮我查下广州天气"}
```

即可获得智能体响应。

------



## **🚀 十、总结：LangChain 智能体开发路线图**



| **阶段**       | **学习重点**                   | **示例**  |
| -------------- | ------------------------------ | --------- |
| **入门阶段**   | initialize_agent() + 简单工具  | 查询天气  |
| **进阶阶段**   | 多工具 + Memory                | 智能助手  |
| **专业阶段**   | 知识检索 + 工具链 + 调试       | RAG Agent |
| **工程化阶段** | LangServe + LangSmith + 持久化 | 生产部署  |



------

