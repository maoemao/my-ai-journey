# LangChain 自定义工具调试

好问题，这个是你在用 LangChain 开发智能 Agent 时必定会碰到的实际环节。

主题是：

> **LangChain 自定义工具（Custom Tools）调试与可视化链路追踪**



------



## **🧩 一、工具（Tool）的本质**

在 LangChain 的设计中：

> **Tool = 一个 LLM 可以调用的可执行函数（Function）**

它允许大模型具备**执行能力**，比如：

- 访问数据库
- 查询天气
- 调用外部 API
- 执行 Python 计算

**每个 Tool 都有输入、输出、描述文档**，LLM 会根据描述决定何时调用它。



------



## **⚙️ 二、创建一个自定义 Tool**



LangChain 提供了两种主流方式：



### **✅ 1️⃣ 使用** 

### **@tool**

###  **装饰器（推荐）**



```
from langchain.tools import tool

@tool("get_exchange_rate", return_direct=False)
def get_exchange_rate(currency: str) -> str:
    """
    获取美元兑指定货币的汇率。
    参数:
        currency: 货币代码，如 CNY, EUR, JPY。
    """
    rates = {"CNY": 7.1, "EUR": 0.9, "JPY": 151.3}
    return f"1 USD = {rates.get(currency.upper(), '未知')} {currency}"
```

> LLM 在看到这个函数的描述后，会自动学会如何调用它。



------



### **✅ 2️⃣ 手动继承** 

### **BaseTool**

### **（高级控制）**



```
from langchain.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field

class WeatherInput(BaseModel):
    city: str = Field(..., description="城市名称")

class WeatherTool(BaseTool):
    name = "get_weather"
    description = "获取城市天气信息"
    args_schema: Type[BaseModel] = WeatherInput

    def _run(self, city: str):
        return f"{city} 今天晴，25°C"

    async def _arun(self, city: str):
        return self._run(city)
```



------



## **🧠 三、将 Tool 注入到 Agent 中**



```
from langchain.agents import initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")
tools = [get_exchange_rate, WeatherTool()]

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

现在 Agent 已经可以自动决定调用哪个工具，例如：

```
agent.invoke("告诉我上海的天气，并换算成美元价格下的旅游预算。")
```



------



## **🧩 四、调试工具调用过程**



### **✅ 1️⃣** 

### **verbose=True**

###  **模式（最简单）**



执行时会在控制台打印：

- 模型思考过程（Thought）
- 工具调用名称与参数
- 工具返回值
- 最终回答



------



### **✅ 2️⃣ 使用** 

### **CallbackHandler**

###  **自定义日志调试**



LangChain 的回调系统允许你在工具调用前后插入钩子。

```
from langchain.callbacks.base import BaseCallbackHandler

class MyDebugHandler(BaseCallbackHandler):
    def on_tool_start(self, serialized, input_str, **kwargs):
        print(f"🧰 [Tool Start]: {serialized['name']}({input_str})")

    def on_tool_end(self, output, **kwargs):
        print(f"✅ [Tool Result]: {output}")

agent = initialize_agent(
    tools,
    llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callbacks=[MyDebugHandler()],
)
```

这能帮你准确定位工具调用输入输出。



------



### **✅ 3️⃣ 使用 LangSmith 可视化调试（推荐）**



LangSmith 是 LangChain 官方推出的链路监控平台。



**使用方法：**

```
pip install langsmith
export LANGCHAIN_TRACING_V2="true"
export LANGCHAIN_API_KEY="你的LangSmith密钥"
```

执行任意 agent 调用时，会自动上传到 LangSmith 平台，你可以在 Web 界面看到：



- 每个工具调用的参数
- LLM 生成的思考链
- 输入输出延迟
- Token 消耗情况



这在团队协作开发 Agent 时非常有用。



------



## **🧠 五、调试要点与常见问题**



| **问题**                    | **可能原因**                  | **解决建议**                            |
| --------------------------- | ----------------------------- | --------------------------------------- |
| 工具从未被调用              | 描述不够清晰，LLM不理解何时用 | 修改 description 用自然语言清楚说明用途 |
| 工具被调用参数错误          | 模型未正确解析输入格式        | 定义 args_schema 并描述字段含义         |
| 工具输出乱码或报错          | 返回类型不标准                | 确保 _run() 返回 str                    |
| Agent 死循环调用同一个 Tool | 工具描述或输出误导模型        | 限制 Tool 使用次数或手动终止            |
| 想在外部可视化调试          | 使用 LangSmith                | 开启 tracing 上传调用链路               |



------



## **🔧 六、进阶技巧**



1. **组合 Tool（复合功能）**

   

   - 让一个 Tool 内部再调用多个 API；
   - 或者让 LLM 自行选择使用多个 Tool 的顺序。

   

2. **使用动态 Tool**



```
from langchain.tools import Tool

Tool.from_function(
    func=my_function,
    name="dynamic_tool",
    description="动态生成的工具"
)
```



1. 

2. **多模态 Tool**

   工具可以返回图片/音频链接，配合 ChatOpenAI 支持多模态输出（如 DALL·E、Whisper）。

3. **链式调试**

   可把 Tool 嵌入 Chain 中，通过 SequentialChain 管理逻辑顺序。

------



## **📊 七、总结**



| **目标**              | **工具/机制**              | **调试方式**           |
| --------------------- | -------------------------- | ---------------------- |
| 快速定义函数型工具    | @tool 装饰器               | verbose=True           |
| 精确控制输入输出      | 继承 BaseTool              | 自定义 CallbackHandler |
| 可视化链路调试        | LangSmith                  | Web 界面查看链路       |
| 实时日志追踪          | BaseCallbackHandler        | 控制台输出             |
| 集成多模态 / 复合调用 | Tool.from_function + Agent | LangSmith 监控         |



------

