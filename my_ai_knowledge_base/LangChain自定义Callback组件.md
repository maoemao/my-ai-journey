# LangChain自定义Callback组件

下面我给你一份 **LangChain 自定义 Callback 组件高阶指南**，直击企业级大模型应用开发痛点，涵盖设计理念、实现方式、最佳实践以及实战案例。你作为程序员，可以直接上手写代码。

------

# 🚀 一、Callback 的核心作用

在 LangChain 中，Callback = **事件监听器 / 钩子函数**，用于监控和扩展 LLM、Agent、Retriever、Chain 的行为。

作用：

1. **训练/推理监控**
   - token 生成、log probabilities、completion stats
2. **日志记录**
   - 对请求/响应做统一监控
3. **自定义行为**
   - 在每次 LLM 调用、Retriever 检索、Agent 执行前后插入逻辑
4. **性能优化**
   - 统计 latency、token usage
5. **调试与可视化**
   - 与 Weights & Biases、TensorBoard 集成

> Callback 是 LangChain 可扩展能力的核心，企业级应用几乎必用。

------

# 🧱 二、Callback 基本结构

LangChain 提供了 **BaseCallbackHandler** 基类：

```python
from langchain.callbacks.base import BaseCallbackHandler

class MyCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("LLM 开始调用:", prompts)

    def on_llm_new_token(self, token, **kwargs):
        print("生成 token:", token)

    def on_llm_end(self, response, **kwargs):
        print("LLM 调用结束:", response)

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain 开始:", inputs)

    def on_chain_end(self, outputs, **kwargs):
        print("Chain 结束:", outputs)
```

> 核心方法：`on_llm_start`, `on_llm_new_token`, `on_llm_end`, `on_chain_start`, `on_chain_end` 等

------

# 🔧 三、自定义 Callback 场景示例

## 1️⃣ Token 级实时日志

```python
class TokenLoggerCallback(BaseCallbackHandler):
    def on_llm_new_token(self, token, **kwargs):
        print(token, end="", flush=True)
```

适合：

- 流式生成展示
- ChatGPT/DeepSeek 实时 token 输出

------

## 2️⃣ LLM 调用计数与性能监控

```python
import time

class PerfMonitorCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()

    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        print(f"LLM 调用耗时: {duration:.2f}s, token数: {len(response['generation'][0]['text'])}")
```

用途：

- 性能分析
- Token 计费预估
- 监控慢请求

------

## 3️⃣ 与 RAG 检索结合

```python
class RetrieverLoggerCallback(BaseCallbackHandler):
    def on_retriever_start(self, serialized, query, **kwargs):
        print(f"检索 query: {query}")

    def on_retriever_end(self, results, **kwargs):
        print(f"检索到文档: {len(results)} 条")
```

> 在大型企业知识库中可用于审计检索效果。

------

# 🛠 四、如何注册 Callback

```python
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

llm = OpenAI(temperature=0)
prompt = PromptTemplate(input_variables=["text"], template="请总结: {text}")
chain = LLMChain(llm=llm, prompt=prompt)

callback = TokenLoggerCallback()
chain.run("LangChain 自定义 Callback 示例", callbacks=[callback])
```

> 注意：`callbacks` 支持列表，可以同时注册多个 Callback

------

# 🔥 五、生产级高级技巧

1. **Callback 链式处理**
   - 多个 Callback 按顺序执行 → 日志 + 性能 + 监控同时处理
2. **异步 Callback**
   - 支持 async 方法，适合流式 token / Agent 执行
3. **事件过滤**
   - 只监听 LLM 或 Chain 或 Retriever，减少性能开销
4. **集成监控平台**
   - Weights & Biases、Prometheus、Grafana → Callback 输出数据直接上报
5. **安全审计**
   - 对敏感文档检索或生成过程做审计日志
6. **自定义流控**
   - Callback 可以在 `on_llm_new_token` 里动态拦截/修改 token

------

# 🧩 六、全功能 Callback 示例（综合示范）

```python
class FullFeatureCallback(BaseCallbackHandler):
    def __init__(self):
        self.start_time = None
        self.tokens = []

    def on_llm_start(self, serialized, prompts, **kwargs):
        self.start_time = time.time()
        print("LLM 开始调用")

    def on_llm_new_token(self, token, **kwargs):
        self.tokens.append(token)

    def on_llm_end(self, response, **kwargs):
        duration = time.time() - self.start_time
        print(f"LLM 完成, 总 token数: {len(self.tokens)}, 耗时: {duration:.2f}s")
        print("生成内容:", "".join(self.tokens))
        self.tokens = []

    def on_chain_start(self, serialized, inputs, **kwargs):
        print("Chain 开始执行:", inputs)

    def on_chain_end(self, outputs, **kwargs):
        print("Chain 执行结束:", outputs)
```

使用：

```python
callback = FullFeatureCallback()
chain.run("写一段LangChain Callback示例代码", callbacks=[callback])
```

> 你会得到 **流式 token 输出 + 总 token 统计 + Chain 输入输出日志**

------

# 🧠 七、总结建议

- Callback = **企业级监控 + 性能分析 + 流式输出 + 安全审计**
- 生产系统建议：
  1. **Token 流式日志**
  2. **性能统计**
  3. **检索日志**
  4. **多 Callback 链式组合**
- 高阶用法：
  - 异步
  - 自定义拦截 token / query
  - 上报监控平台

------

