# LlamaIndex大模型应用开发快速入门

下面我给你一份 **实战派、程序员视角的《LlamaIndex 大模型应用开发快速入门》**，看完你能立刻用它搭建一个可用的 LLM 应用（RAG、Agent、工具调用都能做）。

内容尽量 “直来直去”，避免废话 —— 你看完就能开干。

------

# 🚀 一、LlamaIndex 是什么？

一句话：

> **LlamaIndex = 大模型应用开发框架，用来快速构建 RAG、Agent、工具调用、数据处理流水线。**

它主要解决三个核心问题：

1. **外部数据如何接入大模型（数据连接）**
2. **如何把文本变成向量、索引、检索结构（索引）**
3. **如何让模型“查资料 + 思考 + 回答”（查询/Agent）**

通俗理解：
 它比 LangChain 更偏向 **数据为中心**，RAG 能力非常强。

------

# 🚧 二、最常用的功能（简单记住这 6 个）

LlamaIndex 功能很多，但作为开发者，你最常用的是：

1. **Readers（文档加载器）**
   - PDF、Markdown、网页、数据库
2. **Node Parser（把文档切成 chunk）**
3. **Storage Context（存储）**
4. **Vector Store（向量数据库接口）**
   - 支持 Milvus、Pinecone、Chroma、Elasticsearch…
5. **Index（索引结构）**
   - VectorIndex
   - TreeIndex
   - GraphIndex
   - KeywordTableIndex
6. **Query Engine（查询引擎）**
   - RAG 主逻辑
   - 可接入工具、Agent

------

# 🛠 三、5 分钟跑通一个 RAG Demo（你可以复制直接用）

## 1. 安装

```bash
pip install llama-index
pip install llama-index-embeddings-openai
pip install llama-index-llms-openai
```

## 2. 配置密钥

```python
import os
os.environ["OPENAI_API_KEY"] = "你的key"
```

（也可以换成 DeepSeek、Qwen，本质一样）

## 3. 加载文档

```python
from llama_index.core import SimpleDirectoryReader

documents = SimpleDirectoryReader("docs").load_data()
```

## 4. 构建索引（RAG 核心）

```python
from llama_index.core import VectorStoreIndex

index = VectorStoreIndex.from_documents(documents)
```

## 5. 创建查询引擎

```python
query_engine = index.as_query_engine()
```

## 6. 开始问问题

```python
response = query_engine.query("文档里提到的产品特点是什么？")
print(response)
```

**就是这么简单。**

------

# 🔍 四、核心原理（RAG 的 3 个过程）

### 1）将文档解析成 Node

（等价于 chunk）

### 2）将 Node 编码成 embedding，写入向量库

### 3）用户提问 → 检索 → 组装 Prompt → LLM 回答

这 3 步全部由 LlamaIndex 自动做完。

------

# 🧱 五、推荐的项目结构（你的项目建议这么写）

```
my-rag-app/
│── data/                # PDF、md 文件
│── config.yaml          # LLM、embedding 配置
│── build_index.py       # 构建索引脚本
│── serve.py             # 提供 RAG 接口
│── llm/                 # llm 配置目录
│── vector/              # 向量库
```

------

# 🔥 六、把 LlamaIndex 换成国产模型（Qwen / DeepSeek）

示例（DeepSeek 作为 LLM）：

```bash
pip install llama-index-llms-openai
```

DeepSeek 兼容 OpenAI API：

```python
from llama_index.llms.openai import OpenAI

llm = OpenAI(
    api_base="https://api.deepseek.com/v1",
    api_key="你的key",
    model="deepseek-chat"
)
```

Embedding 用 OpenAI 兼容接口即可：

```python
from llama_index.embeddings.openai import OpenAIEmbedding

embed = OpenAIEmbedding(
    api_base="https://api.deepseek.com/v1",
    api_key="你的key",
    model="deepseek-embedding"
)
```

然后：

```python
query_engine = index.as_query_engine(llm=llm)
```

------

# 🧩 七、LlamaIndex vs LangChain（你一定会问）

| 功能       | LlamaIndex | LangChain |
| ---------- | ---------- | --------- |
| RAG 强度   | ⭐⭐⭐⭐⭐      | ⭐⭐⭐       |
| 可解释性   | ⭐⭐⭐⭐⭐      | ⭐⭐        |
| Agent      | ⭐⭐⭐        | ⭐⭐⭐⭐⭐     |
| 社区生态   | ⭐⭐⭐        | ⭐⭐⭐⭐⭐     |
| 数据为中心 | 强         | 一般      |
| 工具集成   | 一般       | 强        |

总结：

- **做 RAG → 用 LlamaIndex 更简单更强**
- **做 Agent、工具调用 → LangChain 更好**
- **混合用（最常见）→ LlamaIndex 加载数据 + LangChain Agent**

------

# 🚀 八、快速构建一个企业级 RAG（你实际可能需要）

下面是典型企业需求：

- 文档上传
- 自动切片
- 自动索引
- 向量库（Milvus/PGVector）
- LLM（DeepSeek/Qwen）
- Spark/混合检索
- 工程化部署（FastAPI）

LlamaIndex 已有完整支持：

```python
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
```

或搭建混合检索：

```python
from llama_index.core.retrievers import (
    VectorIndexRetriever,
    KeywordTableSimpleRetriever
)
from llama_index.core.query_engine import RouterQueryEngine
```

------

# 🔋 九、实用技巧（你现在就能用）

### ✔ 1. 不要用默认 embeddings

中文任务强烈推荐：

- **text-embedding-3-large**（OpenAI）
- **DeepSeek embedding**
- **Qwen Embedding**

### ✔ 2. 文档必须切片

默认 chunk_size = 512 很合理。

### ✔ 3. 确保索引放在本地或 Redis，不要放 RAM

否则服务重启会丢失。

### ✔ 4. 避免“长文全投喂大模型”

检索到的 chunks 最多 5~10 个。

------

# 🧠 十、作为导师，我给你的入门路线（非常实用）

1. **10 分钟跑通基本 RAG Demo（你已经看到了）**
2. **把索引存入向量数据库（Milvus/Chroma）**
3. **接入 DeepSeek API 做回答**
4. **用 FastAPI 封装成接口**
5. **加入 Hybrid Retrieval（BM25 + vector）**
6. **用 JSON mode + ReAct 做工具增强**
7. **最后结合 LangChain 做 Agent**

这是一个完整的大模型开发入门路线。

------

