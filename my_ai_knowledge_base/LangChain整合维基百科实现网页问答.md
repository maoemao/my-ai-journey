# LangChain整合维基百科实现网页问答

下面给你一套 **LangChain 整合维基百科（Wikipedia）实现网页问答的完整方案**，从依赖、检索器、RAG 流水线、到增强检索、最终问答接口，全流程都讲清楚，让你直接能在项目里用。

内容是实打实的工程级实现，而不是一堆概念。

------

# 🚀 一、整体能力：你能实现什么？

**LangChain + Wikipedia 数据源 + LLM = 任意维基页面问答系统**

包括：

- 指定条目问答
- 自动搜索维基词条
- 多轮对话
- RAG 检索增强
- 网页问答（提取正文）
- 自定义分片、rerank、上下文压缩

------

# 🧱 二、安装依赖

```bash
pip install langchain wikipedia
pip install langchain-openai   # 如果你用 DeepSeek / OpenAI
```

------

# 🧲 三、维基百科加载器（最基础）

LangChain 已内置 WikipediaLoader。

```python
from langchain.document_loaders import WikipediaLoader

docs = WikipediaLoader(query="大语言模型", load_max_docs=3).load()
for d in docs:
    print(d.metadata)
```

你会得到：

```json
{
  "title": "大语言模型",
  "source": "wikipedia",
  "url": "https://zh.wikipedia.org/wiki/..."
}
```

并且 `page_content` 已自动提取好了正文。

------

# ✂️ 四、分片（RAG 的关键步骤）

维基百科页都很长，必须切片。

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(docs)
```

------

# 🔮 五、构建向量库（Chroma 最适合开发）

```python
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

embed = OpenAIEmbeddings(model="text-embedding-3-large")
vectorstore = Chroma.from_documents(
    chunks, embed, persist_directory="./wiki_db"
)
```

> 若用 DeepSeek Embedding，只要换模型名即可。

------

# 🔍 六、构建 Retriever（网页问答核心）

```python
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
```

这是最基础版本。

------

# 🤖 七、构建 LLM（DeepSeek / OpenAI）

（DeepSeek 示例）

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    api_key="你的key",
    temperature=0.2
)
```

------

# 🔗 八、构建 RAG（主链路）

```python
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff"
)
```

使用：

```python
result = qa("介绍一下大语言模型的主要特点")
print(result)
```

------

# 🚀 九、高阶：自动搜索维基百科（搜索 + 加载 + RAG 全自动）

如果用户不知道具体词条，你可以用维基查询 API 自动搜索。

```python
def wiki_query(question):
    from langchain.document_loaders import WikipediaLoader

    # 自动搜索
    docs = WikipediaLoader(
        query=question, 
        load_max_docs=3, 
        lang="zh"
    ).load()

    chunks = splitter.split_documents(docs)

    # 构建临时向量库
    vectorstore = Chroma.from_documents(chunks, embed)

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )
    return qa(question)
```

调用：

```python
print(wiki_query("什么是超导体？"))
```

你已经拥有一个**全自动维基百科问答系统**。

------

# 🧠 十、再升级（强烈推荐）：知识增强检索器

### 1）MultiQueryRetriever（多查询增强）

```python
from langchain.retrievers.multi_query import MultiQueryRetriever

retriever = MultiQueryRetriever.from_llm(
    retriever=retriever, llm=llm
)
```

适用于精度不足时。

------

### 2）Reranker 精排（BGE-Reranker）

```python
from langchain.retrievers.context_compressor import ContextualCompressionRetriever
from langchain_community.document_transformers import BGERerank

reranker = BGERerank(top_n=3)
compression = ContextualCompressionRetriever(
    base_compressor=reranker,
    base_retriever=retriever
)
```

LLM 看到的 context 更精准、短、质量高。

------

### 3）LLM-based 压缩（深度摘要）

```python
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression = ContextualCompressionRetriever(
    base_retriever=retriever,
    base_compressor=compressor
)
```

超长文档问答特别有效。

------

# 🏷️ 十一、构建完整 Web 服务（FastAPI 示例）

```python
from fastapi import FastAPI
app = FastAPI()

@app.post("/ask")
def ask(question: str):
    return {"answer": wiki_query(question)}
```

直接可部署成网页问答 API。

------

# 🔥 十二、实战建议（关键）

1. **维基百科内容非常长** → 必须切片 + 压缩
2. **自动搜索很重要** → 用户提问通常不是词条名
3. **Hybrid 检索胜率更高** → BM25 + 向量
4. **Rerank 提升最明显**
5. **上下文压缩能显著降低 token 用量**

------

# 📌 总结（给你一句最实用的）

> **整合维基百科 = WikipediaLoader + 分片 + 向量库 + RAG。
>  真正效果取决于：多查询检索 + rerank + 压缩 + prompt 设计。**

------

