# LangChain 基于rag实现文档问答



------



## **🧠 一、RAG 是什么？为什么重要？**



RAG（Retrieval-Augmented Generation）中文叫：

> **检索增强生成**

它解决了大模型的两大痛点：

1. **模型不知道最新信息**（比如本地文件内容）
2. **模型容易胡编乱造（幻觉）**

### **🔍 RAG 原理流程：**

```
用户问题 → 检索相关文档 → 构造上下文 → 交给 LLM → 输出最终回答
```



------



## **🧩 二、LangChain RAG 架构图**

```
        ┌────────────────────┐
        │  用户问题（Query） │
        └────────┬───────────┘
                 │
         [Retriever 检索模块]
                 │
        ┌────────▼──────────┐
        │  向量数据库 (FAISS/Chroma) │
        └────────┬──────────┘
                 │
          [文档片段匹配]
                 │
        ┌────────▼────────┐
        │  构造上下文 Prompt │
        └────────┬────────┘
                 │
             [LLM生成回答]
                 │
        ┌────────▼────────┐
        │     最终回答     │
        └─────────────────┘
```



------



## **⚙️ 三、LangChain RAG 核心组件**

| **模块**        | **作用**   | **示例类**                                          |
| --------------- | ---------- | --------------------------------------------------- |
| **Loader**      | 读取文档   | PyPDFLoader, TextLoader, UnstructuredMarkdownLoader |
| **Splitter**    | 文本切分   | RecursiveCharacterTextSplitter                      |
| **Embedding**   | 向量化文本 | OpenAIEmbeddings, OllamaEmbeddings                  |
| **VectorStore** | 存储与检索 | FAISS, Chroma, Milvus, Pinecone                     |
| **Retriever**   | 检索文档块 | .as_retriever()                                     |
| **LLM**         | 生成回答   | ChatOpenAI, Ollama, Gemini                          |
| **Chain**       | 串联逻辑   | RetrievalQA, ConversationalRetrievalChain           |



------



## **🧰 四、最小可运行示例：PDF 问答系统**



### **1️⃣ 安装依赖**

```
pip install langchain langchain-openai faiss-cpu PyPDF2
```



### **2️⃣ 加载文档**

```
from langchain.document_loaders import PyPDFLoader

loader = PyPDFLoader("docs/人工智能白皮书.pdf")
docs = loader.load()
print(f"文档数量: {len(docs)} 段")
```



------



### **3️⃣ 切分文本**

```
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"切分后 {len(chunks)} 段")
```



------



### **4️⃣ 向量化与存储**

```
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding)
```



------



### **5️⃣ 构建检索问答链**

```
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)
```



------



### **6️⃣ 运行查询**

```
query = "这份白皮书中提到了人工智能的核心方向是什么？"
result = qa_chain.invoke({"query": query})

print("回答：", result["result"])
print("\n引用文档片段：")
for doc in result["source_documents"]:
    print("-", doc.page_content[:120])
```



------



## **💬 五、进阶版：支持对话记忆的 RAG**

```
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
)
```

你现在可以实现：

```
用户：这份白皮书的重点是什么？
模型：主要包括AI基础设施、伦理监管等方面。
用户：那它提到的监管原则有哪些？
```

模型会自动参考上文。



------



## **🔍 六、常用优化技巧**

| **优化方向**       | **方法**                                                 |
| ------------------ | -------------------------------------------------------- |
| **提高召回准确度** | 增大 chunk_overlap；使用多路检索（MaxMarginalRelevance） |
| **减少幻觉**       | 增加 context_limit，只返回最相关前 N 段                  |
| **增强响应质量**   | 在 Prompt 中显式要求“仅基于上下文回答”                   |
| **多文档支持**     | 同时加载多个 PDF / MD 文件合并向量库                     |
| **向量缓存**       | 将 FAISS 保存到本地磁盘，避免每次重建                    |

保存示例：

```
vectorstore.save_local("vector_db")
# 重新加载
db = FAISS.load_local("vector_db", embedding, allow_dangerous_deserialization=True)
```



------



## **🧠 七、加上自定义 Prompt（控制输出风格）**

```
from langchain.prompts import PromptTemplate

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "你是AI专家，请基于以下内容回答问题。\n"
        "内容：{context}\n"
        "问题：{question}\n"
        "请用简明专业的方式回答，不要编造。"
    )
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    chain_type_kwargs={"prompt": prompt}
)
```



------



## **🧩 八、部署思路**

你可以：

- 用 **LangServe** 把整个 qa_chain 部署为 REST API
- 用 **FastAPI** 自建 Web 服务
- 或嵌入到 **Streamlit / Gradio** 做前端界面



示例（LangServe）：

```
from langserve import serve
serve(qa_chain, port=8080)
```



------



## **📊 九、RAG 流程总结表**

| **阶段** | **模块**    | **LangChain 类**                          |
| -------- | ----------- | ----------------------------------------- |
| 文档加载 | Loader      | PyPDFLoader, TextLoader                   |
| 切分     | Splitter    | RecursiveCharacterTextSplitter            |
| 向量化   | Embedding   | OpenAIEmbeddings / OllamaEmbeddings       |
| 存储     | VectorStore | FAISS, Chroma                             |
| 检索     | Retriever   | .as_retriever()                           |
| 问答链   | Chain       | RetrievalQA, ConversationalRetrievalChain |
| 部署     | Serve       | LangServe                                 |



------



## **✅ 十、可以拓展的方向**

| **目标**   | **方法**                                 |
| ---------- | ---------------------------------------- |
| 多模态 RAG | 加入图像 OCR + 向量检索                  |
| 增量更新   | 动态追加新文档                           |
| 企业搜索   | 用 Milvus / Weaviate 代替 FAISS          |
| 智能体化   | 将 QA 链封装为一个 Tool，交给 Agent 使用 |
| 自动调优   | 用 LangSmith 追踪召回质量和幻觉率        |



------



