# LangChain 多类型文本分割

下面我把 **LangChain 多类型文本分割（Multi-Type Text Splitting）** 讲透，从底层机制到最佳实践，再到可落地代码示例，让你直接能在企业级 RAG 系统里用。

内容不绕弯子，直接说清楚每种数据如何切片、为什么这么切，以及 LangChain 实战里的常见坑。

------

# 🚀 一、为什么“多类型文本分割”非常关键？

**文本切分（Chunking）几乎决定了 RAG 的上限。**

文档不同 → 最优分割方式完全不同：

| 文档类型               | 最推荐的分割策略              |
| ---------------------- | ----------------------------- |
| Markdown / Wiki        | 按标题层级拆（Header-aware）  |
| PDF（结构混乱）        | 按段落/语义分割               |
| HTML / 网页            | DOM-aware 分割                |
| 表格（CSV/Excel）      | 按行 / 按字段语义             |
| 代码文件               | AST/函数级切分                |
| 长对话                 | 按完整发言 / Session-based    |
| 法律/规范              | 按条目（Article/Chapter）分割 |
| 网页爬虫文本（噪声强） | 清洗 + 语义切片               |

通用的做法（纯字符分割）已经过时，效果会非常差。

------

# 📦 二、LangChain 常用的文本分割工具汇总

LangChain 支持多种切分器：

### 1）RecursiveCharacterTextSplitter（最常用）

按层级字符切分：

- 先尝试按段落
- 再按句子
- 再按短句
- 实在不行按字符

👉 兼容性最强。

------

### 2）MarkdownHeaderTextSplitter（Markdown / Wiki 最强）

按标题层级自动分割：

- `# 一级标题`
- `## 二级标题`
- `### 三级标题`

适合结构清晰的知识手册、Wiki、技术文档。

------

### 3）HTMLSectionSplitter（网页专用）

按 **DOM 树结构** 提取内容：

- section
- nav
- p
- li
- table

避免乱糟糟的文本提取。

------

### 4）TokenTextSplitter（按 Token 分割）

用于控制 LLM token 限制。

------

### 5）PythonCodeTextSplitter（代码文件）

按：

- 类
- 函数
- 方法块

分割，效果远好于普通字符切片。

------

### 6）Semantic Chunking（语义分割）

使用 transform embedding 进行语义段落划分。

LangChain 已支持：

```python
from langchain.text_splitter import SemanticChunker
```

效果是目前最强的，不依赖人工结构。

------

# 🧩 三、各类型文本的最佳分割策略

下面是实战最强的配置，你可以直接用。

------

## 【类型 1】Markdown / Wiki 分割（最推荐）

```python
from langchain.text_splitter import MarkdownHeaderTextSplitter

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]
)
docs = splitter.split_text(markdown_text)
```

🚀 特点：

- 根据标题自动构建 metadata（上下文更精准）
- 检索效果显著提升

------

## 【类型 2】PDF 分割（段落 + Recursive）

PDF 通常结构混乱，所以推荐“两段式方案”：

### Step1: 提取纯文本（避免乱码）

```python
from langchain.document_loaders import PyPDFLoader

docs = PyPDFLoader("a.pdf").load()
```

### Step2: 用 Recursive 分割

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    separators=["\n\n", "\n", ".", "。", "!", "?"]
)
chunks = splitter.split_documents(docs)
```

------

## 【类型 3】网页 HTML 分割（DOM-aware）

```python
from langchain.text_splitter import HTMLSectionSplitter

splitter = HTMLSectionSplitter()
docs = splitter.split_text(html_source)
```

优点：

- 可以保留 DOM metadata，例如 `<h1>`, `<p>`, `<li>`
- 检索网页、百科类文档效果非常好

------

## 【类型 4】代码文件分割（AST-aware）

```python
from langchain.text_splitter import PythonCodeTextSplitter

splitter = PythonCodeTextSplitter()
chunks = splitter.split_text(python_code)
```

适合：

- 代码 QA
- 调试辅助
- 文档自动生成

------

## 【类型 5】表格类（CSV / Excel）

推荐行级切片：

```python
from langchain.document_loaders.csv_loader import CSVLoader

loader = CSVLoader("data.csv")
docs = loader.load()
```

如果太长，按行块分割：

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=0
)
chunks = splitter.split_documents(docs)
```

------

## 【类型 6】长对话（session-based）

按发言拆分 + 会话 metadata：

```python
dialog_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
chunks = dialog_splitter.split_documents(dialog_docs)
```

------

## 【类型 7】法律文档（按“条/款/章”分割）

自定义正则：

```python
legal_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["第[一二三四五六七八九十]+条", "\n\n", "\n"]
)
```

------

# 🔥 四、多类型文本分割组合器（Multi-Source Hybrid Splitter）

实际业务中你会处理：

- PDF
- 网页
- Markdown
- 代码
- 数据库内容

你需要按类型自动选择对应分割器。

👇 最强模板（可以直接用）：

```python
def multi_type_splitter(doc):
    text = doc.page_content
    metadata = doc.metadata
    filetype = metadata.get("source", "")

    if filetype.endswith(".md"):
        return md_splitter.split_text(text)
    elif filetype.endswith(".html") or metadata.get("html"):
        return html_splitter.split_text(text)
    elif filetype.endswith(".py"):
        return code_splitter.split_text(text)
    elif filetype.endswith(".pdf"):
        return pdf_splitter.split_documents([doc])
    else:
        return default_splitter.split_documents([doc])
```

这是企业级知识库常用做法。

------

# 🧠 五、如何选择最佳 chunk_size？（核心）

### 文档结构强 → chunk 大

如：

- Markdown
- 网页（HTML）
- Wiki

推荐：**800~1200**

------

### 文档结构弱 → chunk 小

如：

- PDF（OCR）
- 爬虫内容
- 对话

推荐：**300~500**

------

### 推理困难 → chunk 更小（提高密度）

例如：

- 方案对比
- 技术规范
- 代码解释

推荐：**200~400 + 核心句子提升**

------

# ⚠️ 六、常见坑（你必踩过的）

❌ **切片太大，检索不准**
 ❌ **切片太小，语义破碎（hallucination 增加）**
 ❌ **没有 metadata → 上下文判断失败**
 ❌ **所有文档用一个 splitter → 效果非常差**
 ❌ **PDF 转文本不清洗 → embedding 垃圾**
 ❌ **网页 HTML 没去标签 → 噪声拉低检索准确度**

------

# 🎯 七、最终给你一份最强“多类型切片器”统一配置

```python
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLSectionSplitter,
    PythonCodeTextSplitter
)

md_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "h1"), ("##", "h2"), ("###", "h3")
    ]
)

html_splitter = HTMLSectionSplitter()

code_splitter = PythonCodeTextSplitter()

pdf_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

default_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)
```

结合上一节的 `multi_type_splitter()` 可以直接在项目里用。

------

