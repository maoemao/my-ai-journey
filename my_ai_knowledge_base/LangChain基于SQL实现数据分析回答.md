# LangChain基于SQL实现数据分析回答

下面我给你一份 **LangChain 基于 SQL 实现“数据分析问答（SQL + LLM）”的完整工程级方案**，从原理到可落地代码，一套走通。
 这是企业最常用的 **自然语言 → SQL → 执行 → 分析 → LLM 解读回答** 流水线。

内容直说重点、不绕弯子。

------

# 🚀 一、整体架构（你要实现的能力）

用户输入一句自然语言：

> “统计 2024 年每个月的销售额趋势，并给出分析。”

系统自动完成：

1. **将 NL（自然语言）转 SQL（LLM 生成）**
2. **在真实数据库执行 SQL（Postgres / MySQL / SQLite）**
3. **拿到结果表格**
4. **LLM 解释、分析、总结结果**

最终输出：

- SQL 查询
- 表格结果
- 趋势分析（同比、环比、建议）

这就是 LangChain 中的 **SQL Database Chain / Agent**。

------

# 🧠 二、为什么 LLM + SQL 是目前最强 data QA 方案？

因为：

- 模型负责理解问题 & 生成 SQL
- 数据库负责**真实计算**（不会幻觉）
- LLM 最后负责解释结果

> 这是“AI 数据分析”的底层必备结构，工程界通用。

------

# 📦 三、依赖安装

```bash
pip install langchain langchain-openai sqlalchemy pandas
```

Postgres 所需驱动：

```bash
pip install psycopg2
```

MySQL：

```bash
pip install mysqlclient
```

SQLite：Python 自带，不需要额外安装。

------

# 🏗 四、连接数据库（以 SQLite 为例）

```python
from langchain import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///sales.db")
```

PostgreSQL：

```python
db = SQLDatabase.from_uri("postgresql://user:pwd@host:5432/dbname")
```

MySQL：

```python
db = SQLDatabase.from_uri("mysql://user:pwd@host:3306/dbname")
```

------

# 🤖 五、构建 LLM（DeepSeek / OpenAI 都可）

示例使用 DeepSeek（OpenAI API 格式）：

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="deepseek-chat",
    temperature=0,
    api_key="你的key",
    api_base="https://api.deepseek.com/v1"
)
```

------

# 🔗 六、LangChain SQLChain（最基础）

LangChain 内置一个 SQL 解释执行链：

```python
from langchain.chains import SQLDatabaseChain

chain = SQLDatabaseChain(
    llm=llm,
    database=db,
    verbose=True
)
```

提问：

```python
chain.run("统计 2024 年每个月销售额，并按从高到低排序")
```

系统会自动生成 SQL → 执行 → 返回汇总结果。

------

# ⚙️ 七、升级版：使用 SQL Agent（更智能）

SQL Agent 可以：

- 自动查看表结构
- 生成复杂 SQL
- 自动修复 SQL 错误
- 多步推理

```python
from langchain.agents import create_sql_agent
from langchain.agents.agent_types import AgentType

agent = create_sql_agent(
    llm=llm,
    db=db,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)
```

使用：

```python
agent.run("对比 2023 和 2024 的季度销售额趋势，并给出图表风格的结论")
```

------

# 🔥 八、强力增强：结构化结果 + LLM 分析

LLM 的最佳角色不是“执行 SQL”，而是“理解 SQL 的结果”。

你可以这样做：

------

## Step1：LLM 生成 SQL（但不执行）

```python
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

sql_prompt = PromptTemplate(
    input_variables=["question", "table_info"],
    template="""
你是一名 SQL 专家。根据数据库的表结构回答用户问题。
表结构:
{table_info}

问题:
{question}

生成最合适的 SQL 查询语句，不需要解释。
"""
)

sql_gen = LLMChain(llm=llm, prompt=sql_prompt)
```

调用：

```python
table_info = db.get_table_info()

sql = sql_gen.run({
    "question": "统计 2024 年销售额最高的商品类别",
    "table_info": table_info
})
print(sql)
```

------

## Step2：执行 SQL（用 SQLAlchemy）

```python
import pandas as pd

def execute_sql(db, sql):
    engine = db._engine
    with engine.connect() as conn:
        df = pd.read_sql(sql, conn)
    return df
```

------

## Step3：让 LLM 分析结果（DataFrame → 自然语言）

```python
analysis_prompt = PromptTemplate(
    input_variables=["question", "dataframe"],
    template="""
你是专业数据分析师。
根据以下 DataFrame 的结果，回答用户问题并给出关键原因与结论：

问题: {question}

数据结果:
{dataframe}

请给出：
1. 清晰结论
2. 一句话解释原因
3. 业务建议
"""
)

analysis_chain = LLMChain(llm=llm, prompt=analysis_prompt)
```

调用：

```python
df = execute_sql(db, sql)
answer = analysis_chain.run({
    "question": "统计 2024 年销售额最高的商品类别",
    "dataframe": df.to_string()
})
print(answer)
```

现在你有了：

- 生成 SQL
- 执行 SQL
- 解释 SQL

完整的业务级数据分析问答系统。

------

# 🧠 九、增强检索力的技巧（非常关键）

### 1) **错误检测自动修复（Agent 会做）**

比如 SQL 语法错、字段不存在时，Agent 会自动修复。

### 2) **SQL + 文本 RAG 结合**

数据 + 文档问答混合。

### 3) **自动多步分析**

问题 -> SQL1 -> SQL2 -> SQL3 -> 最终结论

### 4) **LLM 生成可视化语法（echarts / matplotlib）**

下一步就能直接绘图。

------

# 🔐 十、企业生产经验（你应该知道）

真正上线须注意：

| 点              | 原因                       |
| --------------- | -------------------------- |
| 限制 LLM 权限   | 防止危险 SQL DELETE / DROP |
| 提前注入 schema | 减少 hallucination         |
| SQL sandbox     | 沙箱执行防误伤生产库       |
| 加缓存          | 热 SQL 不必每次执行        |
| 加审计日志      | 对话 + SQL 全记录          |

------

# 📌 总结一句话

> **LangChain + SQL = 企业级自然语言数据分析最稳方案。
>  LLM 生成 SQL → SQL 执行 → LLM 解释结果，是刚需架构。**

并且：

- 基础版用 `SQLDatabaseChain`
- 智能版用 `SQL Agent`
- 最强的用“生成 SQL + 执行 + 结果分析”三段式 Pipeline

这是目前业内最通用的做法。

------

