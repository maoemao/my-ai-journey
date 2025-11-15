# LangChain 多模态输入以及自定义输出



------



## **🧩 一、LangChain 多模态输入（Multi-modal Input）**



### **1️⃣ 背景**

LangChain 以前（v0.x）主要面向文本输入。

在 2024–2025 年的版本中（尤其配合 **LCEL** + **ChatOpenAI**），已经可以无缝支持：

- 图像（image）
- 音频（audio）
- 文本（text）
- 文件（pdf、excel、json 等结构化输入）

------



### **2️⃣ 图像输入示例（OpenAI GPT-4o / Claude 3 / Gemini 等）**



```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.schema.messages import HumanMessage

llm = ChatOpenAI(model="gpt-4o")  # 支持多模态

msg = HumanMessage(
    content=[
        {"type": "text", "text": "描述这张图片的内容"},
        {"type": "image_url", "image_url": "https://example.com/dog.jpg"},
    ]
)

response = llm.invoke([msg])
print(response.content)
```

> ✅ GPT-4o、Claude 3、Gemini Pro Vision 都可以直接用这种消息结构。



------



### **3️⃣ 音频输入示例（语音识别 + 文本理解）**



```
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage

llm = ChatOpenAI(model="gpt-4o")

msg = HumanMessage(
    content=[
        {"type": "text", "text": "请帮我总结这段语音的主要内容"},
        {"type": "input_audio", "input_audio": "file://path/to/audio.mp3"},
    ]
)

response = llm.invoke([msg])
print(response.content)
```

> 可以与 Whisper 或第三方语音识别模型结合，实现「音频 → 文字 → 语义分析」。



------



### **4️⃣ 多输入融合（文本 + 图像 + 数据）**



例如：

输入一张截图 + 一段描述 + 一张表格数据，请模型帮你生成报告。

```
from langchain.schema import HumanMessage

msg = HumanMessage(content=[
    {"type": "text", "text": "根据图片和表格，写一段简短的分析"},
    {"type": "image_url", "image_url": "https://example.com/chart.png"},
    {"type": "text", "text": "表格数据：\n年份, 销售额\n2023, 1200\n2024, 1600"},
])
```

这种“多通道输入”就是多模态链路的典型做法。

LangChain 的 **LCEL**（LangChain Expression Language）可以很方便地封装这些输入输出。



------



### **5️⃣ 结合 LangChain 工具的多模态场景**



| **模态**    | **工具 / 模型**                  | **常见用途**        |
| ----------- | -------------------------------- | ------------------- |
| 图像 → 文本 | GPT-4o, Claude 3, Gemini Vision  | 图片理解、OCR       |
| 文本 → 图像 | DALL-E, Stable Diffusion         | 生成图片            |
| 音频 → 文本 | Whisper, OpenAI Audio API        | 语音识别            |
| 文本 → 语音 | TTS API                          | 语音输出            |
| 文件输入    | DocumentLoader（PDF、CSV、DOCX） | 文件问答 / 知识抽取 |



------



## **⚙️ 二、自定义输出（Structured / Controlled Output）**



模型输出如果只是纯文本，很难被后续系统直接利用。

LangChain 提供了多个层级的“结构化输出”能力。



------



### **1️⃣ 最简单：StrOutputParser**



```
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema.output_parser import StrOutputParser

prompt = ChatPromptTemplate.from_template("写一句描述猫的句子")
chain = prompt | ChatOpenAI(model="gpt-4o") | StrOutputParser()
print(chain.invoke({}))
```

> 输出是纯文本。

------



### **2️⃣ 结构化输出：JSON / Pydantic**



LangChain 的 StructuredOutputParser 和 PydanticOutputParser 是关键。



#### **示例（JSON 结构输出）**



```
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema

response_schemas = [
    ResponseSchema(name="name", description="宠物的名字"),
    ResponseSchema(name="species", description="宠物的种类"),
]

parser = StructuredOutputParser.from_response_schemas(response_schemas)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_template(
    "请用JSON格式回答以下问题：\n{format_instructions}\n问题: 描述你理想的宠物。"
)

chain = prompt | ChatOpenAI(model="gpt-4o") | parser
print(chain.invoke({"format_instructions": format_instructions}))
```

输出示例：

```
{"name": "喵喵", "species": "猫"}
```



------



### **3️⃣ Pydantic 模型输出（强类型结构）**



```
from pydantic import BaseModel, Field
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

class ProductInfo(BaseModel):
    name: str = Field(description="商品名称")
    price: float = Field(description="价格")
    category: str = Field(description="类别")

parser = PydanticOutputParser(pydantic_object=ProductInfo)
prompt = ChatPromptTemplate.from_template(
    "根据描述生成商品信息。\n{format_instructions}\n描述：{desc}"
)

chain = prompt | ChatOpenAI(model="gpt-4o") | parser
result = chain.invoke({
    "desc": "一款售价399元的蓝牙耳机，适合运动使用。",
    "format_instructions": parser.get_format_instructions(),
})

print(result)
```

> 输出是一个 ProductInfo 对象，可直接用于数据库或API响应。



------



### **4️⃣ 自定义输出逻辑（自定义 Parser）**



你可以继承 BaseOutputParser 来定义任意解析逻辑：

```
from langchain.schema import BaseOutputParser

class CodeBlockParser(BaseOutputParser):
    def parse(self, text: str):
        code = text.split("```")[1] if "```" in text else text
        return code.strip()

parser = CodeBlockParser()
```

然后放入 LCEL 管道中：

```
chain = prompt | ChatOpenAI(model="gpt-4o") | parser
```



------



### **5️⃣ 结合多模态输出（例如图像描述结构化）**



输入图像 → 输出结构化 JSON（如检测结果）

```
prompt = ChatPromptTemplate.from_template(
    "分析图片中的物体并输出JSON：{format_instructions}"
)

response_schemas = [
    ResponseSchema(name="objects", description="检测到的物体列表"),
    ResponseSchema(name="scene", description="场景描述"),
]
parser = StructuredOutputParser.from_response_schemas(response_schemas)

msg = {
    "type": "image_url",
    "image_url": "https://example.com/street.jpg"
}

chain = (prompt | ChatOpenAI(model="gpt-4o") | parser)
print(chain.invoke({"format_instructions": parser.get_format_instructions(), "image": msg}))
```



------



## **🧠 三、LCEL 流式组合（多模态 + 自定义输出）**



用 LCEL 可以像 Unix 管道一样，将：



> 输入 → 多模态融合 → 模型 → 输出解析 → 返回结构

```
from langchain.schema import HumanMessage
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_template("描述图片内容。")
model = ChatOpenAI(model="gpt-4o")
parser = StrOutputParser()

workflow = prompt | model | parser

msg = HumanMessage(content=[
    {"type": "text", "text": "请描述这张图片"},
    {"type": "image_url", "image_url": "https://example.com/cat.png"}
])

result = workflow.invoke({"input": msg})
print(result)
```



------



## **📊 四、实际工程场景建议**



| **场景**      | **输入模态** | **输出格式**    | **推荐方案**           |
| ------------- | ------------ | --------------- | ---------------------- |
| 智能问答      | 文本         | Markdown / JSON | StructuredOutputParser |
| 图像理解      | 图像 + 文本  | JSON            | ChatOpenAI(gpt-4o)     |
| 文件摘要      | 文本 + 文件  | 段落 / 列表     | LCEL + StrOutputParser |
| 语音助理      | 音频 + 文本  | 纯文本 / JSON   | Whisper + ChatOpenAI   |
| Agent工具调用 | 文本         | Pydantic结构    | PydanticOutputParser   |



------



## **✅ 总结对照表**



| **功能**     | **核心模块**                                 | **关键点**              |
| ------------ | -------------------------------------------- | ----------------------- |
| 多模态输入   | HumanMessage(content=[{"type":...}])         | 支持 text, image, audio |
| 多模态模型   | ChatOpenAI(gpt-4o)、Claude 3                 | Vision 模型             |
| 自定义输出   | StructuredOutputParser, PydanticOutputParser | 结构化JSON              |
| 自定义解析器 | 继承 BaseOutputParser                        | 处理特定格式            |
| 流式组合     | LCEL (`prompt                                | model                   |



------

