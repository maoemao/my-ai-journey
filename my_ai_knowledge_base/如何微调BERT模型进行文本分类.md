# 如何微调BERT模型进行文本分类 

下面我给你 **不绕弯、直接可执行的微调 BERT 文本分类套路**，从原理、数据准备到 PyTorch 代码全流程，一步步讲清楚。你作为程序员，很快能拿去跑。

------

# 1. 微调 BERT 做文本分类的核心思路

BERT 本质是一个 Encoder，输出每个 token 的向量。
 我们要做分类，只需要用 **[CLS] 向量** 接一个 **线性分类头**，然后在你的任务数据上训练几轮就够了。

架构：

```
input text
   ↓
BERT (预训练参数)
   ↓ 取 [CLS] embedding (hidden_size = 768)
Linear Layer (768 → num_labels)
   ↓
Softmax / CrossEntropy
```

训练时微调策略：

- **除了分类头，BERT 主体一起参与训练（finetune）**
- **不需要重新训练分词器**
- **小数据集也能奏效**

------

# 2. 数据准备

你的训练集一般要长这样（CSV）：

| text           | label    |
| -------------- | -------- |
| "我今天很开心" | positive |
| "这产品太差了" | negative |

注意：

- label 最终要映射成数字，如 `{"negative":0, "positive":1}`
- 文本越干净越好（但不需要繁琐预处理）

------

# 3. 使用 HuggingFace Transformers 的微调流程（PyTorch）

以下是你能直接运行的代码（PyTorch）：

## 3.1 安装依赖

```bash
pip install transformers datasets accelerate
```

## 3.2 准备数据集

```python
from datasets import load_dataset

dataset = load_dataset("csv", data_files={"train": "train.csv", "test": "test.csv"})
```

## 3.3 分词

```python
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def encode_batch(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding='max_length',
        max_length=128
    )

dataset = dataset.map(encode_batch, batched=True)
dataset = dataset.rename_column("label", "labels")
dataset.set_format(type="torch", columns=['input_ids','token_type_ids','attention_mask','labels'])
```

------

# 4. 创建 BERT 分类模型

```python
from transformers import BertForSequenceClassification

model = BertForSequenceClassification.from_pretrained(
    "bert-base-chinese",
    num_labels=2
)
```

------

# 5. 微调训练

使用 HuggingFace Trainer：

```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    output_dir="./bert-finetune",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_steps=20,
    evaluation_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

trainer.train()
```

------

# 6. 推理（使用微调后的模型）

```python
text = "这个手机真的很好用！"

inputs = tokenizer(text, return_tensors="pt", truncation=True)
outputs = model(**inputs)
pred = outputs.logits.argmax(-1).item()
print(pred)
```

------

# 7. 微调技巧（避免你掉坑）

### 1）学习率别太大

BERT 微调最佳学习率区间：

- **2e-5**
- **3e-5**
- **5e-5**

再大就容易炸。

### 2）max_length 不要乱设

中文一般用 128 或 256 就够了。

### 3）小数据（几千条）也完全没问题

因为 BERT 已经学了大量语言知识。

### 4）是否要冻结 BERT？

通常 **不需要**。
 但如果你 GPU 很弱：

```python
for param in model.bert.parameters():
    param.requires_grad = False
```

------

