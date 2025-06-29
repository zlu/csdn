---
layout: post
---

# 掩码模型 vs. 因果模型：NLP 核心范式的全面指南

## 引言
在自然语言处理（NLP）中，掩码语言模型（MLMs）和因果语言模型（CLMs）是构建基于 Transformer 模型的基础方法。MLMs（如 BERT）通过利用双向上下文在理解任务方面表现出色，而 CLMs（如 GPT）凭借其单向方法在文本生成方面占据主导地位。本博客探讨了它们的技术差异、数学基础、应用，并提供了代码示例来说明其机制。我们还研究了 MLMs 如何与分类和问答（QA）任务相关，为什么掩码能增强这些任务，以及方向性如何在数学和实现中表示。

## 技术差异

### 掩码语言建模（MLM）
- **定义**：MLMs 在输入序列中随机掩码标记（例如，15% 的标记），并训练模型使用周围上下文预测这些标记。
- **训练目标**：预测掩码标记，例如，在"The cat sat on the [MASK]"中，预测"mat"。
- **上下文**：双向，允许模型关注序列中的所有标记。
- **架构**：仅编码器 Transformer（如 BERT），具有完整的自注意力机制。
- **优势**：非常适合需要深度上下文理解的任务，如分类、QA 和命名实体识别（NER）。

### 因果语言建模（CLM）
- **定义**：CLMs 基于前面的标记预测序列中的下一个标记，从左到右处理。
- **训练目标**：下一个标记预测，例如，给定"The cat sat on the"，预测"mat"。
- **上下文**：单向，仅关注前面的标记。
- **架构**：仅解码器 Transformer（如 GPT），在自注意力中使用因果掩码。
- **优势**：针对文本生成进行了优化，包括聊天机器人、创意写作和代码生成。

| 特性                | 掩码语言模型（MLM）   | 因果语言模型（CLM）   |
|------------------------|-------------------------------|-------------------------------|
| **上下文方向**  | 双向                | 单向（从左到右） |
| **训练目标** | 掩码标记预测      | 下一个标记预测         |
| **注意力**          | 完整序列注意力      | 因果（仅过去标记）     |
| **主要优势**   | 上下文理解     | 顺序生成         |

## 数学基础

### 掩码语言建模
MLMs 旨在最大化预测掩码标记的可能性。给定序列 $ X = [x_1, x_2, \ldots, x_n] $，标记子集 $ M $ 被掩码（例如，替换为 [MASK]）。损失函数为：

$$ L_{MLM} = -\sum_{i \in M} \log P(x_i | X_{\setminus M}; \theta) $$

其中：
- $ M $：掩码标记的索引。
- $ X_{\setminus M} $：排除掩码标记的序列。
- $ \theta $：模型参数。
- $ P(x_i | X_{\setminus M}; \theta) $：预测正确标记 $ x_i $ 的概率。

MLMs 中的自注意力机制是双向的，对注意力没有限制：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

其中 $ Q $、$ K $ 和 $ V $ 是查询、键和值矩阵，$ d_k $ 是键维度。

### 因果语言建模
CLMs 最大化给定前面标记预测下一个标记的可能性：

$$ L_{CLM} = -\sum_{t=1}^n \log P(x_t | x_{<t}; \theta) $$

其中：
- $ x_{<t} $：位置 $ t $ 之前的标记。
- $ P(x_t | x_{<t}; \theta) $：标记 $ x_t $ 的概率。

注意力机制使用因果掩码，一个下三角 $ T \times T $ 矩阵 $ M $，其中 $ M_{ij} = 0 $ 如果 $ i \geq j $（允许关注过去和当前标记），否则 $ M_{ij} = -\infty $：

$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$

这确保了单向性，对顺序生成至关重要。

## 应用

### 掩码语言模型
MLMs 适合需要全面序列理解的任务：
- **文本分类**：情感分析、主题分类、垃圾邮件检测。
- **问答（QA）**：抽取式问答，在上下文中识别答案范围。
- **命名实体识别（NER）**：识别姓名或地点等实体。
- **示例**：BERT ([BERT](https://arxiv.org/abs/1810.04805))、RoBERTa ([RoBERTa](https://arxiv.org/abs/1907.11692))。

### 因果语言模型
CLMs 在生成任务中表现出色：
- **文本生成**：聊天机器人、创意写作、内容创作。
- **代码生成**：如 GitHub Copilot 等工具。
- **语言翻译**：顺序翻译，虽然不如 MLMs 常见。
- **示例**：GPT-2 ([GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf))、GPT-3、LLaMA ([LLaMA](https://arxiv.org/abs/2302.13971))。

## 代码示例

### 掩码语言模型（BERT 用于分类）
以下是使用 Hugging Face Transformers 微调 BERT 进行情感分类的 Python 脚本。

```python
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset

# 示例数据集
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_len, truncation=True, padding='max_length')
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# 数据
texts = ["这部电影很棒！", "糟糕的体验，再也不来了。"]
labels = [1, 0]  # 正面、负面
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = SentimentDataset(texts, labels, tokenizer)

# 模型
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 训练
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    logging_steps=10,
)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()

# 推理
text = "令人惊叹的产品！"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
print(f"情感：{'正面' if prediction == 1 else '负面'}")
```

### 因果语言模型（GPT-2 用于文本生成）
以下是使用 Hugging Face 生成 GPT-2 文本的 Python 脚本。

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 生成文本
prompt = "从前"
inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(inputs['input_ids'], max_length=50, do_sample=True)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"生成：{generated_text}")
```

## 掩码模型用于分类和问答

像 BERT 这样的 MLMs 由于其双向上下文，对分类和问答非常有效。对于**分类**，BERT 使用特殊的 [CLS] 标记，其最终隐藏状态用作情感分析等任务的序列表示。对于**问答**，BERT 被微调以预测上下文中的答案范围，利用双向注意力将问题与相关上下文对齐。

### 为什么掩码有帮助
- **上下文学习**：掩码迫使模型基于周围上下文预测标记，增强其理解词关系的能力。
- **鲁棒性**：在不完整数据上训练使模型为现实世界中部分信息常见的情况做好准备。
- **迁移学习**：预训练的 MLMs 泛化良好，对特定任务需要最少的微调，如 GLUE 基准测试所示 ([GLUE](https://gluebenchmark.com/))。

## 方向性表示

### 掩码语言模型（双向）
- **数学表示**：不应用注意力掩码，允许每个标记关注所有其他标记：
  $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
- **实现**：在 PyTorch 中，注意力分数在没有限制的情况下计算：
  ```python
  attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  attention_weights = torch.softmax(attention_scores, dim=-1)
  ```

### 因果语言模型（单向）
- **数学表示**：因果掩码确保标记仅关注前面的位置：
  $$ M_{ij} = \begin{cases} 
  0 & \text{if } i \geq j \\
  -\infty & \text{otherwise}
  \end{cases} $$
  $$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V $$
- **实现**：在 PyTorch 中，掩码在 softmax 之前应用：
  ```python
  attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
  mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
  attention_scores = attention_scores.masked_fill(mask, float('-inf'))
  attention_weights = torch.softmax(attention_scores, dim=-1)
  ```

## 具体应用

### BERT 用于问答
- **场景**：客户支持系统从产品手册回答查询。
- **示例**：问题："如何重置我的设备？" 上下文："要重置，按住电源按钮 10 秒钟。"
- **过程**：BERT 一起处理问题和上下文，使用双向注意力识别答案范围："按住电源按钮 10 秒钟"。
- **为什么有效**：掩码预训练使 BERT 能够理解上下文关系，有效地对齐问题和上下文。

### GPT 用于故事生成
- **场景**：创意写作工具生成故事延续。
- **示例**：输入："龙在村庄上空翱翔。" 输出："它的鳞片在月光下闪闪发光，在屋顶上投下阴影。"
- **为什么有效**：CLM 的单向性质确保了连贯的顺序叙述流程。

## 结论
MLMs 和 CLMs 是 NLP 中的互补范式。MLMs 利用双向上下文进行需要深度理解的任务，而 CLMs 在顺序生成方面表现出色。它们的数学基础——MLMs 的双向注意力和 CLMs 的因果掩码——支撑了它们的优势。通过理解这些差异，开发人员可以为特定应用选择正确的模型，从情感分析到讲故事。正在进行的研究，如 AntLM ([AntLM](https://arxiv.org/abs/2412.03275)) 等混合方法，表明结合这些范式以提高性能的潜力。

## 关键引用
- [Hugging Face：掩码语言建模文档](https://huggingface.co/docs/transformers/en/tasks/masked_language_modeling)
- [Hugging Face：因果语言建模文档](https://huggingface.co/docs/transformers/en/tasks/language_modeling)
- [理解因果和掩码语言模型](https://medium.com/@sajidc707/understanding-causal-and-masked-language-models-how-scaling-laws-impact-their-power-7768d8a86a68)
- [掩码和因果语言建模探索](https://www.researchgate.net/publication/380757648_Exploration_of_Masked_and_Causal_Language_Modelling_for_Text_Generation)
- [IBM：什么是掩码语言模型？](https://www.ibm.com/think/topics/masked-language-model)
- [BERT：双向编码器表示](https://arxiv.org/abs/1810.04805)
- [RoBERTa：稳健优化的 BERT](https://arxiv.org/abs/1907.11692)
- [GPT-2：语言模型是无监督的](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.png)
- [LLaMA：高效语言模型](https://arxiv.org/abs/2302.13971)
- [GLUE NLP 任务基准测试](https://gluebenchmark.com/)
- [AntLM：桥接因果和掩码模型](https://arxiv.org/abs/2412.03275) 