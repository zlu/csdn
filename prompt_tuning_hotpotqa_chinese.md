## 理解提示调优与提示工程：概念、差异和应用场景

在自然语言处理（NLP）和大语言模型（LLMs）不断发展的领域中，我们与这些模型交互的方式变得更加精细和强大。在LLM可用性和性能方面最重要的进展中，有两种方法：**提示工程**和**提示调优**。虽然这些术语有时可以互换使用，但它们代表了引导语言模型的根本不同的理念和技术策略。理解这种区别对于希望有效利用LLM的研究人员、开发者和实践者来说至关重要。

本博客文章探讨了提示工程和提示调优之间的关键差异，提供了示例和比较结果，并介绍了**硬提示**与**软提示**的概念，特别是在提示调优的背景下。到最后，您将深入了解这些策略在理论和实践中的差异，以及它们如何在不同应用场景中使用。

### 什么是提示工程？

提示工程是一种**手动的**、**人工驱动的**方法，用于设计有效的提示来从预训练语言模型中获得期望的输出。这种方法依赖于理解LLM的行为和限制，并相应地制作输入提示。这个过程类似于编写一个巧妙的查询或指令，以获得最佳可能的结果，而不改变底层模型参数。

例如，考虑一个情感分析任务。一个简单的提示可能是：

> "这部电影还可以。"

这可能不会给您有用的输出，除非您明确指示模型。一个经过提示工程处理的版本会是：

> "请将以下评论的情感分类为积极、消极或中性：'这部电影还可以。'"

提示工程涉及试错迭代、理解模型特性，以及使用少样本学习（在提示中给出示例）或零样本学习（只给出指令）等技术来指导模型。

提示工程的一个流行子领域是**思维链（CoT）提示**，它鼓励模型展示其推理步骤。例如：

> "问题：如果汤姆有3个苹果，吃了1个，还剩多少个？
> 让我们一步步思考。"

这通常比直接要求答案产生更准确和可解释的结果。

### 什么是提示调优？

相比之下，提示调优是一种**机器学习的**、**自动化的**制作提示的方法。它涉及训练一小组参数（提示标记），这些参数被添加到输入前面。这些标记使用梯度下降进行优化，以在特定的下游任务上表现良好。基础模型保持冻结；只有提示嵌入被更新。

提示调优的主要目标是在不更新整个模型的情况下，使大型预训练语言模型适应新任务。这种方法在存储和计算方面非常高效，因为它只需要更新一小部分参数。

假设您有一个具有三个标签的分类任务。提示调优将训练一组标记（可能不是人类可读的），当与输入句子结合时，产生高准确性。与提示工程不同，提示调优不依赖于人类直觉，而是依赖于优化算法。

### 示例：情感分类

让我们使用一个假设的情感分类任务来比较这两种方法。

**提示工程示例：**

> "您是一个情感分类器。请将这句话分类为积极、消极或中性：
> '服务很慢，但食物很棒。'"

这个提示经过精心设计，提供上下文并约束模型的输出空间。它工作得相当好，但可能需要针对边缘情况或特定领域的输入进行调整。

**提示调优示例：**

在提示调优中，我们不是手动设计提示，而是学习一组嵌入向量，如\[P1, P2, P3, ..., Pk]，其中每个P都是一个可训练的向量。这些不是人类可读的。模型输入变为：

> \[P1, P2, ..., Pk] + "服务很慢，但食物很棒。"

在训练过程中，提示向量被调整以在有标签数据集上最大化性能。

### 硬提示与软提示

硬提示和软提示之间的区别是理解提示调优的核心。

**硬提示**是自然语言字符串，就像提示工程中使用的那样。它们是人类可读的，通常是手动制作的。这些可以在模型和任务之间重用，通常是可移植的。

另一方面，**软提示**是**学习的向量**。它们只存在于嵌入空间中，不对应于模型词汇表中的实际标记。它们不能被人类直接解释。这些专门用于提示调优，针对任务性能进行优化，通常在准确性方面优于硬提示，但以可解释性为代价。

打个比方，硬提示就像为Google写一个巧妙的问题，而软提示就像调整内部排名算法来提升某些结果。

### 它们如何相关？

提示调优概括了提示工程的思想。虽然提示工程在离散的、人类可读的空间中运行，但提示调优在连续的、学习的空间中运行。后者可以被认为是自动化和优化提示工程试图手动做的事情。

还有一种混合方法称为**前缀调优**或**提示集成**，其中硬提示通过学习的软提示进行增强。这可以提供两全其美的效果：可解释性和性能。

### 比较结果

几项研究在各种NLP任务上对提示调优和提示工程进行了基准测试。一篇具有里程碑意义的论文，Lester等人（2021）的"规模对参数高效提示调优的力量"发现：

* 仅使用几百个参数的提示调优可以匹配或超过大型模型上完全微调的性能。
* 软提示是高度任务特定的，在没有重新训练的情况下不能很好地跨任务泛化。
* 提示工程在零样本或少样本设置中表现良好，特别是当使用大型LLM（例如，GPT-4）时。

例如，在SuperGLUE基准测试中，使用20个提示向量的提示调优实现了与完全微调相似的准确性，但只需要更新不到0.1%的参数。

在实际使用中，提示工程部署更快，不需要训练，并且与通用模型配合良好。提示调优虽然需要标记数据和训练时间，但在任务特定优化方面表现出色，特别是在计算有限且完全模型微调不切实际的情况下。

### 应用场景和限制

提示工程在以下情况下是理想的：

* 您无法访问训练基础设施
* 任务是通用的或探索性的
* 模型足够大（例如，GPT-4）以在最小指导的情况下表现良好

提示调优在以下情况下是理想的：

* 您想在狭窄的任务上实现高准确性
* 您需要高效部署（例如，在边缘设备上）
* 您可以访问标记数据和适度的计算

然而，提示调优有局限性。它需要训练基础设施，可解释性较差，学习的软提示通常不能在模型或领域之间转移。

相反，提示工程严重依赖人类直觉，当难以找到最佳措辞或模型行为不可预测时，可能会表现不佳。

### 使用T5进行HotpotQA的提示调优
本笔记本演示了使用Hugging Face `peft`在HotpotQA事实问答数据集上进行软提示调优。
- 模型：`t5-small`
- 数据集：`hotpot_qa`（fullwiki）
- 调优方法：软提示调优（提示调优）



```python
!pip install transformers datasets peft accelerate -q
```


```python
from datasets import load_dataset
dataset = load_dataset("hotpot_qa", "fullwiki")
train_data = dataset["train"].select(range(1000))
val_data = dataset["validation"].select(range(100))
```


```python
from transformers import T5Tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")

def preprocess(example):
    question = example["question"]
    context = " ".join(example["context"])
    input_text = f"question: {question} context: {context}"
    return {
        "input_ids": tokenizer(input_text, truncation=True, padding="max_length", max_length=512)["input_ids"],
        "labels": tokenizer(example["answer"], truncation=True, padding="max_length", max_length=64)["input_ids"]
    }

train_dataset = train_data.map(preprocess, remove_columns=train_data.column_names)
val_dataset = val_data.map(preprocess, remove_columns=val_data.column_names)
```
    Map: 100%|██████████| 1000/1000 [00:00<00:00, 2815.83 examples/s]
    Map: 100%|██████████| 100/100 [00:00<00:00, 2925.86 examples/s]

```python
from transformers import T5ForConditionalGeneration
from peft import PromptTuningConfig, get_peft_model, TaskType

base_model = T5ForConditionalGeneration.from_pretrained("t5-small")

peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init="RANDOM",
    num_virtual_tokens=20,
    tokenizer_name_or_path="t5-small"
)

model = get_peft_model(base_model, peft_config)
model.print_trainable_parameters()
```

    trainable params: 20,480 || all params: 60,527,104 || trainable%: 0.0338



```python
from transformers import TrainingArguments, Trainer, DataCollatorForSeq2Seq

training_args = TrainingArguments(
    output_dir="./t5-prompt-tuned-hotpotqa",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-4,
    num_train_epochs=5,
    logging_dir="./logs",
    #evaluation_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=1,
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model)
)

trainer.train()
```

    <div>

      <progress value='625' max='625' style='width:300px; height:20px; vertical-align: middle;'></progress>
      [625/625 04:15, Epoch 5/5]
    </div>
    <table border="1" class="dataframe">
  <thead>
 <tr style="text-align: left;">
      <th>Step</th>
      <th>Training Loss</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>500</td>
      <td>2.982300</td>
    </tr>
  </tbody>
</table><p>

```python
# Add this line before loading the model
import torch
device = torch.device("cpu")

# When loading the model, specify device
model = model.to(device)

def answer_question(question, context):
    input_text = f"question: {question} context: {context}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(input_ids=inputs["input_ids"], max_length=64)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

example_q = "What is the birth place of the president of the country that has the city 'Kazan'?"
example_c = "Kazan is a city in Russia. Vladimir Putin is the president of Russia. Vladimir Putin was born in Leningrad, now Saint Petersburg."
print(answer_question(example_q, example_c))
```

    a city in Russia. Vladimir Putin is the president of Russia.


### 最终思考

总结一下，提示工程和提示调优都是使LLM适应特定任务的强大工具。提示工程是手动的、可解释的，部署快速。提示调优是自动化的、不透明的，通常更准确。在它们之间的选择取决于任务、可用资源和期望的结果。

随着LLM成为更多应用的核心，理解并利用这两种策略将变得至关重要。此外，混合方法的出现，将硬提示的可解释性与软提示的优化能力相结合，代表了一个有前途的前沿。

无论您是开发任务特定模型的研究人员、在生产中部署语言技术的实践者，还是仅仅探索LLM的爱好者，掌握提示工程和提示调优都将使您能够释放它们的全部潜力。 