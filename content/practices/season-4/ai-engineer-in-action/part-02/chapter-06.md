---
title: "第六章：自然语言处理的核心：从词嵌入到Transformer"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第六章：自然语言处理的核心：从词嵌入到Transformer"]
slug: "chapter-06"
---

在上一章中，我们已经掌握了深度学习的基本原理和PyTorch的实战技巧，并成功构建了一个基于LSTM的情感分类器。我们已经能够让机器“处理”语言，但我们离真正“理解”语言还有多远？

人类语言，是思想的载体，其复杂、微妙与歧义性，是计算机科学领域最艰巨的挑战之一。一个词语的意义，往往取决于其上下文；一个句子的情感，可能隐藏在精巧的语法结构和微妙的语序之中。要让机器理解语言，首先必须解决一个根本问题：如何将离散、符号化的文本，转化为机器能够计算和学习的数学表示？

本章，我们将踏上一段解码语言奥秘的旅程。这段旅程，将直接通往我们最终的目标——大语言模型（LLM）的殿堂。因为LLM之所以强大，正是因为它建立在一系列革命性的文本表示和架构创新之上。

我们将从这里开始：

- 文本表示法：我们将回顾文本表示的演进之路，从简单却稀疏的One-Hot编码，到能够捕捉词汇语义关系的词嵌入（Word Embedding）技术，如Word2Vec。你将理解，为什么“国王 - 男性 + 女性 ≈ 女王”这个著名的类比，成为了现代NLP的起点。
- 注意力机制（Attention）：这是深度学习领域最具影响力的思想之一。它模仿了人类视觉的注意力机制，允许模型在处理信息时，动态地将“焦点”放在输入序列中最相关的部分。我们将揭示其工作原理，并理解它是如何解决LSTM等传统序列模型的瓶颈的。
- Transformer架构全解析：2017年，一篇名为《Attention Is All You Need》的论文横空出世，提出了完全基于注意力机制的Transformer模型。它抛弃了RNN的循环结构，实现了真正意义上的并行计算，极大地提升了训练效率和模型性能。Transformer不仅统一了NLP领域，其影响力更辐射至CV、语音等多个领域，是所有现代大语言模型（从BERT到GPT系列）的共同基石。我们将以前所未有的深度，逐层剖析其内部的每一个组件。
- Hugging Face生态入门：理论的最终目的是为了应用。我们将介绍当今NLP领域事实上的“标准”——Hugging Face生态系统。你将学会使用其核心的`transformers`库，轻松地加载和使用数以万计的预训练模型，站在巨人的肩膀上解决实际问题。

最后，为了让你对Transformer的理解达到“知其然，更知其所以然”的境界，我们将通过一个极具挑战性的实战项目——从零开始，使用PyTorch构建一个简化版的Transformer模型。你将亲手实现Self-Attention、多头注意力、位置编码等核心组件，这个过程将彻底巩固你对Transformer架构的理解，其价值无可估量。

本章是本书技术深度的高峰。掌握了Transformer，你就掌握了理解和应用所有现代LLM的“万能钥匙”。现在，让我们集中全部的注意力，开始这场NLP领域最核心、最深刻的探索之旅。

## 6.1 文本表示法：从One-Hot到Word2Vec

### 6.1.1 离散表示法：One-Hot编码

要让计算机处理文本，第一步就是将词语数值化。最直观的方法是构建一个词汇表（Vocabulary），然后为每个词语分配一个唯一的ID。

假设我们的词汇表是：`{"我": 0, "爱": 1, "北京": 2, "天安门": 3}`。

One-Hot编码将每个词语表示为一个非常长的向量，这个向量的维度等于词汇表的大小。向量中，只有在对应词语ID的位置上为1，其余所有位置都为0。

- `我` -> `[1, 0, 0, 0]`
- `爱` -> `[0, 1, 0, 0]`
- `北京` -> `[0, 0, 1, 0]`

One-Hot编码的致命缺陷：

1. 维度灾难（Curse of Dimensionality）：真实世界的词汇表非常巨大（几十万甚至上百万），导致每个词的向量维度极高，且极其稀疏（绝大部分都是0），这在计算和存储上都是巨大的浪费。
2. 语义鸿沟：One-Hot向量是相互正交的。这意味着，从数学上看，任意两个词之间的距离都是一样的。`dist("北京", "天安门")` 和 `dist("北京", "爱")` 没有任何区别。它完全无法表达“北京”和“天安门”在语义上更接近这一事实。

我们需要一种更高级的表示法，它应该是低维、稠密的，并且能够蕴含语义信息。

### 6.1.2 分布式表示法：词嵌入（Word Embedding）

核心思想（分布式假设）：一个词的意义，由它周围的词来定义（"You shall know a word by the company it keeps"）。例如，经常出现在“银行”、“存款”、“利率”周围的词，很可能与“金融”相关。

词嵌入不再使用稀疏的高维向量，而是将每个词映射到一个低维（例如100、300维）、稠密、连续的向量空间中。在这个空间里：

语义上相近的词，其对应的向量在空间中的距离也更近。

向量之间的方向关系可以表示词与词之间的类比关系。这就是著名的 `vector("国王") - vector("男性") + vector("女性") ≈ vector("女王")` 的来源。

### 6.1.3 Word2Vec：训练你自己的词向量

Word2Vec是Google在2013年提出的一个里程碑式的工作，它提供了一种高效训练词嵌入的方法。Word2Vec包含两种模型架构：

1. CBOW (Continuous Bag-of-Words)：根据上下文词来预测中心词。例如，对于句子“我 爱 北京 天安门”，当中心词是“北京”时，上下文词是“我”、“爱”、“天安门”。CBOW模型会接收上下文词的词向量，并将它们（例如通过求平均）合并起来，去预测中心词“北京”。
2. Skip-gram：根据中心词来预测上下文词。还是上面的例子，Skip-gram模型会接收中心词“北京”的词向量，然后去分别预测它周围的词“我”、“爱”、“天安门”。Skip-gram在处理稀有词（低频词）时效果更好，但训练速度比CBOW慢。

训练过程（简述）：

Word2Vec的巧妙之处在于，它将无监督的词向量学习问题，转化为了一个有监督的“伪任务”（predicting surrounding words）。

1. 初始化一个巨大的词向量矩阵 `E`，其中每一行都是一个词的随机初始化的词向量。
2. 通过一个巨大的语料库，创建大量的（中心词，上下文词）训练样本。
3. 使用一个简单的神经网络（通常只有一个隐藏层），根据Skip-gram或CBOW任务进行训练。
4. 在训练过程中，我们真正关心的不是这个伪任务的预测结果，而是作为副产品被不断优化的词向量矩阵 `E`。
5. 训练结束后，这个矩阵 `E` 就是我们得到的词嵌入。

在PyTorch中，词嵌入通常通过一个`nn.Embedding`层来实现。这个层本质上就是一个可学习的查找表（Lookup Table）。

```python
import torch.nn as nn

# 假设词汇表大小为10000，我们想将每个词映射为300维的向量
vocab_size = 10000
embedding_dim = 300

embedding_layer = nn.Embedding(vocab_size, embedding_dim)

# 输入是词的ID序列，形状为 (batch_size, sequence_length)
input_ids = torch.LongTensor([[10, 25, 5, 2], [100, 3, 0, 0]])

# 输出是对应的词向量序列，形状为 (batch_size, sequence_length, embedding_dim)
embedded_vectors = embedding_layer(input_ids)
print(embedded_vectors.shape) # torch.Size([2, 4, 300])
```

在神经网络的训练过程中，这个`embedding_layer`的权重（即词向量矩阵）会随着反向传播一起被优化，从而学习到适合当前任务的词表示。我们也可以加载像Word2Vec或GloVe这样预训练好的词向量来初始化这个层，这称为迁移学习。

## 6.2 注意力机制（Attention）的革命

### 6.2.1 Encoder-Decoder架构的瓶颈

在注意力机制出现之前，处理序列到序列（Seq2Seq）任务（如机器翻译）的主流架构是基于RNN的Encoder-Decoder模型。

Encoder（编码器）：一个RNN（如LSTM），负责读取源语言句子（例如，一个英文句子），并将整个句子的信息压缩成一个固定长度的向量，称为上下文向量（Context Vector） `C`。这个向量 `C` 就是Encoder最后一个时间步的隐藏状态。

Decoder（解码器）：另一个RNN，它接收上下文向量 `C` 作为其初始隐藏状态，然后一个词一个词地生成目标语言句子（例如，法文句子）。

瓶颈在哪里？

整个源句子的所有信息，无论长短，都必须被硬生生地压缩进一个固定长度的上下文向量 `C` 中。这就像让你用一句话总结一部长篇小说，信息的损失是巨大的。对于长句子，模型很难记住开头的细节。这个固定长度的向量 `C` 成为了整个模型的性能瓶颈。

### 6.2.2 注意力机制的诞生

Bahdanau等人在2014年提出的注意力机制，正是为了打破这个瓶颈。其核心思想是：在Decoder生成每个词时，不应该只依赖于一个固定的上下文向量，而应该允许Decoder“回头看”Encoder的所有隐藏状态，并动态地决定在当前时间步，源句子的哪个部分最值得“关注”。

工作流程（以机器翻译为例）：

假设Decoder正在准备生成第 `t` 个目标词。

1. 计算对齐分数（Alignment Score）：Decoder当前的隐藏状态 `s_{t-1}` 会与Encoder的每一个隐藏状态 `h_1, h_2, ..., h_n` 进行比较，计算一个“对齐分数”或“相关性分数” `e_tj = score(s_{t-1}, h_j)`。这个`score`函数可以是一个简单的前馈网络。这个分数衡量了要生成的目标词与源句子中第 `j` 个词的关联程度。

2. 计算注意力权重（Attention Weights）：将所有对齐分数 `e_t1, e_t2, ..., e_tn` 通过一个`Softmax`函数进行归一化，得到一组注意力权重 `α_t1, α_t2, ..., α_tn`。这组权重的和为1，可以看作是一个概率分布，表示在当前时间步，注意力应该如何分配到源句子的各个词上。

3. 计算上下文向量（Context Vector）：将注意力权重 `α_tj` 作为加权系数，对Encoder的所有隐藏状态 `h_j` 进行加权求和，得到一个为当前时间步 `t` 量身定制的上下文向量 `C_t = Σ α_tj * h_j`。如果某个源词的注意力权重高，那么它的信息在 `C_t` 中所占的比重就大。

4. 生成目标词：将这个动态的上下文向量 `C_t` 与Decoder上一步的输出、以及当前的隐藏状态 `s_{t-1}` 结合起来，共同预测当前的目标词 `y_t`。

革命性意义：

- 打破信息瓶颈：不再依赖于单一的固定长度向量，而是为每个解码步骤动态生成上下文向量。
- 可解释性：通过可视化注意力权重矩阵，我们可以直观地看到在生成某个目标词时，模型主要“看”了源句子的哪些部分，这为我们理解和调试模型提供了窗口。
- 解决了长距离依赖：由于可以直接关注到源序列的任意位置，模型处理长距离依赖的能力大大增强。

注意力机制的思想是如此强大和通用，它很快就不再局限于Encoder-Decoder架构，而是演变成了一种更普适的机制，并最终催生了Transformer。

## 6.3 Transformer架构全解析：从Encoder-Decoder到Self-Attention

2017年，Google的论文《Attention Is All You Need》提出了Transformer模型，彻底颠覆了NLP领域。它的核心论点是：我们不再需要RNN的循环结构来处理序列，仅仅依靠注意力机制就足够了。

### 6.3.1 整体架构：基于Attention的Encoder-Decoder

Transformer的宏观结构仍然是一个Encoder-Decoder模型，但其内部的实现被完全重构了。

Encoder：由N个相同的Encoder Layer堆叠而成。负责将输入的ID序列（例如，英文句子）转换为一系列上下文感知的词表示。

Decoder：由N个相同的Decoder Layer堆叠而成。负责接收Encoder的输出和已经生成的目标序列，来预测下一个目标词。

Transformer 整体架构图 (图片来源: The Annotated Transformer)

### 6.3.2 核心组件之一：自注意力机制（Self-Attention）

这是Transformer的灵魂。传统的注意力机制用于连接Encoder和Decoder，而自注意力（Self-Attention）则是在同一个序列内部计算注意力。它的目的是让序列中的每个词，都能“看到”并衡量序列中所有其他词对自己的重要性，从而捕捉句子内部的依赖关系（如语法结构、指代关系等）。

Query, Key, Value (Q, K, V) 的抽象

为了实现Self-Attention，Transformer为输入序列中的每个词向量，都创建了三个新的向量：

- Query (查询向量) `q`：代表当前词，它要去“查询”其他词。
- Key (键向量) `k`：代表被查询的词，它像一个“标签”，用来和Query进行匹配。
- Value (值向量) `v`：代表被查询的词的实际内容。

这三个向量都是通过将原始词向量乘以三个可学习的权重矩阵 `W_Q`, `W_K`, `W_V` 得到的。

计算过程：

假设我们要计算句子 "Thinking Machines" 中 "Thinking" 这个词的Self-Attention输出。

1. 计算分数：将 "Thinking" 的Query向量 `q1`，与句子中所有词（包括它自己）的Key向量 `k1, k2` 进行点积（Dot-Product），得到分数。`score1 = q1 · k1`, `score2 = q1 · k2`。这个分数衡量了其他词对于理解 "Thinking" 的重要性。
2. 缩放（Scale）：将分数除以一个缩放因子，通常是Key向量维度的平方根 `sqrt(d_k)`。这可以防止在维度很高时点积结果过大，导致Softmax进入梯度很小的区域。
3. Softmax：将缩放后的分数通过Softmax函数，得到注意力权重。
4. 加权求和：将得到的注意力权重，与所有词的Value向量 `v1, v2` 进行加权求和，得到最终的输出向量 `z1`。这个 `z1` 就是 "Thinking" 这个词经过Self-Attention之后，融合了整个句子上下文信息的新表示。

这个过程对句子中的每个词都并行地进行，最终得到一系列上下文感知的输出向量。由于整个过程只涉及矩阵乘法，因此可以高度并行化，这是它相比RNN的巨大优势。

### 6.3.3 核心组件之二：多头注意力（Multi-Head Attention）

一次Self-Attention只让一个词从一个“角度”或“子空间”去关注其他词。但一个词的依赖关系可能是多方面的（例如，既有语法上的主谓关系，又有语义上的指代关系）。

多头注意力机制通过并行地运行多个独立的Self-Attention“头”（Head）来解决这个问题。

1. 将原始的Q, K, V通过多组不同的权重矩阵 `W_Q^i, W_K^i, W_V^i`，投影到多个不同的、低维的表示子空间中。
2. 在每个子空间中，独立地执行Self-Attention计算，得到一个输出向量 `z_i`。
3. 将所有头的输出向量 `z_1, z_2, ..., z_h` 拼接（Concatenate）起来。
4. 将拼接后的向量通过一个额外的线性层 `W_O` 进行变换，得到最终的输出。

这使得模型能够同时关注来自不同表示子空间的信息，从而更全面地捕捉复杂的依赖关系。

### 6.3.4 核心组件之三：位置编码（Positional Encoding）

Self-Attention本身并不包含任何关于词序的信息。如果我们打乱一个句子的顺序，Self-Attention的输出是完全一样的。这显然是不行的，因为语序在语言中至关重要。

位置编码就是为了向模型注入位置信息。Transformer的作者没有使用可学习的位置嵌入，而是采用了一种巧妙的、固定的数学方法：

为每个位置 `pos` 和向量的每个维度 `i`，使用不同频率的`sin`和`cos`函数来生成一个位置编码向量 `PE`。

- `PE(pos, 2i) = sin(pos / 10000^(2i/d_model))`
- `PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))`

将这个位置编码向量，直接加到原始的词嵌入向量上。

这种方法的优点是：

它可以推广到比训练时遇到的更长的序列。

由于`sin`和`cos`的周期性，模型可以很容易地学习到相对位置关系。

### 6.3.5 串联一切：Encoder和Decoder Layer的内部结构

Encoder Layer：

1. 一个多头自注意力层（Multi-Head Self-Attention）。
2. 一个残差连接（Residual Connection）和层归一化（Layer Normalization）。即 `LayerNorm(x + Sublayer(x))`。残差连接有助于缓解梯度消失，让深层网络更容易训练。
3. 一个简单的前馈神经网络（Feed-Forward Network），通常由两个线性层和一个ReLU激活函数组成。
4. 另一个残差连接和层归一化。

Decoder Layer：比Encoder Layer多了一个组件。

1. 一个带掩码的多头自注意力层（Masked Multi-Head Self-Attention）。在解码时，为了防止模型“偷看”未来的词，需要将当前位置之后的所有词的注意力权重设置为0。这个“掩码”操作确保了模型的自回归（auto-regressive）特性。
2. 残差连接和层归一化。
3. 一个多头注意力层，它的Q来自上一个Decoder层的输出，而K和V则来自Encoder的最终输出。这是连接Encoder和Decoder的桥梁，与传统的Attention机制作用相同。
4. 残差连接和层归一化。
5. 一个前馈神经网络。
6. 残差连接和层归一化。

最后，Decoder的输出会经过一个线性层和Softmax，来预测词汇表中每个词的概率。

## 6.4 Hugging Face生态入门：`transformers`库的妙用

手动实现和训练一个Transformer模型非常复杂。幸运的是，Hugging Face公司为我们提供了一套无与伦比的开源工具，使得使用最先进的NLP模型变得前所未有的简单。

Hugging Face生态的核心：

Model Hub：一个巨大的模型仓库，托管了数以万计的、由社区和企业贡献的预训练模型（如BERT, GPT-2, T5等），涵盖了100多种语言和各种任务。

`transformers`库：一个Python库，提供了加载、训练和使用Model Hub中所有模型的统一API。

`datasets`库：提供了对数千个常用数据集的便捷访问和处理工具。

`tokenizers`库：提供了高效、可定制的文本分词器。

### 6.4.1 `pipeline`：最简单的入门方式

`pipeline`是`transformers`库中最高级的抽象，能让你用几行代码就完成一个端到端的NLP任务。

```python
from transformers import pipeline

# 情感分析
classifier = pipeline("sentiment-analysis")
result = classifier("I love using Hugging Face, it's so easy!")
print(result) # [{'label': 'POSITIVE', 'score': 0.99...}]

# 文本生成 (使用GPT-2)
generator = pipeline("text-generation", model="gpt2")
text = generator("In a world where AI is becoming more powerful,", max_length=30, num_return_sequences=2)
print(text)

# 填空 (使用BERT)
unmasker = pipeline("fill-mask", model="bert-base-uncased")
result = unmasker("The capital of France is [MASK].")
print(result) # [{'token_str': 'paris', ...}]
```


### 6.4.2 AutoClass：加载任意模型和分词器

当你需要更多控制时，可以使用`AutoModel`和`AutoTokenizer`。它们可以根据你提供的模型名称（例如`"bert-base-uncased"`），自动从Model Hub下载并加载对应的模型类和分词器。

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# 模型名称
model_name = "distilbert-base-uncased-finetuned-sst-2-english"

# 1. 加载分词器
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 2. 加载模型
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 准备输入文本
texts = ["This movie was great!", "This movie was terrible."]

# 3. 分词和编码
# padding=True: 填充到批次中最长句子的长度
# truncation=True: 如果句子超过模型最大长度，则截断
# return_tensors="pt": 返回PyTorch Tensor
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
print(inputs)

# 4. 模型推理
with torch.no_grad():
    outputs = model(inputs) # 使用解包字典作为参数

# 5. 解析输出
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1)
print(predictions) # tensor([1, 0]) (1: positive, 0: negative)
```

这个“加载Tokenizer -> 加载Model -> 编码文本 -> 模型推理 -> 解析输出”的流程，是使用`transformers`库解决几乎所有问题的标准范式。

## 6.5 实战项目：从零构建一个简化版的Transformer模型

这个项目极具挑战性，但完成后你对Transformer的理解将达到一个全新的高度。我们将实现一个用于机器翻译的、简化版的Encoder-Decoder Transformer。

我们将实现的关键组件：

- 位置编码（Positional Encoding）
- 缩放点积注意力（Scaled Dot-Product Attention）
- 多头注意力（Multi-Head Attention）
- 位置前馈网络（Position-wise Feed-Forward Network）
- Encoder Layer 和 Decoder Layer
- 完整的Encoder, Decoder和Transformer模型

```python
# 这是一个高度浓缩和简化的实现，旨在展示核心逻辑
import torch
import torch.nn as nn
import math

# --- 组件 1: 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [seq_len, batch_size, d_model]
        x = x + self.pe[:x.size(0), :]
        return x

# --- 组件 2: 多头注意力 (包含缩放点积注意力) ---
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.head_dim = d_model // nhead
      
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, query, key, value, mask=None):
        # query, key, value: [seq_len, batch_size, d_model]
        batch_size = query.size(1)
      
        # 1. 线性投影
        q = self.q_linear(query)
        k = self.k_linear(key)
        v = self.v_linear(value)
      
        # 2. 重塑以进行多头计算
        # [seq_len, batch_size, d_model] -> [seq_len, batch_size * nhead, head_dim] -> [batch_size * nhead, seq_len, head_dim]
        q = q.view(-1, batch_size * self.nhead, self.head_dim).transpose(0, 1)
        k = k.view(-1, batch_size * self.nhead, self.head_dim).transpose(0, 1)
        v = v.view(-1, batch_size * self.nhead, self.head_dim).transpose(0, 1)
      
        # 3. 缩放点积注意力
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(self.head_dim)
      
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
          
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
      
        output = torch.bmm(attention_weights, v)
      
        # 4. 重塑和最后线性层
        # [batch_size * nhead, seq_len, head_dim] -> [seq_len, batch_size, d_model]
        output = output.transpose(0, 1).contiguous().view(-1, batch_size, self.d_model)
        output = self.out_linear(output)
      
        return output

# --- 组件 3: Encoder Layer ---
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
      
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # Self-Attention
        src2 = self.self_attn(src, src, src, mask=src_mask)
        # Add & Norm
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Feed Forward
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        # Add & Norm
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

# --- 完整的Transformer模型 (简化版) ---
class MyTransformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
      
        # 使用PyTorch内置的TransformerEncoder和DecoderLayer
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
      
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
      
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
      
    def forward(self, src, tgt, src_padding_mask, tgt_padding_mask, memory_key_padding_mask, tgt_mask):
        # src: [src_len, batch_size]
        # tgt: [tgt_len, batch_size]
        src_emb = self.pos_encoder(self.src_embedding(src))
        tgt_emb = self.pos_encoder(self.tgt_embedding(tgt))
      
        memory = self.transformer_encoder(src_emb, src_key_padding_mask=src_padding_mask)
        output = self.transformer_decoder(tgt_emb, memory, tgt_mask=tgt_mask, 
                                          tgt_key_padding_mask=tgt_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
      
        return self.fc_out(output)

# ... 后续还需要编写生成掩码的函数、训练循环等 ...
```

这个实战项目非常复杂，需要对PyTorch和Transformer的每个细节都有深入的理解。完成它，将是你技术能力的一次巨大飞跃。

## 本章小结

在本章中，我们完成了一次穿越现代自然语言处理核心地带的深度旅行。

我们从最基本的文本表示法出发，理解了从One-Hot到词嵌入（Word2Vec）的演进，解决了如何让机器捕捉词汇语义的关键问题。

接着，我们学习了具有革命性意义的注意力机制，它打破了传统RNN模型的瓶颈，实现了对输入序列相关部分的动态聚焦。

在此基础上，我们对Transformer架构进行了“庖丁解牛”式的全解析。我们深入了其内部的每一个核心组件：赋予模型上下文理解能力的自注意力，增强模型表示能力的多头注意力，以及解决语序问题的位置编码。我们清晰地看到了这些组件是如何在Encoder和Decoder中协同工作，最终构建起这个强大的模型。

为了将理论付诸实践，我们入门了Hugging Face生态，学会了使用`transformers`库来轻松调用最先进的预训练模型，这让我们能够快速地解决实际问题。

最后，通过从零构建一个简化版Transformer的极具挑战性的实战项目，我们将所有理论知识内化为了深刻的实践能力。

完成本章后，你已经掌握了通往大语言模型世界的所有前置知识。你不再会对BERT、GPT这些如雷贯耳的名字感到神秘，因为你知道，它们的核心，正是你已经烂熟于心的Transformer架构。在下一章，我们将正式进入LLM的实战领域，学习如何对这些庞大的模型进行微调和应用，真正释放它们的巨大潜能。
