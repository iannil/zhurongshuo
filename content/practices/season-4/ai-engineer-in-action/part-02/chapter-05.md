---
title: "第五章：深度学习之门：从神经网络到PyTorch实战"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第五章：深度学习之门：从神经网络到PyTorch实战"]
slug: "chapter-05"
---

欢迎来到本书的第二篇章。在“基础内功篇”中，我们已经为自己锻造了一身坚实的铠甲：精通了Python编程，掌握了数据科学的利器，并具备了将应用工程化部署的能力。现在，装备齐全的我们，即将踏上一段更为激动人心的征程——深入探索驱动现代人工智能革命的核心引擎：深度学习。

如果说Scikit-learn代表的传统机器学习是基于精巧的数学和统计学构建的“手工作坊”，那么深度学习则更像是一座能够自动学习和提炼特征的“现代化工厂”。它通过模拟人脑中神经网络的结构，构建出由成千上万甚至数十亿个“神经元”连接而成的深度神经网络（Deep Neural Networks, DNNs），从而在图像识别、语音识别、自然语言处理等领域取得了前所未有的突破。

我们今天所惊叹的大语言模型（LLM），正是深度学习发展至今的巅峰之作。要真正理解并驾驭LLM，我们必须回到它的源头，理解其最基本的构成单元和工作原理。本章，就是我们开启这扇“深度学习之门”的钥匙。

我们将从深度学习的“第一性原理”出发：
神经网络与反向传播：我们将用最直观的方式，为你揭示神经网络是如何进行信息传递和学习的。你将理解，看似神秘的“学习”过程，其本质是一场基于微积分的、名为“反向传播”的、优雅的“功劳分配”游戏。
PyTorch框架精讲：理论是枯燥的，而代码是鲜活的。我们将深入学习当今学术界和工业界最受欢迎的深度学习框架——PyTorch。你将掌握其三大核心支柱：`Tensor`（多维数组）、`Autograd`（自动求导引擎）和`nn.Module`（神经网络的构建基石）。
经典网络结构解析：在LLM的Transformer架构一统天下之前，CNN（卷积神经网络）和RNN（循环神经网络）曾是各自领域的王者。理解它们的设计思想，对于我们理解特征提取、序列建模等核心概念，乃至理解Transformer的演进背景，都至关重要。
训练的艺术：构建一个网络只是第一步，如何让它“学得好”则是一门艺术。我们将探讨损失函数（告诉模型错在哪）、优化器（指导模型如何改进）和正则化（防止模型“死记硬背”）这三大核心要素。

最后，我们将通过一个经典的NLP实战项目——使用PyTorch实现一个文本情感分类器——来将本章所有理论和技术点串联起来。我们将从零开始，定义数据集、构建模型、编写训练循环，并最终得到一个能够判断电影评论是正面还是负面的AI模型。

本章是你从“AI应用者”向“AI构建者”转变的关键一步。它将为你后续学习Transformer架构和驾驭大语言模型打下最坚实、最深刻的理论与实践基础。现在，让我们点燃“计算”的熔炉，开启这场充满挑战与创造的“炼丹”之旅。

## 5.1 神经网络与反向传播的核心思想

### 5.1.1 从生物到数学：神经元的抽象

深度学习的灵感来源于生物大脑。一个生物神经元接收来自其他神经元的电信号（输入），当这些信号的累积强度超过某个阈值时，它就会被“激活”，并向其他神经元发送信号（输出）。

数学家和计算机科学家将这个过程抽象成了一个简单的数学模型——人工神经元（或称感知机）：

1. 输入与权重：神经元接收多个输入值 `x1, x2, ..., xn`。每个输入都与一个权重（weight） `w1, w2, ..., wn` 相关联。权重代表了该输入的重要性。
2. 加权求和：神经元将所有输入与对应的权重相乘，然后求和，并加上一个偏置（bias） `b`。这个结果 `z = (w1*x1 + w2*x2 + ... + wn*xn) + b`，在数学上可以简洁地表示为向量点积：`z = w · x + b`。
3. 激活函数：将加权和 `z` 传入一个非线性的激活函数（Activation Function） `f`，得到最终的输出 `a = f(z)`。

为什么需要激活函数？
激活函数的非线性是整个神经网络能够学习复杂模式的关键。如果没有激活函数（或者说激活函数是线性的），那么无论你将多少层神经元堆叠在一起，整个网络本质上都只是一个简单的线性模型，无法学习像图像识别、语言理解这样复杂的非线性关系。

常见的激活函数：
Sigmoid: `f(z) = 1 / (1 + e^(-z))`。将输入压缩到(0, 1)之间，常用于二分类问题的输出层，表示概率。
ReLU (Rectified Linear Unit): `f(z) = max(0, z)`。计算简单，能有效缓解梯度消失问题，是目前最常用的激活函数之一。

### 5.1.2 神经网络：从单个神经元到层层相连

一个神经元的能力是有限的。但当我们将大量的神经元组织成层（Layer），并将这些层前后连接起来，就构成了神经网络。
输入层（Input Layer）：接收最原始的数据，例如一张图片的像素值，或一个句子的词向量。
隐藏层（Hidden Layers）：位于输入层和输出层之间，负责进行大部分的计算和特征提取。一个神经网络可以有零个或多个隐藏层。当隐藏层数量大于等于一时，我们称之为深度神经网络。
输出层（Output Layer）：产生最终的预测结果。例如，在猫狗分类任务中，输出层可能有两个神经元，分别代表“是猫的概率”和“是狗的概率”。

信息在网络中从输入层流向输出层的过程，我们称之为前向传播（Forward Propagation）。每一层的输出，都作为下一层的输入。

### 5.1.3 “学习”的本质：反向传播与梯度下降

我们构建了一个网络，但它的权重 `w` 和偏置 `b` 最初都是随机初始化的。这样一个未经训练的网络，对于任何输入，给出的都是随机的、错误的预测。那么，网络是如何“学习”的呢？

学习的过程，本质上就是一个参数优化的过程。我们希望找到一组最佳的 `w` 和 `b`，使得网络对于给定的输入，能产生最接近真实标签的输出。

这个过程可以分为三步：

第一步：定义损失（Loss）
我们需要一个量化的指标来衡量模型的预测结果“有多差”。这个指标就是损失函数（Loss Function）。例如，在分类任务中，常用的损失函数是交叉熵损失（Cross-Entropy Loss）。损失值越大，说明模型预测得越离谱。我们的目标，就是通过调整参数，让损失值变得尽可能小。

第二步：计算梯度（Gradient）
损失 `L` 是一个关于所有权重 `w` 和偏置 `b` 的函数。微积分告诉我们，函数在某一点的梯度，指向了该函数值增长最快的方向。那么，梯度的反方向，就是函数值下降最快的方向。

我们想让损失 `L` 变小，就需要计算出 `L` 对每一个参数（如 `w_ij`，表示第i层第j个神经元的某个权重）的偏导数 `∂L/∂w_ij`。这个包含了所有偏导数的向量，就是损失函数关于参数的梯度。

如何高效地计算这个梯度？ 这就是反向传播（Backpropagation）算法的核心所在。它利用微积分中的链式法则，从输出层开始，逐层向后计算梯度。

1. 首先计算损失对输出层激活值的梯度。
2. 然后利用这个梯度，计算对输出层加权和的梯度。
3. 再利用这个梯度，计算对连接到输出层的权重和偏置的梯度，以及对前一层激活值的梯度。
4. ……如此循环，直到计算出所有参数的梯度。

反向传播是一个极其巧妙的算法，它避免了大量的重复计算，使得在深层网络中计算梯度成为可能。幸运的是，在现代深度学习框架中，我们无需手动实现它。

第三步：更新参数（Gradient Descent）
有了梯度之后，我们就可以更新参数了。最简单的更新规则是梯度下降（Gradient Descent）：
`new_w = old_w - learning_rate * ∂L/∂w`

`learning_rate`（学习率）是一个超参数，它控制了我们每次更新参数的“步长”。
我们从梯度的反方向（`-`号）迈出一步，来更新权重。

这个“前向传播 -> 计算损失 -> 反向传播 -> 更新参数”的循环，就是神经网络训练的核心。我们会将整个数据集（或一小批数据）反复地送入这个循环中进行迭代，每一次迭代，模型的参数都会向着让损失更小的方向进行微调。经过成千上万次的迭代后，模型就逐渐“学会”了如何进行准确的预测。

## 5.2 PyTorch框架精讲：Tensor、Autograd与`nn.Module`

PyTorch是由Facebook AI研究院（FAIR）推出的一个开源深度学习框架。它以其Pythonic的设计、灵活性和强大的动态计算图机制，深受学术界和研究人员的喜爱，并逐渐在工业界占据主导地位。

### 5.2.1 `Tensor`：带GPU加速的多维数组

PyTorch的`Tensor`，与Numpy的`ndarray`在概念上极其相似，可以看作是`ndarray`的“超级加强版”。
它是一个多维数组，是PyTorch中数据流动的基本单位。
核心优势：`Tensor`可以被无缝地移动到GPU上进行计算，从而利用GPU强大的并行计算能力来加速训练。

```python
import torch

# --- 创建Tensor ---
# 从列表创建
x_list = [[1, 2], [3, 4]]
x_tensor = torch.tensor(x_list)
print(x_tensor)

# 类似Numpy的创建方式
x_zeros = torch.zeros(2, 3)
x_rand = torch.rand(2, 3)
print(x_zeros)
print(x_rand)

# --- Tensor的属性 ---
print(f"Shape of tensor: {x_tensor.shape}")
print(f"Datatype of tensor: {x_tensor.dtype}")
print(f"Device tensor is stored on: {x_tensor.device}")

# --- GPU加速 ---
# 检查是否有可用的GPU
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available! Using device: {device}")
  
    # 将Tensor移动到GPU
    x_gpu = x_tensor.to(device)
    print(f"x_gpu is on device: {x_gpu.device}")
  
    # 在GPU上进行计算
    y_gpu = x_gpu + x_gpu
  
    # 将结果移回CPU（例如，用于打印或与Numpy交互）
    y_cpu = y_gpu.to("cpu")
else:
    device = torch.device("cpu")
    print("GPU not available, using CPU.")

# --- 与Numpy的互操作 ---
# Tensor -> Numpy
np_array = x_tensor.numpy()
print(f"Numpy array: \n {np_array}")

# Numpy -> Tensor
np_array_new = np.ones((2, 2))
tensor_from_np = torch.from_numpy(np_array_new)
print(f"Tensor from numpy: \n {tensor_from_np}")
```

这种与Numpy的无缝转换，使得我们可以轻松地在数据预处理（常用Numpy/Pandas）和模型计算（使用PyTorch Tensor）之间切换。

### 5.2.2 `Autograd`：神奇的自动求导引擎

这是PyTorch最核心、最神奇的功能。它为我们免去了手动实现反向传播的痛苦。`Autograd`会悄悄地记录下所有在`Tensor`上的操作，构建一个动态计算图（Dynamic Computational Graph）。

当一个`Tensor`的属性`.requires_grad`被设置为`True`时，`Autograd`就会开始跟踪它。

```python
# 创建需要计算梯度的Tensor
w = torch.tensor([[2.0], [3.0]], requires_grad=True)
b = torch.tensor(1.0, requires_grad=True)

# 定义输入
x = torch.tensor([[1.0, 2.0]])

# 前向传播
# y = x @ w + b  (@是矩阵乘法的简写)
z = x @ w + b
loss = torch.sum(z) # 假设损失就是z的和

print(f"z: {z}")
print(f"loss: {loss}")

# --- 反向传播 ---
# 调用 .backward()，Autograd会自动计算loss对所有requires_grad=True的Tensor的梯度
loss.backward()

# --- 查看梯度 ---
# 梯度值会累积在 .grad 属性中
print(f"Gradient of w: \n {w.grad}")
print(f"Gradient of b: {b.grad}")
```

发生了什么？

1. 前向传播 `z = x @ w + b`，其中 `x` 是 `[[1, 2]]`，`w` 是 `[[2], [3]]`。
    `x @ w` = `1*2 + 2*3` = `8`。
    `z` = `8 + 1` = `9`。
    `loss` = `9`。
2. `loss.backward()` 被调用。
3. `Autograd`开始反向计算梯度：
    `∂loss/∂z` = `1`
    `∂z/∂w` = `x^T` = `[[1], [2]]` (链式法则)
    `∂loss/∂w` = `∂loss/∂z * ∂z/∂w` = `1 * [[1], [2]]` = `[[1], [2]]`
    `∂z/∂b` = `1`
    `∂loss/∂b` = `∂loss/∂z * ∂z/∂b` = `1 * 1` = `1`
4. 计算出的梯度与我们打印的 `w.grad` 和 `b.grad` 完全一致！

重要提示：
梯度是会累积的。在每次进行参数更新前，都需要手动将梯度清零：`optimizer.zero_grad()`。
只有浮点类型的Tensor才能计算梯度。
在模型评估（inference）阶段，我们不需要计算梯度，应该使用`with torch.no_grad():`上下文管理器来关闭`Autograd`，这样可以节省内存并加速计算。

### 5.2.3 `nn.Module`：构建神经网络的“乐高积木”

`torch.nn`是PyTorch专门用于构建神经网络的模块。所有的网络层、损失函数、激活函数都在这里。而`nn.Module`是所有神经网络模块的基类。

要构建自己的神经网络，我们通常需要：

1. 创建一个继承自`nn.Module`的类。
2. 在`__init__`方法中，定义网络所需的各个层（如卷积层、线性层）。这些层本身也是`nn.Module`的子类。
3. 在`forward`方法中，定义数据在前向传播过程中是如何流经这些层的。

```python
import torch.nn as nn
import torch.nn.functional as F

class SimpleNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 必须调用父类的__init__方法
        super(SimpleNet, self).__init__()
      
        # --- 定义网络层 ---
        # 线性层 (y = Wx + b)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # --- 定义前向传播逻辑 ---
        # x -> fc1 -> relu -> fc2 -> output
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 使用我们定义的网络 ---
input_dim = 784 # 例如，一个28x28的展平图像
hidden_dim = 128
output_dim = 10 # 例如，0-9的10个类别

# 实例化模型
model = SimpleNet(input_dim, hidden_dim, output_dim)
print(model)

# 我们可以像函数一样调用模型实例，它会自动执行forward方法
dummy_input = torch.randn(64, input_dim) # 模拟一个batch的数据
output = model(dummy_input)
print(f"Output shape: {output.shape}") # torch.Size([64, 10])
```

通过继承`nn.Module`，我们的`SimpleNet`类自动获得了许多强大的功能，例如：
`model.parameters()`：可以方便地获取模型中所有可训练的参数（权重和偏置）。
`model.to(device)`：可以将整个模型及其所有参数一键移动到GPU。
`model.train()` / `model.eval()`：切换训练和评估模式（对于像Dropout和BatchNorm这样的层很重要）。

## 5.3 经典网络结构解析：CNN与RNN/LSTM

在Transformer出现之前，CNN和RNN是深度学习在计算机视觉（CV）和自然语言处理（NLP）领域取得成功的两大基石。

### 5.3.1 卷积神经网络（CNN）：图像特征的捕获者

核心思想：
传统的全连接网络处理图像时，会将图像展平成一个长向量，这会丢失像素点的空间结构信息。CNN通过卷积（Convolution）和池化（Pooling）操作，专门设计用来处理具有网格结构的数据（如图像）。

1. 卷积层 (`nn.Conv2d`)：
    它使用一个小的卷积核（Kernel）或滤波器（Filter）（例如3x3或5x5），在输入图像上滑动。
    在每个位置，卷积核与其覆盖的图像区域进行逐元素相乘再求和，得到一个输出值。
    这个过程可以看作是在提取局部特征。例如，一个卷积核可能对图像中的垂直边缘敏感，另一个可能对某种颜色或纹理敏感。
    参数共享：同一个卷积核在整张图上共享同一套权重，这极大地减少了模型的参数量，并使得网络具有平移不变性（无论猫在图的左上角还是右下角，都能被识别）。

2. 池化层 (`nn.MaxPool2d`)：
    也叫下采样层。它在一个区域内（例如2x2），取最大值（Max Pooling）或平均值（Average Pooling）作为输出。
    作用：
        减小特征图的尺寸，从而减少后续层的计算量和参数。
        提供一定程度的平移、旋转不变性，使模型更鲁棒。

一个典型的CNN结构通常是卷积层 -> 激活函数 -> 池化层的重复堆叠，最后接上几个全连接层进行分类。

```python
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 输入: 1个通道 (灰度图), 输出: 16个通道, 卷积核: 3x3, padding=1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        # 假设输入是28x28的图像，经过两次pooling后，尺寸变为 7x7
        # 32个通道 * 7 * 7 = 1568
        self.fc1 = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # 展平特征图
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc1(x)
        return x
```


### 5.3.2 循环神经网络（RNN/LSTM）：序列数据的记忆者

核心思想：
对于文本、语音、时间序列这类具有先后顺序的数据，CNN和普通全连接网络都无法处理。RNN通过引入一个“记忆”单元——隐藏状态（Hidden State）——来解决这个问题。

1. 循环结构：
    在处理序列的每个时间步 `t` 时，RNN不仅接收当前的输入 `x_t`，还接收来自上一个时间步的隐藏状态 `h_{t-1}`。
    它将 `x_t` 和 `h_{t-1}` 一起计算，生成当前的输出 `y_t` 和新的隐藏状态 `h_t`。
    这个新的 `h_t` 会被传递到下一个时间步 `t+1`。
    通过这种方式，`h_t` 就像一个动态的记忆，它编码了从序列开始到当前位置的所有历史信息。

RNN的困境：长期依赖问题
标准的RNN在处理长序列时，会遇到梯度消失/爆炸的问题。这意味着在反向传播时，梯度会随着时间步的增加而指数级地减小或增大，导致模型难以学习到序列中相距较远元素之间的依赖关系（例如，一个长段落开头的主语和结尾的谓语之间的关系）。

LSTM（Long Short-Term Memory）
LSTM是一种特殊的RNN，它通过引入一个更复杂的内部结构——细胞状态（Cell State）和三个门（Gate）——来解决长期依赖问题。
细胞状态 `C_t`：像一条传送带，信息可以在上面直流，只进行少量的线性交互。这使得梯度可以很容易地在长序列中传递。
遗忘门（Forget Gate）：决定从上一个细胞状态 `C_{t-1}` 中丢弃哪些信息。
输入门（Input Gate）：决定将哪些新的信息存入当前的细胞状态 `C_t`。
输出门（Output Gate）：决定从当前的细胞状态 `C_t` 中输出哪些信息作为隐藏状态 `h_t`。

这些门都是由Sigmoid激活函数控制的小型神经网络，它们可以学会何时遗忘、何时记忆、何时输出，从而更有效地捕捉长期依赖。在Transformer出现之前，LSTM及其变体GRU是处理NLP任务的绝对主力。

```python
# 在PyTorch中使用LSTM
# input_size: 每个时间步输入的特征维度 (如词向量维度)
# hidden_size: 隐藏状态的维度
lstm_layer = nn.LSTM(input_size=100, hidden_size=256, num_layers=2, batch_first=True)

# 输入的shape: (batch_size, sequence_length, input_size)
dummy_input = torch.randn(32, 50, 100) 

# 输出: output, (h_n, c_n)
# output: 每个时间步的隐藏状态 (32, 50, 256)
# h_n: 最后一个时间步的隐藏状态 (num_layers, 32, 256)
# c_n: 最后一个时间步的细胞状态 (num_layers, 32, 256)
output, (hidden_state, cell_state) = lstm_layer(dummy_input)
```


## 5.4 训练的艺术：损失函数、优化器与正则化


### 5.4.1 损失函数（Loss Functions）：指引方向的灯塔

损失函数告诉我们模型离目标还有多远。选择正确的损失函数至关重要。
回归任务：
    `nn.MSELoss` (均方误差损失)：`L = (y_pred - y_true)^2`。最常用的回归损失，对异常值敏感。
    `nn.L1Loss` (平均绝对误差损失)：`L = |y_pred - y_true|`。对异常值更鲁棒。
二分类任务：
    `nn.BCELoss` (二元交叉熵损失)：需要模型的输出经过Sigmoid激活，表示概率。
    `nn.BCEWithLogitsLoss`：将Sigmoid和BCELoss合并，数值上更稳定，是二分类的首选。
多分类任务：
    `nn.CrossEntropyLoss`：这是多分类任务中最常用的损失函数。它内部自动包含了Softmax操作和负对数似然损失。因此，模型的原始输出（logits）可以直接传入，无需手动进行Softmax。

### 5.4.2 优化器（Optimizers）：驱动学习的引擎

优化器实现了梯度下降算法，根据计算出的梯度来更新模型的参数。
`torch.optim.SGD` (随机梯度下降)：最基本的优化器。
    `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)`
    `momentum`（动量）是一个改进项，它引入了过去梯度的累积，有助于加速收敛并越过局部最优点。
`torch.optim.Adam` (Adaptive Moment Estimation)：目前最常用、最通用的优化器之一。
    `optimizer = torch.optim.Adam(model.parameters(), lr=0.001)`
    它结合了动量和RMSProp的思想，能为每个参数自适应地计算学习率。在大多数情况下，Adam都能取得良好且快速的收-敛效果，是入门的首选。

一个标准的训练循环（Training Loop）

```python
# 假设 model, train_loader, loss_fn, optimizer 已定义
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        # 1. 将数据移动到GPU
        inputs, labels = inputs.to(device), labels.to(device)
      
        # 2. 前向传播
        outputs = model(inputs)
      
        # 3. 计算损失
        loss = loss_fn(outputs, labels)
      
        # 4. 反向传播
        # a. 清空旧梯度
        optimizer.zero_grad()
        # b. 计算新梯度
        loss.backward()
      
        # 5. 更新参数
        optimizer.step()
      
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
```


### 5.4.3 正则化（Regularization）：防止“死记硬背”的良方

当模型的容量（复杂度）远超训练数据的复杂性时，它可能会“记住”训练集中的每一个样本，包括噪声。这会导致模型在训练集上表现极好，但在未见过的测试集上表现很差。这种现象称为过拟合（Overfitting）。正则化就是一系列用于对抗过拟合的技术。

L1/L2 正则化：
    在损失函数上增加一个与模型权重大小相关的惩罚项。
    L2正则化（权重衰减, Weight Decay）倾向于让权重变得更小、更分散。在PyTorch的优化器中通过`weight_decay`参数实现：`torch.optim.Adam(..., weight_decay=1e-4)`。
Dropout (`nn.Dropout`)：
    在训练过程中的每次前向传播时，以一定的概率 `p` 随机地将一部分神经元的输出设置为0。
    这强迫网络不能过度依赖于任何一个神经元，而是要学习到更鲁棒、更冗余的特征表示。
    重要：Dropout只在训练时生效。调用`model.eval()`会自动关闭Dropout。
早停（Early Stopping）：
    在训练过程中，持续监控模型在验证集上的性能。
    如果验证集上的损失连续多个epoch不再下降，甚至开始上升，就提前终止训练，并保存性能最好的那个模型。

## 5.5 实战项目：使用PyTorch实现一个文本情感分类器

现在，我们将运用本章所学的所有知识，从零开始构建一个能够判断IMDB电影评论是正面还是负面的情感分类器。

项目流程：

1. 数据准备：加载IMDB数据集，进行文本预处理（分词、构建词汇表、数值化）。
2. 定义`Dataset`和`DataLoader`：使用PyTorch的工具来高效地加载和批处理数据。
3. 构建模型：我们将构建一个基于LSTM的模型。
4. 定义训练和评估函数：编写标准的训练循环和评估逻辑。
5. 执行训练：运行训练过程，并观察损失和准确率的变化。

1. 数据准备
我们将使用`torchtext`库来方便地处理数据。

```bash
pip install torchtext spacy
python -m spacy download en_core_web_sm
```

```python
# 1_data_preparation.py
import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import spacy

# 加载英文分词器
tokenizer = get_tokenizer('spacy', language='en_core_web_sm')

# 加载IMDB数据集
train_iter, test_iter = IMDB(split=('train', 'test'))

def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 构建词汇表
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"]) # 设置未知词的默认索引

# 定义文本和标签的处理管道
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

# 封装处理逻辑
def process_data(data_iter):
    processed_data = []
    for label, text in data_iter:
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        processed_label = torch.tensor(label_pipeline(label), dtype=torch.int64)
        processed_data.append((processed_label, processed_text))
    return processed_data

train_data = process_data(IMDB(split='train'))
test_data = process_data(IMDB(split='test'))

print("数据准备完成。")
# 可以保存 vocab 和处理好的数据，以便后续使用
# torch.save(vocab, 'vocab.pth')
# torch.save(train_data, 'train_data.pth')
# torch.save(test_data, 'test_data.pth')
```

2. `Dataset` 和 `DataLoader`
PyTorch的`DataLoader`需要一个`collate_fn`来处理变长的文本序列，将它们填充（padding）到同一批次中最长序列的长度。

```python
# 2_dataloader.py
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载之前处理好的数据
# vocab = torch.load('vocab.pth')
# train_data = torch.load('train_data.pth')
# test_data = torch.load('test_data.pth')

PAD_IDX = vocab['<pad>']

def collate_batch(batch):
    label_list, text_list, lengths = [], [], []
    for (_label, _text) in batch:
        label_list.append(_label)
        text_list.append(_text)
        lengths.append(len(_text))
  
    labels = torch.tensor(label_list, dtype=torch.float32)
    texts = pad_sequence(text_list, batch_first=True, padding_value=PAD_IDX)
    lengths = torch.tensor(lengths, dtype=torch.int64)
  
    return labels.to(device), texts.to(device), lengths.to(device)

BATCH_SIZE = 64
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_batch)
```

3. 构建模型

```python
# 3_model.py
import torch.nn as nn

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, 
                 bidirectional, dropout, pad_idx):
        super().__init__()
      
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, 
                            hidden_dim, 
                            num_layers=n_layers, 
                            bidirectional=bidirectional, 
                            dropout=dropout, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
      
    def forward(self, text, text_lengths):
        # text = [batch size, sent len]
        embedded = self.dropout(self.embedding(text))
        # embedded = [batch size, sent len, emb dim]
      
        # Pack sequence
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, text_lengths.to('cpu'), batch_first=True, enforce_sorted=False)
      
        packed_output, (hidden, cell) = self.lstm(packed_embedded)
      
        # Unpack sequence
        # output, output_lengths = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)

        # Concat the final forward (hidden[-2,:,:]) and backward (hidden[-1,:,:]) hidden layers
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1,:,:])
          
        # hidden = [batch size, hid dim * num directions]
        return self.fc(hidden)
```

4. 训练与评估

```python
# 4_train.py
import torch.optim as optim

# --- 模型超参数 ---
VOCAB_SIZE = len(vocab)
EMBEDDING_DIM = 100
HIDDEN_DIM = 256
OUTPUT_DIM = 1
N_LAYERS = 2
BIDIRECTIONAL = True
DROPOUT = 0.5

model = SentimentLSTM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, OUTPUT_DIM, 
                      N_LAYERS, BIDIRECTIONAL, DROPOUT, PAD_IDX)
model = model.to(device)

optimizer = optim.Adam(model.parameters())
criterion = nn.BCEWithLogitsLoss() # 适用于二分类
criterion = criterion.to(device)

def binary_accuracy(preds, y):
    """返回批次的准确率"""
    rounded_preds = torch.round(torch.sigmoid(preds))
    correct = (rounded_preds == y).float()
    acc = correct.sum() / len(correct)
    return acc

def train(model, iterator, optimizer, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.train()
  
    for labels, text, lengths in iterator:
        optimizer.zero_grad()
        predictions = model(text, lengths).squeeze(1)
        loss = criterion(predictions, labels)
        acc = binary_accuracy(predictions, labels)
        loss.backward()
        optimizer.step()
      
        epoch_loss += loss.item()
        epoch_acc += acc.item()
      
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def evaluate(model, iterator, criterion):
    epoch_loss = 0
    epoch_acc = 0
    model.eval()
  
    with torch.no_grad():
        for labels, text, lengths in iterator:
            predictions = model(text, lengths).squeeze(1)
            loss = criterion(predictions, labels)
            acc = binary_accuracy(predictions, labels)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
          
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

# --- 执行训练 ---
N_EPOCHS = 5
for epoch in range(N_EPOCHS):
    train_loss, train_acc = train(model, train_dataloader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_dataloader, criterion)
  
    print(f'Epoch: {epoch+1:02}')
    print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')
```

## 本章小结

在本章中，我们成功地推开了深度学习这扇厚重而又充满魅力的大门。

我们从最基本的神经元模型出发，理解了神经网络如何通过前向传播进行预测，以及如何通过反向传播和梯度下降这一核心机制来进行“学习”。这为我们后续理解一切复杂的深度学习模型奠定了坚实的理论基础。

接着，我们深入掌握了PyTorch框架。我们学会了使用`Tensor`进行GPU加速计算，领略了`Autograd`自动求导的魔力，并掌握了使用`nn.Module`来像搭积木一样构建我们自己的神经网络。

我们还回顾了CNN和RNN/LSTM这两种经典的神经网络结构，理解了它们分别在处理空间特征和序列数据上的独特设计思想，这对于我们拓宽视野、理解后续更高级架构的演进至关重要。

最后，我们探讨了“训练的艺术”，学习了如何选择合适的损失函数和优化器，以及如何使用正则化技术来防止模型过拟合。并通过一个完整的文本情感分类实战项目，将本章所有零散的知识点，凝聚成了一套行之有效的深度学习项目开发流程。

完成本章后，你已经不再是一个深度学习的门外汉。你具备了理解、构建和训练一个中等复杂度深度学习模型的能力。这块坚实的“基石”，将有力地支撑你向本书的下一个，也是最核心的目标迈进——深入理解并驾驭那颗“皇冠上的明珠”：Transformer架构与大语言模型。
