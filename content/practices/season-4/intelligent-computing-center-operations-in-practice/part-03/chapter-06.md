---
title: "第6章：大模型训练全流程运营"
date: 2025-11-29T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["智算中心运营实战：从基础设施到大模型全栈优化", "第6章：大模型训练全流程运营"]
slug: "chapter-06"
---

在前两篇中，我们已经成功地搭建了一套坚实的、云原生的AI基础设施。我们拥有了由顶尖GPU/NPU构成的算力池，铺设了RDMA高速网络，构建了高性能存储，并部署了能够智能调度AI任务的Volcano/Yunikorn平台。我们已经拥有了一座“硬件精良、软件智能”的现代化工厂。

现在，是时候让这条生产线真正运转起来，去生产它最终的产品——大型语言模型了。

本章，我们将正式从“平台建设者”的角色，部分切换到“生产运营者”的角色。我们将深入大模型训练的全流程，像一位经验丰富的主厨，不仅要了解食材（数据），更要精通各种烹饪技法（训练模式），并能娴熟地指挥一个庞大的后厨团队（分布式并行）。我们将首先从宏观上拆解大模型生命周期中最重要的三种训练模式：预训练、全量微调和高效微调（以LoRA为例），理解它们对资源的不同需求。接着，我们将从运维的独特视角，深入剖析支撑万亿模型训练成为可能的三大并行策略：数据并行、张量并行和流水线并行。最后，我们将理论与实践相结合，以一个极具挑战性和现实意义的案例——在华为昇腾集群上部署一个主流开源大模型的全量微调任务——来结束本章，为你提供一份可落地、可复现的操作指南。

## 6.1 训练模式拆解：Pre-training vs SFT vs LoRA

“训练一个大模型”是一个非常笼统的说法。在实际的LLMOps流程中，“训练”根据其目的和资源消耗，至少可以分为三种截然不同的模式。作为AI Infra工程师，理解它们的区别至关重要，因为这直接决定了你的资源规划、调度策略和成本核算。

### 6.1.1 Pre-training（预训练）：铸造模型的“灵魂”

预训练，是从零开始、在一个超大规模、高质量的通用文本数据集（通常是TB甚至PB级别）上，训练一个基础模型（Foundation Model）的过程。

目标： 让模型学习人类语言的通用规律、语法结构、事实知识和一定的推理能力。它不是为了完成某个特定任务，而是为了构建一个具备广泛“通识”的“胚胎”。

打个比方： 预训练就像是一个人从出生到大学毕业的通识教育阶段。他阅读了图书馆里成千上万本书（海量数据），学习了语言、历史、科学等各种知识，但还没有进入任何具体的专业领域。

对基础设施的诉求（“三高一长”）：

1. 高算力（Massive Compute）： 这是最耗费算力的阶段。训练一个千亿参数级别的模型，通常需要数千张顶级训练卡（如H800或910B）。
2. 高带宽（High Bandwidth Network）： 如此大规模的集群，必须采用InfiniBand或高性能RoCE网络，并结合NCCL/HCCL进行高效的集合通信，否则通信开销将淹没计算收益，线性加速比会非常差。
3. 高吞吐（High Throughput Storage）： 需要一个高性能的并行文件系统（如Lustre/GPFS），能够持续、高吞吐地为数千个计算节点“喂”训练数据。
4. 长周期（Long Duration）： 预训练过程极其漫长，通常持续数周甚至数月。这对整个集群的稳定性和容错能力提出了极致的要求。任何一次长时间的宕机，都可能造成数百万美元的损失。因此，健壮的Checkpoint机制、硬件的健康巡检和故障自愈能力是必备的。

运维视角下的关注点：

- MFU（模型算力利用率）和线性加速比： 这是预训练阶段的“黄金指标”。运营团队的核心KPI就是想尽一切办法提升这两个指标，哪怕只提升5%，都意味着节约了数天的时间和巨额的成本。
- 故障发现与恢复时间（MTTR）： 由于周期长，硬件故障（如GPU掉卡、光模块损坏）是必然会发生的。快速发现问题、定位问题、替换硬件并从最近的Checkpoint恢复训练，是运维团队最重要的日常工作。
- 成本控制： 在规划阶段，就需要使用我们在下一章将详细介绍的算力估算模型，来精确计算所需的资源和时间，制定预算。

### 6.1.2 SFT（Supervised Fine-Tuning，监督微调）：教会模型“说人话”

预训练出的基础模型虽然知识渊博，但它只会“续写”，不会“对话”。它不知道如何遵循指令、如何扮演一个有用的AI助手。SFT就是解决这个问题的过程。

目标： 使用一个规模小得多、但质量极高的“指令-回答”对（Instruction-Response Pairs）数据集，来教会模型如何理解并遵循人类的指令。

打个比方： SFT就像是大学毕业生（基础模型）进入公司后，接受的岗前培训。通过学习大量的“提问-标准答案”案例（指令数据集），他学会了如何以一个合格“员工”的身份与人沟通和工作。

对基础设施的诉求（“中等规模，高频迭代”）：

1. 中等算力： SFT的数据集通常只有几万到几十万条，远小于预训练。所需的算力规模也相应降低，通常在几十到上百张卡的规模。
2. 网络和存储要求降低： 由于集群规模减小，对网络和存储的极致性能要求有所放宽，但仍然需要良好的RDMA网络和稳定的存储。
3. 高频迭代： 算法工程师会频繁地尝试不同的数据集、超参数和模型版本。SFT任务的周期通常是几小时到几天。
4. 全量微调（Full-parameter Fine-tuning）： 在SFT阶段，默认是指对模型的所有参数（例如700亿个参数）都进行梯度更新。这意味着，它对单卡的显存（Memory）要求与预训练阶段是完全相同的！

运维视角下的关注点：

- 作业调度与资源周转率： 运维团队的核心挑战，是如何高效地调度这些高频、中等规模的SFT任务，保证算法团队的实验能够快速启动和完成，最大化GPU的周转率。队列管理、优先级设置、公平调度变得尤为重要。
- 显存占用： 这是SFT（全量微调）最容易出问题的点。一个70B模型，即使只用少量数据微调，其模型权重、梯度、优化器状态所占用的显存，与用海量数据预训练时是一模一样的。因此，精确计算显存需求、选择合适的并行策略至关重要。我们将在下一章详细讲解计算方法。
- 实验管理： 需要与MLOps平台（如MLflow, WandB）集成，帮助算法工程师追踪每次实验的配置、数据、代码和结果。

### 6.1.3 PEFT（Parameter-Efficient Fine-Tuning），以LoRA为例：模型的“微创手术”

全量微调（SFT）虽然有效，但成本依然高昂。每次都更新数百亿个参数，不仅需要大量高端GPU，而且为每个下游任务都保存一个完整的模型副本，也是巨大的存储浪费。PEFT，特别是LoRA的出现，革命性地解决了这个问题。

LoRA（Low-Rank Adaptation）核心思想：

- 在训练过程中，冻结（Freeze）住预训练模型的全部原始权重，使其不可变。
- 在模型的某些层（通常是Transformer的Attention层）的旁边，注入两个小小的、可训练的“旁路”矩阵（称为A和B）。这两个矩阵的“秩（Rank）”很低，意味着它们的参数量非常非常小。
- 在微调时，只训练这两个小矩阵的参数。
- 在推理时，可以将训练好的小矩阵A和B相乘，得到的结果与原始权重矩阵相加，从而在不改变原始模型的情况下，实现对模型行为的适配。

目标： 以极小的代价，实现与全量微调相近的效果，让模型适配特定的下游任务或领域风格。

打个比方： LoRA就像是给一位经验丰富的全科医生（基础模型）做一个“微创手术”或发一个“专业领域指南附录”。医生本身的能力没有变，但通过这个小小的调整，他就能很好地处理某个特定领域的病例（如儿科或心脏科）。

对基础设施的诉求（“低门槛，大规模个性化”）：

1. 极低算力： LoRA需要训练的参数量可能只有原始模型的0.01%。这意味着，原本需要8张A100才能微调的70B模型，现在可能单张消费级显卡（如RTX 4090）就能完成。
2. 显存需求急剧下降： 由于不需要存储庞大的优化器状态（只为LoRA参数存储），对显存的需求大大降低。
3. 快速训练： LoRA训练通常只需要几十分钟到几小时。
4. 部署灵活： 最终产出的LoRA权重文件只有几十MB，而不是几百GB。部署时，可以一个基础模型+多个LoRA“插件”，按需加载，实现大规模个性化服务。

运维视角下的关注点：

- 推理服务优化： LoRA的出现，使得在推理侧动态加载和切换模型适配器成为可能。运维团队需要关注推理服务器（如vLLM, SGLang）对LoRA的集成和优化，例如如何高效地进行LoRA权重的融合（Merging）和批处理（Batching）。
- 资源碎片化管理： LoRA任务对资源的需求小而零散，可能会产生大量GPU资源的“边角料”。如何通过MIG（NVIDIA）或类似技术，将一张物理GPU切分给多个LoRA训练或推理任务，是提升利用率的新挑战。
- 版本管理： 管理成千上万个微小的LoRA权重文件，以及它们与基础模型的对应关系，对MLOps的版本控制和资产管理提出了新的要求。

总结与对比：

| 特性 | Pre-training (预训练) | SFT (全量微调) | LoRA (高效微调) |
| :--- | :--- | :--- | :--- |
| 目标 | 构建通用知识 | 教会模型遵循指令 | 适配特定任务/风格 |
| 打个比方 | 通识教育 (0->1) | 岗前培训 (1->1.1) | 微创手术/专业附录 |
| 算力规模 | 巨大 (数千卡) | 中等 (数十到上百卡) | 极小 (单卡/少数卡) |
| 训练周期 | 长 (数周到数月) | 中 (数小时到数天) | 短 (数分钟到数小时) |
| 显存需求 | 巨大 | 巨大 (与预训练相同) | 小 |
| 产物大小 | 巨大 (数百GB) | 巨大 (数百GB) | 小 (数十MB) |
| 运维核心 | 提升MFU/线性度，保障稳定性 | 提升作业周转率，管理显存 | 优化推理服务，管理碎片资源 |

作为AI Infra工程师，你需要能根据算法团队提交的任务类型，快速判断其资源画像，并为其匹配最合适的集群、队列和调度策略。

## 6.2 分布式并行策略：数据并行、张量并行、流水线并行的运维视角

当一个模型大到无法装入单张GPU显存，或者训练数据多到单卡无法在合理时间内完成时，分布式训练就成了唯一的选择。目前，业界主要采用三种并行策略，或将它们组合使用（所谓的2D、3D并行），来训练万亿级别的大模型。

从运维和资源管理的视角，理解这三种并行策略的原理、通信模式和资源需求，是进行性能优化和故障排查的基础。

### 6.2.1 数据并行 (Data Parallelism, DP)：最简单、最常用的“人海战术”

核心思想：

1. 模型复制： 将完整的模型复制到每一张参与训练的GPU上。
2. 数据切分： 将一个大的训练数据集（Batch）切分成N份（N为GPU数量），每个GPU获得一份数据子集（mini-batch）。
3. 独立计算： 每个GPU独立地进行前向传播和反向传播，计算出各自的梯度。
4. 梯度同步： 这是DP的关键和瓶颈所在。 所有GPU需要通过一次All-Reduce集合通信操作，将各自的梯度进行聚合（通常是求平均）。
5. 同步更新： 所有GPU使用聚合后的全局梯度，以完全相同的方式更新自己的模型副本，确保下一次迭代开始时所有模型保持一致。

打个比方：

DP就像是让一个班的学生（GPU）做同一套练习题（模型）。老师把练习题分成N份（数据切分），每个学生做一部分。做完后，大家把答案（梯度）汇总到黑板上对一下，算出标准答案（All-Reduce），然后每个人都根据标准答案修改自己的卷子（模型更新）。

运维视角下的关键点：

- 通信开销： DP的通信开销与模型参数量成正比。模型越大，每次All-Reduce需要传输的梯度数据就越多。因此，DP对网络的带宽非常敏感。
- 显存瓶颈： DP不能解决单卡显存不足的问题。因为每个GPU都需要装下完整的模型、梯度和优化器状态。如果一个模型本身就放不进单卡，单纯使用DP是无效的。
- 适用场景： 模型可以被单卡装下，但希望通过增加GPU数量来加速训练（缩短时间）。这是最常用、最基础的并行方式。PyTorch的`DistributedDataParallel` (DDP) 就是其标准实现。
- 优化：ZeRO (Zero Redundancy Optimizer)
  - DeepSpeed开发的ZeRO技术，是对DP的重大优化。它通过将模型参数、梯度和优化器状态这三样最耗显存的东西，切分并分布存储在所有GPU上，极大地降低了单卡的显存峰值。
  - 例如，在ZeRO-3阶段，每个GPU只保存模型参数的一部分，在计算时通过All-Gather动态地从其他GPU获取需要的部分。ZeRO使得在显存有限的卡上训练超大模型成为可能，但代价是增加了更多的通信量。

### 6.2.2 张量并行 (Tensor Parallelism, TP)：模型“大卸八块”

核心思想：

- 当一个模型的权重矩阵（Tensor）本身太大，无法装入单卡显存时，TP就派上了用场。它将模型内部的单个大矩阵运算，切分到多个GPU上协同完成。
- 以一个标准的Transformer层为例，其核心是自注意力（Self-Attention）和前馈网络（FFN），这两部分都涉及大量的矩阵乘法。
- TP将这些大矩阵（如Q, K, V的投影矩阵）按行或按列切开，分布到不同的GPU上。
- 每个GPU只持有和计算矩阵的一部分。在计算过程中，它们需要通过All-Gather或Reduce-Scatter等集合通信操作来交换中间结果，以拼凑出最终的完整输出。

打个比方：

TP就像是让多个学生（GPU）协同计算一个超大的矩阵乘法。一个学生负责计算结果矩阵的左半部分，另一个学生负责右半部分。在计算过程中，他们需要互相交换一些中间行或列的数据。

运维视角下的关键点：

- 解决单卡显存瓶颈： 这是TP的核心价值。如果一个模型参数量为P，使用N路TP，则每个GPU上大约只需要存储P/N的模型参数，显存需求降低为1/N。
- 通信模式： TP的通信发生在每次前向和反向传播的内部，非常频繁。它对网络的延迟（Latency）极为敏感。因此，TP通常被限制在节点内部，利用速度最快的NVLink/HCCS进行通信。很少跨节点做TP，因为跨节点的网络延迟会严重拖慢计算。
- 实现复杂： TP需要对模型代码进行深度修改，不是所有模型都能轻易地进行张量并行。Megatron-LM是实现TP的代表性框架。

### 6.2.3 流水线并行 (Pipeline Parallelism, PP)：模型的“工厂流水线”

核心思想：

- 将模型的不同层（Layers）分配到不同的GPU上。
- 例如，一个48层的模型，使用4路PP。GPU 0负责计算第1-12层，GPU 1负责13-24层，以此类推。
- 数据（mini-batch）像在工厂流水线上一样，依次流过每个GPU。GPU 0计算完前12层后，将输出（称为激活值，Activation）发送给GPU 1。GPU 1在此基础上继续计算，再传给GPU 2。
- 流水线气泡（Pipeline Bubble）： 朴素的PP效率很低。因为在流水线刚启动和快结束时，大部分GPU都在空闲等待。例如，当GPU 0在处理第一个mini-batch时，所有其他GPU都在闲置。这个闲置的时间就是“流水线气泡”。

优化：GPipe / PipeDream

- 为了减小气泡，现代PP实现（如GPipe）会将一个大的Batch切分成多个更小的微批次（micro-batches）。
- GPU 0处理完第一个micro-batch后，立刻传给GPU 1，然后马上开始处理第二个micro-batch。
- 这样，经过短暂的“预热”后，所有GPU都可以同时处理不同的micro-batch，流水线被“填满”了，并行效率大大提高。

打个比方：

PP就像一条汽车装配线。GPU 0负责装底盘，GPU 1负责装发动机，GPU 2负责装车身，GPU 3负责喷漆。通过将一个大订单（Batch）拆分成多辆车（micro-batches）连续不断地送上产线，保证了所有工位（GPU）都有活干。

运维视角下的关键点：

- 解决显存瓶颈： 与TP类似，PP也将模型切分，降低了单卡显存。如果一个模型有L层，使用N路PP，则每个GPU大约只需要存储L/N的层，显存需求也大致降低为1/N。
- 通信模式： PP的通信是点对点（Point-to-Point）的，只发生在流水线上相邻的两个GPU之间。通信的数据量是每个micro-batch的激活值大小。
- 负载均衡是关键： PP的效率高度依赖于负载均衡。如果分给每个GPU的计算量（层数、层的复杂度）不均匀，快的GPU就要等待慢的GPU，产生新的气泡。如何合理地切分模型，是一个需要profiling和调优的难题。
- 显存墙： PP会增加额外的显存开销，因为每个GPU需要缓存多个micro-batch的激活值，以便在反向传播时使用。

### 6.2.4 3D并行：终极组合拳

在训练万亿参数级别的超大模型（如GPT-4, PaLM）时，单一的并行策略往往不够用。业界采用了将三者结合起来的“3D并行”策略：

- 数据并行 (DP)： 在最外层，将整个训练任务复制多份，以处理更多的数据，加速收敛。
- 流水线并行 (PP)： 在中层，将巨大的模型纵向切分，分配到不同的计算节点上。
- 张量并行 (TP)： 在最内层，在每个节点内部，将单层的计算横向切分到该节点的所有GPU上。

一个典型的3D并行配置示例：

- 总资源： 1024张GPU，分布在128个节点上（每节点8卡）。
- TP路数： 8路（在每个节点内部使用NVLink进行张量并行）。
- PP路数： 16路（将模型切成16段，分布在16个不同的节点组上）。
- DP路数： 8路（总共有8个这样的128卡（16x8）的完整模型副本，在进行数据并行）。
- 总路数： `8 * 16 * 8 = 1024`。

作为AI Infra工程师，当你面对这样一个复杂的3D并行任务时，你需要能够清晰地分析出其通信模式，并对基础设施进行针对性优化：

- TP的通信依赖节点内的NVLink带宽。
- PP的通信依赖节点间的RDMA网络带宽和延迟。
- DP的通信则消耗大量的跨节点RDMA网络带宽。

任何一个环节出现瓶颈，都会影响整个“3D木桶”的最终性能。

## 6.3 实战案例：在昇腾集群上部署Llama 3 / Qwen 全量微调任务

理论终须落地。本节，我们将挑战一个在当前国产化替代背景下极具现实意义的场景：在一个基于华为昇腾910B的K8s集群中，使用Volcano进行调度，对一个主流的开源大模型（如Llama 3-8B或Qwen-14B）进行全量监督微调（SFT）。

这个案例将贯穿本篇甚至全书的知识点：昇腾硬件、CANN软件栈、K8s Device Plugin、Volcano调度器、分布式并行策略等。

### 6.3.1 准备工作：环境与软件栈

硬件环境：

- 假设我们有一个K8s集群，其中包含若干个计算节点，每台节点配备8张昇腾910B NPU。
- 节点间通过100G RoCE v2网络互联。
- 集群已部署了高性能并行文件系统（如Lustre），并挂载到所有节点的`/mnt/data`。

软件环境（K8s侧）：
  
- K8s集群已安装并配置好Ascend Device Plugin，使得K8s能够识别`huawei.com/npu`资源。
- K8s集群已安装并配置好Volcano调度器。
- 集群的镜像仓库中已准备好基础镜像。

基础镜像构建 (`Dockerfile`)

我们需要一个包含了PyTorch for Ascend、HCCL、Transformers库和DeepSpeed的基础镜像。

```dockerfile
# 使用华为官方提供的PyTorch for Ascend基础镜像
FROM ascendhub.huawei.com/public-ascendhub/pytorch-ascend:2.1.0-cann7.0.1-py39-ubuntu20.04-x86_64

# 安装必要的工具和依赖库
RUN apt-get update && apt-get install -y git vim wget

# 升级pip并安装核心Python包
RUN pip install --upgrade pip
# DeepSpeed需要适配昇腾，通常需要从特定源码分支编译或安装华为提供的版本
# 这里以一个示例说明，实际请参考华为官方文档
RUN pip install deepspeed==0.1x.x-ascend
RUN pip install transformers==4.3x.x
RUN pip install accelerate

# 清理缓存
RUN rm -rf /root/.cache/pip

WORKDIR /workspace
```

注意： DeepSpeed、Transformers等库对昇腾的适配在持续演进中，请务必参考华为官方文档或其CodeLab/Gitee社区，获取与你的CANN版本兼容的正确安装方式。

数据和模型准备：

- 模型权重： 从Hugging Face或国内镜像源下载Llama 3-8B或Qwen-14B的模型权重，并将其存放在共享文件系统上，如`/mnt/data/models/Llama-3-8B`。
- 微调数据： 准备一个SFT数据集，格式通常是JSON lines，每行包含一个指令和期望的输出。例如Alpaca数据集。将其存放在`/mnt/data/datasets/alpaca_data_zh.json`。

### 6.3.2 训练脚本的适配与编写

我们将使用DeepSpeed框架来简化分布式训练的配置。DeepSpeed可以很好地与Transformers的`Trainer` API结合。

DeepSpeed配置文件 (`ds_config.json`)

这是最关键的配置文件之一，它告诉DeepSpeed使用何种并行策略和优化。对于全量微调，我们将使用ZeRO-2来优化显存。

```json
{
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 2,
    "optimizer": {
    "type": "AdamW",
    "params": {
        "lr": 2e-5,
        "betas": [0.9, 0.999],
        "eps": 1e-8
    }
    },
    "fp16": {
    "enabled": true, // 开启混合精度训练
    "loss_scale_window": 1000
    },
    "zero_optimization": {
    "stage": 2, // 使用ZeRO-2
    "allgather_partitions": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
    }
}
```

训练启动脚本 (`run_sft.sh`)

这个脚本负责调用`deepspeed`命令，并传入所有必要的参数。

```bash
#!/bin/bash

# 从Volcano环境变量中获取分布式信息
# Volcano会自动注入这些，HCCL会读取它们
export MASTER_ADDR=${VC_WORKER_0_HOST}
export MASTER_PORT=${VC_WORKER_0_PORT}
export RANK=${VC_TASK_INDEX}
export WORLD_SIZE=${VC_WORKER_NUM}

# DeepSpeed启动命令
deepspeed --num_gpus=8 train_script.py \
    --deepspeed ds_config.json \
    --model_name_or_path /mnt/data/models/Llama-3-8B \
    --data_path /mnt/data/datasets/alpaca_data_zh.json \
    --output_dir /mnt/data/outputs/llama3-8b-sft-v1 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --save_strategy "steps" \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --fp16 True
```

关键点： `train_script.py`是一个标准的、使用Transformers `Trainer` API编写的训练脚本。DeepSpeed的魔力在于，你几乎不需要修改这个Python脚本，只需要通过命令行和`ds_config.json`来注入分布式能力。

### 6.3.3 提交到K8s：VolcanoJob YAML

最后，我们将所有东西打包到一个`VolcanoJob`中，提交给K8s集群。假设我们需要2台节点，共16张NPU。

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: llama3-8b-sft-job
spec:
  schedulerName: volcano
  minAvailable: 2 # 2个Pod，每个8卡，总共16卡
  queue: high-priority-gpu
  tasks:
    - name: worker
      replicas: 2
      template:
        spec:
          containers:
            - name: trainer
              image: my-registry/my-ascend-app:1.0
              command: ["/bin/bash", "-c", "./run_sft.sh"]
              resources:
                limits:
                  huawei.com/npu: 8 # 每个Pod申请8张NPU
              volumeMounts:
                - name: data-storage
                  mountPath: /mnt/data
          restartPolicy: OnFailure
          volumes:
            - name: data-storage
              persistentVolumeClaim:
                claimName: my-pvc-for-lustre # 挂载共享存储
```

工作流程解析：

1. 用户`kubectl apply -f job.yaml`。
2. Volcano调度器收到这个包含2个Pod的Job，`minAvailable`为2。
3. 调度器在`high-priority-gpu`队列中检查资源，发现有2个节点各自拥有8张空闲的NPU。
4. 预检通过，Volcano开始调度。它将`pod-0`调度到`node-A`，`pod-1`调度到`node-B`。
5. Kubelet在`node-A`上创建`pod-0`。Ascend Device Plugin被调用，将8张NPU的环境变量和设备注入容器。
6. `pod-0`的容器启动，执行`run_sft.sh`。脚本从Volcano注入的环境变量中读取到`MASTER_ADDR`（即`pod-0`的IP）等信息。
7. `pod-1`也以同样的方式在`node-B`上启动。
8. 两个Pod中的HCCL库根据环境变量，成功建立通信，一个16卡的分布式训练任务就这样运行起来了！训练产生的Checkpoint会保存在`/mnt/data/outputs`下。

### 6.3.4 运维与排查

Pod Pending： 如果Job长时间处于Pending，使用`kubectl describe podgroup <job-name>`查看Volcano的事件，它会告诉你为什么调度失败（例如，资源不足）。

训练卡住： 如果训练日志长时间没有输出，很可能是HCCL通信出了问题。检查节点间的RoCE网络配置，查看`pod-0`的网络是否能ping通`pod-1`。

OOM（Out of Memory）： 如果日志报显存不足，尝试减小`ds_config.json`中的`train_micro_batch_size_per_gpu`，或者从ZeRO-2升级到ZeRO-3（但这会增加通信量）。

这个案例完整地展示了在一个真实的、国产化的AI基础设施上，如何端到端地运营一个复杂的大模型训练任务。它不仅是对你技术能力的综合考验，更是你作为一名顶级AI Infra工程师价值的最终体现。
