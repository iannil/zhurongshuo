---
title: "第2章：双核心生态：NVIDIA与华为昇腾"
date: 2025-11-29T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["智算中心运营实战：从基础设施到大模型全栈优化", "第2章：双核心生态：NVIDIA与华为昇腾"]
slug: "chapter-02"
---

在第一章中，我们明确了智能算力（以GPU/NPU为代表）与通用算力（以CPU为代表）的本质区别。现在，我们将把显微镜对准智能算力的心脏——AI加速芯片。在这个领域，呈现出“一超多强”的格局。NVIDIA凭借其先发优势和强大的CUDA生态，构建了事实上的行业标准，是那个“一超”。与此同时，以华为昇腾为代表的国产力量正在迅速崛起，因其在自主可控和供应链安全方面的战略价值，以及在特定场景下展现出的优异能效比，成为了智算中心建设中不可或缺的“另一极”。

对于一名智算中心的运营专家而言，仅仅会使用`nvidia-smi`是远远不够的。你必须像一位经验丰富的军械师，不仅能识别不同武器的型号，更能洞悉其内部构造、火力特点和后勤需求。本章将带你深入解密NVIDIA和华为昇腾这两大核心生态的硬件架构与软件栈，并探讨如何在复杂的现实世界中对它们进行统一管理和调度。这不仅是技术选型的必修课，更是实现国产化适配战略的基石。

## 2.1 NVIDIA生态解密：Ampere/Hopper架构、Tensor Core与NVLink原理

谈论AI算力，NVIDIA是绕不开的丰碑。从2006年发布CUDA平台至今，NVIDIA通过十几年的耕耘，成功地将GPU从一个游戏显卡配件，打造成了驱动第四次工业革命的引擎。其成功秘诀在于“硬件迭代”和“软件生态”两条腿走路，构建了深不可测的护城河。

### 2.1.1 架构进化：从Ampere到Hopper

NVIDIA的数据中心GPU架构以物理学家的名字命名，每一代都带来了性能的巨大飞跃。我们将聚焦于当前大模型时代最重要的两个架构：Ampere（安培）和Hopper（赫柏）。

#### Ampere架构 (代表产品: A100, A800)

Ampere架构于2020年发布，其旗舰产品A100是引爆大模型军备竞赛的“第一把火”。A100的成功，源于它针对AI训练负载的几项革命性设计：

1. 第三代Tensor Core： 这是Ampere架构的核心。相比前代，它首次引入了对TF32（TensorFloat-32）数据格式的支持。TF32拥有与FP32（32位单精度）相同的动态范围（8位指数），但只有FP16（16位半精度）的精度（10位尾数）。这意味着，开发者几乎无需修改代码，就能在保持接近FP32稳定性的同时，获得接近FP16的吞吐量（理论上是FP32的8倍）。这极大地降低了混合精度训练的门槛。
2. Multi-Instance GPU (MIG)： A100首次支持将一块物理GPU硬件分区为最多7个独立的GPU实例（Instance）。每个实例都有自己专属的计算、内存和带宽资源，实现了真正的资源隔离。这对于AI推理场景尤其重要，可以在一张卡上安全、高效地部署多个小模型，极大提升了GPU的利用率。
3. 第三代NVLink与NVSwitch： A100 GPU拥有12个NVLink通道，提供了高达600 GB/s的双向带宽。通过NVSwitch，可以在一个8-GPU节点（如DGX A100）内部实现任意GPU对之间的全连接通信，为大规模分布式训练奠定了基础。
4. A800的诞生： 值得注意的是，A800是A100的“合规版”。其主要变化在于将NVLink的互联带宽从600 GB/s降低到了400 GB/s，其他计算和内存规格与A100基本保持一致。

#### Hopper架构 (代表产品: H100, H800)

Hopper架构于2022年发布，专为Transformer模型和万亿参数级别的大模型而设计，堪称“核武器”级别的存在。

1. 第四代Tensor Core与Transformer引擎： Hopper架构的核心是其全新的Transformer引擎。它结合了新一代的Tensor Core，能够智能地分析Transformer模型的每一层，并自动决定在FP8和FP16精度之间进行切换。FP8（8位浮点数）的引入，使得算力再次翻倍。在执行Transformer模型时，Hopper的算力吞吐量是Ampere的数倍之多。
2. 第四代NVLink与NVSwitch： H100将单卡的NVLink带宽提升至900 GB/s。更重要的是，它将NVSwitch芯片直接集成到了GPU die上，使得构建超大规模集群（如高达256-GPU的NVLink网络）变得更为高效，极大地降低了跨节点通信的延迟。
3. DPX指令集： Hopper引入了一套新的指令集，用于加速动态规划算法，这在基因测序、路径优化等领域有重要应用，进一步拓宽了GPU的应用边界。
4. H800的由来： 与A800类似，H800是H100的“合规版”，其芯片间的互联带宽受到了限制，以符合出口管制要求。但在单卡计算能力上，它依然保持着Hopper架构的强大威力。

### 2.1.2 Tensor Core：矩阵运算的“专用流水线”

如果说GPU是“小学生军团”，那么Tensor Core就是军团里的“尖子班”，专门负责处理矩阵乘加（MMA, Matrix Multiply-Accumulate）运算，即 `D = A * B + C`。

工作原理： 一个标准的ALU（算术逻辑单元）一次只能执行一次乘法或加法。而一个Tensor Core，可以在一个时钟周期内，完成一个小型矩阵（如4x4）的乘加运算。它本质上是一个高度优化的、并行的硬件电路，将整个矩阵运算流程固化下来，像一条专用流水线，效率极高。

数据精度与性能： Tensor Core的性能与其支持的数据精度密切相关。

- FP16/BF16 (半精度): 这是目前大模型训练的主流。FP16能显著减少显存占用和通信带宽，并利用Tensor Core进行加速。BF16（脑浮点数）则提供了与FP32相同的动态范围，在训练稳定性上更具优势。
- TF32 (TensorFloat-32): Ampere架构的创举，在易用性和性能之间取得了完美平衡。
- FP8 (八位浮点数): Hopper架构的王牌，将性能推向新高峰，尤其适合推理和训练的某些阶段。
- INT8 (八位整型): 主要用于推理加速。通过量化技术将FP32模型转换为INT8模型，可以获得数倍的性能提升和显著的功耗降低。

对于AI Infra工程师来说，理解并推动算法团队使用正确的数据精度（如开启混合精度训练），是榨干GPU性能、提升MFU的关键一步。

### 2.1.3 NVLink与NVSwitch：构建算力集群的“高速公路网”

在单卡性能逼近物理极限的今天，AI的竞争已演变为集群规模的竞争。如何将成千上万张GPU高效地连接起来，成为了核心挑战。NVIDIA的答案是NVLink和NVSwitch。

PCIe的瓶颈： 传统的GPU间通信依赖于主板上的PCIe总线。但PCIe是一种共享总线，带宽有限（即使是PCIe 5.0 x16也只有128 GB/s双向带宽），且需要CPU中转，延迟较高。当8张甚至更多GPU同时进行梯度交换时，PCIe会迅速成为瓶颈。

#### NVLink：点对点的“私家公路”

NVLink是一种专为GPU间通信设计的高速互联总线，它提供了远高于PCIe的带宽和更低的延迟。它在物理上是GPU之间直接的点对点连接，数据传输无需绕道CPU，就像在两栋大楼间修建了专属的封闭天桥。

#### NVSwitch：GPU间的“高速立交桥”

当GPU数量超过2个时，点对点连接会变得异常复杂。NVSwitch芯片应运而生，它扮演了一个高速交换机的角色。在一个标准的8-GPU服务器（如HGX平台）中，多颗NVSwitch芯片共同构建了一个全连接（Full-Mesh）的内部网络。这意味着，服务器内的任意一张GPU卡都可以用满速的NVLink带宽与其他7张卡通信，这对于All-Reduce等集合通信操作至关重要。

### 2.1.4 CUDA：无法逾越的软件“护城河”

如果说强大的硬件是NVIDIA的利刃，那么CUDA生态就是其坚不可摧的盾牌。

CUDA (Compute Unified Device Architecture)： 它不仅仅是一种编程语言，而是一个完整的并行计算平台和编程模型。它包含：

- 驱动层： 直接与硬件交互。
- API层： 提供C/C++、Fortran等语言的接口，让开发者可以编写能在GPU上运行的程序（Kernel）。
- 运行时库： 管理GPU设备、内存和任务执行。

生态系统： 围绕CUDA，NVIDIA构建了一个庞大的、几乎覆盖所有应用场景的软件库矩阵：

- cuDNN: 深度神经网络库，提供了高度优化的卷积、池化、激活函数等基础算子。
- NCCL (NVIDIA Collective Communications Library): 集合通信库，实现了针对NVLink和InfiniBand优化的All-Reduce、Broadcast等分布式训练核心操作。
- TensorRT: 高性能推理引擎，能将训练好的模型进行优化、量化和编译，以在生产环境中实现最低延迟和最高吞吐。
- Triton Inference Server: 推理服务化框架，支持多种模型格式和硬件后端，简化了模型部署。
- Nsight: 全套的性能分析和调试工具，帮助开发者定位性能瓶颈。

这个生态的成熟度和易用性，使得绝大多数AI框架（TensorFlow, PyTorch, JAX）和上层应用都优先基于CUDA开发。对于开发者而言，“换卡”不仅仅是更换硬件，更意味着可能要放弃整个熟悉的工具链和数十年积累的优化经验，迁移成本极高。

## 2.2 国产之光（重点）：华为昇腾910B架构、DaVinci核心与CANN软件栈详解

在自主可控的国家战略和日益复杂的国际贸易环境下，构建不依赖于外部供应的国产算力体系，已成为中国科技发展的重中之重。在这一浪潮中，华为昇腾（Ascend）系列AI处理器，凭借其自研的达芬奇（DaVinci）架构和完整的软硬件生态，正迅速成为智算中心建设的“主力机型”之一。

### 2.2.1 达芬奇（DaVinci）架构：为AI而生的“立体”计算核心

与GPU源于图形渲染的“通用”背景不同，华为的达芬奇架构从一开始就是为AI计算量身定制的，其设计哲学是“极致的能效比和对AI算子的深度优化”。其核心是AI Core，一个包含了不同计算单元的异构组合。

我们可以把AI Core想象成一个高度专业化的“AI加工车间”，内部有三个协同工作的工种：

1. 3D Cube（矩阵计算单元）： 这是达芬奇架构的灵魂，也是昇腾算力的主要来源。它是一个专门用于执行大规模矩阵乘法（M*N*K）的硬件加速单元。之所以称为“3D”，是因为它可以被看作一个`16x16x16`的立方体计算阵列，在一个时钟周期内可以完成`16x16`的矩阵与`16x16`的矩阵相乘（累加），总计`16*16*16=4096`次乘加运算。这种设计对于Transformer、CNN等以矩阵运算为核心的模型极为高效。它就是车间里那台最先进、最高效的全自动“矩阵加工机床”。
2. Vector Unit（向量计算单元）： 负责处理非矩阵的、元素级的计算，如向量的加减乘除、激活函数（ReLU, Sigmoid等）的计算。它就像是车间里的“多功能操作台”，处理各种零散但必要的加工任务。
3. Scalar Unit（标量计算单元）： 扮演着“车间主任”的角色，负责指令的解码、任务的调度和程序的流程控制，是一个微型的CPU核心。

这种“3D Cube + Vector + Scalar”的异构设计，使得AI Core可以像流水线一样高效处理AI任务：Scalar Unit负责指挥，3D Cube负责最耗时的大规模矩阵运算，Vector Unit负责收尾和辅助计算。

### 2.2.2 昇腾910B：对标主流的旗舰训练芯片

昇腾910B是华为当前主力的AI训练芯片，是昇腾910A的升级版，在性能、互联和软件生态上都进行了大幅优化，直接对标NVIDIA的A100/A800。

核心规格：

- 半精度算力 (FP16): 典型值为320 TFLOPS。（作为对比，A100为312 TFLOPS）
- 整型算力 (INT8): 典型值为640 TOPS。（作为对比，A100为624 TOPS）
- 高带宽内存 (HBM): 配备 64 GB HBM2e 内存，带宽超过 2.2 TB/s。
- 片上互联： 集成了华为自研的HCCS（Huawei-designed Collective Communication System）技术，支持多卡间的高速互联。
- 板间互联： 支持100G RoCE v2网络，用于构建大规模集群。

Atlas 900训练集群： 基于昇腾910B，华为构建了Atlas 900训练集群。该集群通过高速RoCE网络将成百上千个910B节点连接起来，并结合华为的集合通信库HCCL，能够提供强大的总体算力和良好的线性加速比，已经成功支撑了多个国产大模型的训练。

### 2.2.3 CANN（异构计算架构）：昇腾的“CUDA”

如果说910B是昇腾的“躯体”，那么CANN（Compute Architecture for Neural Networks）就是其“灵魂”。CANN是华为打造的、对标NVIDIA CUDA的AI异构计算架构，是上层深度学习框架与底层昇腾硬件之间的桥梁。理解CANN的层次结构，是排查昇腾平台性能问题和进行深度优化的关键。

CANN的架构自下而上可以分为几层：

1. 芯片使能层（Driver）： 最底层，负责驱动昇腾芯片，管理硬件资源。相当于`npu-smi`工具所交互的层面。
2. 计算加速库（Acceleration Library）：
   1. HCCL (Huawei Collective Communication Library): 对标NVIDIA的NCCL，提供了针对昇腾硬件和网络优化的集合通信原语（AllReduce, AllGather等），是进行分布式训练的核心库。
   2. CBLAS (CANN Basic Linear Algebra Subprograms): 对标cuBLAS，提供了高性能的矩阵和向量计算函数。
3. AscendCL (Ascend Computing Language): 这是CANN的核心接口层，对标CUDA API。它为开发者提供了C/C++语言的API，用于管理设备（`aclrtSetDevice`）、管理内存（`aclrtMalloc`）、管理Stream（`aclrtCreateStream`）以及同步启动AI任务。熟悉CUDA编程的开发者会发现其设计思想有很多相似之处。
4. 图引擎（Graph Engine, GE）： 这是CANN最智能的部分。当PyTorch、MindSpore等上层框架下发一个计算图时，GE会接管这个图，并进行一系列优化，包括：
   1. 算子融合： 将多个小算子（如一个卷积+一个激活函数）融合成一个更大的、执行效率更高的算子，减少Kernel启动开销和内存读写。
   2. 并行优化： 分析图中的依赖关系，最大化地并行执行无依赖的算子。
   3. 内存优化： 智能地复用内存，减少显存占用。
5. 算子编译与执行（TBE & AICPU）：
   1. TBE (Tensor Boost Engine): 这是一个强大的算子开发工具。对于标准的NN算子，GE可以直接调用TBE来生成在达芬奇AI Core上高效执行的指令。如果遇到一个框架中有但CANN尚不支持的“生僻”算子，开发者可以利用TBE，使用一种类似Python的DSL（领域特定语言）来定义这个算子的计算逻辑，TBE会自动将其编译成高效的硬件指令。这是CANN生态扩展能力的关键。
   2. AICPU： 对于那些不适合在AI Core上运行的复杂逻辑或Host侧算子，CANN会调度到CPU上执行。

CANN工作流小结：

一个PyTorch for Ascend的任务流程大致是：

Python代码 -> PyTorch前端构建计算图 -> CANN的PyTorch适配层接管图 -> GE进行图优化 -> GE将优化后的图中的算子，一部分调度给TBE编译成AI Core指令，一部分交给AICPU处理 -> AscendCL和驱动层负责将指令下发到硬件执行 -> HCCL负责节点间的通信。

对于AI Infra工程师而言，这意味着：

- 性能问题可能出在任何一环：是图优化没生效？是某个算子没有TBE实现而落到了CPU？还是HCCL通信存在瓶颈？
- 生态适配的核心工作，就是确保上层框架的每一个算子，都能在CANN中找到高效的实现路径。

## 2.3 异构算力统一管理：如何在一个集群中兼容多种芯片

在理想世界中，我们希望所有算力都是同构的（例如全部是H800），这样管理最简单。但在现实世界中，由于历史采购、国产化要求、成本控制、不同负载的最优硬件不同等原因，一个智算中心内往往同时存在NVIDIA GPU和华为昇腾NPU，甚至还有其他品牌的AI芯片。

如何在一个统一的平台（通常是Kubernetes）上，对这些“说不同语言”的硬件进行统一的资源视图、调度、监控和管理，是衡量一个智算中心运营成熟度的重要标志。

### 2.3.1 挑战：各自为政的“方言”

异构管理的根本挑战在于，每种硬件都有自己独立的“技术方言”：

- 驱动和运行时： NVIDIA需要NVIDIA Driver和Container Toolkit；昇腾需要CANN Driver和Ascend Docker Runtime。
- 设备标识： 在容器内，NVIDIA GPU通常表现为`/dev/nvidiaX`；昇腾NPU则表现为`/dev/davinciX`。
- 环境变量： NVIDIA应用依赖`NVIDIA_VISIBLE_DEVICES`；昇腾应用依赖`ASCEND_VISIBLE_DEVICES`。
- 监控工具： `nvidia-smi` vs `npu-smi`。

如果任由这种混乱存在，运维将成为一场噩梦：你需要为不同硬件维护不同的节点池、不同的基础镜像、不同的调度策略，无法实现资源的统一池化和按需分配。

### 2.3.2 解决方案：Kubernetes的抽象与统一

Kubernetes通过其强大的插件化和声明式API机制，为我们提供了抹平底层硬件差异的完美方案。核心武器是 Device Plugin（设备插件）。

#### Device Plugin原理

Device Plugin是K8s提供的一套标准框架，允许第三方厂商将自己的硬件资源（如GPU, NPU, FPGA）接入到K8s的资源管理体系中。

一个Device Plugin通常是一个运行在每个计算节点上的DaemonSet。它的工作流程是：

1. 发现（Discover）： 启动后，它会扫描当前节点，通过调用`nvidia-smi`或`npu-smi`等工具，发现节点上有多少张可用的AI加速卡。
2. 注册（Register）： 它通过gRPC与节点上的Kubelet通信，将发现的硬件资源上报给Kubelet。例如，NVIDIA的Device Plugin会注册一种名为`nvidia.com/gpu`的资源，华为的Device Plugin会注册`huawei.com/npu`（或类似名称）。
3. 分配（Allocate）： 当一个Pod的YAML中声明需要`nvidia.com/gpu: 1`时，Kube-scheduler会找到一个拥有该资源的空闲节点。Pod被调度到该节点后，Kubelet会调用该节点的Device Plugin的`Allocate`接口。Device Plugin会负责准备好GPU/NPU所需的环境（如设置环境变量），并将其挂载到Pod的容器中。

### 2.3.3 实战：构建异构AI集群的K8s实践

下面是一套在K8s中管理NVIDIA GPU和华为NPU的典型实践步骤：

#### 节点打标（Node Labeling）

首先，需要让K8s能够区分不同类型的节点。可以使用`kubectl label node`命令为节点打上清晰的标签：

```bash
# 为装有NVIDIA H800的节点打标
kubectl label node node-01 accelerator=nvidia-h800
# 为装有华为910B的节点打标
kubectl label node node-02 accelerator=ascend-910b
```

#### 部署各自的Device Plugin

在集群中，需要同时部署NVIDIA和华为的官方Device Plugin。它们通常以DaemonSet的形式部署，并使用`nodeSelector`来确保只运行在装有相应硬件的节点上。

- NVIDIA Device Plugin会监听`accelerator: nvidia-h800`的节点。
- Ascend Device Plugin会监听`accelerator: ascend-910b`的节点。

#### 使用Pod Spec声明资源需求

现在，用户可以像申请CPU/Memory一样，在Pod的YAML中声明式地申请AI算力，而无需关心底层的设备细节。

示例1：一个申请1张NVIDIA H800的PyTorch训练任务Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
    name: pytorch-train-nvidia
spec:
    nodeSelector:
    accelerator: nvidia-h800 # 调度到NVIDIA节点
    containers:
    - name: main
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime # 使用CUDA基础镜像
    command: ["python", "train.py"]
    resources:
        limits:
        nvidia.com/gpu: 1 # 申请1张NVIDIA GPU
```

示例2：一个申请1张华为910B的MindSpore训练任务Pod

```yaml
apiVersion: v1
kind: Pod
metadata:
    name: mindspore-train-ascend
spec:
    nodeSelector:
    accelerator: ascend-910b # 调度到Ascend节点
    containers:
    - name: main
    image: mindspore/mindspore-ascend:2.2.0-cann7.0 # 使用CANN基础镜像
    command: ["python", "train.py"]
    resources:
        limits:
        huawei.com/npu: 1 # 申请1张Ascend NPU
```

统一管理的价值：

通过这种方式，我们实现了一个统一的资源视图。对于集群管理员和用户来说：

- 资源池化： 所有的NVIDIA GPU构成一个逻辑资源池，所有的Ascend NPU构成另一个逻辑资源池。
- 调度统一： K8s调度器可以基于标签和资源请求，自动完成“任务到硬件”的匹配。
- 视图统一： 通过`kubectl describe node`，可以清晰地看到每个节点上`nvidia.com/gpu`和`huawei.com/npu`的总量和已分配量。
- 监控整合： 借助Prometheus Operator和Exporter，可以采集`dcgm-exporter`和`npu-exporter`的指标，并在同一个Grafana大盘中进行展示，实现异构算力的统一监控。

至此，我们已经深入了解了NVIDIA和华为昇腾两大生态的硬件核心与软件灵魂，并掌握了在Kubernetes这一现代化云原生平台上对它们进行“求同存异、统一管理”的实战方法。这为您在后续章节中学习AI任务调度、性能优化和故障排查，打下了坚实的硬件认知基础。
