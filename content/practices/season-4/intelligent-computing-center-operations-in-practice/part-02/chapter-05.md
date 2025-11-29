---
title: "第5章：AI任务调度与资源管理"
date: 2025-11-29T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["智算中心运营实战：从基础设施到大模型全栈优化", "第5章：AI任务调度与资源管理"]
slug: "chapter-05"
---

在上一章中，我们已经成功地将AI算力（GPU/NPU）通过Device Plugin机制接入了Kubernetes，并学会了如何构建标准化的容器镜像。至此，我们的K8s集群已经具备了运行AI任务的基本能力。然而，一场新的、更为严峻的挑战正悄然降临，它发生在K8s集群的“中央大脑”——调度器（Scheduler）之中。

想象一个繁忙的机场，调度塔台（K8s Scheduler）负责将一架架飞机（Pods）指派到合适的登机口（Nodes）。对于普通的Web应用这类“客机”，调度相对简单：只要登机口有空位，就可以停靠。但AI训练任务，特别是分布式训练，更像是需要多个登机口同时协同保障的“航天飞机发射任务”。如果塔台仍然用调度客机的方式来调度航天飞机，混乱和灾难将不可避免。

本章，我们将深入剖析原生K8s调度器在AI场景下的“水土不服”，并引入专门为此类“航天飞机任务”设计的“高级调度塔台”——Volcano和Yunikorn。我们将亲手部署和配置它们，并深入理解其背后最核心的“全家桶调度（Gang Scheduling）”、“装箱（Binpack）”和“拓扑感知”等高级调度策略。掌握了这些，你才能真正驾驭大规模AI集群的资源分配，成为一名合格的“智算塔台指挥官”。

#### 5.1 为什么原生K8s调度器不适合AI？（死锁与碎片问题）

Kubernetes的默认调度器（`kube-scheduler`）是一个杰出的通用调度器。它的核心逻辑是：逐个处理Pod，为每个Pod找到一个满足其资源需求的最佳节点。这个“逐个调度”的策略，在处理无状态、可独立运行的微服务时表现出色，但在面对强耦合的AI/HPC（高性能计算）负载时，却会引发两大致命问题：死锁（Deadlock）和资源碎片（Resource Fragmentation）。

### 5.1.1 死锁：永远凑不齐的“麻将搭子”

死锁问题主要发生在分布式训练场景。以一个需要8张GPU进行数据并行的PyTorch `DistributedDataParallel`任务为例。这个任务通常会由8个Pod组成，每个Pod申请1张GPU，它们之间需要通过集合通信（如All-Reduce）进行同步，必须全部启动成功后，训练才能开始。

现在，让我们看看原生调度器会如何处理这个任务：

1. 任务提交： 用户一次性提交了8个Pod（我们称之为`train-job-pod-0`到`train-job-pod-7`）。
2. 逐个调度： `kube-scheduler`从待调度队列中取出`train-job-pod-0`，找到了一个有空闲GPU的`node-A`，成功调度。接着取出`pod-1`，调度到`node-B`……这个过程持续进行。
3. 灾难发生： 假设集群中总共只有10张空闲GPU。当调度器成功调度了前5个Pod后，它们已经占用了5张GPU。此时，集群中还剩5张空闲GPU。不幸的是，另一个用户提交了一个申请5张GPU的推理任务，调度器“公平地”将这5张卡分配给了这个新任务。
4. 死锁形成： 现在，我们的训练任务有5个Pod已经启动并占用了GPU，但它们无法开始工作，因为它们在焦急地等待另外3个兄弟Pod。然而，集群中已经没有任何空闲GPU了！剩下的3个Pod（`pod-5`到`pod-7`）将永远处于`Pending`状态。
5. 资源浪费： 更糟糕的是，那5个已经启动的Pod，因为等不到同伴，只能在原地“空转”，白白占用了5张昂贵的GPU，却不产生任何计算价值。整个集群的资源被部分占用的任务“锁死”了，新的任务也无法被调度。

这个场景就像打麻将，你需要凑齐“东南西北”四张牌才能胡牌。你已经拿到了“东”、“南”、“西”，但“北”被别人拿走了，而且他也不打算打出来。你手里的三张牌就成了废牌，占着位置，还让你无法换牌。

原生调度器的根本缺陷： 它缺乏“All-or-Nothing”（全有或全无）的原子性语义。它不知道这8个Pod是一个不可分割的整体（作业，Job），必须作为一个单元被统一调度。

### 5.1.2 资源碎片：昂贵的“孤岛”

即使不考虑分布式任务，原生调度器在处理大量单卡任务时，也容易导致严重的资源碎片问题。

K8s默认的调度策略之一是`LeastAllocated`（或类似的`BalancedResourceAllocation`），即倾向于将Pod均匀地分布在所有节点上，以实现负载均衡。这对于Web服务是合理的，可以分散风险。但对于GPU集群，这却是灾难。

场景： 假设我们有2台节点，每台8张GPU（共16张）。现在，需要调度8个各申请1张GPU的单卡训练任务。

原生调度器的行为：

第一个Pod来了，调度到`node-A`。
第二个Pod来了，为了“均衡”，调度器会倾向于将它放到负载更低的`node-B`。
第三个Pod，调度到`node-A`...
最终的结果很可能是：`node-A`上运行了4个Pod，`node-B`上也运行了4个Pod。两台节点各有4张GPU被占用，4张空闲。

碎片化后果：

此时，集群看起来“负载均衡”，但资源利用率极低。如果这时来了一个需要8张GPU的大规模分布式训练任务，调度器会发现：没有任何一个节点能满足它的需求！ 尽管集群中总共还有8张空闲GPU，但它们分散在两个节点上，形成了无法被大型任务利用的“资源碎片”。

这就好比电影院卖票，一个8人的大家庭想坐在一起，但售票系统为了让每排都坐上人，把空座位零散地分布在各个角落，导致这个家庭永远找不到连续的8个座位。

原生调度器的根本缺陷： 它缺乏“聚拢（Binpack）”而非“分散（Spread）”的调度意识。对于昂贵的GPU资源，我们的目标应该是尽可能地将任务集中在少数节点上，以保留出完整、连续的“大块”资源给未来的大型任务。

总结：

K8s原生调度器在AI/HPC场景下的失败，根源于其设计的初衷是面向无状态、松耦合的通用应用。它缺乏对“作业（Job）”这一整体概念的认知，也缺乏对昂贵资源“聚拢而非分散”的特殊考量。要解决这些问题，我们必须引入专为批处理和高性能计算设计的“高级调度器”。

#### 5.2 高级调度器实战：Volcano/Yunikorn 的安装、配置与策略详解

为了弥补原生调度器的不足，云原生社区孕育出了两个优秀的面向批处理系统的调度器：Volcano 和 Apache Yunikorn。它们都作为K8s的“第二调度器”运行，专门处理AI/HPC等复杂负载。

### 5.2.1 调度器的工作原理：竞争与合作

在K8s中可以同时运行多个调度器。Pod可以通过`spec.schedulerName`字段来指定自己希望由哪个调度器来处理。

默认调度器 (`default-scheduler`)： 如果Pod不指定`schedulerName`，它就会被默认调度器处理。

第二调度器（如 `volcano` 或 `yunikorn`）： 我们可以配置AI任务的Pod模板，让它们都指定`schedulerName: volcano`。这样，这些Pod就会被Volcano接管，而集群中其他的普通应用（如Nginx）仍然由默认调度器管理，互不干扰。

### 5.2.2 Volcano：CNCF的首个批处理调度系统

Volcano是华为开源并贡献给CNCF（云原生计算基金会）的项目，是社区内第一个也是目前最成熟的云原生批处理系统。它不仅仅是一个调度器，更是一套完整的作业管理系统。

核心概念：

Job & PodGroup： Volcano引入了`Job`这个CRD（自定义资源定义），它代表一个完整的批处理作业。一个`Job`可以包含多个任务（Task），每个任务对应一个Pod模板。更底层地，Volcano将一个`Job`中的所有Pod逻辑上绑定为一个`PodGroup`。`PodGroup`是Volcano进行原子调度的基本单位。

Queue： Volcano引入了队列（Queue）的概念。所有作业都必须提交到某个队列中。管理员可以为不同的队列设置不同的资源配额（Quota）、优先级和调度策略。这为多租户资源隔离和公平共享提供了强大的机制。例如，可以为“算法核心组”创建一个高优先级、高配额的队列，为“实习生组”创建一个低优先级的队列。

安装实战（使用Helm）：

```bash
helm repo add volcano-sh https://volcano-sh.github.io/charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

安装后，你会看到`volcano-scheduler`、`volcano-controller-manager`等核心组件的Pod在`volcano-system`命名空间中运行起来。

如何使用Volcano提交一个作业：

你需要编写一个`VolcanoJob`的YAML文件。

示例：一个需要4个Pod的PyTorch分布式训练VolcanoJob

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
    name: pytorch-dist-job
spec:
    schedulerName: volcano # 指定调度器
    minAvailable: 4 # 这是关键！Gang Scheduling的实现
    queue: default # 提交到默认队列
    tasks:
    - name: worker
        replicas: 4 # 总共需要4个副本
        template:
        spec:
            containers:
            - name: pytorch
                image: my-pytorch-app:1.0
                resources:
                limits:
                    nvidia.com/gpu: 1
            restartPolicy: OnFailure
```

`minAvailable: 4`: 这就是Volcano实现“全家桶调度”（Gang Scheduling）的魔法。它告诉Volcano：这个作业至少需要4个Pod的资源同时被满足，你才能开始调度；否则，一个都不要动，让它们在队列里先等着。这就完美地解决了死锁问题。

Volcano的核心优势：

- 丰富的调度策略： 除了Gang Scheduling，Volcano还支持多种插件化的调度算法，如`Binpack`, `DRF` (Dominant Resource Fairness), `Priority`等。
- 强大的队列管理： 基于队列的资源隔离和抢占机制非常完善。
- 与主流框架集成： 深度集成了Spark, Flink, TensorFlow, PyTorch等，提供了Operator简化作业提交。

### 5.2.3 Apache Yunikorn：为大规模和混合负载而生

Yunikorn是Cloudera和微软等公司联合推动的、源自Hadoop YARN调度器经验的云原生调度器。它在设计上更加强调超大规模（数万节点）、高性能和资源公平性。

核心概念：

- 层次化队列（Hierarchical Queues）： 这是Yunikorn最大的特色。它的队列可以像文件系统目录一样，形成一个树状结构。资源配额可以从父队列继承，也可以在子队列上覆盖。这为大型组织进行精细化的资源划分和管理提供了极大的灵活性。例如：`root.research.team-a` 和 `root.production.online-service`。
- 应用（Application）： Yunikorn将一组相关的Pod（例如一个Spark作业的所有Executor）自动识别为一个“应用”，这也是其调度的基本单位。

安装实战：Yunikorn的安装同样可以通过官方的Helm Chart完成，过程与Volcano类似。

如何使用Yunikorn：

Yunikorn更强调“无感”集成。它会自动识别通过标签等方式关联起来的Pod，并将它们作为一个应用来调度。你提交的仍然是标准的K8s Pod，但需要在Pod的`labels`和`annotations`中加入Yunikorn能识别的信息。

```yaml
apiVersion: v1
kind: Pod
metadata:
    name: my-app-pod-1
    labels:
    applicationId: "app-0001" # 关键！标记属于同一个应用
    queue: "root.my-queue" # 指定队列
spec:
    schedulerName: yunikorn # 指定调度器
    # ...
```

Yunikorn的核心优势：

- 卓越的调度性能： 针对大规模集群做了特别优化，调度吞吐量非常高。
- 灵活的层次化队列： 对于有复杂组织架构和资源管理需求的企业，这是一个杀手级特性。
- 优秀的资源公平性算法： 内置了Fairness和DRF等算法，能很好地平衡多用户、多任务间的资源分配。

### 5.2.4 Volcano vs. Yunikorn：如何选择？

| 对比维度 | Volcano | Apache Yunikorn |
| :--- | :--- | :--- |
| 核心抽象 | Job/PodGroup，显式定义作业 | Application，自动识别应用 |
| 队列模型 | 扁平队列 | 层次化队列 (特色) |
| 调度性能 | 良好 | 卓越，为大规模优化 |
| 易用性 | 需学习`VolcanoJob` CRD | 可继续使用原生Pod/Deployment |
| 社区生态 | CNCF项目，与AI/HPC框架集成更深 | Apache顶级项目，源自大数据生态 |
| 核心特性 | Gang Scheduling，抢占等批处理功能全面 | 资源公平性，层次化资源管理 |

给AI Infra工程师的建议：

- 如果你的团队主要运行深度学习、HPC这类紧耦合的批处理作业，并且希望有开箱即用的Gang Scheduling和作业生命周期管理，Volcano是更直接、更成熟的选择。
- 如果你管理的是一个超大规模、多租户、混合负载（既有AI训练，又有大数据处理，还有在线服务）的复杂环境，需要进行非常精细化的分层级资源预算和公平性保障，那么Yunikorn强大的层次化队列和调度性能会更具吸引力。

在许多实际案例中，两者都能很好地解决原生调度器的问题。选择哪个，更多地取决于你的团队背景、技术栈偏好和组织管理复杂度。

#### 5.3 关键调度策略：Gang Scheduling、Binpack、拓扑感知调度

无论你选择Volcano还是Yunikorn，它们强大的调度能力都源于背后一系列精巧的调度算法（在调度器中通常以“插件”的形式实现）。理解这些核心策略，你才能真正地“调教”好你的调度器。

### 5.3.1 Gang Scheduling（全家桶调度）：解决死锁的银弹

策略目标： All-or-Nothing（全有或全无）。确保一个作业（Gang）中的所有成员（Pods）能够被同时调度。如果资源不足以同时满足所有成员，则整个作业保持等待，不占用任何资源。

实现原理（以Volcano为例）：

1. 排队与校验： 当一个`VolcanoJob`被提交到队列中，Volcano Controller会为其创建一个`PodGroup`对象，并在其中记录`minAvailable`（作业所需的最小副本数）。调度器在开始调度这个作业前，会先进行一次预检（pre-check）：它会模拟一次调度，看看当前集群的空闲资源是否足够满足`minAvailable`个Pod的需求。
2. 资源预留（Gating）： 如果预检通过，Volcano会进入“Gating”阶段。它会暂时“锁定”或“预留”住这部分资源，然后才开始逐个创建和调度Pod。因为资源已经被预留，所以不会发生调度了一半被别人抢走的情况。
3. 超时与回退： 如果预检失败，或者在等待资源的过程中超过了一定的超时时间，整个作业会回到队列中继续等待，并且可能会触发其他低优先级作业的抢占（Preemption），以释放资源。

价值： 彻底根除分布式训练因部分Pod无法启动而导致的死锁问题，是所有批处理调度器的“标配”和核心价值。

### 5.3.2 Binpack（装箱策略）：治愈资源碎片的良药

策略目标： 聚拢而非分散。尽可能地将Pod集中地调度到少数几个节点上，直到这些节点的资源被耗尽，再去使用新的节点。

实现原理：

- Binpack策略在为Pod打分时，会给那些资源使用率已经很高的节点打一个更高的分数。
- 例如，一个简单的Binpack评分函数可以是：`score = (gpu_used / gpu_total) * 10`。
- `node-A`已使用6/8的GPU，得分` (6/8)*10 = 7.5`。
- `node-B`已使用2/8的GPU，得分` (2/8)*10 = 2.5`。
- 调度器会优先选择得分更高的`node-A`。
- 这样，`node-A`会被迅速填满，而`node-B`则保持完全空闲，成为一块可供未来大型任务使用的“整块资源”，从而大大降低了资源碎片。

价值： 极大提升了集群整体的资源利用率，特别是对于需要整机或多机资源的大规模作业，Binpack策略是必不可少的。它与追求负载均衡的`Spread`策略，是AI集群中一对需要仔细权衡的矛盾。通常，对于GPU这类昂贵且不可切分的资源，Binpack是更优的选择。

### 5.3.3 Topology-Aware Scheduling（拓扑感知调度）：追求极致通信性能

策略目标： 将需要频繁通信的Pod，调度到物理上尽可能近的地方，以降低网络延迟、提升分布式训练性能。

“近”的层次：

- Level 0: 同一节点内。 这是最理想的情况。如果一个8卡任务能被调度到一台8卡的服务器上，它们之间的通信将主要通过速度最快的NVLink/HCCS进行，性能最好。
- Level 1: 同一机架内。 如果任务跨节点，那么将它们调度到同一个机架（Rack）内的不同节点，它们之间的通信只需要经过一层汇聚交换机（Top-of-Rack Switch），延迟较低。
- Level 2: 不同机架。 如果Pod被调度到不同机架，通信需要跨越多层交换机，延迟和拥塞风险都会增加。

实现原理：

1. 拓扑信息收集： 这需要管理员预先为K8s的Node对象打上拓扑标签。这些标签通常由自动化脚本或CMDB系统维护。

```bash
# 为节点打上机架和可用区的标签
kubectl label node node-01 topology.kubernetes.io/zone=zone-a topology.kubernetes.io/rack=rack-01
kubectl label node node-02 topology.kubernetes.io/zone=zone-a topology.kubernetes.io/rack=rack-01
kubectl label node node-03 topology.kubernetes.io/zone=zone-b topology.kubernetes.io/rack=rack-03
```

2. 调度器打分： 拓扑感知调度插件会读取这些标签。当调度一个Gang作业时，它会倾向于将这个Gang的所有Pod调度到拥有相同`topology.kubernetes.io/rack`标签的节点组上。它可以作为一个过滤条件（硬约束），也可以作为一个评分项（软约束）。
3. Volcano的实现： Volcano的`Affinity`和`Topology`等插件支持基于这些标签的调度。你可以定义一个PodGroup，要求其所有成员都具有某种亲和性。

价值： 对于大规模、通信密集型的AI训练任务，拓扑感知调度带来的性能提升是实实在在的。它可以将集群的线性加速比提高几个甚至十几个百分点，这意味着节约了大量的训练时间和金钱。这是从“能跑”到“跑得快”的关键一步，是顶级AI Infra团队必须具备的优化能力。

总结：

本章，我们完成了一次从“能用”到“好用”的进化。通过引入Volcano或Yunikorn等高级调度器，并深入理解其背后的Gang Scheduling、Binpack和拓扑感知等核心策略，我们构建了一个真正智能、高效的AI资源调度系统。它不再是一个盲目分配资源的“傻瓜”，而是一个懂得权衡、善于规划的“专家”。至此，我们云原生AI平台的“底座”部分已经全部搭建完毕。接下来，我们将进入全书最核心的第三篇，将我们的目光从平台层转向应用层，去直面大模型训练与推理的全流程运营挑战。
