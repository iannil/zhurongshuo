---
title: "第9章：构建全链路监控体系"
date: 2025-11-29T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["智算中心运营实战：从基础设施到大模型全栈优化", "第9章：构建全链路监控体系"]
slug: "chapter-09"
---

至此，我们已经走过了从硬件选型、平台搭建到模型训练与推理服务化的完整旅程。我们构建的智算中心，如同一座精密的、高速运转的巨型工厂，每一秒都在进行着海量的计算。然而，复杂性与脆弱性是一对孪生兄弟。在这座工厂里，任何一个微小的环节——一块过热的GPU、一个拥堵的网络端口、一个配置不当的推理引擎——都可能引发连锁反应，导致昂贵的训练任务失败或在线服务中断。

我们如何才能洞悉这座黑暗工厂的内部运作，提前预警风险，快速定位故障？答案是构建一个全链路、多层次的可观测性（Observability）体系。

“监控（Monitoring）”告诉我们“系统哪里出错了”，而“可观测性”则要回答“系统为什么出错了”。它不仅仅是收集指标，更是要将来自硬件、平台、应用等不同层面的数据（Metrics, Logs, Traces）关联起来，为我们提供一个可钻取、可分析的全局视图。

本章，我们将以业界主流的Prometheus + Grafana技术栈为核心，从零开始构建一套覆盖“硬件-平台-业务”三层的智算中心监控体系。我们将学习如何从NVIDIA和华为的硬件中“抠”出最核心的指标，如何从vLLM等业务应用中提取关键性能数据，并最终将这些数据汇聚到一块精心设计的“智算运营驾驶舱”Grafana大屏上。这块大屏，将成为你管理整个智算中心的“中控台”。

## 9.1 采集层：DCGM-Exporter（N卡）与 NPU-Exporter（华为）指标抓取

可观测性的第一步是数据采集。对于智算中心，最基础、最重要的监控数据源自于底层的AI加速硬件。我们需要实时了解每一张GPU/NPU的健康状况和负载情况。

Prometheus是一个基于拉（Pull）模型的时序数据库。它会周期性地访问目标（Target）暴露出的一个HTTP端点（通常是`/metrics`），并抓取符合其特定格式的指标数据。为了让Prometheus能够“读懂”GPU/NPU的状态，我们需要一个“翻译官”——Exporter。

### 9.1.1 监控NVIDIA GPU：DCGM-Exporter

NVIDIA提供了一套强大的数据中心GPU管理工具集，名为DCGM (Data Center GPU Manager)。DCGM比我们常用的`nvidia-smi`要强大得多，它能以更高的频率、更低的开销，提供更丰富的指标和健康检查功能。DCGM-Exporter就是将DCGM收集到的海量指标，转换为Prometheus能识别的格式的官方工具。

DCGM的核心优势：

- 高性能采集： DCGM在GPU驱动层面进行数据采集，开销极小。
- 丰富的指标： 除了`nvidia-smi`能看到的GPU利用率、显存使用、温度、功耗，DCGM还能提供更深层次的指标，如：
  - SM Clock/Memory Clock: SM（流多处理器）和显存的实际运行频率。
  - PCIe Replays: PCIe总线的重传次数，是诊断硬件链路问题的重要指标。
  - NVLink Bandwidth/Errors: NVLink的带宽使用率和错误计数。
  - XID Errors: 关键的GPU内部错误代码，是排查“掉卡”等严重故障的线索。
  - ECC Errors: 显存的纠错码错误计数，分为可纠正（Correctable）和不可纠正（Uncorrectable）。不可纠正的ECC错误通常意味着GPU硬件故障。
- 主动健康检查： DCGM可以主动对GPU进行诊断，如显存压力测试、PCIe带宽测试等。

部署DCGM-Exporter（以K8s DaemonSet为例）：

在Kubernetes集群中，最佳实践是将DCGM-Exporter作为DaemonSet部署，确保每个GPU节点上都有一个Exporter实例在运行。

```yaml
apiVersion: apps/v1
kind: DaemonSet
metadata:
    name: dcgm-exporter
    namespace: monitoring
spec:
    selector:
    matchLabels:
        app.kubernetes.io/name: dcgm-exporter
    template:
    metadata:
        labels:
        app.kubernetes.io/name: dcgm-exporter
    spec:
        nodeSelector:
        nvidia.com/gpu: "true" # 只在有NVIDIA GPU的节点上运行
        containers:
        - image: nvcr.io/nvidia/k8s/dcgm-exporter:3.3.0-3.1.8-ubuntu22.04
            name: dcgm-exporter
            ports:
            - name: metrics
                containerPort: 9400
            securityContext:
            runAsUser: 0
        # ... 需要挂载一些必要的设备和目录
```

配置Prometheus抓取

你需要让Prometheus能够自动发现这些Exporter。在K8s中，这通常通过为DaemonSet的Pod添加特定的`annotations`来实现，Prometheus Operator会根据这些注解自动生成抓取配置。

```yaml
# Pod template annotations
metadata:
    annotations:
    prometheus.io/scrape: 'true'
    prometheus.io/path: '/metrics'
    prometheus.io/port: '9400'
```

核心DCGM指标解读（在PromQL中使用）：

- `DCGM_FI_DEV_GPU_UTIL`: GPU利用率 (%)，等同于`nvidia-smi`的`GPU-Util`。
- `DCGM_FI_DEV_FB_USED`: 已使用的显存大小 (MB)。
- `DCGM_FI_DEV_POWER_USAGE`: 功耗 (W)。
- `DCGM_FI_DEV_GPU_TEMP`: GPU核心温度 (°C)。
- `DCGM_FI_DEV_XID_ERRORS`: XID错误计数。`rate(DCGM_FI_DEV_XID_ERRORS[5m]) > 0` 是一个非常关键的告警规则。
- `DCGM_FI_DEV_UNCORRECTED_SBE_ERRORS` / `DBE_ERRORS`: 不可纠正的ECC错误。出现增长通常意味着需要更换硬件。
- `DCGM_FI_PROF_NVLINK_TX_BYTES`, `RX_BYTES`: NVLink的发送/接收字节数。通过`rate()`可以计算出实时带宽。
- `DCGM_FI_PROF_PCIE_RX_BYTES`, `TX_BYTES`: PCIe的发送/接收字节数。

### 9.1.2 监控华为昇腾NPU：NPU-Exporter

对于华为昇腾平台，社区和华为官方也提供了类似的Exporter——通常称为NPU-Exporter。其原理与DCGM-Exporter完全相同：在后台调用`npu-smi`或CANN的底层API来获取NPU状态，并将其转换为Prometheus格式。

部署NPU-Exporter：

同样以DaemonSet的形式部署在所有昇腾节点上。你需要从华为的AscendHub或相关开源社区获取其容器镜像和部署YAML。

```yaml
# 示例DaemonSet
apiVersion: apps/v1
kind: DaemonSet
metadata:
    name: npu-exporter
    namespace: monitoring
spec:
    template:
    spec:
        nodeSelector:
        huawei.com/npu: "true" # 只在昇腾节点运行
        containers:
        - image: ascendhub.huawei.com/public-ascendhub/npu-exporter:latest # 镜像地址请以官方为准
            name: npu-exporter
            ports:
            - name: metrics
                containerPort: 9101 # 端口可能不同
        # ...
```

核心NPU指标解读：

NPU-Exporter暴露的指标名称可能因版本而异，但其核心概念与DCGM是相通的。

- `npu_utilization_ratio`: NPU利用率，可能包含AI Core、AI CPU、Control CPU等多个维度的利用率。`AICore_utilization` 是最重要的，它反映了核心计算单元的繁忙程度。
- `npu_memory_used_bytes`: 已使用的HBM（高带宽内存）大小。
- `npu_temperature_celsius`: NPU芯片温度。
- `npu_power_watts`: 功耗。
- `npu_hbm_bandwidth_usage_ratio`: HBM带宽利用率。这是一个非常关键的性能指标，如果很低，可能意味着存在“内存墙”问题。
- `npu_roce_bandwidth_bytes_total`: RoCE网络的收发字节数，用于监控分布式训练的通信流量。

小结： 通过部署DCGM-Exporter和NPU-Exporter，我们完成了可观测性体系的“物理层”数据采集。现在，Prometheus中已经源源不断地汇入了来自每一张AI加速卡的“心跳”和“血压”数据。这是后续所有告警和可视化分析的基础。

## 9.2 业务层：Token生成速率、请求队列长度、显存碎片率监控

仅仅监控硬件是远远不够的。一个GPU利用率100%的服务，可能因为请求大量积压而导致用户体验极差。我们需要深入到“业务应用”的内部，去采集那些能直接反映服务质量和效率的指标。对于大模型推理服务（如基于vLLM或TensorRT-LLM构建的服务），我们需要关注以下几类核心业务指标。

### 9.2.1 如何从应用中暴露业务指标？

现代的推理引擎（如vLLM, Triton）通常已经内置了Prometheus Exporter的功能。你只需要在启动时开启一个选项，它就会自动暴露一个`/metrics`端点。

- vLLM的例子：vLLM的`AsyncLLMEngine`和API Server已经集成了Prometheus指标。你可以直接从其`/metrics`端点获取丰富的业务数据。
- Triton的例子：Triton Inference Server原生支持Prometheus指标，默认在`http://<triton-server>:8002/metrics`暴露。
- 自定义应用的实现：如果你的应用没有内置支持，可以使用Prometheus的官方Python客户端库（`prometheus-client`）来轻松地添加自定义指标。

```python
from prometheus_client import Counter, Gauge, Histogram, start_http_server

# 定义指标
REQUESTS_IN_QUEUE = Gauge('my_app_requests_in_queue', 'Number of requests waiting in the queue')
TTFT_HISTOGRAM = Histogram('my_app_ttft_seconds', 'Time to first token histogram')

# 在你的代码逻辑中更新指标
def handle_request(request):
    REQUESTS_IN_QUEUE.inc()
    # ... process request ...
    ttft = measure_ttft()
    TTFT_HISTOGRAM.observe(ttft)
    REQUESTS_IN_QUEUE.dec()

# 启动一个HTTP服务器来暴露指标
start_http_server(8000)
```

### 9.2.2 核心业务指标详解

#### 请求与队列指标 (反映服务负载与健康度)

- `llm_requests_in_queue_total` (Gauge): 等待队列中的请求数量。
  - 监控与告警： 这个指标持续增长，是服务即将过载的最直接信号。需要设置一个告警阈值，例如当队列长度持续超过某个值（如100）达5分钟，就立即告警，提示需要扩容。
- `llm_requests_running_total` (Gauge): 正在GPU上处理的请求数量。
  - 监控： 这个值应该与你设置的`max_num_batched_tokens`等参数相关，反映了GPU的并发处理能力。
- `llm_requests_success_total` / `llm_requests_failed_total` (Counter): 成功和失败的请求总数。
  - 监控与告警： `rate(llm_requests_failed_total[5m])` 可以计算出失败率。失败率突然飙升是严重故障的标志。

#### 性能与吞吐量指标 (反映服务效率)

- `llm_generation_tokens_total` (Counter): 生成的总Token数量。
  - 计算TPS (Tokens Per Second): `rate(llm_generation_tokens_total[5m])` 就是整个服务的实时Token生成速率。这是衡量服务总吞吐量的黄金指标。
- `llm_prompt_tokens_total` (Counter): 处理的总Prompt Token数量。
  - 计算Prompt Throughput: `rate(llm_prompt_tokens_total[5m])`。
- `llm_time_to_first_token_seconds_bucket` (Histogram): TTFT的直方图分布。
  - 计算百分位TTFT: `histogram_quantile(0.99, sum(rate(llm_time_to_first_token_seconds_bucket[5m])) by (le))`。计算99百分位的TTFT，是评估服务SLA（服务等级协议）的关键。例如，你可以承诺“99%的请求TTFT在500ms以内”。
- `llm_time_per_output_token_seconds_bucket` (Histogram): TPOT的直方图分布。
  - 计算百分位TPOT: `histogram_quantile(0.99, ...)`。反映了最坏情况下的生成速度。

#### 资源管理指标 (反映引擎内部效率)

- `vllm_gpu_cache_usage_perc` (Gauge): (vLLM特有) KV Cache的利用率。
  - 监控： 这个值应该持续保持在高位（如90%以上），这证明PagedAttention正在高效工作。如果这个值很低，但请求已经开始排队，可能意味着有其他瓶颈。
- 显存碎片率（通常需要间接计算或由特定Exporter提供）：
  - 计算方法： `(显存总块数 - 显存空闲块数 - 显存已使用块数) / 显存总块数`。
  - 监控： 理想情况下，这个值应该接近于0。如果持续升高，表明引擎的内存管理器可能存在问题。
- `vllm_scheduler_running_requests`, `swapped_requests`, `waiting_requests` (Gauges): vLLM调度器内部状态，帮助深入分析请求的处理流程。

通过将这些业务层指标与前一节的硬件层指标结合，我们就能形成一个完整的分析链路。例如：

- 现象： 99百分位TTFT突然飙升。
- 分析：
  - 查看`llm_requests_in_queue_total`，如果队列长度也在飙升 -> 请求积压导致。
  - 查看`DCGM_FI_DEV_GPU_UTIL`，如果GPU利用率已经100% -> 算力瓶颈，需要扩容。
  - 如果GPU利用率不高，但队列积压 -> 可能是CPU瓶颈（Python代码、API服务器）、网络I/O瓶颈，或者推理引擎的调度逻辑出了问题。

## 9.3 可视化实战：从0搭建一套“智算运营驾驶舱”Grafana大屏

数据采集完成，现在我们需要将这些冰冷的数字，变成直观、易懂、可交互的图表。Grafana是这个任务的不二之选。一个好的Grafana Dashboard，不仅是运维人员的“作战指挥室”，更是向管理层展示运营成果、汇报资源利用率的“商业智能（BI）”面板。

我们将设计一个“智算运营驾驶舱”，它将包含三个核心部分：全局概览、训练集群监控 和 推理服务监控。

前提： 你已经部署好了Prometheus和Grafana，并且Prometheus已经配置为抓取DCGM-Exporter, NPU-Exporter, 以及推理服务的业务指标。

### 第一步：设计Dashboard结构 (Layout & Variables)

- 创建新的Dashboard。
- 使用模板变量 (Template Variables)： 这是让Dashboard变得“活”起来的关键。
  - `$node`: 创建一个类型为`Query`的变量，查询表达式为`label_values(node_uname_info, nodename)`，用于在节点间切换。
  - `$gpu`: 创建一个类型为`Query`的变量，查询表达式为`label_values({__name__=~"DCGM_FI_DEV_GPU_UTIL"}, gpu)`，用于在单个GPU间切换。
  - `$service`: 创建一个查询，获取所有推理服务的名称，`label_values(llm_requests_in_queue_total, service_name)`。

### 第二步：构建“全局概览” (The Big Picture)

这部分面向管理者和一线运维的“第一眼”，提供最核心的宏观指标。

- Stat Panel: 核心KPI
  - GPU总数/NPU总数： `count(count by (instance)(DCGM_FI_DEV_GPU_UTIL))`
  - GPU总利用率 (Avg): `avg(DCGM_FI_DEV_GPU_UTIL)`
  - GPU总功耗 (Sum): `sum(DCGM_FI_DEV_POWER_USAGE) / 1000` (单位KW)
  - 告警中的GPU数量： `count(ALERTS{alertstate="firing", alertname=~"GPU.*"})`

- Time Series Panel: 全局资源利用率趋势
  - 查询A (GPU Util): `avg(DCGM_FI_DEV_GPU_UTIL) by (job)` (如果使用K8s，可以按`namespace`或`pod`聚合)
  - 查询B (Memory Util): `avg(DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) * 100`

- Table Panel: 节点资源排行榜
  - 按平均GPU利用率对节点排序，快速发现高负载或低负载的节点。
  - 查询: `avg by (nodename) (DCGM_FI_DEV_GPU_UTIL)`

### 第三步：构建“训练集群钻取视图” (Training Deep Dive)

这部分专注于诊断和分析训练任务。

- Row: 按节点和GPU钻取
  - 使用我们创建的`$node`和`$gpu`变量。
- Time Series Panel: 单GPU核心指标 (重复这个Panel，使其显示`$node`上的所有GPU)
  - GPU/NPU Util: `DCGM_FI_DEV_GPU_UTIL{nodename="$node", gpu="$gpu"}`
  - Memory Used: `DCGM_FI_DEV_FB_USED{nodename="$node", gpu="$gpu"}`
  - Power & Temp: `DCGM_FI_DEV_POWER_USAGE{...}`, `DCGM_FI_DEV_GPU_TEMP{...}`

- Time Series Panel: 网络通信监控
  - NVLink Bandwidth: `rate(DCGM_FI_PROF_NVLINK_TX_BYTES{...}[5m]) / 1024 / 1024` (MB/s)
  - RoCE Bandwidth: `rate(npu_roce_bandwidth_bytes_total{...}[5m])`
  - 意义： 在分布式训练时，这些图上应该能看到规律性的、高峰值的通信流量。如果流量很低或没有，说明分布式通信可能没正常工作。

- Stat Panel / Table: 硬件错误监控
  - XID Errors: `sum(rate(DCGM_FI_DEV_XID_ERRORS{nodename="$node"}[10m])) by (gpu)`
  - ECC Errors: `sum(rate(DCGM_FI_DEV_UNCORRECTED_SBE_ERRORS{...}[10m])) by (gpu)`
  - 关键： 任何非零值都值得高度警惕！

### 第四步：构建“推理服务钻取视图” (Inference Deep Dive)

这部分专注于评估在线服务的性能和健康度。

- Row: 按服务和实例钻取
  - 使用`$service`变量。
- Time Series Panel: 服务质量 (QoS)
  - 99th TTFT: `histogram_quantile(0.99, sum(rate(llm_time_to_first_token_seconds_bucket{service_name="$service"}[5m])) by (le))`
  - Avg TTFT: `sum(rate(llm_time_to_first_token_seconds_sum[5m])) / sum(rate(llm_time_to_first_token_seconds_count[5m]))`
  - Error Rate: `sum(rate(llm_requests_failed_total{service_name="$service"}[5m])) / sum(rate(llm_requests_total{service_name="$service"}[5m]))`

- Time Series Panel: 吞吐量与负载
  - TPS (Tokens/sec): `sum(rate(llm_generation_tokens_total{service_name="$service"}[5m]))`
  - RPS (Requests/sec): `sum(rate(llm_requests_total{service_name="$service"}[5m]))`
  - Queue Length: `llm_requests_in_queue_total{service_name="$service"}`

- Time Series Panel: 引擎内部状态
  - KV Cache Usage: `vllm_gpu_cache_usage_perc{service_name="$service"}`
  - Running vs Waiting Requests: `vllm_scheduler_running_requests`, `vllm_scheduler_waiting_requests`

### 第五步：配置告警 (Alerting)

Grafana集成了强大的告警功能。你可以为几乎任何一个Panel配置告警规则。

关键告警规则示例：

- 硬件告警:
  - `GPU_Too_Hot`: `DCGM_FI_DEV_GPU_TEMP > 85` for 5m
  - `GPU_XID_Error`: `rate(DCGM_FI_DEV_XID_ERRORS[5m]) > 0`
  - `GPU_Uncorrectable_ECC`: `increase(DCGM_FI_DEV_UNCORRECTED_SBE_ERRORS[10m]) > 0`
- 业务告警:
  - `Inference_High_TTFT`: 99th TTFT > 1s for 5m
  - `Inference_High_Queue_Length`: `llm_requests_in_queue_total > 100` for 10m
  - `Inference_High_Error_Rate`: Error Rate > 5% for 1m

将这些告警规则配置好，并对接上你的告警通知渠道（如Slack, PagerDuty, 钉钉），你就拥有了一个7x24小时不间断的“智能哨兵”。

总结：

一个精心设计的Grafana Dashboard，远不止是一堆图表的堆砌。它是一个故事板，讲述了你的智算中心从硬件到业务的完整故事。它是一个诊断仪，能帮助你快速地从宏观现象钻取到微观根因。它更是一个价值放大器，将你和你的团队在幕后所做的繁重而复杂的运维工作，以最直观、最有冲击力的方式，呈现在了所有相关方面前。这，就是可观测性的终极魅力。
