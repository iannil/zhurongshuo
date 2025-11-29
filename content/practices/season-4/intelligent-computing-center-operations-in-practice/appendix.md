---
title: "附录"
date: 2025-11-29T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["智算中心运营实战：从基础设施到大模型全栈优化", "附录"]
slug: "appendix"
---

## 附录A：实操手册

理论的光辉，终需实践的印证。本附录旨在为你提供一套端到端的、可动手操作的实验指南，将我们在正文中探讨的核心概念——从底层的驱动安装，到上层的调度、推理和监控——转化为你本地环境中一行行可执行的命令和一段段可运行的代码。

我们深知，在复杂的AI基础设施领域，环境的差异性是最大的挑战。因此，本手册在设计时，力求简化依赖、明确前提，并对关键步骤提供详尽的注解。我们鼓励你不仅要“复制-粘贴”地完成实验，更要在过程中，回顾和思考每个步骤背后的原理，将其与正文中的知识点一一对应。

请准备好你的终端，让我们开始这场“理论联系实际”的终极演练。

### 环境准备脚本：Terraform/Ansible快速拉起实验环境

在真实的生产环境中，我们会使用Terraform来自动化云资源的创建（如VPC、虚拟机、负载均衡器），然后用Ansible来对这些虚拟机进行精细化的配置（如安装驱动、配置软件）。这是一个复杂但强大的Infrastructure as Code (IaC)流程。

由于完整的IaC脚本与特定的云厂商（AWS, Azure, GCP, 阿里云等）和你的账号配置强相关，在此提供一个通用的、可在任何云上执行的脚本是不现实的。因此，本节将提供一个概念性、模板化的Ansible Playbook，旨在向你展示自动化配置的核心逻辑。你可以将其作为起点，根据你自己的环境进行适配。

对于本地实验，我们推荐使用一台已安装好Linux（如Ubuntu 22.04）并配有NVIDIA GPU或可访问昇腾卡的物理机或虚拟机。

#### Ansible Playbook模板 (用于配置GPU节点)

这个Playbook展示了在一个或多个全新的Ubuntu节点上，自动化安装Docker、NVIDIA驱动、NVIDIA Container Toolkit和轻量级K8s (k3s)的流程。

前提：

1. 你有一台控制机，已安装Ansible (`pip install ansible`)。
2. 你有一个或多个目标GPU节点，控制机可以通过SSH免密登录到这些节点。
3. 在控制机上配置好Ansible的`inventory`文件（例如`/etc/ansible/hosts`），定义你的GPU节点组。

```ini
[gpu_nodes]
192.168.1.101
192.168.1.102
```

Playbook文件 (`setup_gpu_node.yml`):

```yaml
---
- hosts: gpu_nodes
  become: yes # 以root权限执行
  vars:
    nvidia_driver_version: "535" # 指定你想安装的驱动版本

  tasks:
    - name: 1. Update APT cache and install prerequisite
      apt:
        update_cache: yes
        name: ['apt-transport-https', 'ca-certificates', 'curl', 'gnupg-agent', 'software-properties-common']
        state: present

    - name: 2. Add Docker's official GPG key
      apt_key:
        url: https://download.docker.com/linux/ubuntu/gpg
        state: present

    - name: 3. Add Docker repository
      apt_repository:
        repo: deb [arch=amd64] https://download.docker.com/linux/ubuntu {{ ansible_distribution_release }} stable
        state: present

    - name: 4. Install Docker Engine
      apt:
        name: ['docker-ce', 'docker-ce-cli', 'containerd.io']
        state: present

    - name: 5. Add NVIDIA driver repository
      apt_repository:
        repo: ppa:graphics-drivers/ppa
        state: present

    - name: 6. Install NVIDIA Driver
      apt:
        name: "nvidia-driver-{{ nvidia_driver_version }}"
        state: present
      register: driver_install
      notify: Reboot node # 驱动安装后通常需要重启

    - name: 7. Add NVIDIA Container Toolkit repository
      shell: |
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
        sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
        tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

    - name: 8. Install NVIDIA Container Toolkit
      apt:
        update_cache: yes
        name: nvidia-container-toolkit
        state: present

    - name: 9. Configure Docker to use NVIDIA runtime and restart
      shell: |
        nvidia-ctk runtime configure --runtime=docker
        systemctl restart docker

    - name: 10. Install k3s (Lightweight Kubernetes)
      shell: |
        curl -sfL https://get.k3s.io | sh -
        mkdir -p $HOME/.kube
        cp /etc/rancher/k3s/k3s.yaml $HOME/.kube/config
        chown $(id -u):$(id -g) $HOME/.kube/config
      args:
        creates: /usr/local/bin/k3s # 避免重复安装

  handlers:
    - name: Reboot node
      reboot:
        msg: "Rebooting node after NVIDIA driver installation"
        connect_timeout: 5
        reboot_timeout: 300
        pre_reboot_delay: 0
        post_reboot_delay: 30
        test_command: whoami
```

如何运行：

在你的控制机上执行：`ansible-playbook -i /etc/ansible/hosts setup_gpu_node.yml`

这个Playbook会自动完成所有节点的标准化配置，为后续的K8s实验打下坚实的基础。

### Lab 1 - 基础环境：手把手教你安装昇腾驱动与CANN软件栈

本实验将指导你在一个配置有华为昇腾AI处理器的服务器上，完成最基础、也最关键的软件栈安装。

前提：

- 服务器已安装了受支持的操作系统（如EulerOS或指定的Ubuntu版本）。
- 你拥有服务器的root或sudo权限。
- 你已经从华为官网或镜像源下载了对应硬件型号和OS版本的`AIA-Ascend-Driver-*.run`和`AIA-Ascend-Toolkit-*.run`两个文件。

步骤：

Step 1: 检查硬件与环境

在安装前，确认系统能识别到昇腾设备。

```bash
lspci | grep -i ascend
```

你应该能看到类似`Processing accelerators: Huawei Technologies Co., Ltd. Ascend 910 AI Processor`的输出。

Step 2: 安装驱动

驱动是连接操作系统内核与NPU硬件的桥梁。

```bash
# 赋予执行权限
chmod +x AIA-Ascend-Driver-*.run

# 以root权限执行安装
sudo ./AIA-Ascend-Driver-*.run --install
```

安装过程中，请仔细阅读提示。安装完成后，通常需要重启服务器以加载新的内核模块。

```bash
sudo reboot
```

Step 3: 安装CANN工具包

CANN (Compute Architecture for Neural Networks) 是昇腾的应用使能软件栈，包含了编译器、加速库、工具链等。

```bash
# 赋予执行权限
chmod +x AIA-Ascend-Toolkit-*.run

# 以root权限执行安装
# --install-path指定安装路径，--install-for-all表示为所有用户安装
sudo ./AIA-Ascend-Toolkit-*.run --install --install-path=/usr/local/ascend --install-for-all
```

Step 4: 配置环境变量

为了让系统能够找到CANN的命令和库，需要将相关路径添加到环境变量中。CANN的安装包非常贴心地提供了一个脚本来做这件事。
将以下行添加到你的`~/.bashrc`或系统的`/etc/profile`中：

```bash
# 编辑.bashrc文件
vim ~/.bashrc

# 在文件末尾添加以下行 (路径请根据你的实际安装路径修改)
source /usr/local/ascend/ascend-toolkit/set_env.sh

# 使配置立即生效
source ~/.bashrc
```

Step 5: 验证安装

这是最关键的一步，验证我们的安装是否成功。

```bash
# 执行npu-smi命令
npu-smi info
```

如果安装成功，你将看到类似以下的输出，详细列出了服务器上所有NPU卡的信息，包括型号、ID、温度、功耗、HBM使用率等。

```text
+-------------------------------------------------------------------------------------------+
| npu-smi 21.0.2                  Version: 21.0.2                                           |
+-------------------------------+-----------------+-----------------------------------------+
| NPU     Name                  | Health          | Power(W)          Temp(C)               |
| Chip    Device-Chip-Id        | Bus-Id          | AICore(%)         HBM(MB)               |
+===============================+=================+=========================================+
| 0       Ascend 910B           | OK              | 110.0             45                    |
| 0       0-0                   | 0000:C1:00.0    | 0                 0 / 32768             |
+-------------------------------+-----------------+-----------------------------------------+
| 1       Ascend 910B           | OK              | 108.0             44                    |
| 0       1-0                   | 0000:C2:00.0    | 0                 0 / 32768             |
+-------------------------------+-----------------+-----------------------------------------+
... (其他NPU卡)
```

看到这个界面，恭喜你，你已经为昇腾AI处理器构建了最基础的软件运行环境！

### Lab 2 - 调度实战：在K8s中配置Volcano，并提交一个分布式PyTorch Job

本实验将带你实践第5章的核心内容：使用Volcano调度器来解决原生K8s无法处理的分布式训练死锁问题。

前提：

- 你有一个可用的K8s集群，并且集群中的节点已经配置好了GPU/NPU支持（即Device Plugin已安装）。
- 你已经安装了Helm客户端。

Step 1: 安装Volcano

使用Helm可以一键安装Volcano。

```bash
helm repo add volcano-sh https://volcano-sh.github.io/charts
helm repo update
helm install volcano volcano-sh/volcano -n volcano-system --create-namespace
```

验证安装：

```bash
kubectl get pods -n volcano-system
```

你应该能看到`volcano-scheduler`, `volcano-controller`等Pod处于`Running`状态。

Step 2: 准备PyTorch分布式训练应用

我们将使用一个经典的PyTorch MNIST分布式训练脚本。

文件 (`mnist_distributed.py`):

```python
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import datasets, transforms
import os

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

class Net(nn.Module):
    # ... (一个简单的CNN模型定义)
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = torch.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

def train(rank, world_size):
    setup(rank, world_size)
  
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
  
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset = datasets.MNIST('../data', train=True, download=True, transform=transform)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, sampler=train_sampler)

    model = Net().to(device)
    ddp_model = DDP(model, device_ids=[device])

    optimizer = optim.SGD(ddp_model.parameters(), lr=0.01)

    for epoch in range(3):
        train_sampler.set_epoch(epoch)
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % 10 == 0:
                print(f"Rank {rank}, Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item()}")
  
    cleanup()

if __name__ == '__main__':
    # Volcano会注入VC_TASK_INDEX和VC_WORKER_NUM环境变量
    rank = int(os.environ.get("VC_TASK_INDEX", "0"))
    world_size = int(os.environ.get("VC_WORKER_NUM", "1"))
    print(f"Starting training on Rank {rank} of {world_size}...")
    train(rank, world_size)
```

Step 3: 构建Docker镜像

文件 (`Dockerfile`):

```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

COPY mnist_distributed.py .

# 下载数据集，避免在运行时下载
RUN python -c "from torchvision import datasets; datasets.MNIST('../data', download=True)"

CMD ["python", "mnist_distributed.py"]
```

构建并推送到你的镜像仓库：

```bash
docker build -t your-registry/pytorch-dist-mnist:v1 .
docker push your-registry/pytorch-dist-mnist:v1
```

Step 4: 编写并提交VolcanoJob

这是核心步骤，我们定义一个需要2个Pod（每个Pod 1张GPU）的分布式作业。

文件 (`volcano_pytorch_job.yml`):

```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: pytorch-mnist-dist-job
spec:
  schedulerName: volcano
  minAvailable: 2 # 核心：Gang Scheduling，必须凑齐2个Pod才能开始
  queue: default
  tasks:
    - name: worker
      replicas: 2
      template:
        spec:
          containers:
            - name: pytorch-worker
              image: your-registry/pytorch-dist-mnist:v1
              resources:
                limits:
                  nvidia.com/gpu: 1 # 每个Pod申请1张GPU
          restartPolicy: OnFailure
```

提交作业：

```bash
kubectl apply -f volcano_pytorch_job.yml
```

Step 5: 观察与验证

```bash
# 查看PodGroup状态，Volcano调度的核心对象
kubectl get podgroup

# 查看Pod状态，你会看到两个Pod几乎是同时被创建并进入Running状态
kubectl get pods -l volcanosh.dev/job-name=pytorch-mnist-dist-job

# 查看其中一个Pod的日志
kubectl logs pytorch-mnist-dist-job-worker-0
```

在日志中，你将看到来自不同Rank（Rank 0和Rank 1）的训练日志交错打印，这表明分布式训练已成功建立并运行！

### Lab 3 - 压测实战：部署vLLM服务，使用脚本模拟100并发，生成性能报告

本实验将带你体验第8章的推理服务部署与压测。

前提：

- 一台或多台配备NVIDIA GPU的机器，已安装好驱动和Docker。
- Python环境已安装。

Step 1: 部署vLLM服务

我们将以最简单的方式，使用vLLM官方提供的Docker镜像来部署一个Llama 3 8B Instruct模型的服务。

```bash
# 拉取vLLM镜像
docker pull vllm/vllm-openai:latest

# 启动服务
# 注意：这需要较好的网络来下载模型，模型会被缓存到~/.cache/huggingface
docker run --gpus all -d \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --name vllm_server \
  vllm/vllm-openai:latest \
  --model meta-llama/Meta-Llama-3-8B-Instruct
```

注意： Llama 3模型需要Hugging Face的访问授权。请确保你已登录HF并接受了其使用协议。

Step 2: 验证服务

使用`curl`测试服务是否正常工作。

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "meta-llama/Meta-Llama-3-8B-Instruct",
        "messages": [
            {"role": "user", "content": "Hello! What is your name?"}
        ]
    }'
```

如果返回了模型的回答，则服务部署成功。

Step 3: 准备Locust压测脚本

安装Locust：`pip install locust`

文件 (`locustfile.py`): (参考8.3节提供的脚本，这里提供一个非流式的简化版，便于快速上手)

```python
from locust import task, HttpUser
import random

class LLMUser(HttpUser):
    @task
    def generate(self):
        prompts = ["What is the capital of France?", "Write a short poem about the sea.", "Explain black holes to a 5-year-old."]
        payload = {
            "model": "meta-llama/Meta-Llama-3-8B-Instruct",
            "messages": [{"role": "user", "content": random.choice(prompts)}],
            "max_tokens": 100
        }
        self.client.post("/v1/chat/completions", json=payload, name="/v1/chat/completions")
```

Step 4: 启动压测

```bash
locust -f locustfile.py --host http://localhost:8000
```

Step 5: 分析性能报告

1. 打开浏览器访问 `http://localhost:8089`。
2. 输入并发用户数（Total users）为 `100`，增长速率（Spawn rate）为 `10`。
3. 点击 "Start swarming"。
4. 观察 "Charts" 标签页：
    - Total Requests per Second (RPS): 你的服务每秒能完成多少次请求。
    - Response Time (ms): 响应时间的分布，重点关注95%和99%百分位值。这代表了绝大多数用户的体验。
5. 生成报告： 在 "Download Data" 标签页，你可以下载到详细的CSV格式报告，用于离线分析和存档。

通过调整并发用户数，你可以找到服务的“性能拐点”——即在哪个并发水平上，响应时间开始急剧恶化。这为你进行容量规划提供了关键数据。

### Lab 4 - 监控告警：配置Prometheus规则，当GPU温度>80℃或显存使用率>95%时触发告警

本实验将带你实践第9章的可观测性内容，为你的GPU集群配置核心的硬件告警。

前提：

- 一个K8s集群，已部署[Prometheus Operator](https://prometheus-operator.dev/)（通常通过kube-prometheus-stack Helm chart安装）。
- 集群的GPU节点上已部署了DCGM-Exporter。

Step 1: 理解PrometheusRule CRD

在Prometheus Operator生态中，告警规则是通过一个名为`PrometheusRule`的Kubernetes自定义资源来定义的。我们将创建一个YAML文件来定义我们的规则。

Step 2: 编写告警规则YAML文件

文件 (`gpu-alerts.yml`):

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: gpu-alerts
  labels:
    # 这个label需要与你的Prometheus实例的ruleSelector匹配
    prometheus: kube-prometheus 
    role: alert-rules
spec:
  groups:
    - name: gpu.rules
      rules:
        - alert: GPUTemperatureHigh
          expr: DCGM_FI_DEV_GPU_TEMP > 80
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "GPU high temperature on {{ $labels.nodename }}"
            description: "GPU {{ $labels.gpu }} on node {{ $labels.nodename }} has been over 80°C for 5 minutes. Current value is {{ $value }}°C."

        - alert: GPUMemoryHigh
          expr: (DCGM_FI_DEV_FB_USED / DCGM_FI_DEV_FB_TOTAL) * 100 > 95
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "GPU high memory usage on {{ $labels.nodename }}"
            description: "GPU {{ $labels.gpu }} on node {{ $labels.nodename }} memory usage has been over 95% for 10 minutes. Current usage is {{ $value | printf `%.2f` }}%."

        - alert: GPUXidErrorDetected
          expr: rate(DCGM_FI_DEV_XID_ERRORS[5m]) > 0
          for: 1m
          labels:
            severity: critical
          annotations:
            summary: "GPU XID Error Detected on {{ $labels.nodename }}"
            description: "GPU {{ $labels.gpu }} on node {{ $labels.nodename }} is reporting XID errors. This may indicate a software or hardware issue. Please investigate dmesg logs."
```

注解：

- `expr`: PromQL查询表达式，是告警的触发条件。
- `for`: 条件需要持续为真的时间，防止瞬时抖动引发的误报。
- `labels.severity`: 定义告警级别，便于在Alertmanager中进行不同的路由。
- `annotations`: 定义告警的详细信息，`{{ $labels... }}`和`{{ $value }}`是模板变量，会在告警通知中被替换为实际值。

Step 3: 应用规则并验证

```bash
# 应用规则到你的K8s集群 (通常在monitoring命名空间)
kubectl apply -f gpu-alerts.yml -n monitoring
```

几分钟后，Prometheus会加载这些新规则。

1. 打开Prometheus UI（通过`kubectl port-forward`）。
2. 导航到 "Alerts" 页面，你应该能看到新添加的`GPUTemperatureHigh`, `GPUMemoryHigh`, `GPUXidErrorDetected`三条规则，状态为`Inactive`。

Step 4: 模拟触发告警（可选但推荐）

为了验证告警通路是通的，我们需要人为地触发条件。

- 触发温度告警： 在一个GPU节点上，运行一个高强度的GPU压力测试工具，如`gpu-burn`。

```bash
# 在GPU节点上运行
docker run --rm --gpus all -it wshuyi/gpu-burn -t 600 # 运行10分钟
```

同时在另一个终端观察`nvidia-smi`，当温度超过80℃并持续5分钟后，Prometheus中的告警状态会变为`Pending`，然后是`Firing`。

- 触发显存告警： 编写一个简单的PyTorch脚本，分配一个巨大的Tensor。

Step 5: 查看告警

当告警`Firing`时，如果你配置了Alertmanager，你将在配置好的通知渠道（如Slack、钉钉）中收到一条格式化好的告警信息，其中包含了你在`annotations`中定义的详细内容。

至此，你已经成功地为你的AI基础设施安装了最基础的“自动报警器”！

好的，我们来完成最后一个附录——效率工具箱。这个附录将提供一些立即可用的、高度实用的工具，它们是前面章节中复杂公式和排查逻辑的浓缩与自动化。这些工具旨在将你从重复性的计算和检查中解放出来，让你能更专注于架构设计和战略规划。

## 附录B：效率工具箱

在AI基础设施的宏大工程中，我们不仅需要有体系化的知识和深刻的洞察力，还需要一套能将这些智慧快速转化为行动的“利器”。正如优秀的程序员不会重复造轮子，资深的AI Infra工程师也应该善于将重复性的工作自动化、工具化。

本附录为你提供了一个“效率工具箱”，其中包含了两件我们精心打造的工具。第一件是《智算中心算力资源规划计算器》，它将第七章中复杂的数学公式封装成一个简单易用的Excel表格，让你能在几秒钟内完成一个价值数亿元的集群规划估算。第二件是Python运维脚本库，它提供了一系列即插即用的脚本，能帮你一键完成集群的健康巡检、算力需求的快速计算等日常任务。

这些工具是你从“理论家”变为“实干家”的加速器。请将它们收藏到你的“武器库”中，并在日常工作中不断地打磨和扩展它们，使其成为你个人知识体系中最锋利、最得心应手的一部分。

### 《智算中心算力资源规划计算器.xlsx》

这个Excel计算器旨在将第七章中关于显存占用和训练时间的估算模型，转化为一个交互式的、可视化的规划工具。你只需要在“输入”区域填入模型的关键参数和你的业务目标，它就能自动在“输出”区域为你计算出所需的资源规模和相关的性能指标。

#### 计算器结构设计

我们将这个Excel文件设计为包含两个主要的工作表（Sheet）：`Training_Estimator`（训练资源估算器）和`Inference_Memory_Estimator`（推理显存估算器）。

#### Sheet 1: Training_Estimator (训练资源估算器)

这个工作表是整个计算器的核心，用于预训练或全量微调的资源规划。

【输入区 (Input Area)】 - 黄色背景单元格

A. 模型参数 (Model Parameters)

| 参数名 | 符号 | 示例值 | 单位 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 模型参数量 | P | 70 | B (十亿) | 如Llama 2 70B，填70 |
| 模型层数 | L | 80 | | |
| 隐藏层维度 | h | 8192 | | |
| 注意力头数 | a | 64 | | |
| 序列长度 | s | 4096 | Tokens | |

B. 数据与目标 (Data & Goals)

| 参数名 | 符号 | 示例值 | 单位 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| 训练数据量 | D | 2 | T (万亿) | Tokens |
| 目标训练天数 | Days | 90 | 天 | |

C. 硬件与并行策略 (Hardware & Parallelism)

| 参数名 | 符号 | 示例值 | 单位 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| GPU型号 | - | A100-80G | | (下拉菜单选择) |
| 单卡理论算力 | Peak | 312 | TFLOPS (FP16) | (根据GPU型号自动填充) |
| 单卡显存容量 | Mem_GPU| 80 | GB | (根据GPU型号自动填充) |
| 数据并行路数 | DP | 64 | | |
| 张量并行路数 | TP | 8 | | |
| 流水线并行路数 | PP | 8 | | |
| 全局批次大小 | GBS | 1024 | | 全局Batch Size |
| MFU (算力利用率)| MFU | 40% | % | 最关键的经验值 |

【输出区 (Output Area)】 - 绿色背景单元格

A. 训练时间与算力需求 (Time & Compute Requirements)

| 指标名 | 公式 (Excel表达式) | 结果 | 单位 |
| :--- | :--- | :--- | :--- |
| 总计算量 (FLOPs) | `=6 * P * 10^9 * D * 10^12` | 8.40E+23 | FLOPs |
| 所需总有效算力 | `=A2 / (Days * 24 * 3600)` | 1.08E+17 | FLOPS |
| | `=B2 / 10^12` | 107,527 | TFLOPS (Effective) |
| 所需总GPU卡数 | `=B3 / (Peak * MFU)` | 862 | 张 |
| 建议集群规模 | `=ROUNDUP(B4, -2)` | 900 | 张 |
| 实际预估训练天数 | `=(A2 / (B5 * Peak * MFU * 10^12)) / (24*3600)`| 86.4 | 天 |

B. 显存占用分析 (Memory Analysis) - *基于ZeRO-1 + TP + PP*

| 指标名 | 公式 (Excel表达式) | 结果 | 单位 |
| :--- | :--- | :--- | :--- |
| 模型参数占用 (单卡) | `=2 * P / TP` | 17.5 | GB |
| 优化器状态 (单卡) | `=12 * P / (DP * TP)` | 1.64 | GB |
| 梯度占用 (单卡) | `=4 * P / (DP * TP)` | 0.55 | GB |
| 激活值占用 (单卡) [^1] | `=(34 * s * (GBS/DP) * h * L) / (TP * 10^9)`| 19.3 | GB |
| 预估单卡显存峰值 | `=SUM(B8:B11)` | 39.0 | GB |
| 显存是否满足？ | `=IF(B12 <= Mem_GPU, "✅ 满足", "❌ 超出！")` | ✅ 满足 | |

C. 网络带宽需求 (Network Bandwidth Requirement)

| 指标名 | 公式 (Excel表达式) | 结果 | 单位 |
| :--- | :--- | :--- | :--- |
| All-Reduce带宽 (DP) [^2] | `= (2 * P * 10^9 * 2) / (TP * (DP-1)/DP)` | 8.75 | GB |
| All-Gather/Reduce-Scatter带宽 (TP) | `= (2 * P * 10^9 * 2 * (TP-1)/TP) / (TP * PP)` | 0.05 | GB |
| P2P带宽 (PP) | `=(s * (GBS/DP) * h * 2 * 2) / (TP * PP)` | 0.01 | GB |
| 单卡聚合带宽需求 [^3] | `= (B14+B15+B16) * 8 / 10^9 * (1/ (A2/(B5*Peak*MFU*10^12)) ) `| ~21 | Gbps |

[^1]: 激活值公式为简化估算，实际与激活重计算等技术强相关。
[^2]: 网络带宽估算非常复杂，这里提供的是一个基于单次迭代通信量的粗略模型，实际需求受通信计算重叠度影响巨大。
[^3]: 表示单次迭代总通信量除以单次迭代时间，估算所需的平均带宽。

使用说明：

1. 首先在【输入区】填入你的模型、数据和目标参数。
2. 调整【硬件与并行策略】中的`DP`, `TP`, `PP`参数。
3. 观察【输出区】的“所需总GPU卡数”和“显存是否满足？”。
4. 你的目标是：在“显存满足”的前提下，找到一组`DP, TP, PP`组合，使得“建议集群规模”在你的预算范围内，且“实际预估训练天数”满足你的时间要求。
5. 不断调整`DP`, `TP`, `PP`和`MFU`（如果你对你的集群优化有信心，可以调高MFU），进行“What-If”分析，找到最佳的资源配置方案。

### Python运维脚本库

这个脚本库旨在提供一系列开箱即用的运维工具，帮你自动化日常的巡检和计算任务。

#### `check_cluster_health.py`：一键集群健康巡检

这个脚本通过调用Kubernetes Python客户端，获取所有节点，然后SSH到每个节点上执行`nvidia-smi`或`npu-smi`命令，并解析其输出，最后生成一份简洁的健康报告。

前提：

- 安装Python库：`pip install kubernetes paramiko pandas`
- 你的机器上配置好了`~/.kube/config`，可以访问目标K8s集群。
- 你的机器可以SSH免密登录到集群的所有节点。

脚本 (`check_cluster_health.py`):

```python
import argparse
import paramiko
import pandas as pd
from kubernetes import client, config
from datetime import datetime

def ssh_run_command(hostname, command):
    """通过SSH在远程节点上执行命令并返回输出。"""
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(hostname, username='root', timeout=5) # 请根据实际情况修改用户名
        stdin, stdout, stderr = ssh.exec_command(command)
        output = stdout.read().decode('utf-8')
        error = stderr.read().decode('utf-8')
        ssh.close()
        if error and "command not found" not in error:
            return f"ERROR: {error}"
        return output
    except Exception as e:
        return f"SSH_ERROR: {str(e)}"

def parse_nvidia_smi(output):
    """解析nvidia-smi的输出，提取关键健康指标。"""
    if "NVIDIA-SMI has failed" in output or "ERROR" in output:
        return [{"gpu_id": "ALL", "health": "FAIL", "message": output}]
  
    devices = []
    lines = output.split('\n')
    # 非常简化的解析，实际生产中建议使用nvidia-smi --query-gpu=... --format=csv
    for i, line in enumerate(lines):
        if "MiB" in line and "%" in line:
            parts = line.split()
            gpu_id = parts[1]
            temp = parts[3]
            power = parts[5]
            mem_used = parts[9]
            mem_total = parts[11]
            gpu_util = parts[13]
          
            health = "OK"
            message = []
            if int(temp.replace('C','')) > 85:
                health = "WARN"
                message.append(f"Temp>85C({temp})")
            if int(power.replace('W','')) > 350: # 假设功耗阈值为350W
                health = "WARN"
                message.append(f"Power>350W({power})")

            devices.append({
                "gpu_id": gpu_id,
                "health": health if message else "OK",
                "message": ", ".join(message) if message else "N/A"
            })
    return devices

def get_k8s_nodes(label_selector=""):
    """获取K8s集群中符合标签选择器的节点列表。"""
    config.load_kube_config()
    v1 = client.CoreV1Api()
    nodes = v1.list_node(label_selector=label_selector)
    return [node.status.addresses[0].address for node in nodes.items]

def main():
    parser = argparse.ArgumentParser(description="Cluster Health Checker")
    parser.add_argument("--vendor", type=str, default="nvidia", choices=["nvidia", "ascend"], help="GPU/NPU vendor")
    parser.add_argument("--nodes", type=str, help="Comma-separated list of node IPs. Overrides k8s discovery.")
    parser.add_argument("--label", type=str, default="nvidia.com/gpu=true", help="K8s node label selector for discovery.")
    args = parser.parse_args()

    if args.nodes:
        nodes_to_check = args.nodes.split(',')
    else:
        print(f"Discovering nodes with label '{args.label}' from Kubernetes...")
        nodes_to_check = get_k8s_nodes(args.label)
        print(f"Found {len(nodes_to_check)} nodes: {nodes_to_check}")

    if args.vendor == "nvidia":
        command = "nvidia-smi"
        parser_func = parse_nvidia_smi
    elif args.vendor == "ascend":
        command = "npu-smi info"
        # 你需要为npu-smi编写一个类似的解析函数
        # parser_func = parse_npu_smi 
        print("Ascend NPU parser not implemented in this example.")
        return

    results = []
    for node in nodes_to_check:
        print(f"Checking node: {node} ...")
        output = ssh_run_command(node, command)
        health_info = parser_func(output)
        for device in health_info:
            results.append({
                "node": node,
                "device_type": args.vendor.upper(),
                "device_id": device["gpu_id"],
                "health": device["health"],
                "message": device["message"],
            })
  
    report_df = pd.DataFrame(results)
  
    print("\n" + "="*50)
    print(f"  Cluster Health Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*50)
    print(report_df.to_string())

    # 打印问题摘要
    failed_nodes = report_df[report_df['health'] != 'OK']
    if not failed_nodes.empty:
        print("\n" + "!"*20 + "  PROBLEM SUMMARY  " + "!"*20)
        print(failed_nodes.to_string())
        print("!"*59)
    else:
        print("\n" + "✅"*10 + "  All checks passed! Cluster is healthy. " + "✅"*10)

if __name__ == "__main__":
    main()
```

如何使用：

1. 巡检所有带`nvidia.com/gpu=true`标签的K8s节点：`python check_cluster_health.py`
1. 指定节点进行巡检：`python check_cluster_health.py --nodes 192.168.1.101,192.168.1.102`

这个脚本会输出一个清晰的表格，告诉你每个节点的每张卡是否健康，如果不健康，原因可能是什么（如高温）。这对于日常巡检和故障初步排查非常有用。

#### `calc_model_flops.py`：快速计算模型理论算力需求

这个脚本将第七章的`6PD`公式封装成一个简单的命令行工具，帮你快速估算训练一个模型所需的总计算量和在特定集群上所需的时间。

脚本 (`calc_model_flops.py`):

```python
import argparse

def sizeof_fmt(num, suffix=""):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(num) < 1000.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1000.0
    return f"{num:.1f}Y{suffix}"

def main():
    parser = argparse.ArgumentParser(description="Estimate LLM Training FLOPs and Time")
  
    # Model and Data parameters
    parser.add_argument("-p", "--params", type=float, required=True, help="Model parameters in billions (e.g., 70 for 70B)")
    parser.add_argument("-d", "--data_tokens", type=float, required=True, help="Training data size in trillions of tokens (e.g., 2 for 2T)")
  
    # Hardware and Performance parameters
    parser.add_argument("-n", "--num_gpus", type=int, required=True, help="Number of GPUs in the cluster")
    parser.add_argument("--peak_tflops", type=float, required=True, help="Peak TFLOPS of a single GPU at target precision (e.g., 312 for A100 FP16)")
    parser.add_argument("--mfu", type=float, default=0.4, help="Model FLOPs Utilization (MFU), a value between 0 and 1 (default: 0.4)")
  
    args = parser.parse_args()

    # --- Calculations ---
  
    # 1. Total FLOPs
    total_flops = 6 * (args.params * 1e9) * (args.data_tokens * 1e12)
  
    # 2. Cluster effective throughput
    single_gpu_effective_tflops = args.peak_tflops * args.mfu
    cluster_effective_tflops = single_gpu_effective_tflops * args.num_gpus
  
    # 3. Estimated training time
    if cluster_effective_tflops == 0:
        estimated_seconds = float('inf')
    else:
        estimated_seconds = total_flops / (cluster_effective_tflops * 1e12)
  
    estimated_days = estimated_seconds / (24 * 3600)

    # --- Print Report ---
  
    print("\n" + "="*50)
    print("  LLM Training Estimation Report")
    print("="*50)
  
    print("\n[Input Parameters]")
    print(f"  - Model Size: {args.params}B parameters")
    print(f"  - Dataset Size: {args.data_tokens}T tokens")
    print(f"  - Cluster Size: {args.num_gpus} GPUs")
    print(f"  - Single GPU Peak Performance: {args.peak_tflops} TFLOPS")
    print(f"  - Assumed MFU: {args.mfu:.2%}")
  
    print("\n[Estimated Requirements]")
    print(f"  - Total Training FLOPs: {sizeof_fmt(total_flops, 'FLOPs')}")
    print(f"  - Cluster Effective Throughput: {cluster_effective_tflops:,.2f} TFLOPS")
  
    print("\n[Final Estimation]")
    print(f"  - Estimated Training Time: {estimated_days:.2f} days")
    print("="*50)
  
    print("\nNote: This is a theoretical estimation. Actual time may vary based on network, storage, and software stack efficiency.")

if __name__ == "__main__":
    main()
```

如何使用：

假设你想估算用一个1024张A100 (FP16峰值312 TFLOPS)的集群，训练一个70B模型、2T Tokens数据，预估MFU为40%所需的时间：

```bash
python calc_model_flops.py \
  --params 70 \
  --data_tokens 2 \
  --num_gpus 1024 \
  --peak_tflops 312 \
  --mfu 0.4
```

输出：

```text
==================================================
  LLM Training Estimation Report
==================================================

[Input Parameters]
  - Model Size: 70.0B parameters
  - Dataset Size: 2.0T tokens
  - Cluster Size: 1024 GPUs
  - Single GPU Peak Performance: 312.0 TFLOPS
  - Assumed MFU: 40.00%

[Estimated Requirements]
  - Total Training FLOPs: 840.0EFLOPs
  - Cluster Effective Throughput: 127,795.20 TFLOPS

[Final Estimation]
  - Estimated Training Time: 76.08 days
==================================================

Note: This is a theoretical estimation. Actual time may vary based on network, storage, and software stack efficiency.
```

这个脚本为你提供了一个快速、便捷的方式，来验证你在Excel计算器中规划的方案，或者在日常讨论中快速给出算力需求的数量级估算。

### 结语

这个效率工具箱，是你将理论知识转化为生产力的催化剂。它们并不完美，但它们提供了一个坚实的起点。真正的价值，在于你根据自己团队的特定需求，不断地去完善、扩展和定制这些工具。让它们和你一起成长，成为你作为一名顶尖AI Infra工程师，不可或-或缺的“左膀右臂”。
