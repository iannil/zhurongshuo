---
title: "第四章：工程师的必修课：Linux、Docker与性能监控"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第四章：工程师的必修课：Linux、Docker与性能监控"]
slug: "chapter-04"
---

在前三章中，我们已经走过了一段激动人心的旅程：从精通Python编程，到驾驭数据科学工具链，再到将AI模型构建成一个功能完备的在线API服务。至此，我们似乎已经拥有了一个可以工作的“产品原型”。然而，在真实的工业环境中，一个能在你的开发机（通常是Windows或macOS图形界面）上通过`python app.py`运行起来的应用，与一个能够在生产服务器上7x24小时稳定运行、服务成千上万用户的产品之间，还存在着巨大的鸿沟。

弥合这道鸿沟，需要我们掌握一套全新的技能，我们称之为AI工程师的“运维之道”。这并非传统意义上运维工程师的全部职责，而是指AI开发者为了确保其应用能够被顺利部署、高效运行和轻松维护所必须具备的底层系统能力。这套能力，是AI工程化中连接“开发”（Development）与“运维”（Operations）的桥梁，也是DevOps理念在AI领域的具体体现。

本章，我们将深入这片看似与“算法”无关，却对算法落地至关重要的领域。我们将学习：

- Linux高效工作流：生产服务器几乎无一例外地运行在Linux系统上。我们将告别图形界面，学习如何在纯命令行的“黑客帝国”中高效穿行。掌握Shell脚本和常用命令，将让你能够自动化处理繁琐任务，如同拥有了“魔法咒语”。
- 虚拟环境深度解析：AI项目往往依赖于众多特定版本的库。我们将深入理解为何需要环境隔离，并对比`venv`和`Conda`这两种主流虚拟环境管理工具的优劣与适用场景，彻底告别“依赖地狱”。
- Docker容器化：Docker是近年来软件开发领域最具革命性的技术之一。它能将我们的AI应用及其所有依赖（代码、库、系统工具、配置文件）打包成一个轻量、可移植的“集装箱”。我们将学习如何使用Docker来解决“在我机器上能跑”这一经典难题，实现一次构建、处处运行。
- 系统性能分析：当你的模型推理变慢，或者服务器负载飙高时，你不能束手无策。我们将学习使用`top`、`htop`、`nvidia-smi`等工具，像一名侦探一样去监控系统资源，分析CPU、内存、GPU的瓶颈所在，为性能优化提供数据支撑。

最后，我们将通过一个实战项目——将上一章的Flask异步图像分类应用Docker化，并实现一键部署——将本章所有技能融会贯通。我们将编写`Dockerfile`，构建镜像，并通过`docker-compose`一键启动包含Flask应用、Celery Worker和Redis在内的整个服务栈。

掌握本章内容，你将获得一种“掌控全局”的能力。你将不再仅仅是一个算法的实现者，而是一个能够端到端地交付、部署和维护健壮AI系统的全能工程师。这套“内功”将极大地提升你的工程成熟度和解决实际问题的能力，让你在团队中脱颖而出。现在，让我们开启这场通往系统底层的探索之旅。

## 4.1 Linux高效工作流：Shell脚本与常用命令精通

对于许多习惯了图形用户界面（GUI）的开发者来说，初次接触纯命令行的Linux服务器可能会感到一丝畏惧。然而，命令行界面（CLI）一旦熟练，其效率和能力远非GUI所能比拟。

### 4.1.1 基础中的基础：文件系统导航与操作

- `pwd` (Print Working Directory)：显示你当前所在的目录路径。
- `ls` (List)：列出当前目录下的文件和文件夹。
  - `ls -l`：以长格式显示，包含权限、所有者、大小、修改日期等详细信息。
  - `ls -a`：显示所有文件，包括以`.`开头的隐藏文件（如`.bashrc`）。
  - `ls -lh`：`-h`表示`human-readable`，以K, M, G等单位显示文件大小，更易读。
- `cd` (Change Directory)：切换目录。
  - `cd /path/to/directory`：切换到指定绝对路径。
  - `cd relative/path`：切换到相对路径。
  - `cd ..`：切换到上一级目录。
  - `cd ~` 或 `cd`：切换到当前用户的家目录。
  - `cd -`：切换到上一次所在的目录。
- `mkdir` (Make Directory)：创建新目录。
  - `mkdir my_project`
  - `mkdir -p a/b/c`：`-p`表示`parents`，递归创建多级目录。
- `touch` (Touch)：创建一个空文件，或更新一个已存在文件的时间戳。
  - `touch new_file.txt`
- `cp` (Copy)：复制文件或目录。
  - `cp source.txt destination.txt`
  - `cp -r source_dir/ destination_dir/`：`-r`表示`recursive`，用于复制目录。
- `mv` (Move)：移动或重命名文件/目录。
  - `mv old_name.txt new_name.txt` (重命名)
  - `mv file.txt target_dir/` (移动)
- `rm` (Remove)：删除文件或目录。这是一个危险的命令，没有回收站！
  - `rm file.txt`
  - `rm -r directory/`：删除目录。
  - `rm -rf directory/`：`-f`表示`force`，强制删除，不进行任何提示。使用前请三思！

### 4.1.2 文本处理三剑客：`grep`, `sed`, `awk`

在处理日志文件、数据文件时，这三个命令是无价之宝。

- `grep` (Global Regular Expression Print)：强大的文本搜索工具。
  - `grep "error" server.log`：在`server.log`中搜索包含"error"的行。
  - `grep -i "error"`：`-i`表示忽略大小写。
  - `grep -r "my_function" ./src`：`-r`表示在`src`目录及其子目录中递归搜索。
  - `grep -v "debug"`：`-v`表示反向匹配，显示不包含"debug"的行。
  - `grep -E "^[0-9]{3}"`：`-E`表示使用扩展正则表达式。
- `sed` (Stream Editor)：流编辑器，用于对文本进行替换、删除、插入等操作。
  - `sed 's/old_string/new_string/g' file.txt`：将`file.txt`中所有的`old_string`替换为`new_string`。`g`表示全局替换。
  - `sed '/^#/d' config.conf`：删除`config.conf`中所有以`#`开头的注释行。
- `awk`：一个强大的文本分析工具，它将每一行视为一条记录，按字段（默认以空格分隔）进行处理。
  - `ls -l | awk '{print $9, $5}'`：打印`ls -l`输出的第9列（文件名）和第5列（文件大小）。
  - `cat access.log | awk '$9 == "404" {print $7}'`：在`access.log`中，打印所有HTTP状态码（第9个字段）为404的请求路径（第7个字段）。

### 4.1.3 管道（`|`）与重定向（`>`, `>>`）

这是命令行强大组合能力的核心。

管道 `|`：将前一个命令的标准输出（stdout）作为后一个命令的标准输入（stdin）。

`cat server.log | grep "ERROR" | wc -l`：这个命令链做了三件事：1. `cat`读取日志文件内容并输出到stdout；2. `grep`从stdin接收内容，筛选出含"ERROR"的行并输出到stdout；3. `wc -l`从stdin接收内容，统计行数。最终得到错误日志的总行数。

输出重定向 `>` 和 `>>`：

`ls -l > file_list.txt`：将`ls -l`的输出写入`file_list.txt`，会覆盖文件原有内容。

`echo "New log entry" >> server.log`：将字符串追加到`server.log`的末尾，不会覆盖。

输入重定向 `<`：

`wc -l < file.txt`：将`file.txt`的内容作为`wc -l`的输入。

### 4.1.4 Shell脚本：自动化你的工作流

Shell脚本就是将一系列Linux命令按顺序写入一个文本文件，然后让系统执行它。它是实现自动化运维、部署、数据处理的基石。

一个简单的备份脚本 `backup.sh`：

```bash
#!/bin/bash

# 这是一个shebang，告诉系统使用/bin/bash来解释这个脚本

# 定义变量
SOURCE_DIR="/path/to/my_project/src"
BACKUP_DIR="/path/to/backups"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
BACKUP_FILENAME="backup_${TIMESTAMP}.tar.gz"

# 打印日志
echo "开始备份..."
echo "源目录: ${SOURCE_DIR}"
echo "目标文件: ${BACKUP_DIR}/${BACKUP_FILENAME}"

# 使用tar命令创建压缩包
# -c: create, -z: gzip, -v: verbose, -f: file
tar -czvf "${BACKUP_DIR}/${BACKUP_FILENAME}" "${SOURCE_DIR}"

# 检查上一个命令是否成功
if [ $? -eq 0 ]; then
    echo "备份成功！"
else
    echo "备份失败！"
    exit 1
fi

# 删除7天前的旧备份
find "${BACKUP_DIR}" -name "backup_*.tar.gz" -mtime +7 -exec rm {} \;
echo "已清理7天前的旧备份。"

echo "所有操作完成。"
```

如何运行？

1. 赋予执行权限：`chmod +x backup.sh`
2. 执行脚本：`./backup.sh`

在AI项目中，Shell脚本常用于：

- 自动化部署：拉取最新代码、安装依赖、重启服务。
- 数据预处理：批量下载数据、解压、调用Python脚本进行处理。
- 定时任务（Cron Job）：设置定时任务，例如每天凌晨执行模型再训练或数据备份脚本。

### 4.1.5 其他高频命令

- `ssh` (Secure Shell)：远程登录到另一台Linux服务器。`ssh user@hostname`。
- `scp` (Secure Copy)：在本地和远程服务器之间安全地复制文件。`scp local_file.txt user@hostname:/remote/path/`。
- `find`：按条件查找文件。`find . -name "*.py"`。
- `tar`：打包和解包文件。`tar -czvf archive.tar.gz directory/` (打包)，`tar -xzvf archive.tar.gz` (解包)。
- `curl` / `wget`：从网络下载文件或测试API。
- `htop` / `top`：实时监控系统进程和资源使用（详见4.4节）。
- `df -h`：查看磁盘空间使用情况。
- `du -sh *`：查看当前目录下各文件/文件夹的大小。
- `tail -f logfile.log`：实时跟踪日志文件的最新输出。

## 4.2 虚拟环境深度解析：从venv到Conda

### 4.2.1 为什么需要虚拟环境？—— “依赖地狱”

想象一个场景：

项目A，是一个老项目，依赖于`TensorFlow 1.15`和`Python 3.6`。

项目B，是你正在开发的新项目，需要`TensorFlow 2.8`和`Python 3.9`。

如果你在系统的全局Python环境中，通过`pip install`来安装这些库，会发生什么？当你为项目B安装`TensorFlow 2.8`时，它会覆盖掉项目A所依赖的`TensorFlow 1.15`，导致项目A无法运行。这就是典型的“依赖地狱”。

虚拟环境就是为了解决这个问题而生的。它会为每个项目创建一个独立的、隔离的Python环境。在这个环境中，你可以安装任意版本的库，而不会影响到全局环境或其他项目。

### 4.2.2 `venv`：Python官方的轻量级选择

`venv`是自Python 3.3起内置于标准库的虚拟环境管理工具。它轻量、简单，是纯Python项目的首选。

工作流程：

1. 创建环境：在一个项目目录下，运行：

    ```bash
    python3 -m venv venv
    ```

    这会创建一个名为`venv`的文件夹，里面包含了Python解释器的副本和标准库。

2. 激活环境：

    - 在Linux/macOS上：`source venv/bin/activate`
    - 在Windows上：`.\venv\Scripts\activate`
    - 激活后，你的命令行提示符前面会出现`(venv)`字样，表示你现在处于这个虚拟环境中。此时，你使用的`python`和`pip`命令都是指向`venv`文件夹内的版本。

3. 安装依赖：

    ```bash
    pip install flask numpy pandas
    ```

    这些库会被安装到`venv/lib/pythonX.X/site-packages/`目录下，而不是全局环境。

4. 生成依赖列表：

    ```bash
    pip freeze > requirements.txt
    ```

    这会将当前环境中所有已安装的库及其版本号记录到`requirements.txt`文件中。这个文件是项目可复现性的关键。

5. 在另一台机器上复现环境：

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

6. 退出环境：

    ```bash
    deactivate
    ```

`venv`的优点：内置、轻量、标准。

`venv`的缺点：它只能隔离Python包，无法管理Python解释器本身的版本，也无法管理非Python的依赖（如CUDA、cuDNN）。

### 4.2.3 `Conda`：AI与数据科学领域的全能环境管理器

`Conda`是一个开源的、跨平台的包管理和环境管理系统。它最初是为Anaconda发行版创建的，但现在可以独立安装（Miniconda）。对于AI和数据科学项目，`Conda`通常是比`venv`更好的选择。

`Conda`的核心优势：

1. 管理Python版本：`Conda`可以轻松地创建和切换不同版本的Python环境。

    ```bash
    conda create --name tf1_env python=3.6
    conda create --name torch_env python=3.9
    ```

2. 管理非Python包：这是`Conda`的“杀手级”特性。它可以安装和管理C/C++库、CUDA工具包、cuDNN、MKL等AI项目强依赖的底层库。

    ```bash
    # 创建一个环境，同时安装指定版本的Python, PyTorch, 和 CUDA
    conda create --name my_gpu_env python=3.9 pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```

    这一个命令就解决了复杂的GPU环境配置问题，极大地简化了环境搭建。

3. 更强大的依赖解析：`Conda`在安装包时，会进行更复杂的依赖关系求解，以确保所有包之间的兼容性，减少冲突。

工作流程：

1. 创建环境：`conda create --name myenv python=3.9`
2. 激活环境：`conda activate myenv`
3. 安装依赖：`conda install numpy pandas matplotlib scikit-learn`
4. 生成依赖列表：

    ```bash
    conda env export > environment.yml
    ```

    `environment.yml`文件比`requirements.txt`更强大，它记录了环境名、所有包（包括Python和非Python包）及其版本，以及包的来源渠道（channel）。

5. 复现环境：

    ```bash
    conda env create -f environment.yml
    ```

6. 退出环境：`conda deactivate`
7. 查看/删除环境：`conda env list`, `conda env remove --name myenv`

`venv` vs `Conda`，如何选择？

纯Python Web后端、工具脚本等：使用`venv`，因为它更轻量、更标准。

数据科学、机器学习、深度学习项目：强烈推荐使用`Conda`，因为它能完美地处理复杂的非Python依赖，特别是GPU相关的库。

## 4.3 Docker容器化：打包、部署与隔离你的AI应用


### 4.3.1 虚拟机 vs 容器：理解Docker的革命性

在Docker出现之前，如果想隔离应用，我们通常使用虚拟机（Virtual Machine, VM）。VM会在宿主操作系统（Host OS）之上，通过Hypervisor虚拟化一整套硬件（CPU, 内存, 硬盘），然后再安装一个完整的客户操作系统（Guest OS），最后在Guest OS里运行我们的应用。这种方式隔离性极好，但缺点是笨重、资源开销大、启动慢。

Docker容器（Container）则是一种更轻量级的虚拟化技术。它不虚拟化硬件和操作系统内核，而是直接共享宿主机的内核。容器内只打包了应用本身和它所需要的库、二进制文件。这使得容器极其轻量、资源占用小、启动速度极快（秒级甚至毫秒级）。

核心概念：

- 镜像（Image）：一个只读的模板，包含了运行应用所需的一切：代码、运行时、库、环境变量和配置文件。镜像是分层的，可以基于其他镜像构建（例如，基于官方的Python 3.9镜像）。
- 容器（Container）：镜像的一个可运行实例。你可以从同一个镜像启动任意多个容器，它们之间相互隔离。
- Dockerfile：一个文本文件，里面包含了一系列指令，用于告诉Docker如何一步步地构建一个镜像。
- 仓库（Repository）：用于存放和分发镜像的地方，最著名的是Docker Hub。

### 4.3.2 `Dockerfile`：为你的AI应用制作“蓝图”

`Dockerfile`是Docker的核心。让我们为上一章的鸢尾花分类Flask应用编写一个`Dockerfile`。

项目结构：

```text
simple_flask_app/
├── app.py
├── iris_model.pkl
├── requirements.txt
└── Dockerfile
```

`requirements.txt`内容：

```text
Flask==2.2.2
numpy==1.23.5
scikit-learn==1.2.0
```

`Dockerfile`内容：

```dockerfile
# 1. 选择一个基础镜像
# 我们选择官方的Python 3.9 slim版本，它比较小巧
FROM python:3.9-slim

# 2. 设置工作目录
# 在容器内创建一个/app目录，并将其设置为后续命令的执行目录
WORKDIR /app

# 3. 复制依赖文件
# 将requirements.txt复制到容器的/app目录下
COPY requirements.txt .

# 4. 安装依赖
# 在容器内运行pip install命令。--no-cache-dir可以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 5. 复制应用代码和模型文件
# 将当前目录下的所有文件复制到容器的/app目录下
COPY . .

# 6. 暴露端口
# 告诉Docker，容器内的应用将监听5000端口
EXPOSE 5000

# 7. 定义启动命令
# 当容器启动时，执行这个命令。
# 使用gunicorn作为生产级的WSGI服务器，而不是Flask的开发服务器
# CMD ["python", "app.py"] # 开发时可以用这个
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
```

*注意：为了使用`gunicorn`，你需要先`pip install gunicorn`并更新`requirements.txt`。*

### 4.3.3 构建与运行容器

1. 构建镜像：在`Dockerfile`所在的目录下，运行：

    ```bash
    # -t 表示tag，为镜像命名，格式为 name:tag
    docker build -t iris-classifier:1.0 .
    ```

    最后的`.`表示使用当前目录作为构建上下文。Docker会按照`Dockerfile`中的指令，一步步执行并构建镜像。

2. 查看镜像：

    ```bash
    docker images
    ```

    你会看到刚刚创建的`iris-classifier:1.0`镜像。

3. 运行容器：

    ```bash
    # -d: 后台运行 (detached)
    # -p 8080:5000: 端口映射，将宿主机的8080端口映射到容器的5000端口
    # --name: 为容器命名
    docker run -d -p 8080:5000 --name iris_app iris-classifier:1.0
    ```

4. 测试服务：现在，你可以在宿主机上，通过`8080`端口来访问服务了！

    ```bash
    curl -X POST http://localhost:8080/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
    ```

5. 管理容器：

    - `docker ps`：查看正在运行的容器。
    - `docker logs iris_app`：查看容器的日志输出。
    - `docker stop iris_app`：停止容器。
    - `docker rm iris_app`：删除容器。

Docker解决了环境一致性的终极问题。你只需要将`Dockerfile`连同代码一起提交到代码库，任何一个安装了Docker的开发者或服务器，都可以通过`docker build`和`docker run`，完美地复现出与你一模一样的运行环境。

## 4.4 系统性能分析：使用`top`, `htop`, `nvidia-smi`定位瓶颈

### 4.4.1 CPU与内存监控：`top` 和 `htop`

当服务器变慢时，首先要看的就是CPU和内存。

`top`：Linux系统自带的实时性能监控工具。

在终端输入`top`，你会看到一个动态更新的界面。

第一部分（摘要区）：

- `load average`: 系统负载。三个数值分别代表过去1、5、15分钟的平均负载。如果这个值持续高于你的CPU核心数，说明系统过载。
- `%Cpu(s)`: CPU使用率 breakdown。`us`(user), `sy`(system), `id`(idle)是关键。`id`很低说明CPU很忙。
- `MiB Mem` / `MiB Swap`: 物理内存和交换空间的使用情况。

第二部分（进程列表）：

- `PID`: 进程ID。
- `USER`: 进程所有者。
- `%CPU`: 进程占用的CPU百分比。
- `%MEM`: 进程占用的内存百分比。
- `COMMAND`: 进程名。

常用交互：按`P`按CPU排序，按`M`按内存排序，按`q`退出。

`htop`：`top`的增强版，更美观、更易用。需要手动安装（`sudo apt-get install htop`）。

- 提供了彩色的、图形化的CPU和内存使用条。
- 可以用鼠标或方向键选择进程。
- 按`F4`过滤进程，按`F5`显示树状结构，按`F9`杀死进程。
- 对于AI工程师来说，`htop`通常是首选。

### 4.4.2 GPU监控：`nvidia-smi`

对于深度学习任务，GPU是核心资源。`nvidia-smi`（NVIDIA System Management Interface）是监控NVIDIA GPU状态的权威工具。

在终端输入`nvidia-smi`，你会看到一个信息表：

- Driver Version / CUDA Version：驱动和CUDA版本。
- GPU Name / Fan / Temp / Perf / Pwr:Usage/Cap：GPU型号、风扇转速、温度、性能状态、当前功耗/总功耗。温度过高（>85°C）需要警惕。
- Memory-Usage：这是最重要的部分。显示了已用显存 / 总显存。如果显存被占满，新的GPU任务将无法运行（CUDA out of memory error）。
- GPU-Util：GPU利用率百分比。如果你的模型正在训练，这个值应该很高（接近100%）。如果很低，说明可能存在数据加载瓶颈（CPU在准备数据，GPU在等待）。
- Processes：显示正在使用该GPU的进程列表，包括进程ID和占用的显存量。这对于找出是哪个程序占用了GPU资源至关重要。

持续监控：

```bash
# 每秒刷新一次nvidia-smi的输出
watch -n 1 nvidia-smi
```

通过持续监控`nvidia-smi`，你可以清晰地看到模型训练或推理过程中GPU的动态变化，从而判断GPU是否被充分利用，或者是否存在显存泄漏等问题。

## 4.5 实战项目：将Flask应用Docker化并实现一键部署

现在，我们将把上一章构建的、包含Flask、Celery和Redis的异步图像分类应用，完整地进行Docker化，并使用`docker-compose`来实现一键启动整个服务栈。

### 4.5.1 `docker-compose`：编排多容器应用

我们的应用包含三个服务：Web应用（Flask）、任务队列（Celery Worker）、消息中间件（Redis）。手动一个一个地`docker run`来启动和管理它们非常繁琐，而且还需要处理它们之间的网络连接问题。

`docker-compose`正是解决这个问题的工具。它允许我们使用一个YAML文件（`docker-compose.yml`）来定义和配置一个多容器的应用。

### 4.5.2 项目改造与Dockerfile编写

项目结构：

```text
dockerized_async_app/
├── app/
│   ├── __init__.py
│   ├── routes.py
│   └── tasks.py
├── celery_worker.py
├── config.py
├── Dockerfile          # 用于构建app和worker的镜像
├── docker-compose.yml  # 编排文件
├── imagenet_class_index.json
├── requirements.txt
└── run.py
```

1. 修改`config.py`以适应Docker网络

    在Docker Compose创建的网络中，服务之间可以通过服务名直接通信。我们需要将`localhost`修改为Redis服务的名字（我们将在`docker-compose.yml`中定义为`redis`）。

    ```python
    # config.py
    import os

    class Config:
        # 从环境变量获取Redis的主机名，如果不存在则默认为localhost
        # 这使得配置在本地和Docker中都能工作
        REDIS_HOST = os.environ.get('REDIS_HOST', 'localhost')
        CELERY_BROKER_URL = f'redis://{REDIS_HOST}:6379/0'
        CELERY_RESULT_BACKEND = f'redis://{REDIS_HOST}:6379/0'
    ```

2. 编写`Dockerfile`

    这个`Dockerfile`将用于构建我们的Flask应用和Celery Worker的通用镜像。

    ```dockerfile
    # Dockerfile
    FROM python:3.9-slim

    WORKDIR /app

    COPY requirements.txt .
    # 安装PyTorch时指定CPU版本，可以大幅减小镜像体积
    # 如果需要在GPU上运行，需要选择nvidia/cuda基础镜像并安装GPU版PyTorch
    RUN pip install --no-cache-dir -r requirements.txt

    COPY . .

    # 这个镜像可以用于启动Web服务或Worker，启动命令将在docker-compose中指定
    ```

    *注意：`requirements.txt`应包含`flask`, `celery`, `redis`, `torch`, `torchvision`, `pillow`, `gunicorn`。*

3. 编写`docker-compose.yml`

    这是项目的核心编排文件。

    ```yaml
    # docker-compose.yml
    version: '3.8'

    services:
    # Redis服务
    redis:
        image: "redis:alpine"
        ports:
        - "6379:6379"

    # Flask Web应用服务
    web:
        build: .  # 使用当前目录的Dockerfile进行构建
        ports:
        - "5000:5000"
        environment:
        - REDIS_HOST=redis  # 设置环境变量，指向redis服务
        depends_on:
        - redis  # 确保在web启动前，redis服务已经启动
        command: ["gunicorn", "--bind", "0.0.0.0:5000", "run:app"]

    # Celery Worker服务
    worker:
        build: .
        environment:
        - REDIS_HOST=redis
        depends_on:
        - redis
        command: ["celery", "-A", "celery_worker.celery", "worker", "-l", "info"]
    ```

这个文件定义了三个服务：

- `redis`: 直接使用官方的`redis:alpine`镜像。
- `web`: 使用我们自己的`Dockerfile`构建。它将容器的5000端口映射到宿主机的5000端口，并通过环境变量告诉应用Redis的主机名是`redis`。`command`覆盖了`Dockerfile`中的`CMD`，指定了启动Gunicorn的命令。
- `worker`: 同样使用我们的`Dockerfile`构建。它不暴露任何端口，因为它只在内部网络与Redis通信。`command`指定了启动Celery Worker的命令。

### 4.5.3 一键部署与测试

现在，部署整个应用只需要一个命令！在`docker-compose.yml`所在的目录下运行：

```bash
docker-compose up --build
```

`--build`：强制重新构建镜像（第一次运行时需要）。

`docker-compose up`会按照`depends_on`的顺序，依次启动所有服务。你会看到来自三个容器的日志交错输出。

如果想在后台运行，使用：

```bash
docker-compose up -d --build
```

测试：

服务启动后，一切都和之前一样。你可以通过`localhost:5000`来访问你的API。

```bash
curl -X POST http://localhost:5000/predict -F "image=@path/to/your/image.jpg"
```

发起任务后，你可以在`docker-compose up`的日志中，看到Celery Worker接收并处理任务的输出。

管理：

- `docker-compose ps`：查看由compose管理的所有容器的状态。
- `docker-compose logs -f web`：实时跟踪`web`服务的日志。
- `docker-compose down`：停止并删除由compose创建的所有容器、网络。

## 本章小结

在本章中，我们深入到了AI工程师的“引擎室”，掌握了一系列至关重要的底层工程技能。

我们从Linux命令行开始，学习了如何像一个“极客”一样高效地操作服务器。接着，我们通过虚拟环境解决了项目依赖管理的难题，确保了开发环境的纯净与可复现。

然后，我们迎来了本章的高潮——Docker容器化。通过`Dockerfile`和`docker-compose`，我们学会了将一个复杂的多组件AI应用，打包成一个标准的、可移植的“软件集装箱”。这不仅彻底解决了“环境一致性”这一千古难题，更为后续的CI/CD、微服务和大规模部署（如Kubernetes）奠定了基础。

最后，我们学习了如何使用系统性能监控工具，为我们的应用进行“体检”，在出现性能问题时能够有据可查，精准定位瓶颈。

完成本章的学习和实战后，你已经不再仅仅是一个算法开发者。你具备了将一个AI应用从代码，稳健、可靠、高效地交付到生产环境的能力。你理解了隔离、可复现性、自动化这些现代软件工程的核心思想。这套“运维之道”的内功，将使你在面对任何复杂的部署挑战时，都充满信心。至此，我们已经完成了“基础内功篇”的全部修炼，你已经拥有了成为一名优秀AI工程师所需要的坚实地基。在接下来的篇章中，我们将在这块坚实的地基上，开始构建更加宏伟的“核心能力大厦”——深入大语言模型的腹地。
