---
title: "第三章：构建AI服务的后端基石"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第三章：构建AI服务的后端基石"]
slug: "chapter-03"
---

在前面的章节中，我们修炼了Python的内功，并掌握了数据科学的利器。通过Kaggle实战，我们已经能够从原始数据出发，经过一系列精细的处理和分析，最终训练出一个表现不错的机器学习模型，并将其保存为一个文件（例如`.pkl`或`.pth`格式）。

然而，一个孤立的模型文件，无论其内部多么强大，其价值都是有限的。它就像一把铸造好的绝世好剑，却被锁在剑鞘里，无法施展其锋芒。要真正释放AI模型的价值，我们必须将其部署为一个在线服务，让它能够被其他的应用程序、网站、移动端乃至物联网设备所调用，从而赋能万千场景。这个从模型文件到在线服务的过程，就是AI工程化的“最后一公里”。

跨越这“最后一公里”，意味着AI工程师的角色需要从“炼丹师”向“架构师”转变。我们不仅要关心模型的准确率，更要关心服务的可用性（Availability）、响应延迟（Latency）、吞吐量（Throughput）和可扩展性（Scalability）。一个在Jupyter Notebook里运行完美的模型，如果封装成的API服务在100个并发请求下就崩溃，那么它在商业上就是失败的。

本章，我们将聚焦于构建AI服务的后端基石。我们将学习：

Flask：一个轻量级、灵活的Python Web框架。它如同一个精巧的“剑鞘”，能让我们快速地将AI模型封装成一个Web API，对外提供能力。我们将学习如何定义路由、处理HTTP请求和响应。

RESTful API设计：API是服务之间沟通的语言。我们将学习RESTful的设计原则，这是一种业界广泛采纳的、优雅而强大的API设计风格，它能让我们的服务接口清晰、易懂、易于协作。

Celery：AI模型的推理，特别是对于复杂的深度学习模型，可能是非常耗时的。如果让用户在发起请求后一直等待，会带来极差的体验。Celery是一个强大的分布式任务队列，它如同一个“缓冲池”和“加速器”，能让我们将耗时的AI计算作为异步任务在后台处理，从而让API服务能够瞬间响应，极大地提升用户体验和系统吞吐量。

最后，我们将通过一个贯穿始终的实战项目——搭建一个支持异步推理的图像分类API服务——将所有知识点融会贯-通。我们将亲手编写代码，将一个预训练好的图像分类模型，从一个本地文件，一步步打造成一个功能完备、支持高并发的在线AI服务。

掌握本章内容，你将具备将算法模型产品化的核心工程能力，这是区分一名“算法研究员”和一名“AI应用工程师”的关键所在。现在，让我们开始为我们的AI模型，构建一个坚实而高效的“家”。

## 3.1 Flask入门：快速构建轻量级API服务

在Python的Web框架世界里，Django和Flask是最耀眼的双子星。Django是一个“大而全”的框架，自带ORM、后台管理等众多组件，适合构建复杂的大型Web应用。而Flask则是一个“微框架”（Microframework），它核心精简，只保留了Web开发最基本的功能：路由和请求响应处理。这种极简主义的设计哲学，赋予了Flask极高的灵活性和可扩展性，使其成为构建API服务，特别是AI模型API的理想选择。

### 3.1.1 “Hello, AI World!”：你的第一个Flask应用

安装Flask非常简单：

```bash
pip install Flask
```

现在，让我们用最少的代码来创建一个Web服务。新建一个文件 `app.py`：

```python
# app.py
from flask import Flask

# 1. 创建一个Flask应用实例
# __name__ 是一个Python预定义的变量，它指向当前模块的名字。
# Flask用它来确定应用的根目录，以便找到模板和静态文件。
app = Flask(__name__)

# 2. 定义一个路由（Route）和视图函数（View Function）
# @app.route('/') 是一个装饰器，它告诉Flask，当用户访问网站的根URL('/')时，
# 应该调用下面的 home 函数。
@app.route('/')
def home():
    # 3. 视图函数返回的内容，就是用户在浏览器中看到的内容。
    return "Hello, AI World!"

# 4. 启动Web服务器
# 这段代码确保只有在直接运行这个脚本时，服务器才会启动。
# 如果这个文件被其他文件导入，服务器不会启动。
if __name__ == '__main__':
    # app.run() 会启动一个内置的、用于开发的Web服务器。
    # debug=True 会开启调试模式，当代码有改动时服务器会自动重启，
    # 并且在出错时会显示详细的错误信息。生产环境中切勿开启！
    app.run(debug=True)
```

在终端中运行这个文件：

```bash
python app.py
```

你会看到类似下面的输出：

```text
 * Serving Flask app 'app'
 * Debug mode: on
 * Running on http://127.0.0.1:5000
```

现在，打开你的浏览器，访问 `http://127.0.0.1:5000`，你将看到页面上显示的 "Hello, AI World!"。恭喜你，你已经成功构建并运行了你的第一个Web服务！

### 3.1.2 路由、请求与响应：Web服务的核心交互

Web服务的本质，就是接收客户端（如浏览器、手机App）发来的请求（Request），经过处理后，返回一个响应（Response）。

动态路由（Dynamic Routes）

我们可以让URL的一部分是可变的，从而处理更复杂的请求。

```python
# ... (接上文) ...

# <username> 是一个变量，Flask会捕获URL中这部分的值，并作为参数传给视图函数。
@app.route('/user/<username>')
def show_user_profile(username):
    return f"User: {username}"

# 还可以指定变量的类型，例如 <int:post_id>
@app.route('/post/<int:post_id>')
def show_post(post_id):
    return f"Post ID: {post_id}, Type: {type(post_id)}"
```

现在访问 `http://127.0.0.1:5000/user/Alice`，会看到 "User: Alice"。
访问 `http://127.0.0.1:5000/post/123`，会看到 "Post ID: 123, Type: <class 'int'>"`。

HTTP方法（HTTP Methods）

Web通信主要使用不同的HTTP方法来表达操作的意图。对于API服务，最常用的是：

`GET`：获取资源。

`POST`：创建或提交资源（通常带有数据体）。

`PUT`：更新资源。

`DELETE`：删除资源。

默认情况下，Flask的路由只响应`GET`请求。我们可以通过 `methods` 参数来指定。

```python
from flask import request

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # 处理POST请求，通常是接收数据并进行模型推理
        return "Received a POST request. Ready to predict!"
    else:
        # 处理GET请求，可以返回一个说明页面
        return "This is the prediction endpoint. Please use POST to submit data."
```

请求对象（Request Object）

Flask将客户端发来的所有信息都封装在了全局的 `request` 对象中。我们可以从中获取：

`request.method`：请求的方法（'GET', 'POST'等）。

`request.args`：获取URL查询参数（例如 `/search?query=flask` 中的 `query`）。

`request.form`：获取POST请求中表单数据。

`request.json`：如果请求的`Content-Type`是`application/json`，可以直接获取解析后的JSON数据。这是现代API最常用的数据交换方式。

`request.files`：获取上传的文件。

响应（Response）

视图函数不仅可以返回字符串，还可以返回更复杂的响应。最常用的是返回JSON格式的数据，这需要用到 `jsonify` 函数。

```python
from flask import jsonify

@app.route('/api/model/info')
def model_info():
    info = {
        "model_name": "ImageClassifier_ResNet50",
        "version": "1.0",
        "input_type": "image/jpeg",
        "output_type": "json"
    }
    # jsonify 会将Python字典转换为JSON格式的响应，
    # 并设置正确的Content-Type头 (application/json)。
    return jsonify(info)
```

访问 `http://127.0.0.1:5000/api/model/info`，你将看到一个格式化的JSON响应。

### 3.1.3 将AI模型集成到Flask中

现在，让我们来做一个简单的模型集成。假设我们有一个用Scikit-learn训练好的鸢尾花分类模型。

1. 训练并保存模型（一次性操作）

    ```python
    # scripts/train_iris_model.py
    from sklearn.datasets import load_iris
    from sklearn.linear_model import LogisticRegression
    import pickle

    iris = load_iris()
    X, y = iris.data, iris.target

    model = LogisticRegression(max_iter=200)
    model.fit(X, y)

    # 使用pickle将训练好的模型对象序列化到文件
    with open('iris_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    ```

2. 在Flask应用中加载并使用模型

    ```python
    # app.py
    from flask import Flask, request, jsonify
    import pickle
    import numpy as np

    app = Flask(__name__)

    # 在应用启动时，加载一次模型到内存中
    # 避免每次请求都重新加载，提高效率
    try:
        with open('iris_model.pkl', 'rb') as f:
            model = pickle.load(f)
        iris_target_names = ['setosa', 'versicolor', 'virginica']
        print("模型加载成功！")
    except FileNotFoundError:
        model = None
        print("错误：找不到模型文件 'iris_model.pkl'。请先运行训练脚本。")

    @app.route('/')
    def home():
        return "鸢尾花分类API。请POST到 /predict 端点。"

    @app.route('/predict', methods=['POST'])
    def predict():
        if model is None:
            return jsonify({"error": "模型未加载，服务不可用"}), 500

        # 1. 获取并校验输入数据
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({"error": "请求体必须是包含 'features' 键的JSON"}), 400
    
        features = data['features']
        if not isinstance(features, list) or len(features) != 4:
            return jsonify({"error": "'features' 必须是一个包含4个数值的列表"}), 400

        try:
            # 2. 数据预处理
            # 将输入的列表转换为Numpy数组，并reshape成模型需要的(1, 4)形状
            input_data = np.array(features).reshape(1, -1)

            # 3. 模型推理
            prediction_idx = model.predict(input_data)[0]
            prediction_name = iris_target_names[prediction_idx]
        
            probabilities = model.predict_proba(input_data)[0].tolist()
            confidence = dict(zip(iris_target_names, probabilities))

            # 4. 构造响应
            response = {
                "prediction": prediction_name,
                "class_index": int(prediction_idx),
                "confidence": confidence
            }
            return jsonify(response)

        except Exception as e:
            # 捕获潜在的错误，例如输入数据无法转换为数值
            return jsonify({"error": f"处理请求时发生错误: {str(e)}"}), 500

    if __name__ == '__main__':
        app.run(debug=True)
    ```

    现在，你可以使用 `curl` 或 Postman 等工具来测试这个API：

    ```bash
    curl -X POST http://127.0.0.1:5000/predict \
    -H "Content-Type: application/json" \
    -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
    ```

    你会收到一个类似这样的JSON响应：

    ```json
    {
    "class_index": 0,
    "confidence": {
        "setosa": 0.98...,
        "versicolor": 0.01...,
        "virginica": 0.0...
    },
    "prediction": "setosa"
    }
    ```

    这个例子虽然简单，但它包含了构建一个AI API的完整流程：加载模型 -> 定义API端点 -> 接收和校验数据 -> 预处理 -> 模型推理 -> 格式化响应。

## 3.2 RESTful API设计原则与最佳实践

我们刚刚构建了一个能工作的API，但要让它成为一个“好”的API，就需要遵循一定的设计规范。REST（Representational State Transfer）是一种软件架构风格，而不是一个严格的协议。基于REST构建的API，我们称之为RESTful API。

### 3.2.1 核心原则

1. 资源（Resources）：API的核心是“资源”。任何事物都可以是资源，例如一个用户、一张图片、一个模型预测结果。每个资源都由一个唯一的URI（Uniform Resource Identifier）来标识，通常是URL。

   - 坏实践: `/getUser?id=123`, `/createPrediction`
   - 好实践: `/users/123`, `/predictions`

2. 使用HTTP方法表达操作：对资源的操作，应该由HTTP方法来定义。

   - `GET /users/123`：获取ID为123的用户信息。
   - `POST /users`：创建一个新用户。
   - `PUT /users/123`：更新ID为123的用户信息。
   - `DELETE /users/123`：删除ID为123的用户。

URL中应使用名词，而不是动词。

1. 使用HTTP状态码表达结果：API的响应应该使用标准的HTTP状态码来告知客户端操作的结果。

    2xx（成功）:

    - `200 OK`：请求成功。
    - `201 Created`：资源创建成功。
    - `204 No Content`：操作成功，但没有内容返回（例如DELETE）。

    4xx（客户端错误）:

    - `400 Bad Request`：请求无效（例如JSON格式错误、参数缺失）。
    - `401 Unauthorized`：需要认证。
    - `403 Forbidden`：认证成功，但无权限访问。
    - `404 Not Found`：资源不存在。

    5xx（服务器错误）:

    - `500 Internal Server Error`：服务器内部发生未知错误。
    - `503 Service Unavailable`：服务暂时不可用。

2. 无状态（Stateless）：服务器不应该保存任何关于客户端会话的状态。每一次请求都应该包含所有必要的信息，使得服务器能够独立处理它。这极大地简化了服务器的设计，并使其易于水平扩展。

### 3.2.2 AI API设计最佳实践

版本化（Versioning）：API应该有版本。当你的模型或接口发生不兼容的变更时，可以通过升级版本来避免破坏现有客户端的集成。

- URL版本化: `https://api.example.com/v1/predict`
- Header版本化: 在HTTP头中指定 `Accept: application/vnd.example.v1+json`

清晰的请求/响应格式：使用JSON作为主要的数据交换格式。请求和响应的JSON结构应该清晰、一致，并有文档说明。

请求体:

```json
{
    "model_version": "v1.2",
    "data": {
    "image_url": "http://...",
    // or "image_base64": "..."
    },
    "parameters": {
    "top_k": 5
    }
}
```

成功响应体:

```json
{
    "request_id": "xyz-123",
    "prediction": [
    {"label": "cat", "score": 0.95},
    {"label": "dog", "score": 0.04}
    ]
}
```

错误响应体:

```json
{
    "error": {
    "code": "INVALID_INPUT",
    "message": "Input image format not supported."
    }
}
```

异步处理模式：对于耗时长的AI任务（例如视频分析、大模型生成），同步等待是不可行的。应该采用异步模式：

1. 客户端 `POST /jobs` 发起一个任务，请求体包含任务所需数据。
2. 服务器立即验证请求，创建一个任务，并返回 `202 Accepted` 状态码，响应体中包含一个任务ID和查询状态的URL。

    ```json
    {
      "job_id": "job-abc-456",
      "status": "pending",
      "status_url": "/jobs/job-abc-456"
    }
    ```

3. 客户端稍后轮询 `GET /jobs/{job_id}` 来查询任务状态。
4. 当任务完成时，`GET` 请求会返回 `200 OK` 和最终的结果。

我们将在下一节学习如何用Celery来实现这种强大的异步模式。

## 3.3 Celery：用异步任务处理耗时的AI计算

### 3.3.1 为什么需要任务队列？

回到我们的Flask API。当一个`POST /predict`请求进来时，`predict`函数会执行模型推理。如果这个推理过程需要5秒钟，那么这个HTTP连接就会被占用5秒钟，客户端会一直处于等待状态。更糟糕的是，Flask的开发服务器是单线程的，这意味着在这5秒内，它无法处理任何其他请求。

在生产环境中，我们会使用Gunicorn等多进程/多线程的WSGI服务器，但这并不能从根本上解决问题。如果同时有10个请求进来，每个都需要5秒，那么服务器的10个工作进程（worker）都会被占满，第11个请求就必须排队等待。这会导致系统吞吐量极低，且用户体验极差。

Celery引入了任务队列（Task Queue）的架构来解决这个问题。其核心组件包括：

1. 生产者（Producer）：我们的Flask应用。它不直接执行耗时任务，而是将任务描述（例如“请对这张图片进行分类”）发送到一个消息队列中。
2. 消息中间件（Message Broker）：一个消息队列，如 Redis 或 RabbitMQ。它像一个任务的“待办清单”，负责接收和存储生产者发来的任务。
3. 消费者（Consumer），也叫 Worker：一个或多个独立的Celery进程。它们持续地从Broker那里获取任务，并在后台执行这些任务。Worker和Flask应用是完全解耦的，可以部署在不同的机器上。
4. 结果后端（Result Backend）：一个用于存储任务执行结果的数据库，如 Redis 或数据库。这使得Flask应用可以稍后查询任务的状态和结果。

工作流程：

1. Flask API接收到请求，立即将一个推理任务（包含图片数据）发送给Redis。
2. Flask API立刻向客户端返回一个“任务已接收”的响应，包含一个任务ID。整个HTTP请求-响应周期可能只需要几十毫秒。
3. 在后台，一个空闲的Celery Worker从Redis中取出该任务。
4. Worker执行耗时的模型推理。
5. Worker将推理结果存入Redis结果后端。
6. 客户端使用任务ID，通过另一个API端点来查询任务状态，最终获取结果。

这种架构带来了巨大的好处：

- 高响应性：API几乎是瞬间响应的。
- 高吞吐量：API可以快速处理大量请求，因为它的主要工作只是把任务丢进队列。
- 可扩展性：如果任务积压，我们只需要增加更多的Celery Worker进程或机器，而无需改动Flask应用。
- 解耦与健壮性：即使Worker因为某个任务崩溃了，也不会影响到主Web应用。

### 3.3.2 Celery与Flask集成

安装所需库：

```bash
pip install celery redis
```

你还需要安装并运行一个Redis服务器。使用Docker是最简单的方式：

```bash
docker run -d -p 6379:6379 --name my-redis redis
```

项目结构调整：

为了更好地组织代码，我们将创建一个应用工厂模式的Flask项目。

```text
async_image_classifier/
├── app/
│   ├── __init__.py       # 应用工厂
│   ├── tasks.py          # Celery任务定义
│   └── routes.py         # Flask路由定义
├── celery_worker.py      # 启动Celery worker的脚本
├── config.py             # 配置文件
└── run.py                # 启动Flask应用的脚本
```

1. 配置文件 (`config.py`)

    ```python
    class Config:
        CELERY_BROKER_URL = 'redis://localhost:6379/0'
        CELERY_RESULT_BACKEND = 'redis://localhost:6379/0'
    ```

2. 创建Celery实例和Flask应用工厂 (`app/__init__.py`)

    ```python
    from flask import Flask
    from celery import Celery
    from config import Config

    # 创建Celery实例，但尚未配置
    celery = Celery(__name__, broker=Config.CELERY_BROKER_URL)

    def create_app():
        app = Flask(__name__)
        app.config.from_object(Config)
    
        # 将Flask的配置更新到Celery实例中
        celery.conf.update(app.config)

        # 注册路由
        from . import routes
        app.register_blueprint(routes.bp)

        return app
    ```

3. 定义Celery任务 (`app/tasks.py`)

    ```python
    from . import celery
    import time

    # 这是一个模拟的耗时AI任务
    @celery.task
    def long_running_ai_task(data):
        """模拟一个需要5秒钟的AI计算任务"""
        print(f"开始处理任务，接收到数据: {data}")
        time.sleep(5)
        result = {"input": data, "output": "This is the prediction result."}
        print("任务处理完成。")
        return result
    ```

    `@celery.task` 装饰器将一个普通函数转换为了一个Celery任务。

4. 定义Flask路由 (`app/routes.py`)

    ```python
    from flask import Blueprint, request, jsonify, url_for
    from .tasks import long_running_ai_task

    bp = Blueprint('main', __name__)

    @bp.route('/start-task', methods=['POST'])
    def start_task():
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
    
        # 异步地调用任务
        # .delay() 是 .apply_async() 的快捷方式
        task = long_running_ai_task.delay(data)
    
        # 立即返回，并提供一个查询状态的URL
        return jsonify({
            "message": "Task started",
            "task_id": task.id,
            "status_url": url_for('main.task_status', task_id=task.id, _external=True)
        }), 202

    @bp.route('/task-status/<task_id>')
    def task_status(task_id):
        task = long_running_ai_task.AsyncResult(task_id)
    
        response = {
            "task_id": task_id,
            "status": task.state
        }
    
        if task.state == 'PENDING':
            response['info'] = 'Task is waiting to be executed.'
        elif task.state == 'SUCCESS':
            response['result'] = task.result
        elif task.state != 'FAILURE':
            # 任务正在进行中
            response['info'] = task.info or 'No progress info'
        else:
            # 任务失败
            response['info'] = str(task.info) # 异常信息
        
        return jsonify(response)
    ```

5. 启动脚本 (`run.py` 和 `celery_worker.py`)

    ```python
    # run.py (启动Flask)
    from app import create_app

    app = create_app()

    if __name__ == '__main__':
        app.run(debug=True)

    # celery_worker.py (启动Celery)
    from app import create_app, celery

    app = create_app()
    app.app_context().push()
    ```

运行：

你需要打开两个终端窗口。

终端1（启动Celery Worker）:

```bash
# -A 指定Celery应用实例的位置
# -l info 设置日志级别
celery -A celery_worker.celery worker -l info
```

终端2（启动Flask应用）:

```bash
python run.py
```

现在，你可以像之前一样用`curl`来测试了：

1. 发起任务:

    ```bash
    curl -X POST http://127.0.0.1:5000/start-task -H "Content-Type: application/json" -d '{"image_id": 123}'
    ```

    你会立即收到一个包含`task_id`的响应。

2. 查询状态:

    复制上一步返回的`task_id`，然后访问状态URL：

    ```bash
    curl http://127.0.0.1:5000/task-status/your-task-id-here
    ```

    一开始，状态可能是`PENDING`或`STARTED`。等待5秒后，再次查询，状态会变为`SUCCESS`，并且你会看到任务的返回结果。

至此，我们已经成功地搭建了一个完整的异步任务处理系统！

## 3.4 实战项目：搭建一个支持异步推理的图像分类API服务

现在，我们将把本章所有知识点整合起来，完成我们的最终项目。我们将使用一个预训练的深度学习模型（例如ResNet50），通过Flask和Celery，将其部署为一个功能完备的、异步的图像分类服务。

项目目标：

1. 提供一个 `/predict` 端点，接收上传的图像文件。
2. 该端点应立即响应，返回一个任务ID。
3. 后台Celery Worker负责图像的预处理和模型推理。
4. 提供一个 `/results/<task_id>` 端点，用于查询分类结果。

技术栈：

- Flask
- Celery + Redis
- PyTorch + TorchVision (用于模型和图像处理)
- Pillow (用于图像I/O)

安装额外依赖：

```bash
pip install torch torchvision Pillow
```

项目结构： (与3.3节类似)

1. 定义Celery任务 (`app/tasks.py`)

    这次是真正的AI任务。

    ```python
    # app/tasks.py
    from . import celery
    from PIL import Image
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models
    import json
    import io
    import base64

    # --- 模型加载 ---
    # 在worker启动时加载一次模型，避免重复加载
    # 使用预训练的ResNet50模型
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.eval() # 设置为评估模式

    # 加载ImageNet类别标签
    try:
        with open('imagenet_class_index.json') as f:
            class_index = json.load(f)
        imagenet_labels = {int(k): v[1] for k, v in class_index.items()}
        print("ImageNet labels loaded.")
    except FileNotFoundError:
        print("Warning: imagenet_class_index.json not found. Predictions will be class indices.")
        imagenet_labels = None

    # --- 图像预处理 ---
    # 定义与ResNet50训练时相同的预处理流程
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    @celery.task(bind=True)
    def classify_image_task(self, image_b64_string: str):
        """
        Celery任务：解码图像，预处理，进行模型推理。
        bind=True 可以让任务函数访问self，从而可以更新任务状态。
        """
        try:
            self.update_state(state='PROGRESS', meta={'status': 'Decoding image...'})
            # 解码Base64字符串为图像
            image_bytes = base64.b64decode(image_b64_string)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

            self.update_state(state='PROGRESS', meta={'status': 'Preprocessing image...'})
            # 预处理图像
            input_tensor = preprocess(image)
            input_batch = input_tensor.unsqueeze(0) # 创建一个mini-batch

            self.update_state(state='PROGRESS', meta={'status': 'Running model inference...'})
            # 模型推理
            with torch.no_grad(): # 关闭梯度计算，加速推理
                output = model(input_batch)
        
            # 获取Top 5预测结果
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            top5_prob, top5_catid = torch.topk(probabilities, 5)

            self.update_state(state='PROGRESS', meta={'status': 'Formatting results...'})
            results = []
            for i in range(top5_prob.size(0)):
                prob = top5_prob[i].item()
                cat_id = top5_catid[i].item()
                label = imagenet_labels.get(cat_id, "Unknown") if imagenet_labels else "Unknown"
                results.append({"label": label, "probability": f"{prob:.4f}"})
            
            return {'status': 'Completed', 'predictions': results}

        except Exception as e:
            self.update_state(state='FAILURE', meta={'exc_type': type(e).__name__, 'exc_message': str(e)})
            # 在Celery中，最好是抛出异常，而不是返回错误信息
            raise e
    ```

    *注意：你需要从网上下载 `imagenet_class_index.json` 文件并放在项目根目录。*

2. 定义Flask路由 (`app/routes.py`)

    ```python
    # app/routes.py
    from flask import Blueprint, request, jsonify, url_for
    from .tasks import classify_image_task
    import base64

    bp = Blueprint('main', __name__)

    @bp.route('/predict', methods=['POST'])
    def predict():
        if 'image' not in request.files:
            return jsonify({"error": "No image file provided in the request"}), 400
    
        file = request.files['image']
    
        # 检查文件类型 (可选但推荐)
        if file.filename == '' or not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file type. Please upload a PNG, JPG, or JPEG image."}), 400

        try:
            # 读取文件内容并编码为Base64字符串
            # 这是将二进制数据通过JSON安全传输的常用方法
            image_bytes = file.read()
            image_b64_string = base64.b64encode(image_bytes).decode('utf-8')
        
            # 发起异步任务
            task = classify_image_task.delay(image_b64_string)
        
            return jsonify({
                "message": "Image classification task started.",
                "task_id": task.id,
                "status_url": url_for('main.get_result', task_id=task.id, _external=True)
            }), 202

        except Exception as e:
            return jsonify({"error": f"Failed to start task: {str(e)}"}), 500

    @bp.route('/results/<task_id>')
    def get_result(task_id):
        task = classify_image_task.AsyncResult(task_id)
    
        response = {
            "task_id": task_id,
            "status": task.state
        }
    
        if task.state == 'SUCCESS':
            response['result'] = task.result
        elif task.state == 'FAILURE':
            response['error'] = str(task.info)
        elif task.state == 'PROGRESS':
            response['progress'] = task.info
    
        return jsonify(response)
    ```

3. 运行与测试

    启动Celery Worker和Flask应用（同3.3节）。然后使用`curl`上传一张图片进行测试：

    ```bash
    # 将 'path/to/your/image.jpg' 替换为你的图片路径
    curl -X POST http://127.0.0.1:5000/predict -F "image=@path/to/your/image.jpg"
    ```

    你会得到一个任务ID。然后用这个ID去查询结果URL，最终你将看到类似这样的JSON输出：

    ```json
    {
    "result": {
        "predictions": [
        { "label": "golden_retriever", "probability": "0.9213" },
        { "label": "Labrador_retriever", "probability": "0.0345" },
        // ...
        ],
        "status": "Completed"
    },
    "status": "SUCCESS",
    "task_id": "..."
    }
    ```

## 本章小结

在本章中，我们完成了从一个本地模型文件到一套功能完备、健壮、可扩展的在线AI服务的关键跨越。

我们首先学习了使用Flask这个轻量级框架，快速地为我们的AI模型穿上了一层“Web外衣”，让它能够通过HTTP协议与外界沟通。接着，我们深入探讨了RESTful API的设计原则，学会了如何设计出清晰、规范、易于协作的API接口，这是专业软件工程的体现。

最核心的是，我们引入了Celery和任务队列架构，从根本上解决了AI推理耗时导致的API性能瓶颈问题。通过将计算密集型任务异步化，我们构建的服务不仅能为用户提供“秒回”的极致体验，更具备了在高并发场景下水平扩展的能力。

最终的实战项目，将所有这些技术点凝聚在了一起。你现在不仅知道如何训练一个模型，更掌握了如何将它部署为一个能产生实际商业价值的、工业级的AI服务。这是你作为一名AI工程师，在职业道路上迈出的至关重要的一步。

在后续的章节中，我们将继续深入AI工程化的其他领域，例如使用Docker容器化我们的应用，以及探索更高级的AI应用范式。但请始终铭记，本章所学的后端工程基础，将是你未来构建任何复杂AI系统时，都离不开的坚实基石。
