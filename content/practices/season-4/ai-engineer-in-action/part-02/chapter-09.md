---
title: "第九章：释放LLM的潜能：构建RAG与智能体（Agent）"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第九章：释放LLM的潜能：构建RAG与智能体（Agent）"]
slug: "chapter-09"
---

在前面的章节中，我们已经深入探索了大语言模型（LLM）的内部世界。我们学会了如何通过Prompt Engineering引导它，通过高效微调定制它。至此，我们手中的LLM已经像一个知识渊博、训练有素的“超级大脑”。然而，这个大脑在默认状态下，是与世隔绝的。它的知识被“冻结”在训练截止的那一刻，它无法访问最新的信息，也无法与外部工具互动来执行任务。它能“说”，但不能“做”；它能“回忆”，但不能“查询”。

要将LLM从一个强大的“语言模型”真正升级为能够解决现实世界问题的“智能助理”，我们必须打破这层壁垒，让它与外部世界连接起来。本章，我们将专注于实现这一目标的两种核心技术范式：检索增强生成（Retrieval-Augmented Generation, RAG）和智能体（Agent）。

RAG：为LLM装上“外挂知识库”
    我们知道，LLM存在“幻觉”和“知识过时”的问题。RAG架构正是为了解决这一痛点而生。它的核心思想是，在让LLM回答问题之前，先从一个外部的、可信的、可实时更新的知识库（如公司文档、产品手册、数据库）中，检索（Retrieve）出最相关的信息片段，然后将这些信息作为上下文增强（Augment）到Prompt中，最后让LLM基于这些可靠的信息进行生成（Generate）。RAG就像是为LLM配备了一个功能强大的“搜索引擎”和“开放式书架”，使其能够回答基于私有或实时知识的问题，极大地提升了答案的准确性和时效性。

Agent：赋予LLM“思考”与“行动”的能力
    如果说RAG是让LLM“读万卷书”，那么Agent就是让LLM“行万里路”。一个Agent系统将LLM从一个被动的文本生成器，提升为一个主动的、有目标的任务执行者。它以LLM为核心“大脑”，通过一个“思考-行动-观察”的循环，来决策下一步该做什么。它可以被赋予一系列工具（Tools），如调用计算器、查询天气API、执行代码、搜索网络等。当面对一个复杂任务时，Agent会自主地进行任务分解，选择并使用合适的工具，观察结果，并根据结果进行下一步的思考和行动，直到最终完成任务。

本章，我们将深入这两种激动人心的技术：

1. 详解RAG架构：我们将从文本切分、向量化，到向量数据库的应用，再到从检索到生成的完整流程，为你“庖丁解牛”般地解析RAG的每一个环节。
2. 入门Agent开发：我们将学习Agent的核心思想框架（如ReAct），并借助强大的开源框架（如LangChain或LlamaIndex），快速上手开发自己的智能体，并为其赋予实用的工具。
3. 双实战驱动：我们将通过两个紧贴实际应用的实战项目——构建一个基于公司文档的智能问答机器人（RAG）和开发一个能查询天气和计算器的简单智能体（Agent）——将理论知识转化为触手可及的工程实践。

掌握了RAG和Agent，你就掌握了当前LLM应用开发的两大“杀手锏”。你将能够构建出真正有用、可靠、且能与现实世界互动的智能应用。现在，让我们一起，为我们的大模型装上“眼睛”和“双手”，开启它真正的潜能。

## 9.1 检索增强生成（RAG）架构详解

RAG是一种将信息检索（Information Retrieval）与语言模型生成（Language Model Generation）相结合的架构，旨在通过引入外部知识来增强LLM的回答质量。

RAG的核心优势：
缓解幻觉：LLM被强制要求基于提供的上下文来回答，而不是凭空捏造。
知识实时更新：你无需重新训练昂贵的LLM，只需更新外部知识库，模型就能接触到最新的信息。
可追溯性与可解释性：可以向用户展示答案是基于哪些源文档生成的，提高了答案的可信度。
数据安全：私有数据存储在自己的知识库中，无需用其训练模型，降低了数据泄露的风险。

一个典型的RAG流程包含两个阶段：数据索引（Indexing）和检索与生成（Retrieval & Generation）。

RAG 流程示意图

### 9.1.1 数据索引阶段：构建你的知识库

这个阶段是离线进行的，目的是将你的原始文档（如PDF, TXT, Markdown, HTML等）处理成一个可供快速检索的格式。

第一步：加载与切分（Load & Split）
原始文档通常很长，无法直接放入LLM的上下文窗口。因此，第一步就是将长文档切分成更小的、有意义的文本块（Chunks）。

加载器（Loaders）：使用如`LlamaIndex`或`LangChain`中的文档加载器，可以轻松地读取各种格式的文件。
切分器（Splitters）：
    固定大小切分（Fixed-size Chunking）：最简单的方法，按固定字符数（如1000个字符）切分，并设置一定的重叠（Overlap，如100个字符），以保证语义的连续性。
    递归字符切分（Recursive Character Text Splitter）：一种更智能的方法。它会尝试按一系列分隔符（如`\n\n`, `\n`, ` `）来切分，优先保持段落、句子的完整性。
    语义切分（Semantic Chunking）：更高级的方法，通过分析文本块之间的语义相似度来决定切分点，力求每个Chunk都是一个语义完整的单元。

切分的艺术：
Chunk的大小是一个关键超参数。
太小：可能丢失重要的上下文信息，导致检索到的片段过于零散。
太大：可能包含太多与查询无关的噪声，增加了LLM处理的负担。
一个常见的起点是512到1024个token。

第二步：向量化（Embedding）
切分完成后，我们需要将每个文本块（Chunk）转换为一个向量（Vector），这个过程称为嵌入（Embedding）。这个向量是文本块在多维语义空间中的坐标。

嵌入模型（Embedding Model）：我们使用一个预训练好的句子转换模型（Sentence Transformer）来完成这个任务。这些模型专门用于将文本映射到能够捕捉其语义的稠密向量空间。
如何选择嵌入模型？
    MTEB排行榜（Massive Text Embedding Benchmark）：这是评估嵌入模型性能的黄金标准。
    主流选择：
        英文：`BAAI/bge-large-en-v1.5` (当前性能领先), `sentence-transformers/all-MiniLM-L6-v2` (轻量高效)。
        中文/多语言：`BAAI/bge-m3` (强大的多语言模型), `infgrad/stella-base-zh-v2` (优秀的中文模型)。
实现：使用`sentence-transformers`库或Hugging Face的`transformers`库可以轻松加载和使用这些模型。

```python
from sentence_transformers import SentenceTransformer

# 加载嵌入模型
model = SentenceTransformer('BAAI/bge-base-en-v1.5')

# 准备文本块
chunks = ["RAG stands for Retrieval-Augmented Generation.", 
          "It enhances LLMs with external knowledge."]

# 进行向量化
embeddings = model.encode(chunks)
print(embeddings.shape) # (2, 768) -> 2个文本块，每个都是768维的向量
```


### 9.1.2 向量数据库选型与应用

现在我们有了一大堆文本块和它们对应的向量。当用户提出一个问题时，我们需要找到与问题最“相似”的文本块。在一个拥有数百万文本块的知识库中，逐个计算相似度是不可行的。这时，向量数据库（Vector Database）就派上用场了。

向量数据库专门用于高效地存储和检索高维向量。它的核心技术是近似最近邻搜索（Approximate Nearest Neighbor, ANN）。

工作原理（简述）：
ANN算法通过构建特殊的索引结构（如IVF, HNSW），来避免全量搜索。它不能保证100%找到最相似的向量，但在牺牲极小的精度的前提下，将搜索速度提升了几个数量级，这对于实时应用是完全可以接受的。

主流向量数据库选型：

1. 内存型/本地型库：
    FAISS (Facebook AI Similarity Search)：由Facebook AI开发的高性能向量相似度搜索库。它是一个C++库，有Python接口。功能强大，速度极快，但本身不提供数据库的管理功能，更像一个“搜索引擎库”。
    ChromaDB：一个为AI应用设计的开源向量数据库。它非常易于使用，提供了简单的Python API，支持本地持久化存储，非常适合快速原型开发和中小型项目。
2. 服务端/分布式数据库：
    Pinecone, Weaviate, Milvus：这些是功能更全面的、可作为独立服务部署的向量数据库。它们支持分布式扩展、元数据过滤、实时索引更新等高级功能，适合大规模生产环境。

使用ChromaDB示例：

```python
import chromadb

# 1. 初始化ChromaDB客户端 (可以存到磁盘)
client = chromadb.PersistentClient(path="/path/to/db")

# 2. 创建或获取一个集合 (Collection)
collection = client.get_or_create_collection(name="my_knowledge_base")

# 3. 添加数据 (Indexing)
# 假设我们已经有了chunks和embeddings
collection.add(
    embeddings=embeddings.tolist(), # 嵌入向量
    documents=chunks,             # 原始文本块
    metadatas=[{"source": "doc1.pdf"}, {"source": "doc2.txt"}], # 元数据
    ids=[f"chunk_{i}" for i in range(len(chunks))] # 唯一的ID
)

# --- 检索阶段 ---
# 4. 查询 (Query)
query_text = "What is RAG?"
query_embedding = model.encode([query_text])[0].tolist()

# 检索最相似的 top-k 个结果
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2 # 返回最相似的2个
)

print(results['documents'])
# [['RAG stands for Retrieval-Augmented Generation.', 'It enhances LLMs with external knowledge.']]
```


### 9.1.3 从检索到生成的完整流程

现在我们已经打通了索引和检索，可以串联起整个RAG的第二阶段了。

第三步：检索（Retrieve）

1. 接收用户问题 `query`。
2. 使用与索引时相同的嵌入模型，将 `query` 转换为 `query_embedding`。
3. 在向量数据库中，使用 `query_embedding` 进行相似度搜索，检索出Top-K个最相关的文本块 `retrieved_chunks`。

第四步：增强与生成（Augment & Generate）

1. 构建Prompt：将检索到的文本块 `retrieved_chunks` 和用户的原始问题 `query`，一起组合成一个精心设计的Prompt。

    ```
    Context information is below.
    ---------------------
    {context_str}  <-- 将retrieved_chunks拼接成一个字符串
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query_str}
    Answer:
    ```

2. 调用LLM：将这个增强后的Prompt发送给LLM。
3. 获取答案：LLM会基于提供的上下文，生成最终的答案。

这个流程确保了LLM的回答是有据可依的，极大地提升了答案的质量和可靠性。

## 9.2 智能体（Agent）开发入门

如果说RAG是给LLM一个“只读”的外部大脑，那么Agent就是赋予LLM“思考”和“行动”的能力，让它能与外部世界进行“读写”交互。

### 9.2.1 Agent的核心思想：ReAct框架

ReAct (Reasoning and Acting) 是当前Agent系统最核心、最基础的思想框架之一。它将LLM的思考过程显式地分解为一个“Thought -> Action -> Observation”的循环。

Thought (思考)：LLM分析当前的任务目标和已有的信息，进行推理，并决定下一步应该采取什么行动。这个思考过程是LLM自己生成的、对人类可读的文本。
Action (行动)：根据思考，LLM决定调用一个工具（Tool），并指定调用该工具所需的输入（Action Input）。例如，`Action: Calculator, Action Input: 2+2`。
Observation (观察)：Agent系统执行这个Action（例如，运行计算器得到结果`4`），并将工具返回的结果作为“观察”信息，反馈给LLM。

LLM接收到这个Observation后，开始新一轮的循环：根据新的观察结果进行下一步的Thought，决定下一个Action，直到它认为任务已经完成，并生成最终的答案。

ReAct示例（简化版）：
任务: "爱因斯坦的年龄的平方是多少？"

Thought 1: 我需要先知道爱因斯坦的年龄。我没有这个信息，所以我需要搜索。
Action 1: `Search("爱因斯坦的年龄")`
Observation 1: "阿尔伯特·爱因斯坦（1879年3月14日—1955年4月18日），享年76岁。"
Thought 2: 我现在知道爱因斯坦的年龄是76岁。任务要求计算年龄的平方，即76的平方。我需要一个计算器来完成这个计算。
Action 2: `Calculator("76^2")`
Observation 2: "5776"
Thought 3: 我已经得到了计算结果5776。我已经完成了任务的所有步骤，可以给出最终答案了。
Final Answer: 爱因斯坦年龄的平方是5776。

通过这个循环，LLM将一个复杂的、需要外部信息的任务，分解成了多个简单的、可执行的步骤，并利用工具来弥补自身能力的不足。

### 9.2.2 LangChain/LlamaIndex框架快速上手

从零实现一个Agent系统非常复杂，需要处理Prompt模板、工具调用、输出解析、循环控制等大量细节。幸运的是，`LangChain`和`LlamaIndex`这两个强大的开源框架，极大地简化了Agent的开发。

LangChain：一个功能全面、非常灵活的LLM应用开发框架。它提供了构建Agent所需的各种组件（LLM接口、Prompt模板、输出解析器、工具等），并允许你像搭积木一样自由组合。它的学习曲线相对陡峭，但自由度高。
LlamaIndex：最初专注于RAG，但现在也发展出了强大的Agent能力。它的抽象层次更高，通常能用更少的代码实现一个功能完备的RAG或Agent系统，非常适合快速上手。

使用LangChain创建一个简单Agent的流程：

1. 定义工具（Tools）：定义Agent可以使用的工具列表。
2. 初始化LLM：选择并配置一个LLM（如`ChatOpenAI`或`HuggingFaceHub`）。
3. 创建Prompt模板：设计一个符合ReAct框架的Prompt模板，告诉LLM它有哪些工具可用，以及应该如何思考和行动。
4. 构建Agent：将LLM、工具和Prompt组合起来，创建一个Agent。
5. 创建Agent执行器（Agent Executor）：这是一个负责运行Agent循环的控制器。
6. 运行Agent：调用执行器来完成任务。

### 9.2.3 为Agent赋予工具（Tools）

工具是Agent与外部世界交互的桥梁。任何可以被程序化调用的功能，都可以被封装成一个工具。

常见的工具类型：
计算器：执行数学运算。
搜索引擎：通过API（如Google Search API, Tavily）进行网络搜索。
Python REPL：执行Python代码，能力极强但风险也高。
数据库查询：连接数据库，执行SQL查询。
API调用：调用任何第三方API（如天气、股票、地图等）。
RAG检索器：将我们之前构建的RAG检索器本身，也封装成一个工具。当Agent认为需要从私有知识库中查找信息时，就可以调用这个工具。

在LangChain中，定义一个工具通常需要：
`name`: 工具的名称，LLM会通过这个名字来决定调用哪个工具。
`description`: 极其重要。对工具功能的清晰描述。LLM完全依赖这个描述来理解工具的用途和何时使用它。
`func`: 工具背后实际执行的Python函数。

```python
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    # 在这里实现调用天气API的真实逻辑
    if city == "北京":
        return "北京今天晴，25摄氏度。"
    else:
        return f"抱歉，我无法查询{city}的天气。"

# Agent就可以通过 'get_weather' 这个名字来使用这个工具了。
```


## 9.3 实战项目一：构建一个基于公司文档的智能问答机器人（RAG）

项目目标：假设我们有一些关于公司政策的Markdown文档，我们将构建一个RAG系统，让员工可以就这些政策进行提问。

技术栈：`transformers` (for embeddings), `chromadb`, `langchain`

第一步：准备数据和环境

1. 创建一些`.md`文件，如`policy_leave.md`, `policy_expense.md`。
    `policy_leave.md`: "公司提供每年15天的带薪年假。申请年假需提前两周通过HR系统提交。"
    `policy_expense.md`: "员工的出差交通费可以报销。乘坐飞机需选择经济舱。出租车费用需提供发票。"
2. 安装库: `pip install langchain chromadb sentence-transformers`

第二步：索引数据

```python
# rag_indexing.py
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# 1. 加载文档
loader = DirectoryLoader('./company_policies/', glob="/*.md")
documents = loader.load()

# 2. 切分文本
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# 3. 加载嵌入模型
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')

# 4. 创建并持久化向量数据库
# 这会将向量数据存储在 'db' 目录下
vectordb = Chroma.from_documents(documents=texts, 
                                 embedding=embeddings, 
                                 persist_directory="./db")
vectordb.persist()

print("索引创建完成。")
```

第三步：构建问答链（QA Chain）

```python
# rag_qa.py
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOllama # 使用本地Ollama运行的LLM，也可以换成ChatOpenAI等
from langchain.chains import RetrievalQA

# 1. 加载嵌入模型和向量数据库
embeddings = HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5')
vectordb = Chroma(persist_directory="./db", embedding_function=embeddings)

# 2. 初始化LLM
# 假设你已经通过Ollama在本地运行了Llama 3: ollama run llama3
llm = ChatOllama(model="llama3")

# 3. 创建检索器 (Retriever)
retriever = vectordb.as_retriever(search_kwargs={"k": 2}) # 检索最相关的2个块

# 4. 创建RetrievalQA链
# chain_type="stuff" 是最简单的方式，将所有检索到的文档“塞”进一个Prompt里
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True # 同时返回源文档，便于追溯
)

# 5. 进行提问
query = "我每年有多少天年假？"
result = qa_chain({"query": query})

print("回答:", result['result'])
print("来源:", [doc.metadata['source'] for doc in result['source_documents']])
```

运行`rag_qa.py`后，系统会首先从向量数据库中检索到关于年假政策的文本块，然后将其与问题一起发送给LLM，最终得到准确的回答，并能告诉你答案来自哪个文档。

## 9.4 实战项目二：开发一个能查询天气和计算器的简单智能体（Agent）

项目目标：构建一个Agent，它能理解自然语言问题，并自主决定是使用天气查询工具还是计算器工具来回答。

技术栈：`langchain`, `langchain-openai` (或 `langchain-community` for Ollama)

第一步：定义工具

```python
# agent_tools.py
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Returns the current weather for a given city."""
    print(f"--- 调用天气工具，城市: {city} ---")
    if "北京" in city:
        return "北京今天多云转晴，气温15-28摄氏度。"
    elif "上海" in city:
        return "上海今天有小雨，气温20-25摄氏度。"
    else:
        return f"抱歉，我无法查询 {city} 的天气。"

@tool
def calculator(expression: str) -> str:
    """A simple calculator that evaluates a mathematical expression."""
    print(f"--- 调用计算器工具，表达式: {expression} ---")
    try:
        # 使用eval有安全风险，真实项目中应使用更安全的库
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误: {e}"

tools = [get_weather, calculator]
```

第二步：构建并运行Agent

```python
# agent_run.py
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain.agents import create_react_agent, AgentExecutor
from agent_tools import tools

# 1. 初始化LLM
# 需要设置你的OpenAI API Key: os.environ["OPENAI_API_KEY"] = "..."
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 2. 获取ReAct框架的Prompt模板
# 这是LangChain提供的一个经过优化的标准ReAct Prompt
prompt = hub.pull("hwchase17/react")

# 3. 创建Agent
# 这个函数会将LLM、工具和Prompt绑定在一起
agent = create_react_agent(llm, tools, prompt)

# 4. 创建Agent执行器
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True) # verbose=True会打印出完整的思考链

# 5. 运行Agent
# 测试1: 需要调用天气工具
response1 = agent_executor.invoke({"input": "今天北京的天气怎么样？"})
print("最终回答:", response1["output"])

print("\n" + "="*50 + "\n")

# 测试2: 需要调用计算器工具
response2 = agent_executor.invoke({"input": "3的5次方是多少？"})
print("最终回答:", response2["output"])
```

当你运行`agent_run.py`时，`verbose=True`会让你清晰地看到ReAct的每一步：
对于问题1，LLM会Thought: "我需要查询北京的天气"，然后Action: `get_weather("北京")`。
对于问题2，LLM会Thought: "我需要计算3的5次方"，然后Action: `calculator("35")`。

这完美地展示了Agent如何根据任务需求，自主地选择和使用正确的工具。

## 本章小结

在本章中，我们迈出了将LLM从一个“封闭大脑”转变为一个能与外部世界互动的“智能体”的关键两步。

我们首先深入剖析了检索增强生成（RAG）架构。我们学习了其从数据索引（加载、切分、向量化）到检索与生成的完整流程，并掌握了如何使用`ChromaDB`等向量数据库来构建和查询知识库。通过RAG，我们为LLM装上了一个强大的“外挂知识库”，有效解决了其知识局限和幻觉问题。

接着，我们探索了更前沿的智能体（Agent）技术。我们理解了其核心的ReAct思想框架，即通过“思考-行动-观察”的循环，让LLM能够进行任务分解、调用外部工具。借助`LangChain`等框架，我们学会了如何快速地构建一个能够自主决策和行动的Agent。

最后，通过两个紧密结合实际的实战项目，我们将RAG和Agent的理论知识，转化为了可以运行和体验的代码。我们亲手构建了一个企业级的智能问答机器人雏形，并开发了一个能使用工具的简单智能体。

完成本章后，你已经掌握了当前LLL应用层开发最核心、最热门的两大范式。你不再仅仅是LLM的使用者或微调者，你已经成为了一名能够设计和构建复杂、实用的AI应用的“架构师”。你所构建的应用，将不再局限于模型自身的知识，而是能够连接无限的外部数据和功能，从而在真实世界中创造出巨大的价值。在本书的最后，我们将展望AI工程的未来，探讨如何将我们构建的应用，通过CI/CD、监控和评估，打造成一个真正稳定、可靠的生产级系统。
