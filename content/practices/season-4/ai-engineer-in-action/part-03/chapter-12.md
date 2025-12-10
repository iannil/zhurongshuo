---
title: "第十二章：当LLM遇见知识图谱：构建与应用"
date: 2025-12-09T00:00:00+08:00
description: ""
draft: false
hidden: false
tags: ["书稿"]
keywords: ["AI工程师实战：从Python基础到LLM应用与性能优化", "第十二章：当LLM遇见知识图谱：构建与应用"]
slug: "chapter-12"
---

在前面的章节中，我们已经深入探索了如何构建、微调和部署强大的大语言模型（LLM）。我们知道，LLM通过在海量文本上进行预训练，学习到了丰富的世界知识和强大的语言能力。它像一个博览群书、无所不知的“通才”，能够就任何话题侃侃而谈。

然而，我们也清楚地认识到LLM的固有缺陷：

- 知识是隐性的、非结构化的：LLM的知识存储在其数十亿个参数构成的“黑箱”之中，我们无法轻易地对其进行审查、编辑或更新。
- 容易产生幻觉：它的知识是统计性的，而非事实性的。当被问及它不确定或不知道的信息时，它倾向于“编造”看似合理的答案。
- 逻辑推理能力有限：尽管LLM展现出一定的推理能力，但它在处理复杂、多跳的逻辑关系时，仍然容易出错。

与此同时，在人工智能领域，存在着另一条历史悠久、思想迥异的技术路线——知识图谱（Knowledge Graph, KG）。知识图谱以一种结构化的、图的形式来表示世界知识。它由实体（Entities）（如“莱昂纳多·迪卡普里奥”、“《泰坦尼克号》”）和连接这些实体的关系（Relations）（如“主演”、“导演”）组成。

知识图谱像一个严谨、精确的“专家”，它的每一个知识点都是明确的、可验证的、可解释的。它的优势恰好是LLM的劣势：
知识是显性的、结构化的：我们可以清晰地看到、查询和修改图中的每一个事实。

事实性强，无幻觉：图中存储的都是确定的事实，不存在模糊和编造。
强大的多跳推理能力：图数据库天然支持沿着关系路径进行复杂查询，例如“找出所有由詹姆斯·卡梅隆导演，并且由莱昂纳多·迪卡普里奥主演的电影的类型”。

当“博学”的LLM遇见“严谨”的知识图谱，一场深刻的化学反应正在发生。这两种技术范式的结合，被认为是构建下一代更强大、更可靠、更可解释的AI系统的关键。LLM强大的自然语言理解能力，可以被用来自动地从非结构化文本中构建知识图谱；而知识图谱精确的、结构化的知识，则可以反过来增强LLM，为其提供事实依据，提升其推理的精准度。

本章，我们将深入探索这个激动人心的交叉领域。我们将学习：

- 知识图谱基础：我们将从零开始，理解实体、关系、三元组这些基本概念，并了解专门用于存储和查询图数据的图数据库。
- Neo4j与Cypher：我们将上手业界最流行的图数据库——Neo4j，并学习其强大直观的图查询语言Cypher。
- 从文本自动构建知识图谱：我们将利用LLM的强大能力，设计一套工作流，从非结构化文本中自动提取实体和关系，并将其注入知识图谱。
- KG-RAG：我们将学习一种比传统向量检索更高级的RAG范式——知识图谱增强的检索（KG-RAG）。你将理解如何将用户的自然语言问题，转换为对知识图谱的结构化查询，从而获得更精准、更具解释性的答案。
- 实战项目：我们将通过一个完整的实战项目——构建一个小型电影知识图谱，并结合LLM实现自然语言查询——将本章所有技术融会贯通，亲手打造一个能用大白话提问的“电影知识专家”。

掌握LLM与知识图谱的融合技术，将使你站在AI应用开发的最前沿。你将能够构建出不仅“能说会道”，而且“有理有据”的AI系统，真正迈向更可信、更智能的未来。

## 12.1 知识图谱基础：实体、关系与图数据库

### 12.1.1 什么是知识图谱？

知识图谱（Knowledge Graph, KG）本质上是一个语义网络（Semantic Network），它以图（Graph）的数据结构，来描述现实世界中的概念、实体及其相互关系。

一个知识图谱由最基本的单元——三元组（Triple）——构成。一个三元组的形式是 (头实体, 关系, 尾实体)，或者说 (Subject, Predicate, Object)。

例如，对于事实“莱昂纳多主演了《泰坦尼克号》”，我们可以将其表示为：

- 头实体（Subject）：莱昂纳多·迪卡普里奥
- 关系（Predicate）：主演
- 尾实体（Object）：《泰坦尼克号》

当成千上万个这样的三元组汇集在一起时，它们就交织成了一张巨大的、网状的知识图谱。

一个简单的电影知识图谱示例

在这个图中：

- 节点（Nodes） 或 实体（Entities）：代表现实世界中的对象，如`Tom Hanks`（演员）、`Forrest Gump`（电影）。节点可以有标签（Labels）来表示其类型（如`:Person`, `:Movie`），以及属性（Properties）来存储其自身的信息（如`name: "Tom Hanks"`, `born: 1956`）。
- 边（Edges） 或 关系（Relations）：代表实体之间的联系，如`ACTED_IN`。关系也可以有属性（如`roles: ["Forrest"]`）。

### 12.1.2 为什么需要图数据库？

你可能会问，这些信息不也可以用传统的关系型数据库（如MySQL）来存储吗？比如创建一张`actors`表，一张`movies`表，再创建一张`acting_relations`中间表。

对于简单、固定的查询，关系型数据库是可行的。但当我们需要探索实体之间复杂、多跳、未知深度的关系时，关系型数据库的弊端就暴露无遗了。

想象一个查询：“找出与Tom Hanks合作过的演员，这些演员又与其他哪些导演合作过？”

在关系型数据库中，这需要进行多次、代价高昂的`JOIN`操作。随着查询深度的增加，`JOIN`的次数会呈指数级增长，查询性能会急剧下降。

在图数据库中，这个查询非常自然。它就像从`Tom Hanks`这个节点出发，沿着`ACTED_IN`关系找到他演过的电影，再从这些电影节点出发，沿着`ACTED_IN`的反向关系找到其他演员，再从这些演员出发... 这个过程称为图遍历（Graph Traversal）。图数据库对这种遍历操作进行了深度优化，其性能远超关系型数据库。

图数据库的核心优势：Index-free Adjacency（免索引邻接）。每个节点都直接持有指向其邻居节点的“指针”。当需要遍历时，数据库可以直接跟随这些指针，而无需像关系型数据库那样通过索引去查找匹配的行。这使得图数据库在处理深度关联查询时，性能不会随着数据总量的增加而显著下降。

### 12.1.3 知识图谱的类型

通用知识图谱（General KG）：旨在覆盖尽可能广泛的通用领域知识。著名的例子有Google Knowledge Graph, Wikidata, DBpedia, Freebase。它们规模巨大，知识面广，但可能不够深入或实时。

领域知识图谱（Domain-specific KG）：专注于某个特定领域，如金融、医疗、电商、法律等。它们通常由企业自己构建，包含大量私有的、专业的知识，是企业重要的知识资产。我们本章的重点，就是如何构建和应用领域知识图谱。

## 12.2 Neo4j入门与Cypher查询语言

Neo4j是目前最流行、最成熟的图数据库之一。它是一个原生的图数据库，完全围绕图的结构进行设计和优化。

### 12.2.1 安装与启动Neo4j

最简单的方式是使用Docker：

```bash
docker run \
    --name neo4j-llm \
    -p 7474:7474 -p 7687:7687 \
    -d \
    -e NEO4J_AUTH=neo4j/password \
    neo4j:latest
```

`7474`端口是Neo4j Browser的HTTP端口，一个用于交互式查询和可视化的Web界面。

`7687`端口是Bolt协议的端口，是应用程序通过驱动程序连接Neo4j的端口。

启动后，在浏览器中访问`http://localhost:7474`，使用用户名`neo4j`和密码`password`登录，即可进入Neo4j Browser。

### 12.2.2 Cypher：为图而生的查询语言

Cypher是Neo4j的声明式图查询语言。它的设计哲学是“用ASCII艺术来画图”，语法非常直观。

核心语法元素：

- 节点：用圆括号`()`表示。
  - `(n)`：一个匿名的、任意类型的节点。
  - `(p:Person)`：一个标签为`Person`的节点，并用变量`p`来引用它。
  - `(m:Movie {title: 'Forrest Gump'})`：一个标签为`Movie`，且`title`属性为'Forrest Gump'的节点。
- 关系：用方括号`[]`和箭头`-->`或`<--`表示。
  - `-[r]-`：一个匿名的、任意方向的关系。
  - `-[r:ACTED_IN]->`：一个类型为`ACTED_IN`、方向从左到右的关系，并用变量`r`引用它。
  - `-[r:DIRECTED {year: 1994}]->`：一个带属性的关系。
- 模式（Pattern）：将节点和关系组合起来，描述你想要查找的图结构。
  - `(p:Person)-[:ACTED_IN]->(m:Movie)`：描述了一个人参演了一部电影的模式。

常用Cypher子句：

- `CREATE`：创建节点和关系。

    ```cypher
    CREATE (p:Person {name: 'Tom Hanks', born: 1956})
    CREATE (m:Movie {title: 'Forrest Gump', released: 1994})
    CREATE (p)-[:ACTED_IN {roles: ['Forrest']}]->(m)
    ```

- `MATCH`：匹配图中的模式，这是最常用的查询子句。

    ```cypher
    // 查找Tom Hanks演过的所有电影
    MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
    RETURN m.title
    ```

- `RETURN`：指定查询返回的结果。
- `WHERE`：添加过滤条件。

    ```cypher
    // 查找1990年后上映的，由Tom Hanks主演的电影
    MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)
    WHERE m.released > 1990
    RETURN m.title, m.released
    ```

- `MERGE`：智能版的`CREATE`。如果模式不存在，则创建它；如果已存在，则匹配它。这常用于避免创建重复的节点。

    ```cypher
    MERGE (p:Person {name: 'Robert Zemeckis'})
    MERGE (m:Movie {title: 'Forrest Gump'})
    MERGE (p)-[:DIRECTED]->(m)
    ```

- `DELETE`：删除节点和关系。
- `SET`：修改节点或关系的属性。

多跳查询示例：

```cypher
// 查找与Tom Hanks合作过的演员（不包括他自己）
MATCH (tom:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie)<-[:ACTED_IN]-(cofactor:Person)
WHERE tom <> cofactor
RETURN DISTINCT cofactor.name
```

这个查询的直观解释是：找到Tom Hanks，沿着`ACTED_IN`关系找到他演的电影，再从这些电影沿着`ACTED_IN`的反向关系找到其他演员。Cypher的表达能力和直观性，使其成为处理图数据的强大工具。

## 12.3 从文本中自动构建知识图谱

手动构建知识图谱费时费力。利用LLM强大的自然语言理解和结构化信息提取能力，我们可以实现从非结构化文本中自动构建知识图谱（KG Auto-construction）。

工作流程：

1. 定义图谱模式（Schema）：首先，明确你希望图谱中包含哪些类型的实体和关系。例如，在电影领域，实体类型可以是`:Movie`, `:Person`, `:Genre`；关系类型可以是`:ACTED_IN`, `:DIRECTED`, `:BELONGS_TO_GENRE`。
2. 设计提取Prompt：设计一个强大的Prompt，指导LLM从给定的文本中，提取出符合我们定义的Schema的三元组。
3. 文本处理与信息提取：将源文档（如维基百科页面、新闻文章）分块，然后将每个文本块连同提取Prompt一起发送给LLM。
4. 结构化输出解析：要求LLM以JSON等结构化格式返回提取结果，便于程序解析。
5. 注入图数据库：将解析出的三元组，使用`MERGE`语句写入Neo4j，构建或更新知识图谱。

示例：使用LLM从文本中提取电影信息

输入文本（Input Text）：

> "Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis and written by Eric Roth. It is based on the 1986 novel of the same name by Winston Groom. The film stars Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson and Sally Field."

提取Prompt (Extraction Prompt)：

```text
You are an expert in knowledge graph construction. From the text provided, extract entities and relationships according to the following schema.
Return the result in a JSON format with two keys: "entities" and "relationships".

Schema:
- Entities:
  - Person: {name: string}
  - Movie: {title: string, released: integer}
  - Genre: {name: string}
- Relationships:
  - (Person)-[:ACTED_IN]->(Movie)
  - (Person)-[:DIRECTED]->(Movie)
  - (Movie)-[:BELONGS_TO_GENRE]->(Genre)

Text:
"""
Forrest Gump is a 1994 American comedy-drama film directed by Robert Zemeckis and written by Eric Roth. It is based on the 1986 novel of the same name by Winston Groom. The film stars Tom Hanks, Robin Wright, Gary Sinise, Mykelti Williamson and Sally Field.
"""
```

LLM的预期JSON输出 (Expected JSON Output)：

```json
{
  "entities": [
    {"label": "Movie", "properties": {"title": "Forrest Gump", "released": 1994}},
    {"label": "Person", "properties": {"name": "Robert Zemeckis"}},
    {"label": "Person", "properties": {"name": "Tom Hanks"}},
    {"label": "Person", "properties": {"name": "Robin Wright"}},
    {"label": "Genre", "properties": {"name": "Comedy"}},
    {"label": "Genre", "properties": {"name": "Drama"}}
  ],
  "relationships": [
    {"source": {"label": "Person", "name": "Robert Zemeckis"}, "type": "DIRECTED", "target": {"label": "Movie", "title": "Forrest Gump"}},
    {"source": {"label": "Person", "name": "Tom Hanks"}, "type": "ACTED_IN", "target": {"label": "Movie", "title": "Forrest Gump"}},
    {"source": {"label": "Person", "name": "Robin Wright"}, "type": "ACTED_IN", "target": {"label": "Movie", "title": "Forrest Gump"}},
    {"source": {"label": "Movie", "title": "Forrest Gump"}, "type": "BELONGS_TO_GENRE", "target": {"label": "Genre", "name": "Comedy"}},
    {"source": {"label": "Movie", "title": "Forrest Gump"}, "type": "BELONGS_TO_GENRE", "target": {"label": "Genre", "name": "Drama"}}
  ]
}
```

注入Neo4j的Python代码片段：

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))

def add_to_graph(tx, data):
    for entity in data['entities']:
        tx.run(f"MERGE (n:{entity['label']} {{name: $name}})", name=entity['properties'].get('name') or entity['properties'].get('title'))
  
    for rel in data['relationships']:
        source_name = rel['source']['name']
        target_name = rel['target']['title'] if rel['target']['label'] == 'Movie' else rel['target']['name']
        tx.run(f"""
            MATCH (a:{rel['source']['label']} {{name: $source_name}})
            MATCH (b:{rel['target']['label']} {{name: $target_name}})
            MERGE (a)-[:{rel['type']}]->(b)
        """, source_name=source_name, target_name=target_name)

with driver.session() as session:
    session.write_transaction(add_to_graph, llm_output_json)

driver.close()
```

通过重复这个“提取-注入”的过程，我们就可以将大量的非结构化文档，转化为一个结构精良、知识丰富的领域知识图谱。

## 12.4 KG-RAG：用知识图谱增强检索的精准度

我们在第九章学习了基于向量检索的RAG（Vector-RAG）。它的优点是实现简单，能处理任何文本。但它也存在问题：

- 检索不够精准：基于语义相似度的检索，有时会召回一些不完全相关，或者包含噪声的文本块。
- 缺乏可解释性：我们不知道为什么这些文本块被认为是“相似”的。
- 难以回答聚合或比较性问题：例如“A和B共同出演了哪些电影？”，这种问题很难通过检索独立的文本块来回答。

知识图谱增强的检索（KG-RAG）提供了一种更精准、更可解释的解决方案。其核心思想是：将用户的自然语言问题，转换为对知识图谱的结构化查询（如Cypher），直接从图中获取精确的事实，再将这些事实作为上下文提供给LLM来生成最终的自然语言答案。

KG-RAG工作流程：

1. 问题 -> Cypher转换：这是最关键的一步。我们利用LLM的强大代码生成能力，将用户的自然语言问题（如“汤姆·汉克斯演过哪些电影？”）转换为一条Cypher查询语句（`MATCH (p:Person {name: 'Tom Hanks'})-[:ACTED_IN]->(m:Movie) RETURN m.title`）。为了让LLM能生成正确的Cypher，我们需要在Prompt中向它提供图谱的Schema信息（节点标签、属性、关系类型）。
2. 执行Cypher查询：在Neo4j数据库中执行生成的Cypher语句。
3. 获取结构化结果：得到一个精确的、表格形式的结果（例如，一个电影标题列表）。
4. 结果 -> 自然语言：将这个结构化的查询结果，连同原始问题一起，再次发送给LLM，让它将这些事实“润色”成一句通顺的、人类可读的回答。

示例：

用户问题: "Who directed the movie Forrest Gump?"

第一步：Text-to-Cypher

Prompt:

```text
You are an expert Neo4j developer. Given a question and the graph schema, generate a Cypher query to answer the question.

Schema:
Node labels: Person, Movie
Relationship types: ACTED_IN, DIRECTED

Question: Who directed the movie Forrest Gump?
```

LLM生成的Cypher:

```cypher
MATCH (p:Person)-[:DIRECTED]->(m:Movie {title: 'Forrest Gump'})
RETURN p.name
```

第二步：执行查询

在Neo4j中执行该查询，得到结果：`[{"p.name": "Robert Zemeckis"}]`

第三步：生成最终答案

Prompt:

```
You are a helpful assistant. Based on the user's question and the retrieved data, provide a natural language answer.

Question: Who directed the movie Forrest Gump?
Retrieved Data: Robert Zemeckis

Answer:
```

LLM生成的答案: "The movie Forrest Gump was directed by Robert Zemeckis."

KG-RAG的优势：

- 精准性：直接从图中获取事实，避免了向量检索的不确定性。
- 可解释性：生成的Cypher查询本身，就是对答案来源的最好解释。
- 强大的推理能力：能够回答需要多跳推理、聚合、过滤的复杂问题。

混合策略：在实践中，我们常常将Vector-RAG和KG-RAG结合起来。对于事实性、实体性的问题，优先使用KG-RAG；对于更开放、更概念性的问题，则回退到Vector-RAG。

## 12.5 实战项目：构建一个小型电影知识图谱，并结合LLM实现自然语言查询

项目目标：我们将使用一小部分维基百科的电影简介文本，自动构建一个包含电影、演员、导演的Neo4j知识图谱，并实现一个能够将用户自然语言问题转换为Cypher查询并返回答案的问答系统。

技术栈：`openai` (or other LLM library), `neo4j`, `langchain` (for simplification)

第一步：环境准备

1. 启动Neo4j Docker容器（如12.2节所示）。
2. 安装库: `pip install langchain langchain-openai neo4j`
3. 准备一些电影简介文本文件，例如`forrest_gump.txt`, `the_matrix.txt`。

第二步：从文本构建知识图谱（使用LangChain简化）

LangChain提供了方便的工具来简化这个流程。

```python
# build_kg.py
import os
from langchain_openai import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate
from langchain.chains.graph_qa.cypher import GraphCypherQAChain

# --- 1. 连接Neo4j ---
os.environ["OPENAI_API_KEY"] = "..."
graph = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="password"
)

# --- 2. 定义从文本提取图谱的函数 ---
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model="gpt-4", temperature=0)

extraction_prompt = PromptTemplate(
    template="""From the text below, extract the following entities and relationships.
    Return the result as a list of Cypher MERGE statements.

    Schema:
    Nodes: Person, Movie
    Relationships: ACTED_IN, DIRECTED

    Text:
    {text}
    """,
    input_variables=["text"],
)

def extract_and_store_graph(text):
    # Extract graph data using LLM
    chain = LLMChain(llm=llm, prompt=extraction_prompt, output_parser=StrOutputParser())
    cypher_statements = chain.run(text=text)
  
    # Store data in Neo4j
    for stmt in cypher_statements.split('\n'):
        if stmt.strip():
            try:
                graph.query(stmt)
                print(f"Executed: {stmt}")
            except Exception as e:
                print(f"Error executing {stmt}: {e}")

# --- 3. 读取文本并构建图谱 ---
with open("forrest_gump.txt", "r") as f:
    forrest_gump_text = f.read()
extract_and_store_graph(forrest_gump_text)

with open("the_matrix.txt", "r") as f:
    the_matrix_text = f.read()
extract_and_store_graph(the_matrix_text)

print("知识图谱构建完成。")
```

*注意：上述`extraction_prompt`要求LLM直接生成Cypher语句，这是一种更直接高效的方式。你需要确保LLM（如GPT-4）有足够强的代码生成能力。*

第三步：实现Text-to-Cypher问答链

```python
# qa_with_kg.py
import os
from langchain_openai import ChatOpenAI
from langchain.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain

# --- 1. 连接Neo4j和LLM ---
os.environ["OPENAI_API_KEY"] = "..."
graph = Neo4jGraph(
    url="bolt://localhost:7687", 
    username="neo4j", 
    password="password"
)
llm = ChatOpenAI(model="gpt-4", temperature=0)

# --- 2. 创建GraphCypherQAChain ---
# LangChain的这个链封装了Text-to-Cypher和结果合成的完整流程
chain = GraphCypherQAChain.from_llm(
    graph=graph,
    llm=llm,
    verbose=True # 打印出生成的Cypher和中间步骤
)

# --- 3. 进行提问 ---
questions = [
    "Who acted in the movie Forrest Gump?",
    "Which movies did Keanu Reeves act in?",
    "Who directed The Matrix?",
]

for question in questions:
    print(f"Question: {question}")
    result = chain.invoke({"query": question})
    print(f"Answer: {result['result']}\n")
```

当你运行`qa_with_kg.py`时，`verbose=True`会让你看到神奇的幕后过程：

1. 对于问题 "Who acted in the movie Forrest Gump?"，LLM会生成类似 `MATCH (p:Person)-[:ACTED_IN]->(m:Movie {title: 'Forrest Gump'}) RETURN p.name` 的Cypher。
2. `GraphCypherQAChain`会执行这个Cypher，从Neo4j获取演员列表。
3. 最后，LLM会将这个列表格式化成一句通顺的回答，如 "Tom Hanks, Robin Wright, and Gary Sinise acted in the movie Forrest Gump."

这个项目完美地展示了LLM与知识图谱如何协同工作，将非结构化的知识转化为可查询的结构化资产，并最终以自然语言的形式服务于用户，实现了1+1>2的效果。

## 本章小结

在本章中，我们探索了AI领域一个极具深度和价值的前沿方向——大语言模型与知识图谱的融合。

我们从知识图谱的基础出发，理解了其作为一种结构化知识表示方法的强大之处，并学习了如何使用业界领先的图数据库Neo4j及其查询语言Cypher来存储和查询复杂的关联数据。

我们掌握了一项核心的工程能力：利用LLM从非结构化文本中自动构建知识图谱。我们学会了如何设计Prompt，引导LLM提取实体和关系，并将其持久化到图数据库中，将沉睡的文本数据转化为鲜活的知识网络。

在此基础上，我们学习了一种更高级的RAG范式——KG-RAG。我们理解了其如何通过Text-to-Cypher技术，将用户的自然语言问题转换为对知识图谱的精准查询，从而克服了传统向量检索的局限性，获得了更精确、更可解释的答案。

最后，通过一个构建电影知识图谱问答系统的实战项目，我们将所有理论和技术点串联起来，亲手打造了一个LLM与KG协同工作的智能应用。

完成本章后，你的AI工具箱中又增添了一件强大的“神器”。你不再仅仅依赖于LLM自身的、模糊的、不可控的知识，而是学会了如何为其配备一个精确、可靠、可演进的“外置事实大脑”。这种将LLM的语言能力与知识图谱的结构化推理能力相结合的复合型系统构建能力，将使你在面对需要高事实性、强逻辑性的复杂AI应用场景时，游刃有余，展现出卓越的工程设计和创新能力。
