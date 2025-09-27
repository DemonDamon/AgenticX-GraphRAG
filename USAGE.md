# AgenticX GraphRAG 演示系统使用指南

## 概述

本演示系统展示了如何使用 AgenticX 框架构建完整的 GraphRAG（图增强检索生成）应用。系统集成了文档处理、知识图谱构建、多模态存储、智能检索和问答等功能。

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   文档输入      │───▶│   知识图谱      │───▶│   智能检索      │
│                 │    │                 │    │                 │
│ • PDF/TXT/JSON  │    │ • 实体提取      │    │ • 向量检索      │
│ • 智能分块      │    │ • 关系识别      │    │ • 图检索        │
│ • 语义理解      │    │ • 社区检测      │    │ • 混合策略      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   存储管理      │    │   索引构建      │    │   问答系统      │
│                 │    │                 │    │                 │
│ • Neo4j 图存储  │    │ • 向量索引      │    │ • 上下文理解    │
│ • Chroma 向量   │    │ • SPO 索引      │    │ • 智能回答      │
│ • Redis 缓存    │    │ • 缓存优化      │    │ • 结果重排      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 快速开始

### 1. 环境准备

#### 安装依赖
```bash
# 安装 Python 依赖
pip install -r requirements.txt

# 安装 spaCy 语言模型
python -m spacy download en_core_web_sm
python -m spacy download zh_core_web_sm
```

#### 启动外部服务

**Neo4j 图数据库**
```bash
# 使用 Docker 启动 Neo4j
docker run -d \
  --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your-password \
  neo4j:latest
```

**Chroma 向量数据库**
```bash
# 使用 Docker 启动 Chroma
docker run -d \
  --name chroma \
  -p 8000:8000 \
  chromadb/chroma:latest
```

**Redis 缓存**
```bash
# 使用 Docker 启动 Redis
docker run -d \
  --name redis \
  -p 6379:6379 \
  redis:latest
```

### 2. 配置环境

```bash
# 复制环境配置文件
cp .env.example .env

# 编辑配置文件，填入实际的 API 密钥和数据库连接信息
vim .env
```

必须配置的环境变量：
- `OPENAI_API_KEY`: OpenAI API 密钥
- `NEO4J_URI`: Neo4j 连接地址
- `NEO4J_USERNAME`: Neo4j 用户名
- `NEO4J_PASSWORD`: Neo4j 密码

### 3. 准备数据

将要处理的文档放入 `data` 目录：

```bash
# 创建数据目录（如果不存在）
mkdir -p data

# 复制文档到数据目录
cp your_documents.pdf data/
cp your_texts.txt data/
```

支持的文件格式：
- PDF 文档 (`.pdf`)
- 文本文件 (`.txt`, `.md`)
- JSON 文件 (`.json`)
- CSV 文件 (`.csv`)

### 4. 运行演示

#### 方式一：使用快速启动脚本
```bash
python run_demo.py
```

#### 方式二：直接运行主程序
```bash
python main.py
```

## 功能详解

### 1. 文档处理

系统支持多种文档格式的自动处理：

- **PDF 处理**: 自动提取文本内容，保留结构信息
- **文本处理**: 支持纯文本和 Markdown 格式
- **结构化数据**: 处理 JSON 和 CSV 格式的结构化数据

### 2. 智能分块

提供两种分块策略：

- **语义分块 (SemanticChunker)**: 基于语义相似度进行分块
- **智能分块 (AgenticChunker)**: 基于智能体的自适应分块

### 3. 知识图谱构建

使用 `GraphRAGConstructor` 构建层级化知识图谱：

- **实体提取**: 识别人物、组织、地点、概念等实体
- **关系识别**: 发现实体间的各种关系
- **社区检测**: 将相关实体聚类成社区
- **层级结构**: 构建多层级的知识结构

### 4. 多模态存储

- **图存储**: Neo4j 存储实体关系图谱
- **向量存储**: Chroma 存储文档和实体的向量表示
- **键值存储**: Redis 缓存索引和统计信息

### 5. 智能检索

提供多种检索策略：

- **向量检索**: 基于语义相似度的检索
- **图检索**: 基于图结构的关系检索
- **混合检索**: 结合多种策略的智能检索
- **自动检索**: 根据查询类型自动选择最佳策略

### 6. 交互式问答

支持多种查询类型：

- **实体查询**: "什么是 AgenticX？"
- **关系查询**: "AgenticX 和 Neo4j 有什么关系？"
- **路径查询**: "从文档处理到问答系统的完整流程是什么？"
- **社区查询**: "AgenticX 的核心模块有哪些？"
- **开放问答**: "如何使用 AgenticX 构建知识图谱？"

## 配置说明

### 系统配置 (configs.yml)

主要配置项：

```yaml
# 系统基础配置
system:
  workspace:
    base_dir: "./workspace"
    cache_dir: "./workspace/cache"
    logs_dir: "./workspace/logs"

# LLM 配置
llm:
  provider: "openai"
  model: "gpt-3.5-turbo"
  temperature: 0.1

# 知识管理配置
knowledge:
  graphrag:
    entity_extraction:
      max_entities_per_chunk: 20
    relationship_extraction:
      max_relationships_per_chunk: 15
    community_detection:
      algorithm: "louvain"
      resolution: 1.0

# 嵌入服务配置
embeddings:
  router:
    primary_provider: "openai"
    fallback_providers: ["siliconflow"]

# 存储配置
storage:
  vector:
    provider: "chroma"
  graph:
    provider: "neo4j"
  key_value:
    provider: "redis"

# 检索配置
retrieval:
  hybrid:
    vector_weight: 0.6
    graph_weight: 0.4
    enable_reranking: true
```

### 环境变量配置 (.env)

参考 `.env.example` 文件进行配置。

## 故障排除

### 常见问题

1. **连接数据库失败**
   - 检查数据库服务是否启动
   - 验证连接参数是否正确
   - 确认网络连接正常

2. **API 调用失败**
   - 检查 API 密钥是否正确
   - 验证网络连接
   - 确认 API 配额是否充足

3. **内存不足**
   - 减少批处理大小
   - 调整分块参数
   - 增加系统内存

4. **处理速度慢**
   - 启用缓存
   - 调整并发参数
   - 优化数据库索引

### 日志查看

```bash
# 查看系统日志
tail -f workspace/logs/agenticx_graphrag.log

# 查看错误日志
grep ERROR workspace/logs/agenticx_graphrag.log
```

### 性能监控

系统提供了详细的性能监控：

- 处理时间统计
- 内存使用情况
- 数据库连接状态
- API 调用统计

## 扩展开发

### 添加新的文档读取器

```python
from agenticx.knowledge.readers import BaseReader

class CustomReader(BaseReader):
    def read(self, file_path: str) -> Document:
        # 实现自定义读取逻辑
        pass
```

### 添加新的存储后端

```python
from agenticx.storage import BaseVectorStorage

class CustomVectorStorage(BaseVectorStorage):
    async def add_records(self, records: List[VectorRecord]):
        # 实现自定义存储逻辑
        pass
```

### 添加新的检索策略

```python
from agenticx.retrieval import BaseRetriever

class CustomRetriever(BaseRetriever):
    async def retrieve(self, query: str, top_k: int) -> List[RetrievalResult]:
        # 实现自定义检索逻辑
        pass
```

## 最佳实践

1. **数据准备**
   - 确保文档质量，避免过多噪声
   - 合理组织文档结构
   - 预处理特殊格式的文档

2. **参数调优**
   - 根据数据特点调整分块大小
   - 优化实体提取阈值
   - 调整检索权重比例

3. **性能优化**
   - 启用缓存机制
   - 合理设置批处理大小
   - 定期清理过期数据

4. **监控运维**
   - 定期检查日志
   - 监控系统资源使用
   - 备份重要数据

## 技术支持

如果遇到问题，可以通过以下方式获取帮助：

1. 查看 [AgenticX 官方文档](https://github.com/AgenticX/AgenticX)
2. 提交 [GitHub Issue](https://github.com/AgenticX/AgenticX/issues)
3. 参与社区讨论

## 许可证

本演示系统遵循 AgenticX 的开源许可证。