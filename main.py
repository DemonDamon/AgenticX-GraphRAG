import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

#!/usr/bin/env python3
"""
AgenticX GraphRAG 演示系统主程序

基于 AgenticX 核心模块的完整 GraphRAG 实现，包括：
- 文档解析和智能分块
- 层级化知识图谱构建
- 多模态存储和索引
- 智能混合检索
- 交互式问答系统
"""
import absl.logging
absl.logging.set_verbosity('info')

import sys
import yaml
import asyncio
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from loguru import logger

# 过滤警告信息
warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")
warnings.filterwarnings("ignore", message=".*SwigPy.*")
warnings.filterwarnings("ignore", message=".*builtin type.*has no __module__ attribute.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="neo4j")
warnings.filterwarnings("ignore", message=".*session.*")
warnings.filterwarnings("ignore", message=".*connection.*")

# 加载环境变量
from dotenv import load_dotenv
# 确保从脚本所在目录加载 .env 文件
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(env_path)

# 导入 os 模块用于环境变量
import os

# AgenticX 核心模块导入
from agenticx.knowledge import (
    Knowledge,
    Document,
    DocumentProcessor,
    GraphRAGConstructor,
    KnowledgeGraphBuilder,
    SemanticChunker,
    AgenticChunker,
    get_chunker
)
from agenticx.knowledge.readers import (
    PDFReader,
    TextReader,
    JSONReader,
    CSVReader
)
from agenticx.embeddings import (
    EmbeddingRouter,
    OpenAIEmbeddingProvider,
    SiliconFlowEmbeddingProvider,
    BailianEmbeddingProvider
)
from agenticx.storage import (
    StorageManager,
    StorageConfig,
    StorageType,
    Neo4jStorage,
    ChromaStorage,
    RedisStorage,
    VectorRecord
)
from agenticx.retrieval import (
    HybridRetriever,
    GraphRetriever,
    VectorRetriever,
    BM25Retriever,
    AutoRetriever,
    Reranker,
    QueryAnalysisAgent,
    RetrievalAgent
)
from agenticx.llms import LlmFactory


class AgenticXGraphRAGDemo:
    """AgenticX GraphRAG 演示系统主类"""
    
    def __init__(self, config_path: str = "configs.yml"):
        """初始化演示系统"""
        self.config_path = config_path
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # 核心组件
        self.llm_client = None
        self.embedding_router = None
        self.storage_manager = None
        self.knowledge_graph = None
        self.retriever = None
        
        # 数据路径
        self.data_dir = Path("./data")
        self.workspace_dir = Path(self.config['system']['workspace']['base_dir'])
        
        self.logger.info("AgenticX GraphRAG 演示系统初始化完成")
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        config_file = Path(self.config_path)
        if not config_file.exists():
            # 如果当前目录没有配置文件，尝试从文档目录加载
            config_file = Path(".trae/documents/configs.yml")
            
        if not config_file.exists():
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
            
        with open(config_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            
        # 环境变量替换
        self._replace_env_vars(config)
        
        return config
    
    def _replace_env_vars(self, obj: Any) -> None:
        """递归替换配置中的环境变量"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith('${') and value.endswith('}'):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var, value)
                else:
                    self._replace_env_vars(value)
        elif isinstance(obj, list):
            for item in obj:
                self._replace_env_vars(item)
    
    def _setup_logging(self):
        """设置日志系统"""
        # 过滤掉特定的警告
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="importlib._bootstrap")
        warnings.filterwarnings("ignore", message=".*datetime.datetime.utcnow.*")
        warnings.filterwarnings("ignore", message=".*SwigPy.*")
        
        log_config = self.config.get('monitoring', {}).get('logging', {})
        
        # 创建日志目录
        log_dir = Path(self.config['system']['workspace']['logs_dir'])
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # 移除默认的loguru处理器
        logger.remove()
        
        # 配置控制台输出
        logger.add(
            sys.stdout,
            level=log_config.get('level', 'DEBUG'),  # 临时启用DEBUG级别
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # 配置文件输出
        logger.add(
            log_dir / "agenticx_graphrag.log",
            level=log_config.get('level', 'INFO'),
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )
        
        return logger
    
    async def initialize_components(self) -> None:
        """初始化所有核心组件"""
        self.logger.info("正在初始化核心组件...")
        
        # 1. 初始化 LLM 客户端
        await self._initialize_llm()
        
        # 2. 初始化嵌入服务
        await self._initialize_embeddings()
        
        # 3. 初始化存储管理器
        await self._initialize_storage()
        
        # 4. 初始化检索器
        await self._initialize_retriever()
        
        self.logger.info("所有核心组件初始化完成")
    
    async def _initialize_llm(self) -> None:
        """初始化 LLM 客户端"""
        from agenticx.knowledge.graphers.config import LLMConfig
        
        llm_config_dict = self.config['llm']
        provider = llm_config_dict.get('provider', 'litellm')
        
        # 提供商类型映射
        provider_type_mapping = {
            'openai': 'litellm',
            'anthropic': 'litellm', 
            'litellm': 'litellm',
            'bailian': 'bailian',
            'kimi': 'kimi'
        }
        
        llm_type = provider_type_mapping.get(provider, 'litellm')
        
        # 将字典配置转换为 LLMConfig 对象
        llm_config = LLMConfig(
            type=llm_type,
            model=llm_config_dict.get('model'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            provider=provider,
            timeout=llm_config_dict.get('timeout'),
            max_retries=llm_config_dict.get('retry_attempts'),  # 配置文件中是retry_attempts
            temperature=llm_config_dict.get('temperature'),
            max_tokens=llm_config_dict.get('max_tokens')
        )
        
        self.llm_client = LlmFactory.create_llm(llm_config)
        self.logger.info(f"LLM 客户端初始化完成: {llm_config.provider} - {llm_config.model}")
    
    async def _initialize_embeddings(self) -> None:
        """初始化嵌入服务路由器"""
        embed_config = self.config['embeddings']
        
        # 创建嵌入提供商列表
        providers = []
        provider_names = []
        
        # 根据配置的优先级顺序添加提供商
        router_config = embed_config.get('router', {})
        primary_provider = router_config.get('primary_provider', 'bailian')
        fallback_providers = router_config.get('fallback_providers', [])
        
        # 按优先级顺序添加提供商
        provider_order = [primary_provider] + fallback_providers
        
        for provider_name in provider_order:
            if provider_name == 'bailian' and 'bailian' in embed_config:
                config = embed_config['bailian']
                provider = BailianEmbeddingProvider(
                    api_key=config['api_key'],
                    model=config.get('model', 'text-embedding-v4'),
                    api_url=config.get('base_url'),
                    dimensions=config.get('dimensions', 1536),
                    batch_size=config.get('batch_size', 10)  # 修复：添加batch_size参数
                )
                providers.append(provider)
                provider_names.append('bailian')
                
            elif provider_name == 'openai' and 'openai' in embed_config:
                config = embed_config['openai']
                provider = OpenAIEmbeddingProvider(
                    api_key=config['api_key'],
                    model=config.get('model', 'text-embedding-3-small'),
                    base_url=config.get('base_url'),
                    dimensions=config.get('dimensions', 1536)
                )
                providers.append(provider)
                provider_names.append('openai')
                
            elif provider_name == 'siliconflow' and 'siliconflow' in embed_config:
                config = embed_config['siliconflow']
                provider = SiliconFlowEmbeddingProvider(
                    api_key=config['api_key'],
                    model=config.get('model', 'BAAI/bge-large-zh-v1.5'),
                    dimensions=config.get('dimensions', 1024)
                )
                providers.append(provider)
                provider_names.append('siliconflow')
        
        # 创建路由器（只传入providers列表）
        self.embedding_router = EmbeddingRouter(providers=providers)
        
        self.logger.info(f"嵌入服务路由器初始化完成，支持提供商: {provider_names}")

    async def _initialize_storage(self) -> None:
        """初始化存储管理器"""
        from agenticx.storage import StorageConfig, StorageType
        
        storage_config = self.config['storage']
        configs = []
        
        # 动态获取嵌入维度
        embedding_dim = self.embedding_router.get_embedding_dim()
        self.logger.info(f"动态获取的嵌入维度: {embedding_dim}")

        # 配置向量存储
        vector_config = storage_config['vector']
        if vector_config['provider'] == 'milvus':
            try:
                milvus_config = vector_config.get('milvus', {})
                # 构建extra_params，过滤掉None值
                extra_params = {
                    'dimension': embedding_dim,
                    'collection_name': milvus_config.get('collection_name', 'agenticx_graphrag'),
                    'database': milvus_config.get('database', 'default'),
                    'recreate_if_exists': True  # 临时重新创建以匹配新的1024维度
                }
                
                # 只在username和password不为None时才添加
                username = milvus_config.get('username')
                password = milvus_config.get('password')
                if username is not None:
                    extra_params['username'] = username
                if password is not None:
                    extra_params['password'] = password
                
                # 调试信息：打印所有参数
                self.logger.info(f"🔍 Milvus配置参数: host={milvus_config.get('host', 'localhost')}, port={milvus_config.get('port', 19530)}")
                self.logger.info(f"🔍 Milvus extra_params: {extra_params}")
                
                configs.append(StorageConfig(
                    storage_type=StorageType.MILVUS,
                    host=milvus_config.get('host', 'localhost'),
                    port=milvus_config.get('port', 19530),
                    extra_params=extra_params
                ))
            except Exception as e:
                self.logger.error(f"❌ 配置Milvus存储失败: {e}")
                self.logger.warning(f"⚠️ Milvus配置失败: {e}，回退到Chroma")
                # 回退到Chroma
                chroma_config = vector_config.get('chroma', {})
                configs.append(StorageConfig(
                    storage_type=StorageType.CHROMA,
                    extra_params={
                        'dimension': embedding_dim,
                        'persist_directory': chroma_config.get('persist_directory', './storage/chroma'),
                        'collection_name': chroma_config.get('collection_name', 'agenticx_graphrag')
                    }
                ))
                self.logger.info("✅ 配置Chroma向量存储（回退）")
        elif vector_config['provider'] == 'chroma':
            chroma_config = vector_config['chroma'].copy()
            configs.append(StorageConfig(
                storage_type=StorageType.CHROMA,
                extra_params={
                    'dimension': embedding_dim,
                    'persist_directory': chroma_config.get('persist_directory', './storage/chroma'),
                    'collection_name': chroma_config.get('collection_name', 'agenticx_graphrag')
                }
            ))
            self.logger.info("✅ 配置Chroma向量存储")
        elif vector_config['provider'] == 'faiss':
            configs.append(StorageConfig(
                storage_type=StorageType.FAISS,
                extra_params={
                    'dimension': embedding_dim,  # 使用动态获取的维度
                    'index_path': './storage/faiss_index'
                }
            ))
            self.logger.info("✅ 配置FAISS向量存储")
        
        # 配置图存储
        graph_config = storage_config['graph']
        if graph_config['provider'] == 'neo4j':
            neo4j_config = graph_config['neo4j'].copy()
            
            # 处理URI格式的配置
            uri = neo4j_config.get('uri', 'bolt://localhost:7687')
            # 将容器名neo4j替换为localhost（用于外部访问）
            if 'neo4j:' in uri:
                uri = uri.replace('neo4j:', 'localhost:')
            
            configs.append(StorageConfig(
                storage_type=StorageType.NEO4J,
                connection_string=uri,
                username=neo4j_config.get('username', 'neo4j'),
                password=neo4j_config.get('password', None),
                database=neo4j_config.get('database', 'neo4j'),
                extra_params={
                    'max_connection_lifetime': neo4j_config.get('max_connection_lifetime', 3600),
                    'max_connection_pool_size': neo4j_config.get('max_connection_pool_size', 50),
                    'connection_acquisition_timeout': neo4j_config.get('connection_acquisition_timeout', 60)
                }
            ))
        
        # 配置键值存储
        kv_config = storage_config['key_value']
        if kv_config['provider'] == 'redis':
            redis_config = kv_config['redis'].copy()
            configs.append(StorageConfig(
                storage_type=StorageType.REDIS,
                host=redis_config.pop('host', 'localhost'),
                port=redis_config.pop('port', 6379),
                password=redis_config.pop('password', None),
                database=redis_config.pop('database', 0),
                **redis_config
            ))
        elif kv_config['provider'] == 'sqlite':
            sqlite_config = kv_config.get('sqlite', {})
            configs.append(StorageConfig(
                storage_type=StorageType.SQLITE,
                connection_string=sqlite_config.get('database_path', './storage/cache.db')
            ))
            self.logger.info("✅ 配置SQLite键值存储")
        
        # 创建存储管理器
        self.storage_manager = StorageManager(configs=configs)
        await self.storage_manager.initialize()
        
        # 调试：检查键值存储是否可用
        kv_storage = await self.storage_manager.get_key_value_storage('default')
        if kv_storage:
            self.logger.info(f"✅ 键值存储可用: {type(kv_storage).__name__}")
        else:
            self.logger.warning("⚠️ 键值存储不可用")
    
    async def _initialize_retriever(self) -> None:
        """初始化检索器"""
        from agenticx.storage import StorageType
        
        retrieval_config = self.config['retrieval']
        
        # 创建向量检索器
        vector_storage = self.storage_manager.get_storage(StorageType.MILVUS)
        vector_retriever = VectorRetriever(
            tenant_id="default",
            vector_storage=vector_storage,
            embedding_provider=self.embedding_router,
            **retrieval_config.get('vector', {})
        )
        
        # 创建BM25检索器
        bm25_retriever = BM25Retriever(
            tenant_id="default",
            **retrieval_config.get('bm25', {})
        )
        
        # 创建图检索器
        graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
        graph_retriever = GraphRetriever(
            tenant_id="default",
            graph_storage=graph_storage,
            **retrieval_config.get('graph', {})
        )
        
        # 创建混合检索器
        from agenticx.retrieval.hybrid_retriever import HybridConfig
        hybrid_config = HybridConfig(**retrieval_config.get('hybrid', {}))
        self.retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            config=hybrid_config
        )
        
        # 存储图检索器以备后用
        self.graph_retriever = graph_retriever
        
        # 如果配置了重排序器，添加重排序器
        if 'reranker' in retrieval_config:
            reranker = Reranker(retrieval_config['reranker'])
            # 注意：HybridRetriever 可能没有 set_reranker 方法，这里先注释掉
            # self.retriever.set_reranker(reranker)
        
        self.logger.info("检索器初始化完成")
    
    def validate_data_path(self) -> List[Path]:
        """验证数据路径并返回文件列表"""
        self.logger.info(f"验证数据路径: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 支持的文件类型
        supported_extensions = {'.pdf', '.txt', '.json', '.csv', '.md'}
        
        # 扫描文件
        files = []
        for file_path in self.data_dir.rglob('*'):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                files.append(file_path)
        
        if not files:
            raise ValueError(f"数据目录中没有找到支持的文件类型: {supported_extensions}")
        
        self.logger.info(f"找到 {len(files)} 个文件: {[f.name for f in files]}")
        return files
    
    async def load_documents(self, file_paths: List[Path]) -> List[Document]:
        """加载文档"""
        self.logger.info(f"开始加载 {len(file_paths)} 个文档...")
        
        documents = []
        reader_config = self.config['knowledge']['readers']
        
        for file_path in file_paths:
            try:
                # 根据文件类型选择读取器
                if file_path.suffix.lower() == '.pdf' and reader_config['pdf']['enabled']:
                    pdf_config = reader_config['pdf'].copy()
                    pdf_config.pop('enabled', None)  # 移除enabled字段
                    reader = PDFReader(**pdf_config)
                elif file_path.suffix.lower() in ['.txt', '.md'] and reader_config['text']['enabled']:
                    text_config = reader_config['text'].copy()
                    text_config.pop('enabled', None)  # 移除enabled字段
                    self.logger.debug(f"TextReader配置: {text_config}")
                    reader = TextReader(**text_config)
                elif file_path.suffix.lower() == '.json' and reader_config['json']['enabled']:
                    json_config = reader_config['json'].copy()
                    json_config.pop('enabled', None)  # 移除enabled字段
                    reader = JSONReader(**json_config)
                elif file_path.suffix.lower() == '.csv' and reader_config['csv']['enabled']:
                    csv_config = reader_config['csv'].copy()
                    csv_config.pop('enabled', None)  # 移除enabled字段
                    reader = CSVReader(**csv_config)
                else:
                    self.logger.warning(f"不支持的文件类型或已禁用: {file_path}")
                    continue
                
                # 读取文档
                self.logger.debug(f"开始读取文档: {file_path}")
                result = await reader.read(str(file_path))
                self.logger.debug(f"读取结果类型: {type(result)}, 内容: {result if not isinstance(result, list) else f'列表长度: {len(result)}'}")
                
                # 处理返回结果（可能是单个文档或文档列表）
                if isinstance(result, list):
                    # 如果返回的是列表，遍历每个文档
                    for doc in result:
                        documents.append(doc)
                        self.logger.info(f"成功加载文档: {file_path.name} ({len(doc.content)} 字符)")
                else:
                    # 如果返回的是单个文档
                    documents.append(result)
                    self.logger.info(f"成功加载文档: {file_path.name} ({len(result.content)} 字符)")
                
            except Exception as e:
                self.logger.error(f"加载文档失败 {file_path}: {e}")
                continue
        
        self.logger.info(f"文档加载完成，共 {len(documents)} 个文档")
        return documents
    
    async def build_knowledge_graph(self, documents: List[Document]) -> None:
        """构建知识图谱"""
        self.logger.info("开始构建知识图谱...")
        
        # 配置 GraphRAG 构造器
        from agenticx.knowledge.graphers.config import LLMConfig, GraphRagConfig
        from agenticx.knowledge.base import ChunkingConfig
        
        graphrag_config_dict = self.config['knowledge']['graphrag']
        
        # 调试：打印原始配置
        self.logger.info(f"🔍 原始GraphRAG配置字典: {graphrag_config_dict}")
        self.logger.info(f"🔍 extraction_method配置: {graphrag_config_dict.get('extraction_method', '未设置')}")
        
        # 将字典转换为 GraphRagConfig 对象
        graphrag_config = GraphRagConfig.from_dict(graphrag_config_dict)
        self.logger.info(f"🔍 转换后的GraphRAG配置: extraction_method={getattr(graphrag_config, 'extraction_method', '未设置')}")
        self.logger.debug(f"GraphRAG配置: {graphrag_config}")
        llm_config_dict = self.config['llm']
        
        # 转换为 LLMConfig 对象
        provider = llm_config_dict.get('provider', 'litellm')
        
        # 提供商类型映射
        provider_type_mapping = {
            'openai': 'litellm',
            'anthropic': 'litellm', 
            'litellm': 'litellm',
            'bailian': 'bailian',
            'kimi': 'kimi'
        }
        
        llm_type = provider_type_mapping.get(provider, 'litellm')
        
        # 手动创建 LLMConfig，确保 type 字段正确
        llm_config = LLMConfig(
            type=llm_type,
            model=llm_config_dict.get('model'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            provider=provider,
            timeout=llm_config_dict.get('timeout'),
            max_retries=llm_config_dict.get('retry_attempts'),  # 配置文件中是retry_attempts
            temperature=llm_config_dict.get('temperature'),
            max_tokens=llm_config_dict.get('max_tokens')
        )
        
        # 创建强模型配置（用于文档分析和Schema生成）
        strong_model_dict = llm_config_dict.get('strong_model', llm_config_dict)
        strong_provider = strong_model_dict.get('provider', provider)
        strong_llm_type = provider_type_mapping.get(strong_provider, 'litellm')
        
        strong_llm_config = LLMConfig(
            type=strong_llm_type,
            model=strong_model_dict.get('model'),
            api_key=strong_model_dict.get('api_key'),
            base_url=strong_model_dict.get('base_url'),
            provider=strong_provider,
            timeout=strong_model_dict.get('timeout'),
            max_retries=strong_model_dict.get('retry_attempts', 3),
            temperature=strong_model_dict.get('temperature'),
            max_tokens=strong_model_dict.get('max_tokens')
        )
        
        # 将强模型配置添加到graphrag_config中
        graphrag_config.strong_model_config = strong_llm_config
        
        self.logger.info(f"🤖 默认模型: {llm_config.model}")
        self.logger.info(f"🚀 强模型: {strong_llm_config.model}")
        
        self.logger.debug(f"GraphRAG LLMConfig: type={llm_config.type}, provider={llm_config.provider}, model={llm_config.model}")
        
        # 使用分块器对文档进行分块
        chunking_config_dict = self.config['knowledge']['chunking']
        strategy = chunking_config_dict.get('strategy', 'semantic')
        chunk_size = chunking_config_dict.get('chunk_size', 800)
        chunk_overlap = chunking_config_dict.get('chunk_overlap', 150)
        
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.logger.info(f"使用分块策略: {strategy}, 分块大小: {chunk_size}")
        
        # 对所有文档进行分块
        all_chunks = []
        for i, document in enumerate(documents):
            self.logger.debug(f"正在分块文档 {i+1}/{len(documents)}: {document.metadata.name}")
            
            # 检查文档是否需要分块
            if len(document.content) > chunk_size:
                # 使用分块器，确保所有策略都能获得嵌入模型（用于后续向量化）
                chunker_kwargs = {
                    'embedding_model': self.embedding_router  # 所有分块器都需要嵌入模型
                }
                
                if strategy == "semantic":
                    chunker_kwargs['similarity_threshold'] = chunking_config_dict.get('semantic', {}).get('similarity_threshold', 0.8)
                    chunker_kwargs['min_chunk_size'] = chunking_config_dict.get('min_chunk_size', 100)
                    chunker_kwargs['max_chunk_size'] = chunking_config_dict.get('max_chunk_size', 1200)
                    self.logger.info(f"🔍 启用语义分块，相似度阈值: {chunker_kwargs['similarity_threshold']}")
                elif strategy == "agentic":
                    chunker_kwargs['llm_client'] = self.llm_client
                    agentic_config = chunking_config_dict.get('agentic', {})
                    chunker_kwargs.update(agentic_config)
                    self.logger.info("🤖 启用Agentic智能分块")
                elif strategy == "fixed_size":
                    self.logger.info("📏 使用固定大小分块（性能优化）")
                
                chunker = get_chunker(strategy, **chunker_kwargs)
                
                # 详细日志：显示分块器类型和配置
                self.logger.info(f"🔧 分块器类型: {type(chunker).__name__}")
                if hasattr(chunker, 'embedding_model') and chunker.embedding_model:
                    self.logger.info(f"🤖 嵌入模型: {type(chunker.embedding_model).__name__}")
                    self.logger.info(f"📊 相似度阈值: {getattr(chunker, 'similarity_threshold', 'N/A')}")
                else:
                    self.logger.warning("⚠️ 未检测到嵌入模型，可能使用回退策略")
                
                self.logger.info(f"📄 开始分块文档: {document.metadata.name} (长度: {len(document.content)} 字符)")
                chunks = await chunker.chunk_document_async(document)
                
                # 详细日志：显示分块结果
                if hasattr(chunks, 'strategy_used'):
                    self.logger.info(f"🎯 实际使用的分块策略: {chunks.strategy_used}")
                if hasattr(chunks, 'metadata') and chunks.metadata:
                    metadata = chunks.metadata
                    if 'original_sentences' in metadata:
                        self.logger.info(f"📝 原始句子数: {metadata['original_sentences']}")
                    if 'similarity_threshold' in metadata:
                        self.logger.info(f"🎚️ 相似度阈值: {metadata['similarity_threshold']}")
                
                self.logger.info(f"✅ 分块完成: {len(chunks.chunks)} 个块")
                
                if hasattr(chunks, 'chunks'):
                    # 如果返回的是ChunkingResult对象
                    chunk_docs = chunks.chunks
                else:
                    # 如果直接返回文档列表
                    chunk_docs = chunks
                
                all_chunks.extend(chunk_docs)
                self.logger.info(f"文档 {document.metadata.name} 被分成 {len(chunk_docs)} 个块")
            else:
                # 文档足够小，不需要分块
                all_chunks.append(document)
                self.logger.debug(f"文档 {document.metadata.name} 无需分块")
        
        self.logger.info(f"分块完成，共 {len(all_chunks)} 个文本块")
        
        # 使用新的KnowledgeGraphBuilder进行两阶段SPO抽取
        builder = KnowledgeGraphBuilder(
            config=graphrag_config,
            llm_config=llm_config
        )
        
        # 提取分块后的文档文本
        texts = [chunk.content for chunk in all_chunks]
        metadata = [chunk.metadata.__dict__ for chunk in all_chunks]
        
        # 构建图谱（使用批处理SPO抽取）
        self.knowledge_graph = await builder.build_from_texts(
            texts=texts,
            metadata=metadata
        )
        
        self.logger.info(
            f"知识图谱构建完成: "
            f"{len(self.knowledge_graph.entities)} 个实体, "
            f"{len(self.knowledge_graph.relationships)} 个关系"
        )
    
    async def store_and_index(self) -> None:
        """存储和索引知识图谱"""
        self.logger.info("开始存储和索引知识图谱...")
        
        # 1. 存储到图数据库
        try:
            self.logger.debug("🔍 开始查找图数据库存储...")
            graph_storage = await self.storage_manager.get_graph_storage('default')
            
            if graph_storage:
                self.logger.info(f"✅ 找到图存储: {type(graph_storage).__name__}")
                graph_storage.store_graph(self.knowledge_graph, clear_existing=True) # 确保清空
                self.logger.info("知识图谱已存储到图数据库")
            else:
                self.logger.warning("⚠️ 未找到图数据库存储，跳过图谱存储")
                # 调试信息：检查所有存储
                self.logger.debug(f"📊 当前存储实例数量: {len(self.storage_manager.storages)}")
                for i, storage in enumerate(self.storage_manager.storages):
                    storage_type = type(storage).__name__
                    has_store_graph = hasattr(storage, 'store_graph')
                    has_add_triplet = hasattr(storage, 'add_triplet')
                    has_add_node = hasattr(storage, 'add_node')
                    self.logger.debug(f"  存储 {i+1}: {storage_type} - store_graph:{has_store_graph}, add_triplet:{has_add_triplet}, add_node:{has_add_node}")
        except Exception as e:
            self.logger.error(f"❌ 图数据库存储失败: {e}")
            import traceback
            self.logger.debug(f"❌ 错误堆栈: {traceback.format_exc()}")
            self.logger.warning("💡 继续执行其他索引步骤...")
        
        # 2. 构建向量索引
        try:
            await self._build_vector_index()
        except Exception as e:
            self.logger.error(f"❌ 向量索引构建失败: {e}")
        
        # 3. 构建 SPO 索引
        try:
            await self._build_spo_index()
        except Exception as e:
            self.logger.error(f"❌ SPO索引构建失败: {e}")
        
        # 4. 缓存关键数据
        try:
            await self._cache_key_data()
        except Exception as e:
            self.logger.error(f"❌ 数据缓存失败: {e}")
        
        self.logger.info("存储和索引完成")
    
    async def _build_vector_index(self) -> None:
        """构建向量索引"""
        from agenticx.storage import StorageType
        
        self.logger.info("构建向量索引...")
        
        # 尝试获取向量存储
        vector_storage = await self.storage_manager.get_vector_storage('default')
        if not vector_storage:
            # 回退到通过类型获取
            vector_storage = self.storage_manager.get_storage(StorageType.CHROMA)
            if not vector_storage:
                self.logger.warning("⚠️ 未找到向量存储，跳过向量索引构建")
                return
        
        # 为实体构建向量索引
        entity_records = []
        for entity in self.knowledge_graph.entities.values():
            # 生成实体描述文本
            entity_text = f"{entity.name}: {entity.description or ''}"
            
            # 生成嵌入
            embedding = await self.embedding_router.aembed_text(entity_text)
            
            # 创建向量记录
            record = VectorRecord(
                id=entity.id,
                vector=embedding,
                metadata={
                    'type': 'entity',
                    'entity_type': entity.entity_type.value,
                    'name': entity.name,
                    'confidence': entity.confidence
                },
                content=entity_text
            )
            entity_records.append(record)
        
        # 批量存储
        vector_storage.add(entity_records)
        self.logger.info(f"实体向量索引构建完成: {len(entity_records)} 条记录")
        
        # 为关系构建向量索引
        relationship_records = []
        for relationship in self.knowledge_graph.relationships.values():
            # 生成关系描述文本
            source_entity = self.knowledge_graph.get_entity(relationship.source_entity_id)
            target_entity = self.knowledge_graph.get_entity(relationship.target_entity_id)
            
            if source_entity and target_entity:
                rel_text = f"{source_entity.name} {relationship.relation_type.value} {target_entity.name}"
                
                # 生成嵌入
                embedding = await self.embedding_router.aembed_text(rel_text)
                
                # 创建向量记录
                record = VectorRecord(
                    id=relationship.id,
                    vector=embedding,
                    metadata={
                        'type': 'relationship',
                        'relation_type': relationship.relation_type.value,
                        'source_entity': source_entity.name,
                        'target_entity': target_entity.name,
                        'confidence': relationship.confidence
                    },
                    content=rel_text
                )
                relationship_records.append(record)
        
        vector_storage.add(relationship_records)
        self.logger.info(f"关系向量索引构建完成: {len(relationship_records)} 条记录")
    
    async def _build_spo_index(self) -> None:
        """构建 SPO 三元组索引"""
        import json
        self.logger.info("构建 SPO 三元组索引...")
        
        # 调试：检查存储管理器状态
        self.logger.info(f"存储管理器初始化状态: {self.storage_manager.initialized}")
        self.logger.info(f"存储实例数量: {len(self.storage_manager.storages)}")
        
        kv_storage = await self.storage_manager.get_key_value_storage('default')
        self.logger.info(f"获取到的键值存储: {type(kv_storage) if kv_storage else 'None'}")
        
        if not kv_storage:
            self.logger.warning("⚠️ 未找到键值存储，跳过SPO索引构建")
            return
        
        # 构建 SPO 索引
        spo_index = {
            'subject_index': {},  # 主语索引
            'predicate_index': {},  # 谓语索引
            'object_index': {}  # 宾语索引
        }
        
        for relationship in self.knowledge_graph.relationships.values():
            source_entity = self.knowledge_graph.get_entity(relationship.source_entity_id)
            target_entity = self.knowledge_graph.get_entity(relationship.target_entity_id)
            
            if source_entity and target_entity:
                subject = source_entity.name
                predicate = relationship.relation_type.value
                object_name = target_entity.name
                
                # 主语索引
                if subject not in spo_index['subject_index']:
                    spo_index['subject_index'][subject] = []
                spo_index['subject_index'][subject].append({
                    'predicate': predicate,
                    'object': object_name,
                    'relationship_id': relationship.id
                })
                
                # 谓语索引
                if predicate not in spo_index['predicate_index']:
                    spo_index['predicate_index'][predicate] = []
                spo_index['predicate_index'][predicate].append({
                    'subject': subject,
                    'object': object_name,
                    'relationship_id': relationship.id
                })
                
                # 宾语索引
                if object_name not in spo_index['object_index']:
                    spo_index['object_index'][object_name] = []
                spo_index['object_index'][object_name].append({
                    'subject': subject,
                    'predicate': predicate,
                    'relationship_id': relationship.id
                })
        
        # 存储索引（序列化为JSON）
        kv_storage.set('spo_index', json.dumps(spo_index, ensure_ascii=False))
        
        self.logger.info(
            f"SPO 索引构建完成: "
            f"{len(spo_index['subject_index'])} 个主语, "
            f"{len(spo_index['predicate_index'])} 个谓语, "
            f"{len(spo_index['object_index'])} 个宾语"
        )
    
    async def _cache_key_data(self) -> None:
        """缓存关键数据"""
        import json
        from datetime import datetime, timezone
        self.logger.info("缓存关键数据...")
        
        kv_storage = await self.storage_manager.get_key_value_storage('default')
        if not kv_storage:
            self.logger.warning("⚠️ 未找到键值存储，跳过数据缓存")
            return
        
        # 缓存图谱统计信息
        stats = {
            'entity_count': len(self.knowledge_graph.entities),
            'relationship_count': len(self.knowledge_graph.relationships),
            'entity_types': {},
            'relationship_types': {},
            'build_time': datetime.now(timezone.utc).isoformat()
        }
        
        # 统计实体类型
        for entity in self.knowledge_graph.entities.values():
            entity_type = entity.entity_type.value
            stats['entity_types'][entity_type] = stats['entity_types'].get(entity_type, 0) + 1
        
        # 统计关系类型
        for relationship in self.knowledge_graph.relationships.values():
            rel_type = relationship.relation_type.value
            stats['relationship_types'][rel_type] = stats['relationship_types'].get(rel_type, 0) + 1
        
        kv_storage.set('graph_stats', json.dumps(stats, ensure_ascii=False))
        self.logger.info("关键数据缓存完成")
    
    async def interactive_qa(self) -> None:
        """交互式问答系统"""
        self.logger.info("启动交互式问答系统...")
        
        print("\n" + "="*60)
        print("🤖 AgenticX GraphRAG 问答系统")
        print("="*60)
        print("支持的查询类型:")
        print("  1. 实体查询: 查找特定实体的信息")
        print("  2. 关系查询: 查找实体间的关系")
        print("  3. 路径查询: 查找实体间的连接路径")
        print("  4. 社区查询: 查找相关实体群组")
        print("  5. 自由问答: 基于知识图谱的开放式问答")
        print("\n输入 'quit' 或 'exit' 退出系统")
        print("="*60 + "\n")
        
        while True:
            try:
                # 获取用户输入
                query = input("🔍 请输入您的问题: ").strip()
                
                if query.lower() in ['quit', 'exit', '退出']:
                    print("👋 感谢使用 AgenticX GraphRAG 系统！")
                    break
                
                if not query:
                    continue
                
                # 处理查询
                await self._process_query(query)
                
            except KeyboardInterrupt:
                print("\n👋 感谢使用 AgenticX GraphRAG 系统！")
                break
            except Exception as e:
                self.logger.error(f"查询处理错误: {e}")
                print(f"❌ 查询处理出错: {e}")
    
    async def _process_query(self, query: str) -> None:
        """处理用户查询"""
        print(f"\n🔄 正在处理查询: {query}")
        
        try:
            # 获取检索配置
            retrieval_config = self.config.get('retrieval', {})
            graph_config = retrieval_config.get('graph', {})
            vector_config = retrieval_config.get('vector', {})
            
            # 先尝试图检索（适合实体查询）
            graph_results = []
            if hasattr(self, 'graph_retriever') and self.graph_retriever:
                try:
                    graph_top_k = min(graph_config.get('max_nodes', 50), 10)  # 限制在10以内
                    graph_results = await self.graph_retriever.retrieve(query, top_k=graph_top_k)
                    if graph_results:
                        print(f"🔍 图检索找到 {len(graph_results)} 条结果 (top_k={graph_top_k})")
                except Exception as e:
                    self.logger.warning(f"图检索失败: {e}")
            
            # 使用混合检索器进行查询
            hybrid_top_k = vector_config.get('top_k', 20)
            hybrid_results = await self.retriever.retrieve(query, top_k=hybrid_top_k)
            print(f"🔍 混合检索找到 {len(hybrid_results)} 条结果 (top_k={hybrid_top_k})")
            
            # 合并结果
            all_results = graph_results + hybrid_results
            
            # 去重并按相似度排序
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = getattr(result, 'id', str(hash(result.content)))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # 按相似度排序
            unique_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            results = unique_results[:5]  # 取前5个
            
            if not results:
                print("❌ 没有找到相关信息")
                # 尝试直接在Neo4j中搜索实体
                await self._search_entity_directly(query)
                return
            
            # 显示检索配置信息
            similarity_threshold = vector_config.get('similarity_threshold', 0.7)
            print(f"\n✅ 找到 {len(results)} 条相关信息 (相似度阈值: {similarity_threshold}):\n")
            
            # 显示检索结果
            for i, result in enumerate(results, 1):
                score = getattr(result, 'score', 0)
                score_status = "✅" if score >= similarity_threshold else "⚠️"
                print(f"📄 结果 {i} {score_status} (相似度: {score:.3f})")
                print(f"   内容: {result.content[:200]}...")
                if result.metadata:
                    print(f"   元数据: {result.metadata}")
                print()
            
            # 生成答案
            await self._generate_answer(query, results)
        
        except Exception as e:
            self.logger.error(f"查询处理错误: {e}")
            print(f"❌ 查询处理出错: {e}")
    
    async def _search_entity_directly(self, query: str) -> None:
        """直接在Neo4j中搜索实体"""
        try:
            from agenticx.storage import StorageType
            graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
            
            if not graph_storage:
                print("⚠️ 图存储不可用")
                return
            
            # 提取可能的实体名称
            entity_name = query.replace("是啥", "").replace("是什么", "").replace("?", "").replace("？", "").strip()
            
            # 在Neo4j中搜索实体
            cypher_query = """
            MATCH (n:Entity) 
            WHERE n.name CONTAINS $entity_name OR n.name =~ ('(?i).*' + $entity_name + '.*')
            RETURN n.name as name, n.description as description, labels(n) as type
            LIMIT 5
            """
            
            results = graph_storage.execute_query(cypher_query, {"entity_name": entity_name})
            
            if results:
                print(f"\n🔍 在知识图谱中找到相关实体:")
                for result in results:
                    print(f"📍 实体: {result['name']}")
                    print(f"   类型: {result['type']}")
                    if result['description']:
                        print(f"   描述: {result['description']}")
                    print()
            else:
                print(f"❌ 在知识图谱中未找到 '{entity_name}' 相关的实体")
                
        except Exception as e:
            self.logger.error(f"直接实体搜索失败: {e}")
            print(f"❌ 实体搜索出错: {e}")
    
    async def _generate_answer(self, query: str, results: List[Any]) -> None:
        """基于检索结果生成答案"""
        try:
            # 构建上下文
            context = "\n".join([result.content for result in results[:3]])
            
            # 构建提示词
            prompt = f"""
基于以下知识图谱信息回答用户问题。请提供准确、简洁的答案。

知识图谱信息:
{context}

用户问题: {query}

请回答:
"""
            
            # 调用 LLM 生成答案
            response = await self.llm_client.ainvoke(prompt)
            
            print("🤖 AI 回答:")
            print("-" * 40)
            print(response.content)
            print("-" * 40)
            
        except Exception as e:
            self.logger.error(f"答案生成错误: {e}")
            print(f"❌ 答案生成出错: {e}")
    
    async def run_demo(self) -> None:
        """运行完整演示流程"""
        try:
            print("🚀 启动 AgenticX GraphRAG 演示系统...")
            
            # 1. 初始化组件
            await self.initialize_components()
            
            # 2. 验证数据路径
            file_paths = self.validate_data_path()
            
            # 3. 加载文档
            documents = await self.load_documents(file_paths)
            
            # 4. 构建知识图谱
            await self.build_knowledge_graph(documents)
            
            # 5. 存储和索引
            await self.store_and_index()
            
            # 6. 启动交互式问答
            await self.interactive_qa()
            
        except Exception as e:
            self.logger.error(f"演示运行错误: {e}")
            print(f"❌ 演示运行出错: {e}")
            raise


async def main():
    """主函数"""
    try:
        # 创建演示系统
        demo = AgenticXGraphRAGDemo()
        
        # 运行演示
        await demo.run_demo()
        
    except Exception as e:
        print(f"❌ 系统启动失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())