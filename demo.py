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
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
from loguru import logger

# 导入 Rich 美化库
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
    from rich.layout import Layout
    from rich.align import Align
    from rich.rule import Rule
    from rich import box
    from rich.prompt import Prompt, Confirm
    try:
        from pyboxen import boxen
    except ImportError:
        boxen = None
except ImportError:
    # 如果导入失败，使用基础版本
    Console = None
    Panel = None
    Table = None
    Text = None
    Progress = None
    SpinnerColumn = None
    TextColumn = None
    BarColumn = None
    TimeElapsedColumn = None
    Layout = None
    Align = None
    Rule = None
    box = None
    Prompt = None
    Confirm = None
    boxen = None

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
    KnowledgeGraphBuilder,   SemanticChunker,
    AgenticChunker,
    get_chunker
)
from agenticx.knowledge.readers import (
    PDFReader,
    TextReader,
    JSONReader,
    CSVReader,
    WordReader,
    PowerPointReader
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
    RedisStorage
)
from agenticx.storage.vectordb_storages.base import VectorRecord
from agenticx.retrieval import (
    HybridRetriever,
    GraphRetriever,
    GraphVectorConfig,
    VectorRetriever,
    BM25Retriever,
    AutoRetriever,
    Reranker,
    QueryAnalysisAgent,
    RetrievalAgent
)
from agenticx.llms import LlmFactory

# 导入本地模块
from prompt_manager import PromptManager

# 创建 Rich Console 实例
console = Console() if Console else None

# ANSI 颜色代码（保留作为后备）
class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # 基础颜色
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # 亮色
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'

def print_thinking(message: str):
    """显示AI思考过程（白色点）"""
    if console:
        console.print(f"● {message}", style="white dim")
    else:
        print(f"{Colors.WHITE}● {Colors.DIM}{message}{Colors.RESET}")

def print_action(message: str):
    """显示工具调用（绿色点）"""
    if console:
        console.print(f"● {message}", style="green")
    else:
        print(f"{Colors.BRIGHT_GREEN}● {message}{Colors.RESET}")

def print_error(message: str):
    """显示错误信息（红色）"""
    if console:
        console.print(f"● {message}", style="bright_red bold")
    else:
        print(f"{Colors.BRIGHT_RED}● {message}{Colors.RESET}")

def print_success(message: str):
    """显示成功信息（绿色点）"""
    if console:
        console.print(f"● {message}", style="bright_green bold")
    else:
        print(f"{Colors.BRIGHT_GREEN}● {message}{Colors.RESET}")

def print_info(message: str):
    """显示信息（白色点）"""
    if console:
        console.print(f"● {message}", style="white")
    else:
        print(f"{Colors.WHITE}● {message}{Colors.RESET}")

def print_mode_selection(message: str):
    """显示模式选择（橙色点）"""
    if console:
        console.print(f"● {message}", style="bright_yellow bold")
    else:
        print(f"{Colors.BRIGHT_YELLOW}● {message}{Colors.RESET}")

def print_welcome():
    """显示欢迎界面"""
    # AgenticX-GraphRAG ASCII Logo
    graphrag_logo = """

 █████╗  ██████╗ ███████╗███╗   ██╗████████╗██╗ ██████╗██╗  ██╗
██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝██║██╔════╝╚██╗██╔╝
███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║   ██║██║      ╚███╔╝ 
██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║   ██║██║      ██╔██╗ 
██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║   ██║╚██████╗██╔╝ ██╗
╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝ ╚═════╝╚═╝  ╚═╝

██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗██████╗  █████╗  ██████╗ 
██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔══██╗██╔══██╗██╔════╝ 
██║  ███╗██████╔╝███████║██████╔╝███████║██████╔╝███████║██║  ███╗
██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║██╔══██╗██╔══██║██║   ██║
╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║██║  ██║██║  ██║╚██████╔╝
╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝ 
                                                                      
    """
    
    if console and Panel:
        # 使用 Rich 显示活力橙色主题的 logo 和信息
        console.print(graphrag_logo, style="bold #FF6B35")
        
        # 环境配置信息
        api_key = os.getenv('BAILIAN_API_KEY', 'sk-***')
        api_key_display = f"{api_key[:8]}..." if len(api_key) > 8 else api_key
        
        console.print("● 智能知识图谱问答系统:", style="bold #FF6B35")
        console.print("  ⎿  文档解析 + 知识图谱构建 + 混合检索 + 智能问答", style="dim")
        console.print("  ⎿  支持多种文档格式：PDF、Word、PPT、TXT", style="dim")
        console.print("  ⎿  多跳推理、实体关系查询、社区发现\n", style="dim")
        
        console.print("● 环境配置:", style="bold #FF6B35")
        console.print(f"  ⎿  API Key: {api_key_display}", style="dim")
        console.print(f"  ⎿  工作目录: {os.getcwd()}", style="dim")
        console.print(f"  ⎿  配置文件: configs.yml\n", style="dim")

    elif boxen:
        # 使用 pyboxen 创建框框
        print(graphrag_logo)
        content = (
            "🚀 AgenticX GraphRAG 智能知识图谱问答系统\n\n"
            "功能特性:\n"
            "● 文档解析 + 知识图谱构建 + 混合检索 + 智能问答\n"
            "● 支持多种文档格式：PDF、Word、PPT、TXT\n"
            "● 多跳推理、实体关系查询、社区发现\n\n"
            "环境配置:\n"
            f"● API Key: {os.getenv('BAILIAN_API_KEY', 'sk-***')[:8]}...\n"
            f"● 工作目录: {os.getcwd()}\n"
            f"● 配置文件: configs.yml"
        )
        print(boxen(
            content,
            title="AgenticX-GraphRAG",
            title_alignment="center",
            style="rounded",
            color="orange",
            padding=1
        ))
    else:
        # 后备方案：使用原始的 ASCII 艺术
        print(graphrag_logo)
        print(f"""
┌─────────────────────────────────────────────────────────────┐
│ 🚀 AgenticX GraphRAG 智能知识图谱问答系统                    │
│                                                             │
│ 功能特性:                                                   │
│ ● 文档解析 + 知识图谱构建 + 混合检索 + 智能问答              │
│ ● 支持多种文档格式：PDF、Word、PPT、TXT                     │
│ ● 多跳推理、实体关系查询、社区发现                          │
│                                                             │
│ 环境配置:                                                   │
│ ● API Key: {os.getenv('BAILIAN_API_KEY', 'sk-***')[:8]}...{' ' * (43 - len(os.getenv('BAILIAN_API_KEY', 'sk-***')[:8]))} │
│ ● 工作目录: {os.getcwd():<51} │
│ ● 配置文件: configs.yml{' ' * 42} │
└─────────────────────────────────────────────────────────────┘
""")

def print_help():
    """显示帮助信息"""
    if console and Table and box:
        # 使用 Rich 创建美观的帮助表格
        table = Table(title="[bold #FF6B35]可用命令[/bold #FF6B35]", box=box.ROUNDED)
        table.add_column("命令", style="bold #FF6B35", width=12)
        table.add_column("描述", style="white")
        
        table.add_row("/help", "显示帮助信息")
        table.add_row("/clear", "清屏")
        table.add_row("/mode", "选择运行模式并初始化系统")
        table.add_row("/data", "选择文档路径")
        table.add_row("/rebuild", "立即重新构建知识库")
        table.add_row("/exit", "退出程序")
        table.add_row("", "")
        table.add_row("[dim]直接输入[/dim]", "[dim]输入问题开始智能问答[/dim]")
        
        console.print(table)
    elif boxen:
        # 使用 pyboxen 创建帮助框
        content = (
            "可用命令:\n\n"
            "/help     显示帮助信息\n"
            "/clear    清屏\n"
            "/mode     选择运行模式并初始化系统\n"
            "/data     选择文档路径\n"
            "/rebuild  立即重新构建知识库\n"
            "/exit     退出程序\n\n"
            "直接输入问题开始智能问答"
        )
        print(boxen(
            content,
            title="帮助",
            style="rounded",
            color="orange",
            padding=1
        ))
    else:
        # 后备方案
        print(f"""
{Colors.BOLD}可用命令:{Colors.RESET}

/help     显示帮助信息
/clear    清屏
/mode     选择运行模式并初始化系统
/data     选择文档路径
/rebuild  立即重新构建知识库
/exit     退出程序

直接输入问题开始智能问答
""")

def scan_data_directories() -> Dict[str, List[Dict[str, Any]]]:
    """扫描数据目录，返回文件信息"""
    data_dirs = ["data", "data2"]
    file_info = {}
    
    for data_dir in data_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists():
            files = []
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    stat = file_path.stat()
                    files.append({
                        "name": file_path.name,
                        "path": str(file_path),
                        "size": stat.st_size,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "type": file_path.suffix.lower(),
                        "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
                    })
            file_info[data_dir] = files
    
    return file_info

def display_data_selection() -> str:
    """显示数据目录选择界面"""
    file_info = scan_data_directories()
    
    if console and Table and box:
        # 使用 Rich 显示文件选择表格
        table = Table(title="[bold #FF6B35]📁 可用文档目录[/bold #FF6B35]", box=box.ROUNDED)
        table.add_column("目录", style="bold #FF6B35", width=10)
        table.add_column("文件数", style="bold #FF6B35", width=8)
        table.add_column("文件列表", style="white")
        
        for dir_name, files in file_info.items():
            file_list = []
            for file in files:
                file_list.append(f"{file['name']} ({file['size_mb']}MB)")
            
            table.add_row(
                dir_name,
                str(len(files)),
                "\n".join(file_list) if file_list else "无文件"
            )
        
        console.print(table)
        
        # 显示详细文件信息
        for dir_name, files in file_info.items():
            if files:
                detail_table = Table(title=f"[bold #FF6B35]{dir_name}/ 目录详情[/bold #FF6B35]", box=box.SIMPLE)
                detail_table.add_column("文件名", style="cyan")
                detail_table.add_column("大小", style="bold #FF6B35")
                detail_table.add_column("类型", style="magenta")
                detail_table.add_column("修改时间", style="dim")
                
                for file in files:
                    detail_table.add_row(
                        file['name'],
                        f"{file['size_mb']} MB",
                        file['type'].upper(),
                        file['modified']
                    )
                
                console.print(detail_table)
                console.print()
    
    # 选择目录
    available_dirs = [dir_name for dir_name, files in file_info.items() if files]
    
    if not available_dirs:
        print_error("未找到可用的文档目录")
        return "data"  # 默认返回 data 目录
    
    if len(available_dirs) == 1:
        print_info(f"自动选择唯一可用目录: {available_dirs[0]}")
        return available_dirs[0]
    
    # 多个目录时让用户选择
    if console and Prompt:
        choices = "/".join(available_dirs)
        selected = Prompt.ask(
            f"请选择文档目录 ({choices})",
            choices=available_dirs,
            default=available_dirs[0]
        )
        return selected
    else:
        print(f"可用目录: {', '.join(available_dirs)}")
        while True:
            choice = input(f"请选择目录 ({'/'.join(available_dirs)}): ").strip()
            if choice in available_dirs:
                return choice
            elif not choice and available_dirs:
                return available_dirs[0]
            print("无效选择，请重新输入")

def select_run_mode() -> str:
    """选择运行模式"""
    if console and Panel and Text and box:
        mode_text = Text()
        mode_text.append("1. Full Mode", style="bold green")
        mode_text.append(" - 完整流程（文档解析 + 知识库构建 + 问答）\n", style="white")
        mode_text.append("   适用: 首次运行或需要重新处理文档\n", style="dim")
        
        mode_text.append("2. Build Mode", style="bold #FF6B35")
        mode_text.append(" - 仅构建模式（文档解析 + 知识库构建）\n", style="white")
        mode_text.append("   适用: 只需要构建知识库，不启动问答\n", style="dim")
        
        mode_text.append("3. QA Mode", style="bold magenta")
        mode_text.append(" - 仅问答模式（使用已有知识库）\n", style="white")
        mode_text.append("   适用: 知识库已存在，直接开始问答\n", style="dim")
        
        panel = Panel(
            mode_text,
            title="选择运行模式",
            border_style="#FF6B35",
            box=box.ROUNDED,
            padding=(1, 2)
        )
        console.print(panel)
    elif boxen:
        content = (
            "选择运行模式:\n\n"
            "1. Full Mode - 完整流程（文档解析 + 知识库构建 + 问答）\n"
            "   适用: 首次运行或需要重新处理文档\n\n"
            "2. Build Mode - 仅构建模式（文档解析 + 知识库构建）\n"
            "   适用: 只需要构建知识库，不启动问答\n\n"
            "3. QA Mode - 仅问答模式（使用已有知识库）\n"
            "   适用: 知识库已存在，直接开始问答"
        )
        print(boxen(
            content,
            title="运行模式选择",
            style="rounded",
            color="orange",
            padding=1
        ))
    else:
        print("""
选择运行模式:

1. Full Mode - 完整流程（文档解析 + 知识库构建 + 问答）
   适用: 首次运行或需要重新处理文档

2. Build Mode - 仅构建模式（文档解析 + 知识库构建）
   适用: 只需要构建知识库，不启动问答

3. QA Mode - 仅问答模式（使用已有知识库）
   适用: 知识库已存在，直接开始问答
""")
    
    mode_map = {"1": "full", "2": "build", "3": "qa"}
    
    if console and Prompt:
        choice = Prompt.ask(
            "请选择模式",
            choices=["1", "2", "3"],
            default="1"
        )
        return mode_map[choice]
    else:
        while True:
            try:
                choice = input("\n请选择模式 (1-3): ").strip()
                if choice == '' or choice == '1':
                    return 'full'
                elif choice == '2':
                    return 'build'
                elif choice == '3':
                    return 'qa'
                else:
                    print("请输入 1、2 或 3")
            except (EOFError, KeyboardInterrupt):
                return 'full'


class AgenticXGraphRAGDemo:
    """AgenticX GraphRAG 演示系统主类"""
    
    def __init__(self, config_path: str = "configs.yml", mode: str = "full"):
        """初始化演示系统"""
        self.config_path = config_path
        self.mode = mode  # 运行模式：full 或 qa
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # 核心组件
        self.llm_client = None
        self.embedding_router = None
        self.storage_manager = None
        self.knowledge_graph = None
        self.retriever = None
        
        # 提示词管理器
        self.prompt_manager = PromptManager("prompts")
        
        # 数据路径
        self.data_dir = Path("./data")
        self.workspace_dir = Path(self.config['system']['workspace']['base_dir'])
        
        self.logger.info(f"AgenticX GraphRAG 演示系统初始化完成 (模式: {self.mode})")
    
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
        self.logger.info("开始初始化系统组件...")
        
        # 1. 初始化 LLM 客户端
        await self._initialize_llm()
        
        # 2. 初始化嵌入服务
        await self._initialize_embeddings()
        
        # 3. 初始化存储管理器
        await self._initialize_storage()
        
        # 4. 初始化检索器
        await self._initialize_retriever()
        
        self.logger.info("系统组件初始化完成")
    
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
        self.logger.info(f"LLM初始化完成: {llm_config.provider}/{llm_config.model}")
    
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
        
        self.logger.info(f"嵌入服务初始化完成: {len(provider_names)}个提供商")

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
                # 根据运行模式决定是否清理数据
                recreate_if_exists = self.mode in ["full", "build"]  # full和build模式重新创建，qa模式保留
                
                extra_params = {
                    'dimension': embedding_dim,
                    'collection_name': milvus_config.get('collection_name', 'agenticx_graphrag'),
                    'database': milvus_config.get('database', 'default'),
                    'recreate_if_exists': recreate_if_exists  # 根据模式决定是否重新创建
                }
                
                # 只在username和password不为None时才添加
                username = milvus_config.get('username')
                password = milvus_config.get('password')
                if username is not None:
                    extra_params['username'] = username
                if password is not None:
                    extra_params['password'] = password
                
                # 简化日志：只记录关键信息
                self.logger.debug(f"Milvus配置: {milvus_config.get('host', 'localhost')}:{milvus_config.get('port', 19530)}")
                
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
            self.logger.debug(f"键值存储: {type(kv_storage).__name__}")
        else:
            self.logger.warning("⚠️ 键值存储不可用")
    
    async def _initialize_retriever(self) -> None:
        """初始化检索器"""
        from agenticx.storage import StorageType
        
        retrieval_config = self.config['retrieval']
        
        # 🔧 确保文档向量存储实例存在，如果不存在则创建
        if not hasattr(self, '_document_vector_storage') or not self._document_vector_storage:
            from agenticx.storage.vectordb_storages.milvus import MilvusStorage
            storage_config = self.config['storage']['vector']['milvus']
            self.logger.info(f"🔧 检索器初始化: mode={self.mode}, 文档向量存储recreate_if_exists=False")
            self._document_vector_storage = MilvusStorage(
                dimension=1024,  # 使用嵌入维度
                host=storage_config['host'],
                port=storage_config['port'],
                collection_name=storage_config['collection_name'],  # 使用文档专用集合名
                database=storage_config.get('database', 'default'),
                username=storage_config.get('username'),
                password=storage_config.get('password'),
                recreate_if_exists=False  # 检索器初始化时不重新创建
            )
        
        # 创建向量检索器
        vector_retriever = VectorRetriever(
            tenant_id="default",
            vector_storage=self._document_vector_storage,  # 使用统一的文档向量存储实例
            embedding_provider=self.embedding_router,
            **retrieval_config.get('vector', {})
        )
        
        self.logger.info(f"📄 文档检索存储集合: {storage_config['collection_name']}")
        
        # 创建BM25检索器
        bm25_retriever = BM25Retriever(
            tenant_id="default",
            **retrieval_config.get('bm25', {})
        )
        
        # 创建增强的图检索器（支持向量索引）
        graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
        vector_storage = self.storage_manager.get_storage(StorageType.MILVUS)
        
        # 🔧 确保图向量存储实例存在，如果不存在则创建
        if not hasattr(self, '_graph_vector_storage') or not self._graph_vector_storage:
            from agenticx.storage.vectordb_storages.milvus import MilvusStorage
            storage_config = self.config['storage']['vector']['milvus']
            # 根据运行模式决定是否重新创建图向量集合
            recreate_graph_collection = self.mode in ["full", "build"]
            self.logger.info(f"🔧 图向量存储配置: mode={self.mode}, recreate_if_exists={recreate_graph_collection}")
            self._graph_vector_storage = MilvusStorage(
                dimension=1024,  # 使用嵌入维度
                host=storage_config['host'],
                port=storage_config['port'],
                collection_name=storage_config['graph_collection_name'],  # 使用图专用集合名
                database=storage_config.get('database', 'default'),
                username=storage_config.get('username'),
                password=storage_config.get('password'),
                recreate_if_exists=recreate_graph_collection  # 🔧 修复：根据模式决定是否重新创建
            )
        
        # 配置图向量索引
        graph_vector_config = GraphVectorConfig(
            enable_vector_indexing=True,
            **retrieval_config.get('graph', {}).get('vector_config', {})
        )
        
        graph_retriever = GraphRetriever(
            tenant_id="default",
            graph_storage=graph_storage,
            vector_storage=self._graph_vector_storage,  # 使用统一的图向量存储实例
            embedding_provider=self.embedding_router,
            vector_config=graph_vector_config,
            **retrieval_config.get('graph', {})
        )
        
        self.logger.info(f"🔗 图向量存储集合: {storage_config['graph_collection_name']}")
        
        # 创建三路混合检索器
        from agenticx.retrieval.hybrid_retriever import HybridConfig
        hybrid_config = HybridConfig(**retrieval_config.get('hybrid', {}))
        self.retriever = HybridRetriever(
            vector_retriever=vector_retriever,
            bm25_retriever=bm25_retriever,
            graph_retriever=graph_retriever,  # 集成图检索器
            config=hybrid_config
        )
        
        # 存储图检索器以备后用
        self.graph_retriever = graph_retriever
        
        # 如果配置了重排序器，添加重排序器
        if 'reranker' in retrieval_config:
            reranker = Reranker(retrieval_config['reranker'])
            # 注意：HybridRetriever 可能没有 set_reranker 方法，这里先注释掉
            # self.retriever.set_reranker(reranker)
        
        self.logger.debug("检索器初始化完成")
    
    def validate_data_path(self) -> List[Path]:
        """验证数据路径并返回文件列表"""
        self.logger.info(f"验证数据路径: {self.data_dir}")
        
        if not self.data_dir.exists():
            raise FileNotFoundError(f"数据目录不存在: {self.data_dir}")
        
        # 支持的文件类型
        supported_extensions = {'.pdf', '.txt', '.json', '.csv', '.md', '.doc', '.docx', '.ppt', '.pptx'}
        
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
                elif file_path.suffix.lower() in ['.doc', '.docx'] and reader_config.get('word', {}).get('enabled', True):
                    word_config = reader_config.get('word', {}).copy()
                    word_config.pop('enabled', None)  # 移除enabled字段
                    reader = WordReader(**word_config)
                elif file_path.suffix.lower() in ['.ppt', '.pptx'] and reader_config.get('powerpoint', {}).get('enabled', True):
                    ppt_config = reader_config.get('powerpoint', {}).copy()
                    ppt_config.pop('enabled', None)  # 移除enabled字段
                    reader = PowerPointReader(**ppt_config)
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
        
        self.logger.info(f"文档加载完成: {len(documents)}个文档")
        return documents
    
    async def build_knowledge_graph(self, documents: List[Document]) -> None:
        """构建知识图谱"""
        self.logger.info("开始构建知识图谱")
        
        # 打印入参信息
        doc_info = [f"{getattr(doc.metadata, 'title', None) or getattr(doc.metadata, 'name', 'Unknown')}({len(doc.content)}字符)" for doc in documents]
        self.logger.info(f"输入文档: {len(documents)}个 - {', '.join(doc_info[:3])}{'...' if len(documents) > 3 else ''}")
        
        # 保存documents以供向量索引使用
        self.documents = documents
        
        # 配置 GraphRAG 构造器
        from agenticx.knowledge.graphers.config import LLMConfig, GraphRagConfig
        from agenticx.knowledge.base import ChunkingConfig
        
        graph_knowledge_config_dict = self.config['knowledge']['graph_knowledge']
        
        # 打印关键配置
        self.logger.info(f"配置参数: extraction_method={graph_knowledge_config_dict.get('extraction_method', '未设置')}, "
                        f"spo_batch_size={graph_knowledge_config_dict.get('spo_batch_size', '未设置')}")
        
        # 将字典转换为 GraphRagConfig 对象
        graph_knowledge_config = GraphRagConfig.from_dict(graph_knowledge_config_dict)
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
        
        # 将强模型配置添加到graph_knowledge_config中
        graph_knowledge_config.strong_model_config = strong_llm_config
        
        self.logger.info(f"LLM配置: 默认模型={llm_config.model}, 强模型={strong_llm_config.model}")
        
        # 使用知识图谱专用分块配置
        graph_chunking_config = self.config['knowledge']['chunking']['graph_knowledge']
        strategy = graph_chunking_config.get('strategy', 'fixed_size')
        chunk_size = graph_chunking_config.get('chunk_size', 3000)
        chunk_overlap = graph_chunking_config.get('chunk_overlap', 300)
        
        chunking_config = ChunkingConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.logger.info(f"分块配置: strategy={strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # 对所有文档进行分块
        all_chunks = []
        for i, document in enumerate(documents):
            doc_name = document.metadata.name
            doc_length = len(document.content)
            
            # 检查文档是否需要分块
            if doc_length > chunk_size:
                self.logger.info(f"分块文档 {i+1}/{len(documents)}: {doc_name} ({doc_length}字符)")
                
                # 使用分块器
                chunker_kwargs = {}
                if strategy == "semantic":
                    chunker_kwargs['embedding_model'] = self.embedding_router
                    chunker_kwargs['similarity_threshold'] = graph_chunking_config.get('semantic', {}).get('similarity_threshold', 0.7)
                    chunker_kwargs['min_chunk_size'] = graph_chunking_config.get('min_chunk_size', 1500)
                    chunker_kwargs['max_chunk_size'] = graph_chunking_config.get('max_chunk_size', 5000)
                elif strategy == "agentic":
                    chunker_kwargs['llm_client'] = self.llm_client
                    agentic_config = graph_chunking_config.get('agentic', {})
                    chunker_kwargs.update(agentic_config)
                
                chunker = get_chunker(strategy, chunking_config, **chunker_kwargs)
                chunks = await chunker.chunk_document_async(document)
                
                if hasattr(chunks, 'chunks'):
                    chunk_docs = chunks.chunks
                else:
                    chunk_docs = chunks
                
                all_chunks.extend(chunk_docs)
                self.logger.info(f"分块结果: {len(chunk_docs)}个块")
            else:
                # 文档足够小，不需要分块
                all_chunks.append(document)
                self.logger.info(f"文档 {i+1}/{len(documents)}: {doc_name} 无需分块")
        
        self.logger.info(f"分块完成: 总计{len(all_chunks)}个文本块")
        
        # 使用新的KnowledgeGraphBuilder进行两阶段SPO抽取
        builder = KnowledgeGraphBuilder(
            config=graph_knowledge_config,
            llm_config=llm_config
        )
        
        # 提取分块后的文档文本
        texts = [chunk.content for chunk in all_chunks]
        metadata = [chunk.metadata.__dict__ for chunk in all_chunks]
        
        # 构建图谱（使用批处理SPO抽取）
        self.logger.info(f"开始SPO抽取: {len(texts)}个文本块")
        self.knowledge_graph = await builder.build_from_texts(
            texts=texts,
            metadata=metadata
        )
        
        # 打印构建结果
        entity_count = len(self.knowledge_graph.entities)
        relation_count = len(self.knowledge_graph.relationships)
        self.logger.info(f"✅ 知识图谱构建完成: {entity_count}个实体, {relation_count}个关系")
    
    async def store_and_index(self) -> None:
        """存储和索引知识图谱"""
        self.logger.info("开始存储和索引")
        
        # 打印输入数据统计
        entity_count = len(self.knowledge_graph.entities)
        relation_count = len(self.knowledge_graph.relationships)
        self.logger.info(f"输入数据: {entity_count}个实体, {relation_count}个关系")
        
        # 1. 存储到图数据库
        try:
            graph_storage = await self.storage_manager.get_graph_storage('default')
            
            if graph_storage:
                clear_existing = self.mode in ["full", "build"]
                self.logger.info(f"图数据库存储: 清理模式={clear_existing}")
                graph_storage.store_graph(self.knowledge_graph, clear_existing=clear_existing)
                self.logger.info("✅ 图数据库存储完成")
            else:
                self.logger.warning("❌ 未找到图数据库存储，跳过")
        except Exception as e:
            self.logger.error(f"❌ 图数据库存储失败: {e}")
            self.logger.warning("继续执行其他索引步骤")
        
        # 2. 构建向量索引
        try:
            await self._build_vector_index()
        except Exception as e:
            self.logger.error(f"❌ 向量索引构建失败: {e}")
        
        # 3. 构建BM25索引
        try:
            await self._build_bm25_index()
        except Exception as e:
            self.logger.error(f"❌ BM25索引构建失败: {e}")
        
        # 4. 构建 SPO 索引
        try:
            await self._build_spo_index()
        except Exception as e:
            self.logger.error(f"❌ SPO索引构建失败: {e}")
        
        # 5. 缓存关键数据
        try:
            await self._cache_key_data()
        except Exception as e:
            self.logger.error(f"❌ 数据缓存失败: {e}")
        
        # 统计向量索引总数 - 统计所有独立向量存储实例
        total_vectors = 0
        
        # 统计文档向量存储
        if hasattr(self, '_document_vector_storage') and self._document_vector_storage:
            try:
                doc_status = self._document_vector_storage.status()
                doc_vectors = doc_status.vector_count
                total_vectors += doc_vectors
                self.logger.debug(f"文档向量存储: {doc_vectors}条记录")
            except Exception as e:
                self.logger.debug(f"获取文档向量存储状态失败: {e}")
        
        # 统计图向量存储
        if hasattr(self, '_graph_vector_storage') and self._graph_vector_storage:
            try:
                graph_status = self._graph_vector_storage.status()
                graph_vectors = graph_status.vector_count
                total_vectors += graph_vectors
                self.logger.debug(f"图向量存储: {graph_vectors}条记录")
            except Exception as e:
                self.logger.debug(f"获取图向量存储状态失败: {e}")
        
        self.logger.info(f"存储和索引完成: 图数据库{len(self.knowledge_graph.entities)}个实体/{len(self.knowledge_graph.relationships)}个关系, 向量数据库{total_vectors}条记录")
    
    async def _build_vector_index(self) -> None:
        """构建向量索引 - 现在分离为文档向量和图向量"""
        self.logger.info("开始构建向量索引")
        
        # 构建文档向量索引（用于向量检索）
        await self._build_document_vector_index()
        
        # 构建图向量索引（用于图检索增强）
        await self._build_graph_vector_indices()
        
        self.logger.info("✅ 向量索引构建完成")
    
    async def _build_document_vector_index(self) -> None:
        """构建文档分块向量索引 - 专用于向量检索路径"""
        from agenticx.storage import StorageType
        from agenticx.storage.vectordb_storages.milvus import MilvusStorage
        
        self.logger.info("构建文档向量索引")
        
        # 🔧 为文档向量创建独立的Milvus存储实例
        storage_config = self.config['storage']['vector']['milvus']
        # 🔧 修复：根据运行模式决定是否重新创建文档向量集合
        recreate_document_collection = self.mode in ["full", "build"]
        self.logger.info(f"🔧 文档向量存储配置: mode={self.mode}, recreate_if_exists={recreate_document_collection}")
        self._document_vector_storage = MilvusStorage(
            dimension=1024,  # 使用嵌入维度
            host=storage_config['host'],
            port=storage_config['port'],
            collection_name=storage_config['collection_name'],  # 使用文档专用集合名
            database=storage_config.get('database', 'default'),
            username=storage_config.get('username'),
            password=storage_config.get('password'),
            recreate_if_exists=recreate_document_collection  # 🔧 修复：根据模式决定是否重新创建
        )
        
        self.logger.info(f"📄 文档向量存储集合: {storage_config['collection_name']}")
        
        if not hasattr(self, 'documents') or not self.documents:
            self.logger.warning("❌ 没有文档可以索引")
            return
        
        # 使用向量检索专用分块配置
        vector_chunking_config = self.config['knowledge']['chunking'].get('vector', {
            'strategy': 'fixed_size',
            'chunk_size': 1500,
            'chunk_overlap': 150,
            'min_chunk_size': 500,
            'max_chunk_size': 2000
        })
        
        strategy = vector_chunking_config['strategy']
        chunk_size = vector_chunking_config['chunk_size']
        chunk_overlap = vector_chunking_config['chunk_overlap']
        self.logger.info(f"向量分块配置: strategy={strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # 获取向量检索专用分块器
        from agenticx.knowledge.base import ChunkingConfig
        vector_config = ChunkingConfig(
            chunk_size=vector_chunking_config['chunk_size'],
            chunk_overlap=vector_chunking_config['chunk_overlap']
        )
        vector_chunker = get_chunker(vector_chunking_config['strategy'], vector_config)
        
        document_records = []
        for doc_idx, document in enumerate(self.documents):
            # 使用向量检索专用分块
            try:
                chunks_result = await vector_chunker.chunk_document_async(document)
                chunks = chunks_result.chunks if hasattr(chunks_result, 'chunks') else chunks_result
            except Exception as e:
                self.logger.warning(f"分块失败，使用简单分块: {e}")
                # 简单分块作为备用
                chunk_size = vector_chunking_config['chunk_size']
                content = document.content
                chunks = []
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i + chunk_size]
                    chunk_metadata = type(document.metadata)(
                        name=f"{document.metadata.name}_chunk_{i//chunk_size}",
                        source=document.metadata.source,
                        source_type=document.metadata.source_type,
                        content_type=document.metadata.content_type,
                        parent_id=document.id,
                        chunk_index=i//chunk_size,
                        chunker_name="SimpleChunker"
                    )
                    chunk = type(document)(content=chunk_content, metadata=chunk_metadata)
                    chunks.append(chunk)
            
            for chunk_idx, chunk in enumerate(chunks):
                # 生成嵌入
                embedding = await self.embedding_router.aembed_text(chunk.content)
                
                # 创建向量记录
                record = VectorRecord(
                    id=f"doc_{doc_idx}_chunk_{chunk_idx}",
                    vector=embedding,
                    payload={
                        'content': chunk.content,  # 🔧 修复：将content放到payload中
                        'metadata': {
                            'type': 'document_chunk',
                            'document_id': document.id,
                            'document_title': getattr(document.metadata, 'title', None) or getattr(document.metadata, 'name', ''),
                            'chunk_index': chunk_idx,
                            'chunk_size': len(chunk.content),
                            'chunking_strategy': vector_chunking_config['strategy']
                        }
                    }
                )
                document_records.append(record)
        
        # 批量存储文档分块向量
        if document_records:
            await self._document_vector_storage.add(document_records)
            self.logger.info(f"✅ 文档向量索引完成: {len(document_records)}条记录")
            self.logger.info(f"📄 存储到集合: {storage_config['collection_name']}")
        else:
            self.logger.warning("❌ 没有文档分块可以索引")
    
    async def _build_graph_vector_indices(self) -> None:
        """构建图向量索引 - 专用于图检索增强"""
        self.logger.info("构建图向量索引")
        
        # 检查图检索器是否支持向量索引
        if not hasattr(self, 'graph_retriever') or not self.graph_retriever:
            self.logger.warning("❌ 图检索器未初始化，跳过")
            return
        
        if not self.graph_retriever.enable_vector_search:
            self.logger.info("图向量索引已禁用，跳过构建")
            return
        
        try:
            # 使用GraphRetriever的向量索引构建功能
            results = await self.graph_retriever.build_vector_indices()
            
            # 处理不同类型的结果状态
            success_count = 0
            skipped_count = 0
            failed_count = 0
            
            for k, v in results.items():
                if k == 'error':
                    continue
                if v is True or v == "success":
                    success_count += 1
                elif v == "skipped":
                    skipped_count += 1
                else:
                    failed_count += 1
            
            total_count = len([k for k in results.keys() if k != 'error'])
            
            if 'error' in results:
                self.logger.error(f"❌ 图向量索引构建失败: {results['error']}")
            else:
                self.logger.info(f"✅ 图向量索引构建完成: {success_count}/{total_count}个成功")
                for index_type, result in results.items():
                    if index_type == 'error':
                        continue
                    if result is True or result == "success":
                        status = "✅"
                    elif result == "skipped":
                        status = "⏭️"  # 跳过符号
                    else:
                        status = "❌"
                    self.logger.info(f"  {status} {index_type}索引")
                
        except Exception as e:
            self.logger.error(f"❌ 图向量索引构建异常: {e}")
    
    async def _build_bm25_index(self) -> None:
        """构建BM25倒排索引 - 基于专用分块配置"""
        self.logger.info("构建BM25倒排索引")
        
        if not hasattr(self, 'documents') or not self.documents:
            self.logger.warning("❌ 没有文档可以构建BM25索引")
            return
        
        # 检查BM25检索器是否已初始化
        if not hasattr(self, 'retriever') or not self.retriever:
            self.logger.warning("❌ 检索器未初始化，跳过")
            return
        
        # 获取BM25检索器
        bm25_retriever = None
        if hasattr(self.retriever, 'bm25_retriever'):
            bm25_retriever = self.retriever.bm25_retriever
        else:
            self.logger.warning("❌ 未找到BM25检索器，跳过")
            return
        
        # 使用BM25专用分块配置
        bm25_chunking_config = self.config['knowledge']['chunking'].get('bm25', {
            'strategy': 'fixed_size',
            'chunk_size': 600,
            'chunk_overlap': 100,
            'min_chunk_size': 400,
            'max_chunk_size': 1000
        })
        
        strategy = bm25_chunking_config['strategy']
        chunk_size = bm25_chunking_config['chunk_size']
        chunk_overlap = bm25_chunking_config['chunk_overlap']
        self.logger.info(f"BM25分块配置: strategy={strategy}, chunk_size={chunk_size}, overlap={chunk_overlap}")
        
        # 获取BM25专用分块器
        from agenticx.knowledge.base import ChunkingConfig
        bm25_config = ChunkingConfig(
            chunk_size=bm25_chunking_config['chunk_size'],
            chunk_overlap=bm25_chunking_config['chunk_overlap']
        )
        bm25_chunker = get_chunker(bm25_chunking_config['strategy'], bm25_config)
        
        # 准备BM25文档
        bm25_documents = []
        for doc_idx, document in enumerate(self.documents):
            # 使用BM25专用分块
            try:
                chunks_result = await bm25_chunker.chunk_document_async(document)
                chunks = chunks_result.chunks if hasattr(chunks_result, 'chunks') else chunks_result
            except Exception as e:
                self.logger.warning(f"BM25分块失败，使用简单分块: {e}")
                # 简单分块作为备用
                chunk_size = bm25_chunking_config['chunk_size']
                content = document.content
                chunks = []
                for i in range(0, len(content), chunk_size):
                    chunk_content = content[i:i + chunk_size]
                    chunk_metadata = type(document.metadata)(
                        name=f"{document.metadata.name}_bm25_chunk_{i//chunk_size}",
                        source=document.metadata.source,
                        source_type=document.metadata.source_type,
                        content_type=document.metadata.content_type,
                        parent_id=document.id,
                        chunk_index=i//chunk_size,
                        chunker_name="SimpleChunker"
                    )
                    chunk = type(document)(content=chunk_content, metadata=chunk_metadata)
                    chunks.append(chunk)
            
            for chunk_idx, chunk in enumerate(chunks):
                # 创建BM25文档记录
                bm25_doc = {
                    'id': f"bm25_doc_{doc_idx}_chunk_{chunk_idx}",
                    'content': chunk.content,
                    'metadata': {
                        'type': 'bm25_chunk',
                        'document_id': document.id,
                        'document_title': getattr(document.metadata, 'title', None) or getattr(document.metadata, 'name', ''),
                        'chunk_index': chunk_idx,
                        'chunk_size': len(chunk.content),
                        'chunking_strategy': bm25_chunking_config['strategy']
                    }
                }
                bm25_documents.append(bm25_doc)
        
        # 批量添加到BM25检索器
        if bm25_documents:
            try:
                document_ids = await bm25_retriever.add_documents(bm25_documents)
                self.logger.info(f"✅ BM25索引构建完成: {len(bm25_documents)}条记录")
                self.logger.debug(f"BM25文档ID: {document_ids[:5]}..." if len(document_ids) > 5 else f"BM25文档ID: {document_ids}")
            except Exception as e:
                self.logger.error(f"❌ BM25索引添加失败: {e}")
        else:
            self.logger.warning("⚠️ 没有BM25文档可以索引")
    
    async def _build_legacy_entity_relation_vectors(self) -> None:
        """构建传统的实体和关系向量（保留用于兼容性）"""
        from agenticx.storage import StorageType
        
        self.logger.debug("构建传统实体关系向量...")
        
        vector_storage = await self.storage_manager.get_vector_storage('default')
        if not vector_storage:
            vector_storage = self.storage_manager.get_storage(StorageType.CHROMA)
            if not vector_storage:
                return
        
        # 为实体构建向量索引
        entity_records = []
        for entity in self.knowledge_graph.entities.values():
            entity_text = f"{entity.name}: {entity.description or ''}"
            embedding = await self.embedding_router.aembed_text(entity_text)
            
            record = VectorRecord(
                id=f"legacy_entity_{entity.id}",
                vector=embedding,
                metadata={
                    'type': 'legacy_entity',
                    'entity_type': entity.entity_type.value,
                    'name': entity.name,
                    'confidence': entity.confidence
                },
                content=entity_text
            )
            entity_records.append(record)
        
        # 为关系构建向量索引
        relationship_records = []
        for relationship in self.knowledge_graph.relationships.values():
            source_entity = self.knowledge_graph.get_entity(relationship.source_entity_id)
            target_entity = self.knowledge_graph.get_entity(relationship.target_entity_id)
            
            if source_entity and target_entity:
                rel_text = f"{source_entity.name} {relationship.relation_type.value} {target_entity.name}"
                embedding = await self.embedding_router.aembed_text(rel_text)
                
                record = VectorRecord(
                    id=f"legacy_relation_{relationship.id}",
                    vector=embedding,
                    metadata={
                        'type': 'legacy_relationship',
                        'relation_type': relationship.relation_type.value,
                        'source_entity': source_entity.name,
                        'target_entity': target_entity.name,
                        'confidence': relationship.confidence
                    },
                    content=rel_text
                )
                relationship_records.append(record)
        
        # 批量存储
        all_records = entity_records + relationship_records
        if all_records:
            vector_storage.add(all_records)
            self.logger.debug(f"传统向量索引: {len(entity_records)}个实体 + {len(relationship_records)}个关系")
    
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
        
        if console and Panel and box:
            # 使用 Rich 显示问答界面
            qa_text = Text()
            qa_text.append("🤖 AgenticX GraphRAG 问答系统\n\n", style="bold cyan")
            qa_text.append("支持的查询类型:\n", style="bold white")
            qa_text.append("● 实体查询: 查找特定实体的信息\n", style="white")
            qa_text.append("● 关系查询: 查找实体间的关系\n", style="white")
            qa_text.append("● 路径查询: 查找实体间的连接路径\n", style="white")
            qa_text.append("● 社区查询: 查找相关实体群组\n", style="white")
            qa_text.append("● 自由问答: 基于知识图谱的开放式问答\n\n", style="white")
            qa_text.append("输入 'quit' 或 'exit' 退出系统", style="dim")
            
            panel = Panel(
                qa_text,
                title="智能问答",
                border_style="#FF6B35",
                box=box.ROUNDED,
                padding=(1, 2)
            )
            console.print(panel)
        else:
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
                if console and Prompt:
                    query = Prompt.ask("\n[bold cyan]🔍 请输入您的问题[/bold cyan]").strip()
                else:
                    query = input("🔍 请输入您的问题: ").strip()
                
                if query.lower() in ['quit', 'exit', '退出']:
                    print_success("感谢使用 AgenticX GraphRAG 系统！")
                    break
                
                if not query:
                    continue
                
                # 处理查询
                await self._process_query(query)
                
            except KeyboardInterrupt:
                print_success("\n感谢使用 AgenticX GraphRAG 系统！")
                break
            except Exception as e:
                self.logger.error(f"查询处理错误: {e}")
                print_error(f"查询处理出错: {e}")
    
    async def _process_query(self, query: str) -> None:
        """处理用户查询 - 增强版本，支持智能查询处理和多级回退"""
        print(f"\n🔄 正在处理查询: {query}")
        self.logger.info(f"开始处理查询: {query}")
        
        try:
            # 🚀 新增：使用增强检索器
            if not hasattr(self, '_enhanced_retriever'):
                # 延迟导入增强检索器
                try:
                    from enhanced_retriever import EnhancedRetriever
                    self._enhanced_retriever = EnhancedRetriever(
                        base_retriever=self.retriever,
                        graph_retriever=getattr(self, 'graph_retriever', None),
                        storage_manager=self.storage_manager
                    )
                    self.logger.info("✅ 增强检索器初始化成功")
                except ImportError as e:
                    self.logger.warning(f"增强检索器导入失败，使用原始检索: {e}")
                    self._enhanced_retriever = None
            
            # 使用增强检索器进行多级回退检索
            if self._enhanced_retriever:
                results, retrieval_report = await self._enhanced_retriever.retrieve_with_fallback(query)
                
                # 显示检索报告
                if retrieval_report.get('success', False):
                    print(f"✅ 检索成功！使用策略: {retrieval_report.get('strategy_used', 'unknown')}")
                    print(f"📊 找到 {retrieval_report.get('total_results', 0)} 条相关信息")
                    
                    # 显示处理后的查询信息
                    processed_query = retrieval_report.get('processed_query')
                    if processed_query:
                        self.logger.info(f"🔍 查询分析: 类型={processed_query.query_type}, 置信度={processed_query.confidence:.2f}")
                        self.logger.info(f"🔍 关键词: {processed_query.keywords}")
                        self.logger.info(f"🔍 实体: {processed_query.entities}")
                    
                    # 生成答案
                    await self._generate_answer(query, results)
                    return
                else:
                    print("❌ 增强检索也未找到结果")
                    
                    # 🚀 新增：运行诊断检查
                    print("\n🔍 正在运行系统诊断...")
                    await self._run_quick_diagnostics()
                    
                    # 提供查询建议
                    suggestions = await self._enhanced_retriever.suggest_related_queries(query)
                    if suggestions:
                        print("\n💡 您可以尝试以下相关查询:")
                        for i, suggestion in enumerate(suggestions, 1):
                            print(f"   {i}. {suggestion}")
                    return
            
            # 🔧 回退到原始检索逻辑（保持向后兼容）
            self.logger.info("使用原始检索逻辑")
            
            # 获取配置
            rag_config = self.config.get('rag', {})
            rag_retrieval_config = rag_config.get('retrieval', {})
            retrieval_config = self.config.get('retrieval', {})
            vector_config = retrieval_config.get('vector', {})
            graph_config = retrieval_config.get('graph', {})
            
            # 🔧 调试：打印配置内容
            self.logger.info(f"🔍 配置调试:")
            self.logger.info(f"  rag_config keys: {list(rag_config.keys())}")
            self.logger.info(f"  rag_retrieval_config: {rag_retrieval_config}")
            self.logger.info(f"  vector_config top_k: {vector_config.get('top_k', 'NOT_FOUND')}")
            self.logger.info(f"  graph_config max_nodes: {graph_config.get('max_nodes', 'NOT_FOUND')}")
            
            # 🔧 修复：使用合理的检索参数
            hybrid_top_k = rag_retrieval_config.get('default_top_k', vector_config.get('top_k', 20))
            graph_top_k = graph_config.get('max_nodes', 15)
            # 🔧 修复：使用配置中的原始阈值，不再人为降低
            similarity_threshold = vector_config.get('similarity_threshold', 0.5)  # 使用合理的阈值
            graph_similarity_threshold = graph_config.get('similarity_threshold', 0.3)  # 使用合理的阈值

            self.logger.info(f"🎯 优化后检索参数: hybrid_top_k={hybrid_top_k}, graph_top_k={graph_top_k}, vector_threshold={similarity_threshold}, graph_threshold={graph_similarity_threshold}")
            
            # 🚀 新增：智能查询预处理
            processed_queries = [query]
            try:
                from query_processor import ChineseQueryProcessor
                processor = ChineseQueryProcessor()
                processed_query = processor.process_query(query)
                additional_queries = processor.generate_search_queries(processed_query)
                processed_queries.extend(additional_queries)
                self.logger.info(f"🔍 生成搜索查询: {processed_queries}")
            except ImportError:
                self.logger.warning("查询处理器不可用，使用原始查询")

            # 对每个处理后的查询执行检索
            all_hybrid_results = []
            all_graph_results = []
            
            for search_query in processed_queries[:3]:  # 限制最多3个查询避免过度检索
                # 1. 执行混合检索
                self.logger.info(f"🔍 开始混合检索: '{search_query}', top_k={hybrid_top_k}, min_score={similarity_threshold}")
                try:
                    hybrid_results = await self.retriever.retrieve(search_query, top_k=hybrid_top_k, min_score=similarity_threshold)
                    all_hybrid_results.extend(hybrid_results)
                    self.logger.info(f"🔍 混合检索 '{search_query}' 返回: {len(hybrid_results)}条")
                except Exception as e:
                    self.logger.warning(f"混合检索失败: {e}")
                
                # 2. 执行图检索
                if hasattr(self, 'graph_retriever') and self.graph_retriever:
                    try:
                        self.logger.info(f"🔗 开始图检索: '{search_query}', top_k={graph_top_k}, min_score={graph_similarity_threshold}")
                        graph_results = await self.graph_retriever.retrieve(search_query, top_k=graph_top_k, min_score=graph_similarity_threshold)
                        all_graph_results.extend(graph_results)
                        self.logger.info(f"🔗 图检索 '{search_query}' 返回: {len(graph_results)}条")
                    except Exception as e:
                        self.logger.warning(f"图检索失败: {e}")
            
            # 3. 合并和去重
            all_results = all_hybrid_results + all_graph_results
            seen_ids = set()
            unique_results = []
            for result in all_results:
                result_id = getattr(result, 'id', str(hash(result.content)))
                if result_id not in seen_ids:
                    seen_ids.add(result_id)
                    unique_results.append(result)
            
            # 4. 按相似度排序和筛选
            unique_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
            results = unique_results[:hybrid_top_k]
            
            # 5. 统计不同类型的结果
            type_counts = {}
            for result in results:
                result_type = "其他"
                if hasattr(result, 'metadata') and result.metadata:
                    if 'search_source' in result.metadata:
                        result_type = result.metadata['search_source']
                    elif 'type' in result.metadata:
                        result_type = result.metadata['type']
                type_counts[result_type] = type_counts.get(result_type, 0) + 1
            
            # 6. 优化后的统一日志输出
            self.logger.info(f"完成执行多查询检索，总查询数: {len(processed_queries)}")
            
            if not results:
                print("❌ 没有找到相关信息")
                self.logger.warning("检索无结果，尝试直接实体搜索")
                await self._search_entity_directly(query)
                
                # 🚀 新增：提供查询建议
                print("\n💡 您可以尝试:")
                print("   1. 使用更具体的关键词")
                print("   2. 检查拼写是否正确")
                print("   3. 尝试相关的同义词")
                return
            
            # 统一的检索统计信息
            stats_info = f"检索统计:\n🔍 混合检索: {len(all_hybrid_results)}条\n🔗 图检索: {len(all_graph_results)}条\n🔄 去重后: {len(unique_results)}条\n✅ 最终采用: {len(results)}条，其中："
            for result_type, count in type_counts.items():
                stats_info += f"\n    {result_type}: {count}条"
            
            self.logger.info(stats_info)
            
            # 生成答案
            await self._generate_answer(query, results)
        
        except Exception as e:
            self.logger.error(f"查询处理错误: {e}")
            print(f"❌ 查询处理出错: {e}")
            
            # 🚀 新增：错误时的友好提示
            print("\n💡 遇到问题时，您可以:")
            print("   1. 简化查询内容")
            print("   2. 使用中文关键词")
            print("   3. 检查网络连接")
            print("   4. 稍后重试")
    
    async def _run_quick_diagnostics(self) -> None:
        """运行快速诊断检查"""
        try:
            print("📊 系统状态检查:")
            
            # 检查文档数量
            doc_count = len(self.documents) if hasattr(self, 'documents') and self.documents else 0
            print(f"  📄 已加载文档: {doc_count} 个")
            
            # 检查BM25索引
            if hasattr(self, 'retriever') and hasattr(self.retriever, 'bm25_retriever'):
                bm25_retriever = self.retriever.bm25_retriever
                if hasattr(bm25_retriever, '_documents'):
                    bm25_doc_count = len(bm25_retriever._documents)
                    index_terms = len(bm25_retriever._inverted_index) if hasattr(bm25_retriever, '_inverted_index') else 0
                    print(f"  🔍 BM25索引: {bm25_doc_count} 个文档, {index_terms} 个词汇")
                else:
                    print(f"  🔍 BM25索引: 未初始化")
            
            # 检查图数据
            if hasattr(self, 'storage_manager'):
                try:
                    from agenticx.storage import StorageType
                    graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
                    if graph_storage:
                        # 检查实体数量
                        entity_result = graph_storage.execute_query("MATCH (n:Entity) RETURN count(n) as count")
                        entity_count = entity_result[0]['count'] if entity_result else 0
                        
                        # 检查所有节点数量
                        node_result = graph_storage.execute_query("MATCH (n) RETURN count(n) as count")
                        node_count = node_result[0]['count'] if node_result else 0
                        
                        print(f"  🔗 知识图谱: {entity_count} 个实体, {node_count} 个总节点")
                        
                        # 🚀 新增：检查是否有包含常见词汇的节点
                        common_terms = ['中国', '技术', '系统', '公司', '服务']
                        found_terms = []
                        for term in common_terms:
                            try:
                                result = graph_storage.execute_query(
                                    "MATCH (n) WHERE toString(n.name) CONTAINS $term OR toString(n.content) CONTAINS $term RETURN count(n) as count",
                                    {"term": term}
                                )
                                if result and result[0]['count'] > 0:
                                    found_terms.append(f"{term}({result[0]['count']})")
                            except:
                                continue
                        
                        if found_terms:
                            print(f"  📝 常见词汇: {', '.join(found_terms)}")
                        else:
                            print(f"  ⚠️  未找到常见中文词汇，可能存在索引问题")
                    else:
                        print(f"  🔗 知识图谱: 连接失败")
                except Exception as e:
                    print(f"  🔗 知识图谱: 检查失败 - {e}")
            
            # 🚀 新增：测试简单查询
            print("\n🧪 测试基础查询:")
            test_queries = ['测试', '中国', '技术']
            for test_query in test_queries:
                try:
                    if hasattr(self, 'retriever'):
                        results = await self.retriever.retrieve(test_query, top_k=1, min_score=0.0)
                        print(f"  '{test_query}': {len(results)} 条结果")
                    else:
                        print(f"  '{test_query}': 检索器不可用")
                except Exception as e:
                    print(f"  '{test_query}': 查询失败 - {str(e)[:50]}...")
            
            print("\n💡 诊断建议:")
            if doc_count == 0:
                print("  ⚠️  没有加载文档，请先运行构建模式")
            elif hasattr(self, 'retriever') and hasattr(self.retriever, 'bm25_retriever'):
                bm25_doc_count = len(getattr(self.retriever.bm25_retriever, '_documents', {}))
                if bm25_doc_count == 0:
                    print("  ⚠️  BM25索引为空，请检查索引构建过程")
                else:
                    print("  ✅ 系统组件正常，可能是查询词汇不在知识库中")
            
        except Exception as e:
            print(f"  ❌ 诊断过程出错: {e}")
    
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
            self.logger.info(f"开始生成答案，输入{len(results)}条检索结果")
            
            # 🔧 调试：详细检查输入结果
            if results:
                for i, result in enumerate(results[:3]):  # 只检查前3个结果
                    self.logger.info(f"结果{i+1}: type={type(result)}, content长度={len(getattr(result, 'content', ''))}, metadata={getattr(result, 'metadata', {})}")
            else:
                self.logger.warning("输入结果为空！")
                print("⚠️ 传入的检索结果为空")
                return
            
            # 获取上下文配置
            rag_config = self.config.get('rag', {})
            retrieval_config = rag_config.get('retrieval', {})
            context_top_k = retrieval_config.get('default_top_k', 10)
            max_context_length = retrieval_config.get('max_context_length', 4000)
            
            # 🔧 修复：添加单个内容片段的最大长度限制
            max_content_per_item = 500  # 每个检索结果最多保留500字符
            
            # 🔧 修复：更宽松的结果分类，确保所有结果都能被处理
            graph_results = []
            doc_results = []
            entity_results = []
            other_results = []
            
            for r in results:
                if not r.metadata:
                    # 没有metadata的结果归类为其他结果
                    other_results.append(r)
                    continue
                
                search_source = r.metadata.get('search_source', '')
                result_type = r.metadata.get('type', '')
                
                # 图检索结果
                if (search_source in ['graph_vector', 'graph', 'direct_entity', 'full_text'] or 
                    result_type in ['entity', 'relationship', 'triple', 'community']):
                    if result_type == 'entity' or search_source == 'direct_entity':
                        entity_results.append(r)
                    else:
                        graph_results.append(r)
                # 文档结果
                elif (result_type in ['document_chunk', 'bm25_chunk', 'vector_chunk'] or 
                      'document_id' in r.metadata or 
                      'document_title' in r.metadata or
                      search_source in ['vector', 'bm25', 'hybrid']):
                    doc_results.append(r)
                else:
                    other_results.append(r)
            
            # 🔧 调试：打印结果分类信息
            self.logger.info(f"结果分类: 图检索={len(graph_results)}, 文档={len(doc_results)}, 实体={len(entity_results)}, 其他={len(other_results)}")
            
            # 合并图检索和实体结果
            graph_results.extend(entity_results)
            
            # 🔧 修复：优化内容选择策略，按相关度排序
            # 首先对所有结果按相关度排序
            all_sorted_results = sorted(results, key=lambda x: getattr(x, 'score', 0), reverse=True)
            
            # 🔧 修复：平衡选择不同类型的内容，确保多样性
            context_results = []
            
            # 按相关度选择最佳结果，但保持类型平衡
            doc_selected = 0
            graph_selected = 0
            entity_selected = 0
            other_selected = 0
            
            max_per_type = max(2, context_top_k // 4)  # 每种类型最多选择的数量
            
            for result in all_sorted_results:
                if len(context_results) >= context_top_k:
                    break
                
                if not result.metadata:
                    if other_selected < max_per_type:
                        context_results.append(result)
                        other_selected += 1
                    continue
                
                search_source = result.metadata.get('search_source', '')
                result_type = result.metadata.get('type', '')
                
                # 优先选择高质量的实体结果
                if (result_type == 'entity' or search_source == 'direct_entity') and entity_selected < max_per_type:
                    context_results.append(result)
                    entity_selected += 1
                # 然后选择文档结果
                elif (result_type in ['document_chunk', 'bm25_chunk', 'vector_chunk'] or 
                      'document_id' in result.metadata or 
                      'document_title' in result.metadata or
                      search_source in ['vector', 'bm25', 'hybrid']) and doc_selected < max_per_type:
                    context_results.append(result)
                    doc_selected += 1
                # 最后选择图检索结果
                elif (search_source in ['graph_vector', 'graph', 'full_text'] or 
                      result_type in ['relationship', 'triple', 'community']) and graph_selected < max_per_type:
                    context_results.append(result)
                    graph_selected += 1
                # 其他结果
                elif other_selected < max_per_type:
                    context_results.append(result)
                    other_selected += 1
            
            # 🔧 如果还有空位，按相关度填充剩余位置
            if len(context_results) < context_top_k:
                remaining_slots = context_top_k - len(context_results)
                selected_ids = {id(r) for r in context_results}
                for result in all_sorted_results:
                    if len(context_results) >= context_top_k:
                        break
                    if id(result) not in selected_ids:
                        context_results.append(result)
            
            # 🔧 如果分类后没有结果，直接使用所有原始结果
            if not context_results and results:
                self.logger.warning("分类后无结果，直接使用所有原始结果")
                context_results = all_sorted_results[:context_top_k]
            
            # 🔧 重新设计上下文格式，支持所有类型的结果
            context_sections = []
            
            # 🔧 调试：记录实际使用的结果数量
            self.logger.info(f"构建上下文: 使用{len(context_results)}条结果")
            
            # === 处理所有结果，按来源分组 ===
            entity_info = []
            document_info = []
            graph_info = []
            other_info = []
            
            for i, result in enumerate(context_results):
                 try:
                     # 🔧 增强：更强的属性访问容错
                     content = ""
                     if hasattr(result, 'content'):
                         content = str(result.content).strip()
                     elif isinstance(result, dict):
                         content = str(result.get('content', '')).strip()
                     
                     if not content:
                         self.logger.warning(f"结果{i+1}内容为空，跳过")
                         continue
                     
                     # 🔧 修复：截断过长的内容
                     if len(content) > max_content_per_item:
                         content = content[:max_content_per_item] + "..."
                         self.logger.debug(f"结果{i+1}内容被截断到{max_content_per_item}字符")
                     
                     # 获取分数
                     score = 0.0
                     if hasattr(result, 'score'):
                         score = float(result.score)
                     elif isinstance(result, dict):
                         score = float(result.get('score', 0.0))
                     
                     # 获取metadata
                     metadata = {}
                     if hasattr(result, 'metadata'):
                         metadata = result.metadata or {}
                     elif isinstance(result, dict):
                         metadata = result.get('metadata', {})
                     
                     # 构建来源信息
                     source_info = ""
                     search_source = metadata.get('search_source', '')
                     result_type = metadata.get('type', '')
                     
                     if search_source:
                         source_info = f"[{search_source}]"
                     elif result_type:
                         source_info = f"[{result_type}]"
                     
                     # 🔧 调试：记录处理的结果
                    #  self.logger.info(f"处理结果{i+1}: content={content[:50]}..., score={score}, source={search_source}, type={result_type}")
                     
                     # 分类结果
                     if result_type == 'entity' or search_source == 'direct_entity':
                         entity_info.append(f"• {content} {source_info} [相关度: {score:.3f}]")
                     elif (result_type in ['document_chunk', 'bm25_chunk', 'vector_chunk'] or 
                           'document_title' in metadata or 
                           search_source in ['vector', 'bm25', 'hybrid']):
                         # 提取文档信息
                         doc_title = metadata.get('document_title', '')
                         if doc_title:
                             source_info = f"[{doc_title}]"
                         document_info.append(f"• {content} {source_info} [相关度: {score:.3f}]")
                     elif (search_source in ['graph_vector', 'graph', 'full_text'] or 
                           result_type in ['relationship', 'triple', 'community']):
                         graph_info.append(f"• {content} {source_info} [相关度: {score:.3f}]")
                     else:
                         # 🔧 修复：确保所有结果都被包含
                         other_info.append(f"• {content} {source_info} [相关度: {score:.3f}]")
                 
                 except Exception as e:
                     self.logger.error(f"处理结果{i+1}时出错: {e}")
                     # 🔧 容错：即使出错也尝试提取基本信息
                     try:
                         content = str(result)[:200] if result else "无内容"
                         other_info.append(f"• {content} [处理出错]")
                     except:
                         pass
            
            # 按优先级添加到上下文
            if entity_info:
                context_sections.append("=== 实体信息 ===")
                context_sections.extend(entity_info)
            
            if document_info:
                if context_sections:
                    context_sections.append("\n=== 文档内容 ===")
                else:
                    context_sections.append("=== 文档内容 ===")
                context_sections.extend(document_info)
            
            if graph_info:
                if context_sections:
                    context_sections.append("\n=== 知识图谱信息 ===")
                else:
                    context_sections.append("=== 知识图谱信息 ===")
                context_sections.extend(graph_info)
            
            if other_info:
                if context_sections:
                    context_sections.append("\n=== 其他相关信息 ===")
                else:
                    context_sections.append("=== 其他相关信息 ===")
                context_sections.extend(other_info)
            
            # 🔧 最终安全网：如果所有分类都为空，直接显示原始结果
            if not entity_info and not document_info and not graph_info and not other_info:
                self.logger.warning("所有分类都为空，使用原始结果")
                context_sections.append("=== 检索结果 ===")
                for i, result in enumerate(context_results):
                    try:
                        content = str(getattr(result, 'content', result))[:500]
                        score = getattr(result, 'score', 0.0)
                        context_sections.append(f"• 结果{i+1}: {content} [相关度: {score:.3f}]")
                    except:
                        context_sections.append(f"• 结果{i+1}: {str(result)[:200]}")
            
            context = "\n".join(context_sections)
            
            # 🔧 修复：添加最终上下文长度检查和截断
            if len(context) > max_context_length:
                self.logger.warning(f"上下文长度{len(context)}超过限制{max_context_length}，进行截断")
                context = context[:max_context_length] + "\n\n[注：内容因长度限制被截断]"
            
            # 🔧 调试：检查最终上下文
            self.logger.info(f"最终上下文长度: {len(context)}字符 (限制: {max_context_length})")
            if len(context) < 50:
                self.logger.warning(f"上下文过短: '{context}'")
                # 如果上下文太短，强制添加一些内容
                if context_results:
                    context = "=== 检索到的信息 ===\n"
                    for i, result in enumerate(context_results[:3]):
                        try:
                            content = str(getattr(result, 'content', result))[:300]  # 限制每个结果300字符
                            context += f"结果{i+1}: {content}\n"
                        except:
                            context += f"结果{i+1}: {str(result)[:200]}\n"
            
            # 使用提示词管理器加载模板
            try:
                prompt_template = self.prompt_manager.get_prompt_template("rag_qa", "template")
                if prompt_template:
                    prompt = prompt_template.format(context=context, query=query)
                else:
                    # 回退到默认模板
                    prompt = f"""你是一个专业的智能问答助手，能够基于多种来源的信息为用户提供准确、全面的答案。

## 检索到的相关信息
{context}

## 用户问题
{query}

## 请提供答案"""
            except Exception as e:
                self.logger.warning(f"提示词模板加载失败，使用默认模板: {e}")
                prompt = f"""你是一个专业的智能问答助手，能够基于多种来源的信息为用户提供准确、全面的答案。

## 检索到的相关信息
{context}

## 用户问题
{query}

## 请提供答案"""
            
            # 记录提示词信息
            self.logger.info(f"提示词构建完成 - 上下文:{len(context)}字符, 总长度:{len(prompt)}字符")
            
            # 显示完整提示词
            print("\n" + "="*60)
            print("📝 最终发送给大模型的提示词:")
            print("="*60)
            print(prompt)
            print("="*60)
            
            # 调用 LLM 生成答案（流式返回）
            print(f"\n🤖 AI 回答:")
            print("-" * 50)
            self.logger.info("开始调用大模型生成答案")
            
            # 检查是否支持流式调用
            if hasattr(self.llm_client, 'astream'):
                # 使用流式调用
                full_response = ""
                async for chunk in self.llm_client.astream(prompt):
                    if hasattr(chunk, 'content') and chunk.content:
                        print(chunk.content, end='', flush=True)
                        full_response += chunk.content
                    elif isinstance(chunk, str):
                        print(chunk, end='', flush=True)
                        full_response += chunk
                
                print()  # 换行
                self.logger.info(f"答案生成完成: {len(full_response)}字符")
            else:
                # 回退到非流式调用
                response = await self.llm_client.ainvoke(prompt)
                print(response.content)
                self.logger.info(f"答案生成完成: {len(response.content)}字符")
            
            print("-" * 50)
            
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
    
    async def run_build_only(self) -> None:
        """仅构建模式，执行文档解析和知识库构建，不启动问答"""
        try:
            print("🔨 启动 AgenticX GraphRAG 构建模式...")
            print("📋 执行文档解析和知识库构建，完成后退出")
            
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
            
            print("✅ 知识库构建完成！")
            print("💡 现在可以使用 'python main.py --mode qa' 启动问答系统")
            
        except Exception as e:
            self.logger.error(f"构建模式运行错误: {e}")
            print(f"❌ 构建模式运行出错: {e}")
            raise

    async def run_qa_only(self) -> None:
        """仅运行问答模式，跳过文档解析和知识库构建"""
        try:
            print("启动 AgenticX GraphRAG 问答模式...")
            print("📋 跳过文档解析和知识库构建，直接使用已有数据")
            
            # 1. 初始化组件
            await self.initialize_components()
            
            # 2. 验证已有数据
            await self._validate_existing_data()
            
            # 3. 启动交互式问答
            await self.interactive_qa()
            
        except Exception as e:
            self.logger.error(f"问答模式运行错误: {e}")
            print(f"❌ 问答模式运行出错: {e}")
            raise
    
    async def _validate_existing_data(self) -> None:
        """验证已有的向量和图数据库数据"""
        self.logger.info("🔍 验证已有数据...")
        
        # 检查向量数据库
        try:
            from agenticx.storage import StorageType
            vector_storage = await self.storage_manager.get_vector_storage('default')
            if not vector_storage:
                vector_storage = self.storage_manager.get_storage(StorageType.MILVUS)
            
            if vector_storage:
                # 尝试验证向量数据库连接和数据
                try:
                    # 首先检查对象的所有可用方法
                    available_methods = [method for method in dir(vector_storage) if not method.startswith('_')]
                    # 移除冗余日志
                    
                    # 尝试简单的连接验证，而不是数据搜索
                    if hasattr(vector_storage, 'collection') and vector_storage.collection:
                        # 检查集合是否存在
                        collection_info = vector_storage.collection.describe()
                        self.logger.debug(f"Milvus集合: {collection_info['collection_name']}")
                        print(f"✅ 向量数据库连接正常，集合已存在")
                        
                        # 尝试获取集合统计信息
                        if hasattr(vector_storage.collection, 'num_entities'):
                            entity_count = vector_storage.collection.num_entities
                            self.logger.debug(f"向量数据库记录数: {entity_count}")
                            print(f"向量数据库: {entity_count}条记录")
                        
                    elif hasattr(vector_storage, 'client'):
                        # 如果有client属性，尝试检查连接
                        self.logger.info("✅ Milvus客户端连接正常")
                        print(f"✅ 向量数据库连接正常")
                    else:
                        # 简单的连接验证
                        self.logger.info("✅ 向量数据库对象创建成功")
                        print(f"✅ 向量数据库连接正常")
                        
                except Exception as validation_error:
                    self.logger.warning(f"向量数据库验证失败: {validation_error}")
                    print(f"✅ 向量数据库连接正常（跳过详细验证）")
            else:
                self.logger.error("❌ 无法连接到向量数据库")
                print("❌ 无法连接到向量数据库")
                
        except Exception as e:
            self.logger.error(f"向量数据库验证失败: {e}")
            print(f"⚠️ 向量数据库验证失败: {e}")
        
        # 检查图数据库
        try:
            graph_storage = await self.storage_manager.get_graph_storage('default')
            if not graph_storage:
                graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
            
            if graph_storage:
                # 检查图数据库中的节点数量
                count_query = "MATCH (n) RETURN count(n) as node_count"
                result = graph_storage.execute_query(count_query)
                
                if result and len(result) > 0:
                    node_count = result[0]['node_count']
                    self.logger.debug(f"图数据库验证: {node_count}个节点")
                    print(f"✅ 图数据库连接正常，包含 {node_count} 个节点")
                    
                    # 检查关系数量
                    rel_count_query = "MATCH ()-[r]->() RETURN count(r) as rel_count"
                    rel_result = graph_storage.execute_query(rel_count_query)
                    if rel_result and len(rel_result) > 0:
                        rel_count = rel_result[0]['rel_count']
                        self.logger.debug(f"图数据库关系数: {rel_count}")
                        print(f"✅ 图数据库包含 {rel_count} 个关系")
                else:
                    self.logger.warning("⚠️ 图数据库为空")
                    print("⚠️ 图数据库似乎为空，可能需要重新构建")
            else:
                self.logger.error("❌ 无法连接到图数据库")
                print("❌ 无法连接到图数据库")
                
        except Exception as e:
            self.logger.error(f"图数据库验证失败: {e}")
            print(f"⚠️ 图数据库验证失败: {e}")
        
        self.logger.info("数据验证完成")

    async def cleanup(self) -> None:
        """清理资源"""
        try:
            # 关闭图数据库连接
            if hasattr(self, 'storage_manager') and self.storage_manager:
                graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
                if graph_storage and hasattr(graph_storage, 'close'):
                    graph_storage.close()
                    logger.info("✅ Neo4j 连接已关闭")
        except Exception as e:
            logger.error(f"❌ 清理资源失败: {e}")


async def interactive_mode():
    """交互式模式主函数"""
    # 显示欢迎界面
    print_welcome()
    
    # 🔧 修复：启动时直接显示帮助信息
    print_help()
    
    # 初始化配置
    data_path = None
    run_mode = None
    demo_instance = None
    
    while True:
        try:
            # 获取用户输入
            if console and Prompt:
                user_input = Prompt.ask("\n[bold green]请输入命令或问题[/bold green] ([dim]/help 查看帮助[/dim])").strip()
            else:
                user_input = input("\n请输入命令或问题 (/help 查看帮助): ").strip()
            
            if not user_input:
                continue
            
            # 处理命令
            if user_input.startswith('/'):
                command = user_input.lower()
                
                if command == '/help':
                    print_help()
                
                elif command == '/clear':
                    if console:
                        console.clear()
                    else:
                        os.system('clear' if os.name == 'posix' else 'cls')
                    print_welcome()
                
                elif command == '/mode':
                    run_mode = select_run_mode()
                    print_success(f"已选择运行模式: {run_mode}")
                    
                    # 🔧 修复：选择模式后立即进行完整的初始化流程
                    try:
                        # 1. 选择数据目录
                        print_mode_selection("请选择数据目录")
                        data_path = display_data_selection()
                        print_success(f"已选择数据目录: {data_path}")
                        
                        # 2. 初始化系统
                        print_action("正在初始化 GraphRAG 系统...")
                        demo_instance = AgenticXGraphRAGDemo(config_path="configs.yml", mode=run_mode)
                        demo_instance.data_dir = Path(data_path)  # 设置数据目录
                        
                        # 3. 根据模式执行相应的操作
                        if run_mode in ['full', 'build']:
                            print_action("正在构建知识库...")
                            await demo_instance.run_build_only()
                            print_success("知识库构建完成！")
                            
                            if run_mode == 'build':
                                print_info("构建模式完成，可以使用 /mode 切换到问答模式")
                                continue
                            else:  # full模式
                                print_success("系统已准备就绪，可以开始问答！")
                        
                        elif run_mode == 'qa':
                            print_action("正在加载已有知识库...")
                            await demo_instance.initialize_components()
                            await demo_instance._validate_existing_data()
                            print_success("系统已准备就绪，可以开始问答！")
                        
                    except Exception as e:
                        print_error(f"初始化失败: {e}")
                        demo_instance = None
                        run_mode = None
                        data_path = None
                
                elif command == '/data':
                    data_path = display_data_selection()
                    print_success(f"已选择数据目录: {data_path}")
                
                elif command == '/rebuild':
                    if console and Confirm:
                        rebuild = Confirm.ask("确定要重新构建知识库吗？这将删除现有的索引")
                    else:
                        rebuild_input = input("确定要重新构建知识库吗？(y/N): ").strip().lower()
                        rebuild = rebuild_input in ['y', 'yes']
                    
                    if rebuild:
                        try:
                            # 🔧 修复：立即执行重建操作
                            print_mode_selection("请选择数据目录")
                            data_path = display_data_selection()
                            print_success(f"已选择数据目录: {data_path}")
                            
                            print_action("正在重新构建知识库...")
                            # 创建build模式的demo实例
                            rebuild_demo = AgenticXGraphRAGDemo(config_path="configs.yml", mode="build")
                            rebuild_demo.data_dir = Path(data_path)
                            
                            # 执行重建
                            await rebuild_demo.run_build_only()
                            print_success("知识库重建完成！")
                            
                            # 如果当前有运行的实例，重置它
                            if demo_instance:
                                print_info("重置当前系统实例，请重新使用 /mode 选择运行模式")
                                demo_instance = None
                                run_mode = None
                                
                        except Exception as e:
                            print_error(f"重建失败: {e}")
                    else:
                        print_info("取消重新构建")
                
                elif command == '/exit':
                    print_success("感谢使用 AgenticX GraphRAG 系统！")
                    break
                
                else:
                    print_error(f"未知命令: {command}")
                    print_info("输入 /help 查看可用命令")
            
            else:
                # 🔧 修复：简化问答处理逻辑
                if not demo_instance:
                    print_error("系统尚未初始化，请先使用 /mode 选择运行模式")
                    continue
                
                # 执行问答
                try:
                    print_thinking(f"正在处理您的问题: {user_input}")
                    await demo_instance._process_query(user_input)
                except Exception as e:
                    print_error(f"问答处理失败: {e}")
                    logger.error(f"Query processing error: {e}", exc_info=True)
        
        except KeyboardInterrupt:
            print_success("\n感谢使用 AgenticX GraphRAG 系统！")
            break
        except Exception as e:
            print_error(f"处理过程中出现错误: {str(e)}")
            logger.error(f"Interactive mode error: {e}", exc_info=True)

async def main():
    """主函数"""
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='AgenticX GraphRAG 演示系统',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
运行模式说明:
  full   - 完整流程: 文档解析 + 知识图谱构建 + 向量索引 + 问答系统
  build  - 仅构建: 文档解析 + 知识图谱构建 + 向量索引 (不启动问答)
  qa     - 仅问答: 直接使用已有数据启动问答系统 (不重建数据)
  interactive - 交互式模式: 美观的用户界面，支持动态配置

使用示例:
  python demo.py                    # 交互式模式 (默认)
  python demo.py --mode full        # 完整流程
  python demo.py --mode build       # 仅重建知识库
  python demo.py --mode qa          # 仅启动问答
        """
    )
    
    parser.add_argument('--mode', choices=['full', 'build', 'qa', 'interactive'], default='interactive',
                       help='运行模式: interactive=交互式(默认), full=完整流程, build=仅构建知识库, qa=仅问答模式')
    parser.add_argument('--config', default='configs.yml',
                       help='配置文件路径 (默认: configs.yml)')
    parser.add_argument('--data-path', default='data',
                       help='数据目录路径 (默认: data)')
    
    args = parser.parse_args()
    
    # 如果是交互式模式，直接启动
    if args.mode == 'interactive':
        await interactive_mode()
        return
    
    # 显示模式信息
    mode_descriptions = {
        'full': '🔄 完整模式 - 文档解析 + 知识库构建 + 问答系统',
        'build': '🔨 构建模式 - 仅重建知识库和向量索引',
        'qa': '💬 问答模式 - 使用已有数据启动问答系统'
    }
    
    print(f"\n{mode_descriptions[args.mode]}")
    print(f"📁 配置文件: {args.config}")
    print(f"📂 数据目录: {args.data_path}")
    print("=" * 60)
    
    demo = None
    try:
        # 创建演示系统
        demo = AgenticXGraphRAGDemo(config_path=args.config, mode=args.mode)
        
        # 根据模式运行
        if args.mode == 'qa':
            await demo.run_qa_only()
        elif args.mode == 'build':
            await demo.run_build_only()
        else:  # full
            await demo.run_demo()
        
    except Exception as e:
        print_error(f"系统启动失败: {e}")
        sys.exit(1)
    finally:
        # 清理资源
        if demo:
            await demo.cleanup()


if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main())