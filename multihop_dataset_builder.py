#!/usr/bin/env python3
"""
多跳复杂推理问答对数据集构建器

功能：
- 支持多种文档格式（PDF、TXT、JSON、CSV等）
- 集成提示词管理器，支持领域通用配置
- 支持多种LLM提供商（百炼、OpenAI等）
- 输出标准JSON格式的多跳问答对数据集

使用方法：
python multihop_dataset_builder.py --data_path ./documents --output ./output.json --llm_provider bailian --llm_model qwen3-max
"""

import os
import sys
import json
import yaml
import asyncio
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量
script_dir = Path(__file__).parent
env_path = script_dir / ".env"
load_dotenv(env_path)

# 导入AgenticX组件
from agenticx.knowledge import Document
from agenticx.knowledge.readers import PDFReader, TextReader, JSONReader, CSVReader
from agenticx.llms import LlmFactory
from agenticx.knowledge.graphers.config import LLMConfig

# 导入本地组件
from prompt_manager import PromptManager


class MultihopDatasetBuilder:
    """多跳数据集构建器"""
    
    def __init__(self, config_path: str = "configs.yml"):
        self.config_path = config_path
        self.config = self._load_config()
        self.prompt_manager = PromptManager("prompts")
        self.llm_client = None
        
        # 设置日志
        logger.add(
            "logs/multihop_builder.log",
            rotation="10 MB",
            retention="7 days",
            level="INFO"
        )
        
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 替换环境变量
            self._replace_env_vars(config)
            return config
            
        except Exception as e:
            logger.error(f"❌ 加载配置文件失败: {e}")
            return {}
    
    def _replace_env_vars(self, obj: Any) -> None:
        """递归替换配置中的环境变量"""
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    obj[key] = os.getenv(env_var, value)
                elif isinstance(value, (dict, list)):
                    self._replace_env_vars(value)
        elif isinstance(obj, list):
            for item in obj:
                self._replace_env_vars(item)
    
    async def initialize_llm(self, provider: str = None, model: str = None) -> None:
        """初始化LLM客户端"""
        # 使用命令行参数或配置文件中的强模型配置
        if provider and model:
            llm_config_dict = {
                'provider': provider,
                'model': model,
                'api_key': self.config['llm']['api_key'],
                'base_url': self.config['llm']['base_url'],
                'temperature': 0.1,
                'max_tokens': 128000,
                'timeout': 600,
                'retry_attempts': 3
            }
        else:
            # 使用强模型配置
            llm_config_dict = self.config['llm']['strong_model']
        
        # 提供商类型映射
        provider_type_mapping = {
            'openai': 'litellm',
            'anthropic': 'litellm', 
            'litellm': 'litellm',
            'bailian': 'bailian',
            'kimi': 'kimi'
        }
        
        llm_type = provider_type_mapping.get(llm_config_dict['provider'], 'litellm')
        
        # 创建LLM配置对象
        llm_config = LLMConfig(
            type=llm_type,
            model=llm_config_dict.get('model'),
            api_key=llm_config_dict.get('api_key'),
            base_url=llm_config_dict.get('base_url'),
            provider=llm_config_dict.get('provider'),
            timeout=llm_config_dict.get('timeout'),
            max_retries=llm_config_dict.get('retry_attempts'),
            temperature=llm_config_dict.get('temperature'),
            max_tokens=llm_config_dict.get('max_tokens')
        )
        
        self.llm_client = LlmFactory.create_llm(llm_config)
        logger.info(f"✅ LLM初始化完成: {llm_config.provider}/{llm_config.model}")
    
    async def load_documents(self, data_path: str, file_types: List[str] = None) -> List[Document]:
        """加载文档
        
        Args:
            data_path: 数据文件路径（文件或目录）
            file_types: 支持的文件类型列表，如['pdf', 'txt', 'json']
            
        Returns:
            文档列表
        """
        documents = []
        data_path = Path(data_path)
        
        if file_types is None:
            file_types = ['pdf', 'txt', 'md', 'json', 'csv']
        
        # 获取文件列表
        if data_path.is_file():
            file_paths = [data_path]
        elif data_path.is_dir():
            file_paths = []
            for file_type in file_types:
                file_paths.extend(data_path.glob(f"*.{file_type}"))
                file_paths.extend(data_path.glob(f"**/*.{file_type}"))
        else:
            logger.error(f"❌ 数据路径不存在: {data_path}")
            return []
        
        logger.info(f"📄 发现 {len(file_paths)} 个文档文件")
        
        # 加载文档
        reader_config = self.config['knowledge']['readers']
        
        for file_path in file_paths:
            try:
                # 根据文件类型选择读取器
                if file_path.suffix.lower() == '.pdf' and reader_config['pdf']['enabled']:
                    pdf_config = reader_config['pdf'].copy()
                    pdf_config.pop('enabled', None)
                    reader = PDFReader(**pdf_config)
                elif file_path.suffix.lower() in ['.txt', '.md'] and reader_config['text']['enabled']:
                    text_config = reader_config['text'].copy()
                    text_config.pop('enabled', None)
                    reader = TextReader(**text_config)
                elif file_path.suffix.lower() == '.json' and reader_config['json']['enabled']:
                    json_config = reader_config['json'].copy()
                    json_config.pop('enabled', None)
                    reader = JSONReader(**json_config)
                elif file_path.suffix.lower() == '.csv' and reader_config['csv']['enabled']:
                    csv_config = reader_config['csv'].copy()
                    csv_config.pop('enabled', None)
                    reader = CSVReader(**csv_config)
                else:
                    logger.warning(f"⚠️ 不支持的文件类型: {file_path}")
                    continue
                
                # 读取文档
                result = await reader.read(str(file_path))
                
                # 处理返回结果
                if isinstance(result, list):
                    for doc in result:
                        documents.append(doc)
                        logger.info(f"✅ 加载文档: {file_path.name} ({len(doc.content)} 字符)")
                else:
                    documents.append(result)
                    logger.info(f"✅ 加载文档: {file_path.name} ({len(result.content)} 字符)")
                
            except Exception as e:
                logger.error(f"❌ 加载文档失败 {file_path}: {e}")
                continue
        
        logger.info(f"📚 总计加载 {len(documents)} 个文档")
        return documents
    
    def prepare_prompt_variables(self, domain_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """准备提示词变量
        
        Args:
            domain_config: 领域特定配置
            
        Returns:
            提示词变量字典
        """
        # 加载多跳数据集生成配置
        prompt_config = self.prompt_manager.load_prompt("multihop_dataset_generation")
        
        # 基础变量
        variables = {
            'sample_nums': prompt_config.get('config', {}).get('sample_nums', 20),
            'min_documents_per_question': prompt_config.get('config', {}).get('min_documents_per_question', 2),
            'max_answer_sentences': prompt_config.get('config', {}).get('max_answer_sentences', 3),
            'target_language': prompt_config.get('config', {}).get('target_language', '中文'),
            'allowed_terms': prompt_config.get('config', {}).get('allowed_terms', '英文术语'),
        }
        
        # 默认变量值
        default_variables = prompt_config.get('variables', {})
        for key, value in default_variables.items():
            variables[key] = value
        
        # 应用领域特定配置
        if domain_config:
            variables.update(domain_config)
        
        return variables
    
    async def generate_dataset(self, 
                             documents: List[Document], 
                             domain_config: Dict[str, Any] = None,
                             custom_variables: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """生成多跳数据集
        
        Args:
            documents: 文档列表
            domain_config: 领域特定配置
            custom_variables: 自定义变量
            
        Returns:
            多跳问答对列表
        """
        if not self.llm_client:
            logger.error("❌ LLM客户端未初始化")
            return []
        
        # 准备文档内容
        documents_text = ""
        for i, doc in enumerate(documents, 1):
            doc_name = getattr(doc.metadata, 'name', f'document_{i}')
            documents_text += f"\n\n=== 文档 {i}: {doc_name} ===\n{doc.content}"
        
        # 准备提示词变量
        variables = self.prepare_prompt_variables(domain_config)
        variables['documents'] = documents_text
        
        # 应用自定义变量
        if custom_variables:
            variables.update(custom_variables)
        
        # 格式化提示词
        prompt = self.prompt_manager.format_prompt(
            "multihop_dataset_generation", 
            "template", 
            **variables
        )
        
        if not prompt:
            logger.error("❌ 提示词格式化失败")
            return []
        
        logger.info(f"🚀 开始生成多跳数据集，目标数量: {variables['sample_nums']}")
        logger.debug(f"📝 提示词长度: {len(prompt)} 字符")
        
        try:
            # 调用LLM生成数据集
            response = await self.llm_client.achat(prompt)
            
            # 解析JSON响应
            response_text = response.strip()
            
            # 尝试提取JSON部分
            if "```json" in response_text:
                start = response_text.find("```json") + 7
                end = response_text.find("```", start)
                if end != -1:
                    response_text = response_text[start:end].strip()
            
            # 解析JSON
            dataset = json.loads(response_text)
            
            if not isinstance(dataset, list):
                logger.error("❌ 响应格式错误：期望JSON数组")
                return []
            
            logger.info(f"✅ 成功生成 {len(dataset)} 个多跳问答对")
            
            # 验证数据质量
            valid_dataset = self._validate_dataset(dataset)
            logger.info(f"📊 验证通过: {len(valid_dataset)}/{len(dataset)} 个问答对")
            
            return valid_dataset
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON解析失败: {e}")
            logger.debug(f"原始响应: {response_text[:500]}...")
            return []
        except Exception as e:
            logger.error(f"❌ 数据集生成失败: {e}")
            return []
    
    def _validate_dataset(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """验证数据集质量"""
        valid_items = []
        
        for i, item in enumerate(dataset):
            try:
                # 检查必需字段
                required_fields = ['query', 'from_docs', 'expected_output', 'criteria_clarify']
                if not all(field in item for field in required_fields):
                    logger.warning(f"⚠️ 第{i+1}个问答对缺少必需字段")
                    continue
                
                # 检查字段类型
                if not isinstance(item['query'], str) or not item['query'].strip():
                    logger.warning(f"⚠️ 第{i+1}个问答对的query无效")
                    continue
                
                if not isinstance(item['from_docs'], list) or len(item['from_docs']) < 1:
                    logger.warning(f"⚠️ 第{i+1}个问答对的from_docs无效")
                    continue
                
                if not isinstance(item['expected_output'], str) or not item['expected_output'].strip():
                    logger.warning(f"⚠️ 第{i+1}个问答对的expected_output无效")
                    continue
                
                valid_items.append(item)
                
            except Exception as e:
                logger.warning(f"⚠️ 第{i+1}个问答对验证失败: {e}")
                continue
        
        return valid_items
    
    async def save_dataset(self, dataset: List[Dict[str, Any]], output_path: str) -> None:
        """保存数据集到文件"""
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 数据集已保存到: {output_path}")
            logger.info(f"📊 数据集统计: {len(dataset)} 个问答对")
            
        except Exception as e:
            logger.error(f"❌ 保存数据集失败: {e}")
    
    async def build_dataset(self, 
                          data_path: str,
                          output_path: str,
                          llm_provider: str = None,
                          llm_model: str = None,
                          domain_config: Dict[str, Any] = None,
                          file_types: List[str] = None) -> None:
        """构建多跳数据集的主要流程
        
        Args:
            data_path: 数据文件路径
            output_path: 输出文件路径
            llm_provider: LLM提供商
            llm_model: LLM模型名称
            domain_config: 领域特定配置
            file_types: 支持的文件类型
        """
        try:
            # 1. 初始化LLM
            await self.initialize_llm(llm_provider, llm_model)
            
            # 2. 加载文档
            documents = await self.load_documents(data_path, file_types)
            
            if not documents:
                logger.error("❌ 没有成功加载任何文档")
                return
            
            # 3. 生成数据集
            dataset = await self.generate_dataset(documents, domain_config)
            
            if not dataset:
                logger.error("❌ 数据集生成失败")
                return
            
            # 4. 保存数据集
            await self.save_dataset(dataset, output_path)
            
            logger.info("🎉 多跳数据集构建完成！")
            
        except Exception as e:
            logger.error(f"❌ 数据集构建失败: {e}")
    
    async def load_documents(self, data_path: str, file_types: List[str] = None) -> List[Document]:
        """加载文档（复用main.py中的逻辑）"""
        documents = []
        data_path = Path(data_path)
        
        if file_types is None:
            file_types = ['pdf', 'txt', 'md', 'json', 'csv']
        
        # 获取文件列表
        if data_path.is_file():
            file_paths = [data_path]
        elif data_path.is_dir():
            file_paths = []
            for file_type in file_types:
                file_paths.extend(data_path.glob(f"*.{file_type}"))
                file_paths.extend(data_path.glob(f"**/*.{file_type}"))
        else:
            logger.error(f"❌ 数据路径不存在: {data_path}")
            return []
        
        logger.info(f"📄 发现 {len(file_paths)} 个文档文件")
        
        # 加载文档
        reader_config = self.config['knowledge']['readers']
        
        for file_path in file_paths:
            try:
                # 根据文件类型选择读取器
                if file_path.suffix.lower() == '.pdf' and reader_config['pdf']['enabled']:
                    pdf_config = reader_config['pdf'].copy()
                    pdf_config.pop('enabled', None)
                    reader = PDFReader(**pdf_config)
                elif file_path.suffix.lower() in ['.txt', '.md'] and reader_config['text']['enabled']:
                    text_config = reader_config['text'].copy()
                    text_config.pop('enabled', None)
                    reader = TextReader(**text_config)
                elif file_path.suffix.lower() == '.json' and reader_config['json']['enabled']:
                    json_config = reader_config['json'].copy()
                    json_config.pop('enabled', None)
                    reader = JSONReader(**json_config)
                elif file_path.suffix.lower() == '.csv' and reader_config['csv']['enabled']:
                    csv_config = reader_config['csv'].copy()
                    csv_config.pop('enabled', None)
                    reader = CSVReader(**csv_config)
                else:
                    logger.warning(f"⚠️ 不支持的文件类型: {file_path}")
                    continue
                
                # 读取文档
                result = await reader.read(str(file_path))
                
                # 处理返回结果
                if isinstance(result, list):
                    for doc in result:
                        documents.append(doc)
                        logger.debug(f"📄 加载文档: {file_path.name}")
                else:
                    documents.append(result)
                    logger.debug(f"📄 加载文档: {file_path.name}")
                
            except Exception as e:
                logger.error(f"❌ 加载文档失败 {file_path}: {e}")
                continue
        
        logger.info(f"📚 总计加载 {len(documents)} 个文档")
        return documents


def create_domain_config(domain: str) -> Dict[str, Any]:
    """创建领域特定配置
    
    Args:
        domain: 领域名称 (technology, finance, medical, general等)
        
    Returns:
        领域配置字典
    """
    domain_configs = {
        'technology': {
            'domain_guidance': '''
请根据文档内容自动识别技术主题（如算法、框架、系统架构等），然后选择两个不同技术主题进行组合，生成跨文档多跳问题。
注意识别文档中的核心技术概念、实现方法、性能指标，并寻找它们之间的技术关联和对比点。
            '''.strip(),
            'domain_specific_terms': '算法名称、性能指标、技术架构、框架组件',
            'comparison_aspect': '技术架构/算法设计/性能表现/实现复杂度',
            'mechanism_name': '核心算法/技术机制/架构设计',
            'target_aspect': '性能/效率/准确性/可扩展性',
            'challenge': '技术挑战/性能瓶颈/扩展性问题'
        },
        'finance': {
            'domain_guidance': '''
请根据文档内容自动识别金融主题（如交易策略、风险管理、市场分析等），然后选择两个不同金融主题进行组合，生成跨文档多跳问题。
注意识别文档中的金融概念、策略方法、风险指标，并寻找它们之间的业务关联和对比点。
            '''.strip(),
            'domain_specific_terms': '交易策略、风险指标、收益率、金融工具',
            'comparison_aspect': '策略设计/风险控制/收益表现/市场适应性',
            'mechanism_name': '交易机制/风控策略/分析方法',
            'target_aspect': '收益率/风险控制/流动性/稳定性',
            'challenge': '市场波动/风险控制/监管要求'
        },
        'medical': {
            'domain_guidance': '''
请根据文档内容自动识别医学主题（如诊断方法、治疗方案、药物机制等），然后选择两个不同医学主题进行组合，生成跨文档多跳问题。
注意识别文档中的医学概念、治疗方法、临床指标，并寻找它们之间的医学关联和对比点。
            '''.strip(),
            'domain_specific_terms': '疾病名称、药物成分、治疗方案、临床指标',
            'comparison_aspect': '治疗效果/副作用/适应症/治疗机制',
            'mechanism_name': '治疗机制/药物作用/诊断方法',
            'target_aspect': '疗效/安全性/适用性/便利性',
            'challenge': '复杂病例/药物耐受/个体差异'
        },
        'general': {
            'domain_guidance': '''
请根据文档内容自动识别主要主题和概念，然后选择两个不同主题进行组合，生成跨文档多跳问题。
注意识别文档中的核心概念、方法、系统或理论，并寻找它们之间的潜在关联。
            '''.strip(),
            'domain_specific_terms': '专业术语、关键概念、方法名称、指标数据',
            'comparison_aspect': '设计理念/实现方法/效果表现/应用场景',
            'mechanism_name': '核心机制/主要方法/关键策略',
            'target_aspect': '效果/效率/准确性/适用性',
            'challenge': '复杂场景/性能要求/实际限制'
        }
    }
    
    return domain_configs.get(domain, domain_configs['general'])


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多跳复杂推理问答对数据集构建器")
    
    # 必需参数
    parser.add_argument("--data_path", required=True, help="数据文件路径（文件或目录）")
    parser.add_argument("--output", required=True, help="输出JSON文件路径")
    
    # LLM配置参数
    parser.add_argument("--llm_provider", help="LLM提供商 (bailian, openai, anthropic等)")
    parser.add_argument("--llm_model", help="LLM模型名称")
    
    # 可选参数
    parser.add_argument("--config", default="configs.yml", help="配置文件路径")
    parser.add_argument("--domain", default="general", 
                       choices=['technology', 'finance', 'medical', 'general'],
                       help="领域类型")
    parser.add_argument("--file_types", nargs='+', 
                       default=['pdf', 'txt', 'md', 'json', 'csv'],
                       help="支持的文件类型")
    parser.add_argument("--sample_nums", type=int, help="生成问题数量")
    parser.add_argument("--min_docs", type=int, help="每个问题最少涉及文档数")
    
    args = parser.parse_args()
    
    try:
        # 创建构建器
        builder = MultihopDatasetBuilder(args.config)
        
        # 准备领域配置
        domain_config = create_domain_config(args.domain)
        
        # 准备自定义变量
        custom_variables = {}
        if args.sample_nums:
            custom_variables['sample_nums'] = args.sample_nums
        if args.min_docs:
            custom_variables['min_documents_per_question'] = args.min_docs
        
        # 构建数据集
        await builder.build_dataset(
            data_path=args.data_path,
            output_path=args.output,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            domain_config=domain_config,
            custom_variables=custom_variables if custom_variables else None
        )
        
    except KeyboardInterrupt:
        logger.info("👋 用户中断操作")
    except Exception as e:
        logger.error(f"❌ 程序执行失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())