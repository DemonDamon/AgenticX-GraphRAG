#!/usr/bin/env python3
"""
查询分解器模块
基于youtu-graphrag的先进思想，实现智能查询分解功能
支持多跳推理查询的自动分解和并行检索
"""

import json
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml
from loguru import logger

try:
    import json_repair
except ImportError:
    # 如果没有json_repair，使用标准json
    import json as json_repair

from agenticx.llms import LlmFactory
from agenticx.knowledge.graphers.config import LLMConfig


@dataclass
class SubQuestion:
    """子问题数据结构"""
    question: str
    confidence: float
    reasoning_type: str  # "factual", "relational", "comparative", "temporal"
    entities: List[str]
    relations: List[str]
    priority: int = 1  # 1-高优先级, 2-中优先级, 3-低优先级


@dataclass
class DecompositionResult:
    """查询分解结果"""
    original_query: str
    sub_questions: List[SubQuestion]
    decomposition_confidence: float
    reasoning_complexity: str  # "simple", "medium", "complex"
    involved_types: Dict[str, List[str]]
    decomposition_strategy: str
    should_decompose: bool


class QueryDecomposer:
    """
    智能查询分解器
    基于youtu-graphrag的agentic_decomposer思想，结合AgenticX框架
    """
    
    def __init__(self, config: Dict[str, Any], schema_path: Optional[str] = None):
        """初始化查询分解器"""
        self.config = config
        self.decomposition_config = config.get('retrieval', {}).get('query_decomposition', {})
        self.schema_path = schema_path or "schema.json"
        
        # 初始化LLM客户端
        llm_config = config.get('llm', {})
        decomposition_model = self.decomposition_config.get('decomposition_model', 'qwen3-max')
        
        # 使用强模型进行查询分解
        strong_model_config = config.get('llm', {}).get('strong_model', llm_config)
        
        # 创建LLMConfig对象
        llm_config_obj = LLMConfig(
            type=strong_model_config.get('provider', 'bailian'),
            model=decomposition_model,
            api_key=strong_model_config.get('api_key'),
            base_url=strong_model_config.get('base_url'),
            timeout=strong_model_config.get('timeout'),
            max_retries=strong_model_config.get('max_retries'),
            temperature=strong_model_config.get('temperature'),
            max_tokens=strong_model_config.get('max_tokens')
        )
        
        self.llm_client = LlmFactory.create_llm(llm_config_obj)
        
        # 分解配置参数
        self.enable_decomposition = self.decomposition_config.get('enable_decomposition', True)
        self.decomposition_count = self.decomposition_config.get('decomposition_count', 3)
        self.min_query_length = self.decomposition_config.get('min_query_length', 10)
        self.confidence_threshold = self.decomposition_config.get('confidence_threshold', 0.7)
        self.enable_schema_aware = self.decomposition_config.get('enable_schema_aware', True)
        self.enable_entity_focus = self.decomposition_config.get('enable_entity_focus', True)
        self.fallback_to_original = self.decomposition_config.get('fallback_to_original', True)
        
        # 加载图模式
        self.schema = self._load_schema()
        
        logger.info(f"查询分解器初始化完成，模型: {decomposition_model}")

    def _load_schema(self) -> str:
        """加载图模式"""
        try:
            if Path(self.schema_path).exists():
                with open(self.schema_path, 'r', encoding='utf-8') as f:
                    schema_data = json.load(f)
                    return json.dumps(schema_data, ensure_ascii=False, indent=2)
            else:
                logger.warning(f"Schema文件不存在: {self.schema_path}")
                return "{}"
        except Exception as e:
            logger.error(f"加载Schema失败: {e}")
            return "{}"

    def _analyze_query_complexity(self, query: str) -> Tuple[str, float]:
        """分析查询复杂度"""
        # 简单启发式规则
        complexity_indicators = {
            'simple': ['是什么', '什么是', '定义', '介绍'],
            'medium': ['如何', '为什么', '原因', '方法', '过程'],
            'complex': ['比较', '对比', '哪个更', '最', '关系', '影响', '导致']
        }
        
        query_lower = query.lower()
        
        # 检查复杂度指标
        for complexity, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                confidence = 0.8 if complexity == 'complex' else 0.6
                return complexity, confidence
        
        # 基于查询长度判断
        if len(query) > 50:
            return 'complex', 0.7
        elif len(query) > 20:
            return 'medium', 0.6
        else:
            return 'simple', 0.5

    def _should_decompose(self, query: str) -> bool:
        """判断是否需要分解查询"""
        if not self.enable_decomposition:
            return False
            
        if len(query) < self.min_query_length:
            return False
            
        complexity, confidence = self._analyze_query_complexity(query)
        
        # 复杂查询需要分解
        if complexity in ['medium', 'complex'] and confidence > 0.6:
            return True
            
        # 包含多个实体或关系的查询
        entity_keywords = ['和', '与', '以及', '还有', '对比', '比较']
        if any(keyword in query for keyword in entity_keywords):
            return True
            
        return False

    def should_decompose(self, query: str) -> bool:
        """公共方法：判断是否需要分解查询"""
        return self._should_decompose(query)

    def _create_decomposition_prompt(self, query: str, language: str = "chinese") -> str:
        """创建分解提示词"""
        if language == "chinese":
            return f"""
你是一个专业的问题分解大师，擅长将复杂问题分解为简单的子问题。
请根据以下问题和图本体模式，将问题分解为{self.decomposition_count}个子问题。

核心要求：
1. 每个子问题必须：
   - 明确且专注于一个事实或关系
   - 能够独立回答，不依赖其他子问题的结果
   - 明确引用原始问题中的实体和关系
   - 设计为检索最终答案所需的相关知识

2. 分解策略：
   - 对于简单问题（1-2跳推理），返回原始问题作为单个子问题
   - 对于复杂问题，按照逻辑推理链分解
   - 优先分解实体识别、关系查询、属性获取等基础问题

3. 返回格式：
   请返回一个JSON对象，包含以下字段：
   - sub_questions: 子问题列表，每个包含question、confidence、reasoning_type、entities、relations
   - decomposition_confidence: 分解置信度(0-1)
   - reasoning_complexity: 推理复杂度("simple", "medium", "complex")
   - involved_types: 涉及的节点类型、关系类型、属性类型

原始问题：{query}

图本体模式：
{self.schema}

示例输出：
{{
    "sub_questions": [
        {{
            "question": "什么是智取生辰纲事件？",
            "confidence": 0.9,
            "reasoning_type": "factual",
            "entities": ["智取生辰纲"],
            "relations": ["定义", "描述"]
        }},
        {{
            "question": "智取生辰纲事件中的主要人物有哪些？",
            "confidence": 0.8,
            "reasoning_type": "relational",
            "entities": ["智取生辰纲", "人物"],
            "relations": ["参与", "涉及"]
        }}
    ],
    "decomposition_confidence": 0.85,
    "reasoning_complexity": "medium",
    "involved_types": {{
        "nodes": ["EVENT", "PERSON"],
        "relations": ["PARTICIPATES_IN", "INVOLVES"],
        "attributes": ["name", "role"]
    }}
}}
"""
        else:
            return f"""
You are a professional question decomposition expert specializing in multi-hop reasoning.
Given the following schema and question, decompose the complex question into {self.decomposition_count} focused sub-questions.

CRITICAL REQUIREMENTS:
1. Each sub-question must be:
   - Specific and focused on a single fact or relationship
   - Answerable independently with the given schema
   - Explicitly reference entities and relations from the original question
   - Designed to retrieve relevant knowledge for the final answer

2. Decomposition Strategy:
   - For simple questions (1-2 hop), return the original question as a single sub-question
   - For complex questions, decompose along logical reasoning chains
   - Prioritize entity identification, relationship queries, and attribute retrieval

3. Return Format:
   Return a JSON object with the following fields:
   - sub_questions: List of sub-questions with question, confidence, reasoning_type, entities, relations
   - decomposition_confidence: Confidence in decomposition (0-1)
   - reasoning_complexity: Reasoning complexity ("simple", "medium", "complex")
   - involved_types: Involved node types, relation types, attribute types

Original Question: {query}

Graph Schema:
{self.schema}

Example Output:
{{
    "sub_questions": [
        {{
            "question": "Who is the director of Ethnic Notions?",
            "confidence": 0.9,
            "reasoning_type": "factual",
            "entities": ["Ethnic Notions"],
            "relations": ["directed_by"]
        }},
        {{
            "question": "When did the director of Ethnic Notions die?",
            "confidence": 0.8,
            "reasoning_type": "temporal",
            "entities": ["director", "Ethnic Notions"],
            "relations": ["death_date", "directed_by"]
        }}
    ],
    "decomposition_confidence": 0.85,
    "reasoning_complexity": "medium",
    "involved_types": {{
        "nodes": ["FILM", "PERSON"],
        "relations": ["DIRECTED_BY", "DEATH_DATE"],
        "attributes": ["name", "date"]
    }}
}}
"""

    async def decompose_query(self, query: str) -> DecompositionResult:
        """分解查询为子问题"""
        logger.info(f"开始分解查询: {query}")
        
        # 检查是否需要分解
        should_decompose = self._should_decompose(query)
        if not should_decompose:
            logger.info("查询无需分解，返回原查询")
            return DecompositionResult(
                original_query=query,
                sub_questions=[SubQuestion(
                    question=query,
                    confidence=1.0,
                    reasoning_type="simple",
                    entities=[],
                    relations=[]
                )],
                decomposition_confidence=1.0,
                reasoning_complexity="simple",
                involved_types={"nodes": [], "relations": [], "attributes": []},
                decomposition_strategy="no_decomposition",
                should_decompose=False
            )
        
        try:
            # 检测语言
            language = "chinese" if any('\u4e00' <= char <= '\u9fff' for char in query) else "english"
            
            # 创建分解提示词
            prompt = self._create_decomposition_prompt(query, language)
            
            # 调用LLM进行分解
            response = await self.llm_client.ainvoke(prompt)
            
            # 解析响应
            try:
                # 清理响应内容，移除markdown代码块标记
                raw_content = response.content.strip()
                
                # 移除markdown代码块标记
                if raw_content.startswith('```json'):
                    raw_content = raw_content[7:]  # 移除 ```json
                if raw_content.startswith('```'):
                    raw_content = raw_content[3:]   # 移除 ```
                if raw_content.endswith('```'):
                    raw_content = raw_content[:-3]  # 移除结尾的 ```
                
                raw_content = raw_content.strip()
                logger.debug(f"清理后的JSON内容: {raw_content[:200]}...")
                
                content = json_repair.loads(raw_content)
                logger.info(f"✅ JSON解析成功")
            except Exception as e:
                logger.error(f"❌ JSON解析失败: {e}")
                logger.error(f"原始响应: {response.content}")
                logger.error(f"清理后内容: {raw_content if 'raw_content' in locals() else 'N/A'}")
                # 回退到原查询
                if self.fallback_to_original:
                    return self._create_fallback_result(query)
                raise
            
            # 构建子问题列表
            sub_questions = []
            for i, sub_q in enumerate(content.get('sub_questions', [])):
                sub_question = SubQuestion(
                    question=sub_q.get('question', ''),
                    confidence=sub_q.get('confidence', 0.5),
                    reasoning_type=sub_q.get('reasoning_type', 'factual'),
                    entities=sub_q.get('entities', []),
                    relations=sub_q.get('relations', []),
                    priority=i + 1
                )
                sub_questions.append(sub_question)
            
            # 构建分解结果
            result = DecompositionResult(
                original_query=query,
                sub_questions=sub_questions,
                decomposition_confidence=content.get('decomposition_confidence', 0.7),
                reasoning_complexity=content.get('reasoning_complexity', 'medium'),
                involved_types=content.get('involved_types', {}),
                decomposition_strategy="llm_decomposition",
                should_decompose=True
            )
            
            logger.info(f"查询分解完成，生成{len(sub_questions)}个子问题")
            return result
            
        except Exception as e:
            logger.error(f"查询分解失败: {e}")
            if self.fallback_to_original:
                return self._create_fallback_result(query)
            raise

    def _create_fallback_result(self, query: str) -> DecompositionResult:
        """创建回退结果"""
        return DecompositionResult(
            original_query=query,
            sub_questions=[SubQuestion(
                question=query,
                confidence=1.0,
                reasoning_type="fallback",
                entities=[],
                relations=[]
            )],
            decomposition_confidence=0.5,
            reasoning_complexity="unknown",
            involved_types={"nodes": [], "relations": [], "attributes": []},
            decomposition_strategy="fallback",
            should_decompose=False
        )

    async def parallel_decompose_and_retrieve(self, query: str, retriever) -> Tuple[List[Any], Dict[str, Any]]:
        """并行分解和检索"""
        # 1. 分解查询
        decomposition_result = await self.decompose_query(query)
        
        if not decomposition_result.should_decompose:
            # 不需要分解，直接检索
            results = await retriever.retrieve_single_query(query)
            return results, {
                'decomposition_result': decomposition_result,
                'retrieval_strategy': 'direct',
                'sub_question_count': 1
            }
        
        # 2. 并行检索子问题
        parallel_retrieval = self.decomposition_config.get('parallel_retrieval', True)
        
        # 记录子问题详情
        logger.info(f"📋 开始检索{len(decomposition_result.sub_questions)}个子问题:")
        for i, sub_q in enumerate(decomposition_result.sub_questions, 1):
            logger.info(f"  {i}. [{sub_q.reasoning_type}] {sub_q.question} (置信度: {sub_q.confidence:.2f})")
        
        if parallel_retrieval:
            # 并行检索
            logger.info("🔄 执行并行检索...")
            tasks = []
            for sub_q in decomposition_result.sub_questions:
                task = retriever.retrieve_single_query(sub_q.question)
                tasks.append(task)
            
            sub_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # 顺序检索
            logger.info("🔄 执行顺序检索...")
            sub_results = []
            for i, sub_q in enumerate(decomposition_result.sub_questions, 1):
                try:
                    logger.info(f"  检索子问题 {i}/{len(decomposition_result.sub_questions)}: {sub_q.question[:50]}...")
                    result = await retriever.retrieve_single_query(sub_q.question)
                    sub_results.append(result)
                    logger.info(f"  ✅ 子问题 {i} 检索完成: {len(result)}个结果")
                except Exception as e:
                    logger.error(f"  ❌ 子问题 {i} 检索失败: {e}")
                    sub_results.append([])
        
        # 记录每个子问题的检索统计
        logger.info("📊 子问题检索统计:")
        total_results = 0
        for i, (sub_q, result) in enumerate(zip(decomposition_result.sub_questions, sub_results), 1):
            if isinstance(result, Exception):
                logger.warning(f"  {i}. ❌ 异常: {result}")
                result_count = 0
            else:
                result_count = len(result) if isinstance(result, list) else 0
                total_results += result_count
                status = "✅" if result_count > 0 else "⚠️"
                logger.info(f"  {i}. {status} [{sub_q.reasoning_type}] {result_count}个结果 - {sub_q.question[:60]}...")
        
        logger.info(f"📈 检索汇总: 总计{total_results}个结果来自{len(decomposition_result.sub_questions)}个子问题")
        
        # 3. 合并结果
        logger.info("🔀 开始合并子问题结果...")
        merged_results = self._merge_sub_results(
            sub_results, 
            decomposition_result.sub_questions
        )
        logger.info(f"✨ 合并完成: 最终获得{len(merged_results)}个去重后的结果")
        
        return merged_results, {
            'decomposition_result': decomposition_result,
            'retrieval_strategy': 'parallel' if parallel_retrieval else 'sequential',
            'sub_question_count': len(decomposition_result.sub_questions),
            'sub_results_count': [len(r) if isinstance(r, list) else 0 for r in sub_results]
        }

    def _merge_sub_results(self, sub_results: List[List[Any]], sub_questions: List[SubQuestion]) -> List[Any]:
        """合并子问题的检索结果"""
        merge_strategy = self.decomposition_config.get('merge_strategy', 'weighted_score')
        
        all_results = []
        seen_contents = set()
        
        for i, (results, sub_q) in enumerate(zip(sub_results, sub_questions)):
            if isinstance(results, Exception):
                logger.error(f"子问题检索异常: {results}")
                continue
                
            if not isinstance(results, list):
                continue
                
            for result in results:
                # 去重
                content_key = getattr(result, 'content', str(result))[:100]
                if content_key in seen_contents:
                    continue
                seen_contents.add(content_key)
                
                # 根据合并策略调整分数
                if merge_strategy == 'weighted_score':
                    # 根据子问题置信度和优先级调整分数
                    weight = sub_q.confidence * (1.0 / sub_q.priority)
                    if hasattr(result, 'score'):
                        result.score = result.score * weight
                elif merge_strategy == 'relevance_ranking':
                    # 保持原始分数，后续重新排序
                    pass
                
                # 添加分解信息到元数据
                if hasattr(result, 'metadata'):
                    result.metadata.update({
                        'sub_question': sub_q.question,
                        'sub_question_confidence': sub_q.confidence,
                        'reasoning_type': sub_q.reasoning_type,
                        'decomposition_priority': sub_q.priority
                    })
                
                all_results.append(result)
        
        # 排序和限制结果数量
        if merge_strategy == 'weighted_score':
            all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        elif merge_strategy == 'relevance_ranking':
            # 可以在这里实现更复杂的相关性排序
            all_results.sort(key=lambda x: getattr(x, 'score', 0), reverse=True)
        
        # 限制结果数量
        max_results = self.decomposition_config.get('max_merged_results', 100)
        return all_results[:max_results]

    def get_decomposition_stats(self) -> Dict[str, Any]:
        """获取分解统计信息"""
        return {
            'config': self.decomposition_config,
            'schema_loaded': bool(self.schema and self.schema != "{}"),
            'llm_model': self.decomposition_config.get('decomposition_model', 'unknown')
        }


# 工厂函数
def create_query_decomposer(config: Dict[str, Any], schema_path: Optional[str] = None) -> QueryDecomposer:
    """创建查询分解器实例"""
    return QueryDecomposer(config, schema_path)