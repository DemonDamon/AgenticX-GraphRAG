#!/usr/bin/env python3
"""
增强检索器
实现多级回退策略、动态阈值调整和智能查询处理
"""

import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from query_processor import ChineseQueryProcessor, ProcessedQuery

# 🔧 修复：使用AgenticX框架的标准RetrievalResult
try:
    from agenticx.retrieval.base import RetrievalResult
    logger.info("使用AgenticX标准RetrievalResult")
except ImportError:
    # 回退到自定义定义
    @dataclass
    class RetrievalResult:
        """检索结果 - 回退定义"""
        content: str
        score: float
        metadata: Dict[str, Any]
        source: Optional[str] = None
        chunk_id: Optional[str] = None
        bm25_score: Optional[float] = None
    logger.warning("使用自定义RetrievalResult定义")


@dataclass
class RetrievalStrategy:
    """检索策略配置"""
    name: str
    vector_threshold: float
    graph_threshold: float
    bm25_min_score: float
    top_k: int
    description: str


class EnhancedRetriever:
    """增强检索器 - 支持多级回退和智能查询处理"""
    
    def __init__(self, base_retriever, graph_retriever=None, storage_manager=None):
        """初始化增强检索器"""
        self.base_retriever = base_retriever
        self.graph_retriever = graph_retriever
        self.storage_manager = storage_manager
        self.query_processor = ChineseQueryProcessor()
        
        # 定义多级检索策略 - 🚀 调整为用户期望的阈值：文档>0.2，图谱>0.1
        self.strategies = [
            RetrievalStrategy(
                name="strict",
                vector_threshold=0.5,   # 🔧 严格模式保持较高阈值
                graph_threshold=0.4,    # 🔧 严格图检索阈值
                bm25_min_score=0.25,    # 🔧 严格BM25阈值
                top_k=30,               # 🔧 适中返回数量
                description="严格模式 - 高质量结果"
            ),
            RetrievalStrategy(
                name="standard",
                vector_threshold=0.3,   # 🔧 标准模式适中阈值
                graph_threshold=0.2,    # 🔧 标准图检索阈值
                bm25_min_score=0.15,    # 🔧 标准BM25阈值
                top_k=60,               # 🔧 增加返回数量
                description="标准模式 - 平衡质量和召回"
            ),
            RetrievalStrategy(
                name="relaxed",
                vector_threshold=0.2,   # 🔧 用户期望：文档相似度>0.2
                graph_threshold=0.1,    # 🔧 用户期望：知识图谱>0.1
                bm25_min_score=0.08,    # 🔧 适中的BM25阈值
                top_k=100,              # 🔧 增加返回数量
                description="宽松模式 - 高召回率"
            ),
            RetrievalStrategy(
                name="fuzzy",
                vector_threshold=0.15,  # 🔧 模糊模式稍低阈值
                graph_threshold=0.08,   # 🔧 模糊图检索阈值
                bm25_min_score=0.04,    # 🔧 模糊BM25分数
                top_k=150,              # 🔧 较多返回数量
                description="模糊模式 - 最大召回"
            ),
            RetrievalStrategy(
                name="aggressive",
                vector_threshold=0.1,   # 🔧 激进模式低阈值，兜底策略
                graph_threshold=0.05,   # 🔧 激进图阈值
                bm25_min_score=0.02,    # 🔧 激进BM25阈值
                top_k=200,              # 🔧 最多返回数量
                description="激进模式 - 兜底策略"
            )
        ]
        
        # 统计信息
        self.stats = {
            'total_queries': 0,
            'successful_queries': 0,
            'strategy_usage': {s.name: 0 for s in self.strategies},
            'avg_results_per_query': 0.0
        }

    async def retrieve_with_fallback(self, query: str, **kwargs) -> Tuple[List[RetrievalResult], Dict[str, Any]]:
        """多级回退检索"""
        self.stats['total_queries'] += 1
        
        try:
            # 1. 智能查询预处理
            processed_query = self.query_processor.process_query(query)
            logger.info(f"查询预处理完成: {processed_query.query_type}, 置信度: {processed_query.confidence}")
            
            # 🔧 修复：特殊查询类型的直接处理
            if processed_query.query_type == 'greeting':
                logger.info("检测到问候语，返回友好回复")
                greeting_result = RetrievalResult(
                    content="你好！我是AgenticX GraphRAG智能问答助手。我可以帮您查询知识库中的信息，请告诉我您想了解什么？",
                    score=1.0,
                    metadata={'type': 'greeting_response', 'search_source': 'system'}
                )
                return [greeting_result], {
                    'processed_query': processed_query,
                    'search_queries': [query],
                    'strategy_used': 'greeting_handler',
                    'total_results': 1,
                    'success': True
                }
            
            if processed_query.query_type == 'meaningless':
                logger.info("检测到无意义查询，返回提示")
                help_result = RetrievalResult(
                    content="请输入具体的问题，我可以帮您查询相关信息。例如：\n• 查询特定概念或实体\n• 询问技术原理\n• 了解产品服务等",
                    score=1.0,
                    metadata={'type': 'help_response', 'search_source': 'system'}
                )
                return [help_result], {
                    'processed_query': processed_query,
                    'search_queries': [query],
                    'strategy_used': 'help_handler',
                    'total_results': 1,
                    'success': True
                }
            
            # 2. 生成多个搜索查询
            search_queries = self.query_processor.generate_search_queries(processed_query)
            logger.info(f"生成搜索查询: {search_queries}")
            
            # 3. 根据查询特征选择起始策略
            start_strategy_idx = self._select_start_strategy(processed_query)
            
            # 4. 多级回退检索
            all_results = []
            strategy_used = None
            
            for i in range(start_strategy_idx, len(self.strategies)):
                strategy = self.strategies[i]
                logger.info(f"尝试策略: {strategy.name} - {strategy.description}")
                
                # 对每个搜索查询执行检索
                strategy_results = []
                for idx, search_query in enumerate(search_queries):
                    results = await self._execute_strategy(search_query, strategy)
                    logger.debug(f"查询{idx+1}[{search_query[:30]}...] 返回 {len(results)} 条结果")
                    strategy_results.extend(results)
                
                logger.info(f"策略 {strategy.name} 总计获得 {len(strategy_results)} 条原始结果")
                
                # 去重和排序
                unique_results = self._deduplicate_results(strategy_results)
                logger.info(f"去重后剩余 {len(unique_results)} 条结果")
                
                if unique_results:
                    all_results = unique_results
                    strategy_used = strategy
                    self.stats['strategy_usage'][strategy.name] += 1
                    logger.info(f"策略 {strategy.name} 成功，返回 {len(unique_results)} 条结果")
                    break
                else:
                    logger.info(f"策略 {strategy.name} 无结果，尝试下一级")
            
            # 5. 如果所有策略都失败，尝试实体搜索
            if not all_results:
                logger.warning("所有检索策略失败，尝试直接实体搜索")
                entity_results = await self._direct_entity_search(processed_query)
                if entity_results:
                    all_results = entity_results
                    strategy_used = RetrievalStrategy("entity_search", 0, 0, 0, 10, "直接实体搜索")
            
            # 6. 更新统计信息
            if all_results:
                self.stats['successful_queries'] += 1
                self.stats['avg_results_per_query'] = (
                    (self.stats['avg_results_per_query'] * (self.stats['total_queries'] - 1) + len(all_results)) 
                    / self.stats['total_queries']
                )
            
            # 7. 生成检索报告
            retrieval_report = {
                'original_query': query,
                'processed_query': processed_query,
                'search_queries': search_queries,
                'strategy_used': strategy_used.name if strategy_used else 'none',
                'total_results': len(all_results),
                'success': len(all_results) > 0
            }
            
            return all_results, retrieval_report
            
        except Exception as e:
            logger.error(f"增强检索失败: {e}")
            return [], {'error': str(e), 'success': False}

    def _select_start_strategy(self, processed_query: ProcessedQuery) -> int:
        """根据查询特征选择起始策略"""
        # 🔧 优化：对特定查询类型使用更宽松的策略
        complex_query_types = ['specific_inquiry', 'commitment_inquiry', 'enumeration', 'classification', 'service_inquiry']
        
        # 复杂查询类型直接从宽松模式开始，提高召回率
        if processed_query.query_type in complex_query_types:
            return 2  # relaxed
        
        # 长查询（通常包含更多上下文）从宽松模式开始
        if len(processed_query.original) > 20:
            return 2  # relaxed
        
        # 包含多个关键词的查询从标准模式开始
        if len(processed_query.keywords) >= 3:
            return 1  # standard
        
        # 高置信度且有明确实体的简单查询从严格模式开始
        if processed_query.confidence > 0.8 and len(processed_query.entities) > 0 and len(processed_query.original) < 15:
            return 0  # strict
        
        # 中等置信度从标准模式开始
        elif processed_query.confidence > 0.6:
            return 1  # standard
        
        # 其他情况从宽松模式开始，确保更好的召回率
        else:
            return 2  # relaxed

    async def _execute_strategy(self, query: str, strategy: RetrievalStrategy) -> List[RetrievalResult]:
        """执行特定策略的检索"""
        try:
            results = []
            
            # 1. 混合检索（向量 + BM25）
            if hasattr(self.base_retriever, 'retrieve'):
                hybrid_results = await self.base_retriever.retrieve(
                    query, 
                    top_k=strategy.top_k,
                    min_score=strategy.vector_threshold
                )
                # 🔧 修复：对向量检索结果应用向量阈值过滤
                filtered_hybrid = [
                    r for r in hybrid_results 
                    if r.score >= strategy.vector_threshold
                ]
                # 标记结果来源
                for result in filtered_hybrid:
                    if not hasattr(result, 'metadata') or result.metadata is None:
                        result.metadata = {}
                    result.metadata['source_type'] = 'vector'
                results.extend(filtered_hybrid)
            
            # 2. 图检索
            if self.graph_retriever:
                try:
                    graph_results = await self.graph_retriever.retrieve(
                        query,
                        top_k=strategy.top_k // 2,
                        min_score=strategy.graph_threshold
                    )
                    # 🔧 修复：对图检索结果应用图阈值过滤
                    filtered_graph = [
                        r for r in graph_results 
                        if r.score >= strategy.graph_threshold
                    ]
                    # 标记结果来源
                    for result in filtered_graph:
                        if not hasattr(result, 'metadata') or result.metadata is None:
                            result.metadata = {}
                        result.metadata['source_type'] = 'graph'
                    results.extend(filtered_graph)
                except Exception as e:
                    logger.warning(f"图检索失败: {e}")
            
            # 3. 🔧 修复：根据结果来源应用不同的最终过滤标准
            final_filtered = []
            for r in results:
                source_type = r.metadata.get('source_type', 'unknown') if hasattr(r, 'metadata') and r.metadata else 'unknown'
                
                if source_type == 'graph':
                    # 图检索结果：使用图阈值
                    if r.score >= strategy.graph_threshold:
                        final_filtered.append(r)
                elif source_type == 'vector':
                    # 向量检索结果：使用向量阈值
                    if r.score >= strategy.vector_threshold:
                        final_filtered.append(r)
                else:
                    # 未知来源：使用BM25阈值作为兜底
                    if r.score >= strategy.bm25_min_score:
                        final_filtered.append(r)
            
            return final_filtered
            
        except Exception as e:
            logger.error(f"策略执行失败: {e}")
            return []

    async def _direct_entity_search(self, processed_query: ProcessedQuery) -> List[RetrievalResult]:
        """直接实体搜索 - 🚀 增强版本，支持多种搜索策略"""
        try:
            if not self.storage_manager:
                return []
            
            from agenticx.storage import StorageType
            graph_storage = self.storage_manager.get_storage(StorageType.NEO4J)
            
            if not graph_storage:
                return []
            
            results = []
            
            # 🚀 策略1: 搜索实体
            search_terms = processed_query.entities + processed_query.keywords + [processed_query.original]
            
            for term in search_terms:
                if len(term) < 1:  # 降低最小长度要求
                    continue
                
                # 🚀 增强的模糊匹配实体查询
                cypher_queries = [
                    # 精确匹配
                    """
                    MATCH (n:Entity) 
                    WHERE toLower(n.name) = toLower($term)
                    RETURN n.name as name, n.description as description, 
                           labels(n) as type, n.id as id, 1.0 as relevance
                    LIMIT 5
                    """,
                    # 包含匹配
                    """
                    MATCH (n:Entity) 
                    WHERE toLower(n.name) CONTAINS toLower($term)
                       OR toLower($term) CONTAINS toLower(n.name)
                    RETURN n.name as name, n.description as description, 
                           labels(n) as type, n.id as id, 0.8 as relevance
                    LIMIT 10
                    """,
                    # 正则匹配
                    """
                    MATCH (n:Entity) 
                    WHERE n.name =~ ('(?i).*' + $term + '.*')
                       OR n.description =~ ('(?i).*' + $term + '.*')
                    RETURN n.name as name, n.description as description, 
                           labels(n) as type, n.id as id, 0.6 as relevance
                    LIMIT 15
                    """,
                    # 🚀 新增：搜索所有节点（包括Chunk等）
                    """
                    MATCH (n) 
                    WHERE (n.name IS NOT NULL AND toLower(n.name) CONTAINS toLower($term))
                       OR (n.content IS NOT NULL AND toLower(n.content) CONTAINS toLower($term))
                       OR (n.description IS NOT NULL AND toLower(n.description) CONTAINS toLower($term))
                    RETURN n.name as name, 
                           COALESCE(n.description, n.content, '') as description, 
                           labels(n) as type, n.id as id, 0.4 as relevance
                    LIMIT 20
                    """
                ]
                
                for cypher_query in cypher_queries:
                    try:
                        entity_results = graph_storage.execute_query(
                            cypher_query, 
                            {"term": term}
                        )
                        
                        for result in entity_results:
                            if result['name']:  # 确保有名称
                                # 转换为RetrievalResult格式
                                content = f"实体: {result['name']}"
                                if result.get('description'):
                                    content += f"\n描述: {result['description'][:500]}..."  # 限制描述长度
                                
                                retrieval_result = RetrievalResult(
                                    content=content,
                                    score=result.get('relevance', 0.5),
                                    metadata={
                                        'type': 'entity',
                                        'entity_name': result['name'],
                                        'entity_type': result.get('type', []),
                                        'entity_id': result.get('id'),
                                        'search_source': 'direct_entity',
                                        'search_term': term
                                    },
                                    source='knowledge_graph'
                                )
                                results.append(retrieval_result)
                    
                    except Exception as e:
                        logger.warning(f"实体查询失败: {e}")
                        continue
            
            # 🚀 策略2: 如果实体搜索无结果，尝试全文搜索
            if not results:
                logger.info("实体搜索无结果，尝试全文搜索")
                results.extend(await self._full_text_search(processed_query, graph_storage))
            
            # 🚀 策略3: 去重和排序
            unique_results = self._deduplicate_results(results)
            
            # 🚀 策略4: 如果还是无结果，返回一些通用信息
            if not unique_results:
                logger.info("所有搜索策略无结果，返回系统信息")
                unique_results = await self._get_fallback_results(processed_query)
            
            return unique_results[:10]  # 限制返回数量
            
        except Exception as e:
            logger.error(f"直接实体搜索失败: {e}")
            return []
    
    async def _full_text_search(self, processed_query: ProcessedQuery, graph_storage) -> List[RetrievalResult]:
        """全文搜索"""
        results = []
        
        try:
            # 搜索所有包含关键词的节点
            search_terms = [processed_query.original] + processed_query.keywords
            
            for term in search_terms:
                if len(term) < 2:
                    continue
                
                # 全文搜索查询
                cypher_query = """
                MATCH (n) 
                WHERE ANY(prop IN keys(n) WHERE toString(n[prop]) =~ ('(?i).*' + $term + '.*'))
                RETURN n, labels(n) as node_type, 
                       [prop IN keys(n) WHERE toString(n[prop]) =~ ('(?i).*' + $term + '.*')] as matched_props
                LIMIT 20
                """
                
                search_results = graph_storage.execute_query(cypher_query, {"term": term})
                
                for result in search_results:
                    node = result['n']
                    node_type = result['node_type']
                    matched_props = result['matched_props']
                    
                    # 构建内容
                    content_parts = []
                    if hasattr(node, 'name') and node.name:
                        content_parts.append(f"名称: {node.name}")
                    if hasattr(node, 'content') and node.content:
                        content_parts.append(f"内容: {node.content[:300]}...")
                    if hasattr(node, 'description') and node.description:
                        content_parts.append(f"描述: {node.description[:200]}...")
                    
                    if content_parts:
                        retrieval_result = RetrievalResult(
                            content="\n".join(content_parts),
                            score=0.3,
                            metadata={
                                'type': 'full_text_match',
                                'node_type': node_type,
                                'matched_properties': matched_props,
                                'search_source': 'full_text',
                                'search_term': term
                            },
                            source='knowledge_graph'
                        )
                        results.append(retrieval_result)
        
        except Exception as e:
            logger.warning(f"全文搜索失败: {e}")
        
        return results
    
    async def _get_fallback_results(self, processed_query: ProcessedQuery) -> List[RetrievalResult]:
        """获取回退结果"""
        fallback_results = []
        
        # 提供一些通用的帮助信息
        fallback_content = f"""
很抱歉，没有找到与 "{processed_query.original}" 相关的具体信息。

💡 建议您尝试：
1. 使用更具体的关键词
2. 检查拼写是否正确
3. 尝试相关的同义词
4. 使用更简单的表达方式

🔍 系统分析：
- 查询类型: {processed_query.query_type}
- 识别的关键词: {', '.join(processed_query.keywords) if processed_query.keywords else '无'}
- 识别的实体: {', '.join(processed_query.entities) if processed_query.entities else '无'}
- 置信度: {processed_query.confidence:.2f}
"""
        
        fallback_result = RetrievalResult(
            content=fallback_content.strip(),
            score=0.1,
            metadata={
                'type': 'fallback_help',
                'search_source': 'system_fallback',
                'original_query': processed_query.original
            },
            source='system'
        )
        
        fallback_results.append(fallback_result)
        
        return fallback_results

    def _deduplicate_results(self, results: List[RetrievalResult]) -> List[RetrievalResult]:
        """去重和排序结果 - 🚀 优化版本：减少过度去重"""
        if not results:
            return []
        
        logger.debug(f"开始去重，原始结果数: {len(results)}")
        
        # 🔧 优化：更宽松的去重策略
        seen_chunk_ids = set()
        unique_results = []
        duplicate_count = 0
        
        for result in results:
            is_duplicate = False
            
            # 策略1: 基于chunk_id去重（最严格，完全相同的块）
            if hasattr(result, 'chunk_id') and result.chunk_id:
                if result.chunk_id in seen_chunk_ids:
                    duplicate_count += 1
                    is_duplicate = True
                else:
                    seen_chunk_ids.add(result.chunk_id)
            
            # 策略2: 只对非常相似的内容进行去重（提高阈值到0.95）
            if not is_duplicate:
                for existing in unique_results[-3:]:  # 🔧 减少检查范围，只检查最近3个
                    if self._is_content_similar(result.content, existing.content, threshold=0.95):
                        duplicate_count += 1
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_results.append(result)
        
        logger.debug(f"去重完成: 保留 {len(unique_results)} 条，去除 {duplicate_count} 条重复")
        
        # 按分数排序
        unique_results.sort(key=lambda x: x.score, reverse=True)
        
        return unique_results
    
    def _is_content_similar(self, content1: str, content2: str, threshold: float = 0.95) -> bool:
        """检查两个内容是否相似 - 🚀 优化：更严格的相似度判断"""
        # 🔧 改进：使用更精确的相似度计算
        
        # 如果内容长度差异很大，不太可能是重复
        len1, len2 = len(content1), len(content2)
        if abs(len1 - len2) / max(len1, len2) > 0.3:  # 长度差异超过30%
            return False
        
        # 计算词汇重叠度（Jaccard相似度）
        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())
        
        if not words1 or not words2:
            return False
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # 🔧 额外检查：如果内容开头和结尾都很相似，可能是重复
        start_similar = content1[:100].lower() == content2[:100].lower()
        end_similar = content1[-100:].lower() == content2[-100:].lower()
        
        # 只有在Jaccard相似度很高，或者开头结尾都相同时才认为是重复
        return jaccard_similarity > threshold or (start_similar and end_similar and jaccard_similarity > 0.8)

    def get_stats(self) -> Dict[str, Any]:
        """获取检索统计信息"""
        success_rate = (
            self.stats['successful_queries'] / self.stats['total_queries'] 
            if self.stats['total_queries'] > 0 else 0
        )
        
        return {
            'total_queries': self.stats['total_queries'],
            'successful_queries': self.stats['successful_queries'],
            'success_rate': success_rate,
            'avg_results_per_query': self.stats['avg_results_per_query'],
            'strategy_usage': self.stats['strategy_usage'],
            'most_used_strategy': max(
                self.stats['strategy_usage'].items(), 
                key=lambda x: x[1]
            )[0] if any(self.stats['strategy_usage'].values()) else 'none'
        }

    async def suggest_related_queries(self, query: str, max_suggestions: int = 5) -> List[str]:
        """基于查询失败情况提供相关查询建议"""
        try:
            processed_query = self.query_processor.process_query(query)
            suggestions = []
            
            # 1. 基于实体的建议
            for entity in processed_query.entities:
                suggestions.append(f"{entity}是什么")
                suggestions.append(f"{entity}的作用")
                suggestions.append(f"{entity}的特点")
            
            # 2. 基于关键词的建议
            for keyword in processed_query.keywords:
                if len(keyword) > 2:
                    suggestions.append(f"{keyword}相关信息")
                    suggestions.append(f"关于{keyword}")
            
            # 3. 基于查询类型的建议
            if processed_query.query_type == 'definition':
                suggestions.extend([
                    "相关概念介绍",
                    "基本概念说明",
                    "术语解释"
                ])
            
            # 去重并限制数量
            unique_suggestions = []
            for suggestion in suggestions:
                if suggestion not in unique_suggestions and suggestion != query:
                    unique_suggestions.append(suggestion)
                if len(unique_suggestions) >= max_suggestions:
                    break
            
            return unique_suggestions
            
        except Exception as e:
            logger.error(f"生成查询建议失败: {e}")
            return ["请尝试更具体的查询", "使用关键词搜索", "检查拼写是否正确"]


# 使用示例
async def test_enhanced_retriever():
    """测试增强检索器"""
    # 这里需要实际的检索器实例
    # enhanced_retriever = EnhancedRetriever(base_retriever, graph_retriever, storage_manager)
    
    test_queries = [
        "铁塔是啥",
        "中国铁塔公司",
        "nihao",
        "人工智能技术"
    ]
    
    print("增强检索器测试")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\n测试查询: {query}")
        # results, report = await enhanced_retriever.retrieve_with_fallback(query)
        # print(f"结果数量: {len(results)}")
        # print(f"使用策略: {report.get('strategy_used', 'unknown')}")
        # print(f"成功: {report.get('success', False)}")


if __name__ == "__main__":
    asyncio.run(test_enhanced_retriever())