#!/usr/bin/env python3
"""
智能查询预处理器
解决中文分词、查询扩展和实体识别问题
"""

import re
import jieba
from typing import List, Dict, Any, Set, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class ProcessedQuery:
    """处理后的查询结果"""
    original: str
    normalized: str
    keywords: List[str]
    entities: List[str]
    expanded_terms: List[str]
    query_type: str
    confidence: float


class ChineseQueryProcessor:
    """中文查询预处理器"""
    
    def __init__(self):
        """初始化处理器"""
        # 初始化jieba分词
        jieba.initialize()
        
        # 常见的查询模式
        self.question_patterns = {
            r'(.+?)是什么': 'definition',
            r'(.+?)是啥': 'definition', 
            r'什么是(.+?)': 'definition',
            r'(.+?)怎么样': 'evaluation',
            r'(.+?)如何': 'method',
            r'(.+?)的作用': 'function',
            r'(.+?)的特点': 'feature',
        }
        
        # 同义词词典
        self.synonyms = {
            '是啥': ['是什么', '是', '定义', '含义'],
            '怎么样': ['如何', '怎样', '效果'],
            '作用': ['功能', '用途', '目的'],
            '特点': ['特征', '性质', '属性'],
        }
        
        # 实体类型词典
        self.entity_types = {
            '公司': ['公司', '企业', '集团', '有限公司', '股份有限公司'],
            '技术': ['技术', '算法', '方法', '框架', '系统'],
            '产品': ['产品', '服务', '平台', '工具'],
            '概念': ['概念', '理论', '模型', '原理'],
        }
        
        # 停用词
        self.stop_words = {
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', 
            '都', '一', '一个', '上', '也', '很', '到', '说', '要', '去',
            '你', '会', '着', '没有', '看', '好', '自己', '这'
        }

    def process_query(self, query: str) -> ProcessedQuery:
        """处理查询"""
        try:
            # 1. 标准化查询
            normalized = self._normalize_query(query)
            
            # 2. 识别查询类型
            query_type, confidence = self._identify_query_type(normalized)
            
            # 3. 提取关键词
            keywords = self._extract_keywords(normalized)
            
            # 4. 实体识别
            entities = self._extract_entities(normalized)
            
            # 5. 查询扩展
            expanded_terms = self._expand_query(keywords, entities)
            
            return ProcessedQuery(
                original=query,
                normalized=normalized,
                keywords=keywords,
                entities=entities,
                expanded_terms=expanded_terms,
                query_type=query_type,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"查询处理失败: {e}")
            # 返回基础处理结果
            return ProcessedQuery(
                original=query,
                normalized=query.strip(),
                keywords=[query.strip()],
                entities=[],
                expanded_terms=[query.strip()],
                query_type='unknown',
                confidence=0.5
            )

    def _normalize_query(self, query: str) -> str:
        """标准化查询"""
        # 去除多余空格
        normalized = re.sub(r'\s+', ' ', query.strip())
        
        # 统一标点符号
        normalized = normalized.replace('？', '?').replace('！', '!')
        
        # 处理常见的口语化表达
        replacements = {
            '是啥': '是什么',
            '咋样': '怎么样',
            '咋办': '怎么办',
            '啥意思': '什么意思',
        }
        
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        
        return normalized

    def _identify_query_type(self, query: str) -> tuple[str, float]:
        """识别查询类型"""
        for pattern, query_type in self.question_patterns.items():
            if re.search(pattern, query):
                return query_type, 0.9
        
        # 基于关键词判断
        if any(word in query for word in ['什么', '是', '定义']):
            return 'definition', 0.7
        elif any(word in query for word in ['如何', '怎么', '方法']):
            return 'method', 0.7
        elif any(word in query for word in ['为什么', '原因']):
            return 'reason', 0.7
        else:
            return 'general', 0.5

    def _extract_keywords(self, query: str) -> List[str]:
        """提取关键词"""
        # 使用jieba分词
        words = list(jieba.cut(query))
        
        # 过滤停用词和短词
        keywords = []
        for word in words:
            word = word.strip()
            if (len(word) > 1 and 
                word not in self.stop_words and 
                not re.match(r'^[？！。，、；：""''（）【】\s]+$', word)):
                keywords.append(word)
        
        return keywords

    def _extract_entities(self, query: str) -> List[str]:
        """提取实体"""
        entities = []
        
        # 基于模式匹配提取实体
        patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 英文实体
            r'([\u4e00-\u9fff]{2,}(?:公司|企业|集团|技术|系统|平台))',  # 中文机构/技术实体
            r'([\u4e00-\u9fff]{2,})',  # 一般中文实体
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, query)
            entities.extend(matches)
        
        # 去重并过滤
        unique_entities = []
        for entity in entities:
            if entity not in unique_entities and len(entity) > 1:
                unique_entities.append(entity)
        
        return unique_entities

    def _expand_query(self, keywords: List[str], entities: List[str]) -> List[str]:
        """扩展查询词汇"""
        expanded = set(keywords + entities)
        
        # 添加同义词
        for keyword in keywords:
            if keyword in self.synonyms:
                expanded.update(self.synonyms[keyword])
        
        # 添加相关词汇
        for entity in entities:
            # 如果是公司名，添加相关词汇
            if any(suffix in entity for suffix in ['公司', '企业', '集团']):
                expanded.update(['业务', '服务', '产品'])
            # 如果是技术名，添加相关词汇
            elif any(suffix in entity for suffix in ['技术', '系统', '平台']):
                expanded.update(['应用', '功能', '特点'])
        
        return list(expanded)

    def generate_search_queries(self, processed_query: ProcessedQuery) -> List[str]:
        """生成多个搜索查询"""
        queries = []
        
        # 1. 原始查询
        queries.append(processed_query.original)
        
        # 2. 标准化查询
        if processed_query.normalized != processed_query.original:
            queries.append(processed_query.normalized)
        
        # 3. 关键词组合
        if len(processed_query.keywords) > 1:
            queries.append(' '.join(processed_query.keywords))
        
        # 4. 实体查询
        for entity in processed_query.entities:
            if len(entity) > 2:  # 过滤太短的实体
                queries.append(entity)
        
        # 5. 扩展词查询
        if processed_query.expanded_terms:
            # 选择最重要的扩展词
            important_terms = [term for term in processed_query.expanded_terms 
                             if len(term) > 2 and term not in processed_query.keywords]
            if important_terms:
                queries.append(' '.join(important_terms[:3]))  # 最多3个扩展词
        
        # 去重
        unique_queries = []
        for query in queries:
            if query not in unique_queries:
                unique_queries.append(query)
        
        return unique_queries

    def should_use_fuzzy_search(self, processed_query: ProcessedQuery) -> bool:
        """判断是否应该使用模糊搜索"""
        # 如果查询很短或者置信度很低，建议使用模糊搜索
        return (len(processed_query.original) < 5 or 
                processed_query.confidence < 0.6 or
                len(processed_query.keywords) < 2)


# 使用示例
if __name__ == "__main__":
    processor = ChineseQueryProcessor()
    
    # 测试查询
    test_queries = [
        "铁塔是啥",
        "中国铁塔公司",
        "nihao",
        "什么是人工智能",
        "AgenticX框架怎么样"
    ]
    
    for query in test_queries:
        print(f"\n原始查询: {query}")
        result = processor.process_query(query)
        print(f"标准化: {result.normalized}")
        print(f"关键词: {result.keywords}")
        print(f"实体: {result.entities}")
        print(f"扩展词: {result.expanded_terms}")
        print(f"查询类型: {result.query_type} (置信度: {result.confidence})")
        
        search_queries = processor.generate_search_queries(result)
        print(f"搜索查询: {search_queries}")
        print(f"建议模糊搜索: {processor.should_use_fuzzy_search(result)}")