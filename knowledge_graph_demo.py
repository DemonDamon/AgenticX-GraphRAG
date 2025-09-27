"""AgenticX Knowledge Graph Construction Demo

This demo showcases the powerful knowledge graph construction capabilities of AgenticX,
including entity extraction, relationship extraction, graph building, and GraphRAG features.

Features demonstrated:
- Basic knowledge graph construction
- Entity and relationship extraction
- Graph quality validation
- Advanced GraphRAG construction with communities
- Incremental graph updates
- Graph optimization and analysis

Usage:
    python examples/knowledge_graph_demo.py
"""

import asyncio
import json
import sys
import yaml
from pathlib import Path
from typing import List
from loguru import logger

# 配置loguru日志
logger.remove()  # 移除默认的日志处理器
logger.add(
    sys.stdout,
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="DEBUG",
    colorize=True
)
logger.add(
    "knowledge_graph_demo.log",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB"
)

from agenticx.knowledge.graphers import (
    # Data models
    Document, DocumentMetadata, Entity, Relationship, KnowledgeGraph,
    EntityType, RelationType,
    # Builders
    KnowledgeGraphBuilder, EntityExtractor, RelationshipExtractor,
    GraphQualityValidator,
    # Advanced constructors
    GraphRAGConstructor, CommunityDetector, GraphOptimizer
)
from agenticx.knowledge.graphers.config import GrapherConfig

# 日志已在文件开头配置


def basic_graph_construction_demo():
    """Basic knowledge graph construction example"""
    print("=== Basic Knowledge Graph Construction Demo ===")
    
    # Sample documents
    documents = [
        Document(
            content="""
            张三是北京大学的计算机科学教授。他专门研究人工智能和机器学习。
            张三与李四是同事，他们一起在AI实验室工作。
            北京大学位于北京市海淀区，是中国顶尖的研究型大学。
            """,
            metadata=DocumentMetadata(
                name="doc1",
                source_type="text",
                content_type="text/plain"
            )
        ),
        Document(
            content="""
            李四是北京大学的数据科学研究员。她的研究重点是深度学习和神经网络。
            李四曾在清华大学获得博士学位。
            清华大学也位于北京市，与北京大学齐名。
            """,
            metadata=DocumentMetadata(
                name="doc2",
                source_type="text",
                content_type="text/plain"
            )
        ),
        Document(
            content="""
            AI实验室是北京大学计算机学院的重要研究机构。
            实验室主要研究方向包括机器学习、深度学习、自然语言处理等。
            张三担任实验室主任，李四是核心研究员。
            """,
            metadata=DocumentMetadata(
                name="doc3",
                source_type="text",
                content_type="text/plain"
            )
        )
    ]
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "agenticx" / "configs" / "knowledge_graphers_config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grapher_config = GrapherConfig.from_dict(config_data)

    # Build knowledge graph
    builder = KnowledgeGraphBuilder(
        config=grapher_config.graphrag,
        llm_config=grapher_config.llm
    )
    texts = [doc.content for doc in documents]
    metadata = [vars(doc.metadata) for doc in documents]
    graph = builder.build_from_texts(texts, metadata)
    graph.name = "University Research Graph"
    
    print(f"Graph Name: {graph.name}")
    print(f"Entities: {len(graph.entities)}")
    print(f"Relationships: {len(graph.relationships)}")
    
    # Show entities
    print("\nExtracted Entities:")
    for entity in graph.entities.values():
        print(f"  - {entity.name} ({entity.entity_type.value}) [confidence: {entity.confidence:.2f}]")
    
    # Show relationships
    print("\nExtracted Relationships:")
    for rel in graph.relationships.values():
        source_name = graph.get_entity(rel.source_entity_id).name
        target_name = graph.get_entity(rel.target_entity_id).name
        rel_type = rel.relation_type.value if hasattr(rel.relation_type, 'value') else rel.relation_type
        print(f"  - {source_name} --[{rel_type}]--> {target_name} [confidence: {rel.confidence:.2f}]")
    
    return graph


def graph_quality_validation_demo(graph: KnowledgeGraph):
    """Graph quality validation example"""
    print("\n=== Graph Quality Validation Demo ===")
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "configs" / "knowledge_graphers_config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grapher_config = GrapherConfig.from_dict(config_data)

    validator = GraphQualityValidator(config=grapher_config.graphrag.quality_validation)
    quality_report = validator.validate(graph)
    
    print(f"Overall Quality Score: {quality_report.overall_score:.3f}")
    print(f"Quality Level: {quality_report.quality_level}")
    
    print("\nQuality Metrics:")
    metrics = quality_report.metrics
    print(f"  Entity Count: {metrics.entity_count}")
    print(f"  Relationship Count: {metrics.relationship_count}")
    print(f"  Entity Coverage: {metrics.entity_coverage:.3f}")
    print(f"  Relationship Diversity: {metrics.relationship_diversity}")
    print(f"  Confidence Score: {metrics.confidence_score:.3f}")
    print(f"  Entities with Attributes: {metrics.entities_with_attributes}")
    print(f"  Entities with Descriptions: {metrics.entities_with_descriptions}")
    
    if quality_report.issues:
        print("\nIdentified Issues:")
        for issue in quality_report.issues:
            print(f"  - {issue}")
    
    if quality_report.recommendations:
        print("\nRecommendations:")
        for rec in quality_report.recommendations:
            print(f"  - {rec}")


def graphrag_construction_demo():
    """Advanced GraphRAG construction example"""
    print("\n=== GraphRAG Construction Demo ===")
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "configs" / "knowledge_graphers_config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grapher_config = GrapherConfig.from_dict(config_data)

    # Extended documents for better community detection
    documents = [
        Document(
            content="""
            人工智能是计算机科学的一个分支，致力于创建能够执行通常需要人类智能的任务的系统。
            机器学习是人工智能的核心技术，包括监督学习、无监督学习和强化学习。
            深度学习是机器学习的一个子领域，使用神经网络来模拟人脑的工作方式。
            """,
            metadata=DocumentMetadata(name="ai_overview", source_type="text")
        ),
        Document(
            content="""
            自然语言处理(NLP)是人工智能的重要应用领域。
            NLP技术包括文本分析、语言理解、机器翻译等。
            BERT、GPT等预训练模型推动了NLP技术的快速发展。
            """,
            metadata=DocumentMetadata(name="nlp_tech", source_type="text")
        ),
        Document(
            content="""
            计算机视觉是另一个重要的AI应用领域。
            卷积神经网络(CNN)是计算机视觉的核心技术。
            图像识别、目标检测、图像分割是计算机视觉的主要任务。
            """,
            metadata=DocumentMetadata(name="cv_tech", source_type="text")
        ),
        Document(
            content="""
            北京大学、清华大学、中科院是中国AI研究的重要机构。
            这些机构在深度学习、自然语言处理、计算机视觉等领域都有重要贡献。
            产学研合作推动了AI技术的产业化应用。
            """,
            metadata=DocumentMetadata(name="ai_institutions", source_type="text")
        )
    ]
    
    # Build GraphRAG
    constructor = GraphRAGConstructor(
        config=grapher_config.graphrag,
        llm_config=grapher_config.llm
    )
    
    texts = [doc.content for doc in documents]
    metadata = [vars(doc.metadata) for doc in documents]
    
    graph = constructor.construct_from_texts(texts, metadata)
    graph.name = "AI Knowledge Graph"
    
    print(f"GraphRAG Name: {graph.name}")
    print(f"Total Nodes: {graph.graph.number_of_nodes()}")
    print(f"Total Edges: {graph.graph.number_of_edges()}")
    
    # Show hierarchical structure
    level_distribution = graph.metadata.get('level_distribution', {})
    print(f"\nHierarchical Structure:")
    for level, count in sorted(level_distribution.items()):
        level_name = {1: "Attributes", 2: "Entities", 3: "Keywords", 4: "Communities"}.get(level, f"Level {level}")
        print(f"  {level_name}: {count} nodes")
    
    # Show communities
    community_nodes = [
        (node_id, data) for node_id, data in graph.graph.nodes(data=True)
        if data.get('level') == 4  # Community level
    ]
    
    if community_nodes:
        print(f"\nDetected Communities:")
        for node_id, data in community_nodes:
            properties = data.get('properties', {})
            name = properties.get('name', 'Unknown')
            member_count = properties.get('member_count', 0)
            description = properties.get('description', 'No description')
            print(f"  - {name}: {member_count} members")
            print(f"    Description: {description}")
    
    return graph


def incremental_update_demo(graph: KnowledgeGraph):
    """Incremental graph update example"""
    print("\n=== Incremental Update Demo ===")
    
    print(f"Original graph: {len(graph.entities)} entities, {len(graph.relationships)} relationships")
    
    # New documents to add
    new_documents = [
        Document(
            content="""
            强化学习是机器学习的第三大类，与监督学习和无监督学习并列。
            强化学习通过智能体与环境的交互来学习最优策略。
            AlphaGo就是强化学习的成功应用案例。
            """,
            metadata=DocumentMetadata(name="rl_intro", source_type="text")
        ),
        Document(
            content="""
            OpenAI是人工智能研究的重要机构，开发了GPT系列模型。
            GPT-4是目前最先进的大语言模型之一。
            大语言模型的发展推动了通用人工智能的研究。
            """,
            metadata=DocumentMetadata(name="openai_gpt", source_type="text")
        )
    ]
    
    # Load config from YAML
    config_path = Path(__file__).parent.parent / "configs" / "knowledge_graphers_config.yml"
    with open(config_path, "r", encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grapher_config = GrapherConfig.from_dict(config_data)

    # Perform incremental update
    constructor = GraphRAGConstructor(
        config=grapher_config.graphrag,
        llm_config=grapher_config.llm
    )
    new_texts = [doc.content for doc in new_documents]
    updated_graph = constructor.construct_incremental(graph, new_texts)
    
    print(f"Updated graph: {len(updated_graph.entities)} entities, {len(updated_graph.relationships)} relationships")
    
    # Show new entities
    print("\nNew entities added:")
    for entity in updated_graph.entities.values():
        if any("rl_intro" in chunk or "openai_gpt" in chunk for chunk in entity.source_chunks):
            print(f"  - {entity.name} ({entity.entity_type.value})")
    
    return updated_graph


async def graph_analysis_demo(graph: KnowledgeGraph):
    """Graph analysis and statistics example"""
    print("\n=== Graph Analysis Demo ===")
    
    # Basic statistics
    stats = graph.get_statistics()
    print("Basic Statistics:")
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Find most connected entities
    entity_degrees = []
    for entity_id in graph.entities.keys():
        if entity_id in graph.graph:
            degree = graph.graph.degree(entity_id)
            entity_name = graph.entities[entity_id].name
            entity_degrees.append((entity_name, degree))
    
    entity_degrees.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nMost Connected Entities:")
    for name, degree in entity_degrees[:5]:
        print(f"  - {name}: {degree} connections")
    
    # Find entities by type
    print(f"\nEntities by Type:")
    for entity_type in EntityType:
        entities_of_type = graph.find_entities_by_type(entity_type)
        if entities_of_type:
            print(f"  {entity_type.value}: {len(entities_of_type)} entities")
            for entity in entities_of_type[:3]:  # Show first 3
                print(f"    - {entity.name}")
            if len(entities_of_type) > 3:
                print(f"    ... and {len(entities_of_type) - 3} more")


def graph_export_demo(graph: KnowledgeGraph):
    """Graph export and serialization example"""
    print("\n=== Graph Export Demo ===")
    
    # Export to JSON
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / "knowledge_graph.json"
    graph.to_json(str(json_file))
    print(f"Graph exported to JSON: {json_file}")
    
    # Show JSON structure (first few lines)
    with open(json_file, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
        print(f"JSON file preview (first 10 lines):")
        for line in lines[:10]:
            print(f"  {line}")
        if len(lines) > 10:
            print(f"  ... and {len(lines) - 10} more lines")
    
    # Test loading from JSON
    loaded_graph = KnowledgeGraph.from_json(str(json_file))
    print(f"Loaded graph: {len(loaded_graph.entities)} entities, {len(loaded_graph.relationships)} relationships")
    
    # Export graph statistics
    stats_file = output_dir / "graph_statistics.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(graph.get_statistics(), f, ensure_ascii=False, indent=2)
    print(f"Graph statistics exported to: {stats_file}")


def main():
    """Run all knowledge graph demos"""
    logger.info("🚀 启动AgenticX知识图谱构建演示")
    print("AgenticX Knowledge Graph Construction Demo")
    print("=" * 60)
    
    try:
        logger.info("📊 开始基础知识图谱构建演示")
        # Basic construction
        basic_graph = basic_graph_construction_demo()
        logger.success(f"✅ 基础知识图谱构建完成，包含 {len(basic_graph.entities)} 个实体，{len(basic_graph.relationships)} 个关系")
        
        logger.info("🔍 开始图谱质量验证演示")
        # Quality validation
        graph_quality_validation_demo(basic_graph)
        logger.success("✅ 图谱质量验证完成")
        
        # Advanced GraphRAG construction
        graphrag_graph = graphrag_construction_demo()
        
        # Incremental updates
        updated_graph = incremental_update_demo(graphrag_graph)
        
        # Graph analysis
        graph_analysis_demo(updated_graph)
        
        # Export and serialization
        graph_export_demo(updated_graph)
        
        print("\n" + "=" * 60)
        print("All knowledge graph demos completed successfully!")
        
        # Final summary
        print(f"\nFinal Graph Summary:")
        print(f"  Name: {updated_graph.name}")
        print(f"  Entities: {len(updated_graph.entities)}")
        print(f"  Relationships: {len(updated_graph.relationships)}")
        print(f"  Total Nodes: {updated_graph.graph.number_of_nodes()}")
        print(f"  Total Edges: {updated_graph.graph.number_of_edges()}")
        print(f"  Created: {updated_graph.created_at}")
        print(f"  Last Updated: {updated_graph.updated_at}")
        
    except Exception as e:
        logger.error(f"Demo execution failed: {e}")
        raise


if __name__ == "__main__":
    main()