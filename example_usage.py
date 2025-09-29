#!/usr/bin/env python3
"""
多跳数据集构建器使用示例

展示如何使用 multihop_dataset_builder.py 构建不同领域的多跳问答对数据集
"""

import asyncio
from pathlib import Path
from multihop_dataset_builder import MultihopDatasetBuilder, create_domain_config


async def example_technology_domain():
    """技术领域示例"""
    print("🔧 技术领域多跳数据集构建示例")
    
    builder = MultihopDatasetBuilder("configs.yml")
    
    # 技术领域配置
    domain_config = create_domain_config('technology')
    
    # 自定义变量
    custom_variables = {
        'sample_nums': 10,  # 生成10个问题
        'min_documents_per_question': 2,
        'reasoning_pattern': '基于{context}技术实现，分析{target}算法的{aspect}如何影响{outcome}性能表现？'
    }
    
    await builder.build_dataset(
        data_path="./documents/tech",  # 技术文档目录
        output_path="./output/tech_multihop_dataset.json",
        llm_provider="bailian",
        llm_model="qwen3-max",
        domain_config=domain_config,
        custom_variables=custom_variables
    )


async def example_finance_domain():
    """金融领域示例"""
    print("💰 金融领域多跳数据集构建示例")
    
    builder = MultihopDatasetBuilder("configs.yml")
    
    # 金融领域配置
    domain_config = create_domain_config('finance')
    
    # 自定义变量
    custom_variables = {
        'sample_nums': 15,
        'target_language': '中文',
        'reasoning_pattern': '基于{context}交易策略，分析{target}风控机制的{aspect}如何影响{outcome}收益表现？'
    }
    
    await builder.build_dataset(
        data_path="./documents/finance",  # 金融文档目录
        output_path="./output/finance_multihop_dataset.json",
        domain_config=domain_config,
        custom_variables=custom_variables
    )


async def example_custom_domain():
    """自定义领域示例"""
    print("🎯 自定义领域多跳数据集构建示例")
    
    builder = MultihopDatasetBuilder("configs.yml")
    
    # 完全自定义配置
    custom_domain_config = {
        'domain_guidance': '''
请根据文档内容自动识别教育相关主题（如教学方法、学习理论、评估体系等），然后选择两个不同教育主题进行组合，生成跨文档多跳问题。
注意识别文档中的教育理念、教学策略、学习效果评估，并寻找它们之间的教育学关联。
        '''.strip(),
        'domain_specific_terms': '教学方法、学习理论、评估指标、教育工具',
        'comparison_aspect': '教学效果/学习体验/评估准确性/适用范围',
        'mechanism_name': '教学机制/学习策略/评估方法',
        'target_aspect': '学习效果/教学效率/评估准确性/学生满意度',
        'challenge': '个体差异/资源限制/评估难度'
    }
    
    custom_variables = {
        'sample_nums': 8,
        'max_answer_sentences': 2,  # 更简洁的答案
        'reasoning_pattern': '基于{context}教学实践，分析{target}学习方法的{aspect}如何影响{outcome}教学效果？'
    }
    
    await builder.build_dataset(
        data_path="./documents/education",
        output_path="./output/education_multihop_dataset.json",
        domain_config=custom_domain_config,
        custom_variables=custom_variables
    )


async def example_single_file():
    """单文件处理示例"""
    print("📄 单文件多跳数据集构建示例")
    
    builder = MultihopDatasetBuilder("configs.yml")
    
    await builder.build_dataset(
        data_path="./documents/single_paper.pdf",  # 单个PDF文件
        output_path="./output/single_file_dataset.json",
        llm_provider="bailian",
        llm_model="qwen3-max",
        domain_config=create_domain_config('general')
    )


async def main():
    """运行所有示例"""
    print("🚀 多跳数据集构建器使用示例")
    print("=" * 50)
    
    # 确保输出目录存在
    Path("./output").mkdir(exist_ok=True)
    Path("./logs").mkdir(exist_ok=True)
    
    try:
        # 运行不同领域的示例
        await example_technology_domain()
        print()
        
        await example_finance_domain()
        print()
        
        await example_custom_domain()
        print()
        
        await example_single_file()
        print()
        
        print("✅ 所有示例执行完成！")
        
    except Exception as e:
        print(f"❌ 示例执行失败: {e}")


if __name__ == "__main__":
    asyncio.run(main())