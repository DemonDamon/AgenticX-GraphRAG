#!/usr/bin/env python3
"""
AgenticX GraphRAG 演示系统快速启动脚本

简化的启动入口，包含环境检查和错误处理
"""

import os
import sys
import asyncio
from pathlib import Path

# 添加 AgenticX 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from main import AgenticXGraphRAGDemo
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保已安装 AgenticX 及其依赖")
    sys.exit(1)


def check_environment():
    """检查运行环境"""
    print("🔍 检查运行环境...")
    
    # 检查必要的环境变量
    required_env_vars = [
        'OPENAI_API_KEY',
        'NEO4J_URI',
        'NEO4J_USERNAME', 
        'NEO4J_PASSWORD'
    ]
    
    missing_vars = []
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ 缺少必要的环境变量: {', '.join(missing_vars)}")
        print("\n请设置以下环境变量:")
        for var in missing_vars:
            print(f"  export {var}=<your_value>")
        return False
    
    # 检查数据目录
    data_dir = Path("./data")
    if not data_dir.exists():
        print(f"❌ 数据目录不存在: {data_dir}")
        print("请创建 data 目录并放入要处理的文档")
        return False
    
    # 检查配置文件
    config_files = ["configs.yml", ".trae/documents/configs.yml"]
    config_exists = any(Path(f).exists() for f in config_files)
    
    if not config_exists:
        print(f"❌ 配置文件不存在: {config_files}")
        return False
    
    print("✅ 环境检查通过")
    return True


async def main():
    """主函数"""
    print("🚀 AgenticX GraphRAG 演示系统")
    print("=" * 50)
    
    # 环境检查
    if not check_environment():
        sys.exit(1)
    
    try:
        # 创建并运行演示
        demo = AgenticXGraphRAGDemo()
        await demo.run_demo()
        
    except KeyboardInterrupt:
        print("\n👋 用户中断，退出系统")
    except Exception as e:
        print(f"❌ 系统运行错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())