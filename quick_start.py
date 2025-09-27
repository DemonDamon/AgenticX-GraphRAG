#!/usr/bin/env python3
"""
AgenticX GraphRAG 演示系统快速启动脚本

这个脚本提供了一个简化的启动流程，帮助用户快速体验 GraphRAG 功能
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

# 添加 AgenticX 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class QuickStartDemo:
    """快速启动演示"""
    
    def __init__(self):
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self):
        """设置日志"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
    
    def check_environment(self) -> bool:
        """检查环境"""
        print("🔍 检查环境配置...")
        
        # 检查必要的环境变量
        required_vars = {
            'OPENAI_API_KEY': 'OpenAI API 密钥',
            'NEO4J_URI': 'Neo4j 连接地址',
            'NEO4J_USERNAME': 'Neo4j 用户名',
            'NEO4J_PASSWORD': 'Neo4j 密码'
        }
        
        missing_vars = []
        for var_name, description in required_vars.items():
            if not os.getenv(var_name):
                missing_vars.append(f"  • {var_name}: {description}")
        
        if missing_vars:
            print("❌ 缺少必要的环境变量:")
            for var in missing_vars:
                print(var)
            print("\n请设置这些环境变量后重试。")
            return False
        
        print("✅ 环境变量检查通过")
        return True
    
    def check_data_directory(self) -> bool:
        """检查数据目录"""
        print("📁 检查数据目录...")
        
        data_dir = Path("./data")
        if not data_dir.exists():
            print("❌ data 目录不存在")
            return False
        
        # 查找支持的文档文件
        supported_extensions = ['.pdf', '.txt', '.json', '.csv', '.md']
        doc_files = []
        for ext in supported_extensions:
            doc_files.extend(data_dir.rglob(f'*{ext}'))
        
        if not doc_files:
            print("❌ data 目录中没有找到支持的文档文件")
            print(f"支持的文件格式: {', '.join(supported_extensions)}")
            return False
        
        print(f"✅ 找到 {len(doc_files)} 个文档文件")
        return True
    
    def create_workspace_directories(self):
        """创建工作空间目录"""
        print("📂 创建工作空间目录...")
        
        workspace_dirs = [
            "./workspace",
            "./workspace/cache",
            "./workspace/logs",
            "./workspace/exports"
        ]
        
        for dir_path in workspace_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        print("✅ 工作空间目录已准备就绪")
    
    async def run_simple_demo(self):
        """运行简化演示"""
        print("\n🚀 启动 AgenticX GraphRAG 演示...")
        
        try:
            # 导入主演示类
            from main import AgenticXGraphRAGDemo
            
            # 创建演示实例
            demo = AgenticXGraphRAGDemo()
            
            # 运行演示
            await demo.run_demo()
            
        except ImportError as e:
            print(f"❌ 导入演示模块失败: {e}")
            print("请确保 main.py 文件存在且可以正常导入")
        except Exception as e:
            print(f"❌ 演示运行失败: {e}")
            self.logger.exception("演示运行异常")
    
    async def run_interactive_mode(self):
        """运行交互模式"""
        print("\n💬 进入交互模式...")
        print("输入 'quit' 或 'exit' 退出")
        print("输入 'help' 查看帮助")
        
        try:
            from main import AgenticXGraphRAGDemo
            demo = AgenticXGraphRAGDemo()
            
            # 初始化系统
            await demo.initialize_system()
            
            while True:
                try:
                    user_input = input("\n🤖 请输入您的问题: ").strip()
                    
                    if user_input.lower() in ['quit', 'exit', '退出']:
                        print("👋 再见！")
                        break
                    
                    if user_input.lower() in ['help', '帮助']:
                        self.show_help()
                        continue
                    
                    if not user_input:
                        continue
                    
                    # 处理查询
                    print("\n🔍 正在处理您的问题...")
                    response = await demo.process_query(user_input)
                    
                    print(f"\n📝 回答: {response}")
                    
                except KeyboardInterrupt:
                    print("\n👋 再见！")
                    break
                except Exception as e:
                    print(f"❌ 处理查询时出错: {e}")
                    self.logger.exception("查询处理异常")
        
        except Exception as e:
            print(f"❌ 交互模式启动失败: {e}")
            self.logger.exception("交互模式异常")
    
    def show_help(self):
        """显示帮助信息"""
        help_text = """
🔧 AgenticX GraphRAG 演示系统帮助

📋 可用命令:
  • help/帮助    - 显示此帮助信息
  • quit/exit/退出 - 退出系统

💡 查询示例:
  • "什么是 AgenticX？"
  • "AgenticX 有哪些核心模块？"
  • "如何使用 GraphRAG 构建知识图谱？"
  • "AgenticX 的技术优势是什么？"

🔍 查询类型:
  • 实体查询: 查询特定概念或技术
  • 关系查询: 查询实体之间的关系
  • 路径查询: 查询概念之间的连接路径
  • 摘要查询: 获取主题的综合摘要

💭 提示:
  • 尽量使用具体、清晰的问题
  • 可以询问技术细节或应用场景
  • 系统会基于知识图谱提供准确答案
"""
        print(help_text)
    
    async def main_menu(self):
        """主菜单"""
        print("\n🎯 AgenticX GraphRAG 演示系统")
        print("="*50)
        print("1. 运行完整演示 (构建知识图谱 + 问答)")
        print("2. 仅运行交互问答 (需要已构建的知识图谱)")
        print("3. 运行系统测试")
        print("4. 退出")
        print("="*50)
        
        while True:
            try:
                choice = input("请选择操作 (1-4): ").strip()
                
                if choice == '1':
                    await self.run_simple_demo()
                    break
                elif choice == '2':
                    await self.run_interactive_mode()
                    break
                elif choice == '3':
                    from test_demo import SystemTester
                    tester = SystemTester()
                    await tester.run_all_tests()
                    break
                elif choice == '4':
                    print("👋 再见！")
                    break
                else:
                    print("❌ 无效选择，请输入 1-4")
            
            except KeyboardInterrupt:
                print("\n👋 再见！")
                break
            except Exception as e:
                print(f"❌ 操作失败: {e}")
    
    async def run(self):
        """运行快速启动"""
        print("🌟 欢迎使用 AgenticX GraphRAG 演示系统！")
        print("="*60)
        
        # 环境检查
        if not self.check_environment():
            return
        
        if not self.check_data_directory():
            return
        
        # 创建工作空间
        self.create_workspace_directories()
        
        # 显示主菜单
        await self.main_menu()


async def main():
    """主函数"""
    demo = QuickStartDemo()
    await demo.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 程序已退出")
    except Exception as e:
        print(f"❌ 程序运行失败: {e}")
        sys.exit(1)