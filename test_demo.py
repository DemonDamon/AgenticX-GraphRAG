#!/usr/bin/env python3
"""
AgenticX GraphRAG 演示系统测试脚本

用于验证系统的基本功能和配置
"""

import os
import sys
import asyncio
import yaml
from pathlib import Path
from typing import Dict, Any

# 添加 AgenticX 到 Python 路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SystemTester:
    """系统测试器"""
    
    def __init__(self):
        self.test_results = []
        self.config = None
    
    def log_test(self, test_name: str, success: bool, message: str = ""):
        """记录测试结果"""
        status = "✅ PASS" if success else "❌ FAIL"
        result = f"{status} {test_name}"
        if message:
            result += f": {message}"
        print(result)
        self.test_results.append((test_name, success, message))
    
    def test_environment_variables(self):
        """测试环境变量"""
        print("\n🔍 测试环境变量...")
        
        required_vars = [
            ('OPENAI_API_KEY', 'OpenAI API 密钥'),
            ('NEO4J_URI', 'Neo4j 连接地址'),
            ('NEO4J_USERNAME', 'Neo4j 用户名'),
            ('NEO4J_PASSWORD', 'Neo4j 密码')
        ]
        
        for var_name, description in required_vars:
            value = os.getenv(var_name)
            if value:
                # 隐藏敏感信息
                display_value = value[:8] + "..." if len(value) > 8 else "***"
                self.log_test(f"环境变量 {var_name}", True, f"{description} ({display_value})")
            else:
                self.log_test(f"环境变量 {var_name}", False, f"{description} 未设置")
    
    def test_config_file(self):
        """测试配置文件"""
        print("\n📄 测试配置文件...")
        
        config_files = ["configs.yml", ".trae/documents/configs.yml"]
        config_found = False
        
        for config_file in config_files:
            config_path = Path(config_file)
            if config_path.exists():
                try:
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self.config = yaml.safe_load(f)
                    self.log_test(f"配置文件 {config_file}", True, "加载成功")
                    config_found = True
                    break
                except Exception as e:
                    self.log_test(f"配置文件 {config_file}", False, f"加载失败: {e}")
            else:
                self.log_test(f"配置文件 {config_file}", False, "文件不存在")
        
        if not config_found:
            self.log_test("配置文件", False, "未找到有效的配置文件")
    
    def test_data_directory(self):
        """测试数据目录"""
        print("\n📁 测试数据目录...")
        
        data_dir = Path("./data")
        if data_dir.exists():
            files = list(data_dir.rglob('*'))
            doc_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.pdf', '.txt', '.json', '.csv', '.md']]
            
            if doc_files:
                self.log_test("数据目录", True, f"找到 {len(doc_files)} 个文档文件")
                for doc_file in doc_files[:5]:  # 只显示前5个文件
                    file_size = doc_file.stat().st_size
                    size_str = f"{file_size} bytes" if file_size < 1024 else f"{file_size/1024:.1f} KB"
                    self.log_test(f"  文档 {doc_file.name}", True, f"大小: {size_str}")
            else:
                self.log_test("数据目录", False, "目录存在但没有找到支持的文档文件")
        else:
            self.log_test("数据目录", False, "data 目录不存在")
    
    def test_python_dependencies(self):
        """测试 Python 依赖"""
        print("\n🐍 测试 Python 依赖...")
        
        dependencies = [
            ('yaml', 'PyYAML'),
            ('asyncio', 'asyncio'),
            ('pathlib', 'pathlib'),
            ('logging', 'logging')
        ]
        
        # 尝试导入 AgenticX 相关模块
        agenticx_modules = [
            ('agenticx.knowledge', 'AgenticX Knowledge 模块'),
            ('agenticx.embeddings', 'AgenticX Embeddings 模块'),
            ('agenticx.storage', 'AgenticX Storage 模块'),
            ('agenticx.retrieval', 'AgenticX Retrieval 模块')
        ]
        
        # 测试基础依赖
        for module_name, description in dependencies:
            try:
                __import__(module_name)
                self.log_test(f"依赖 {module_name}", True, description)
            except ImportError as e:
                self.log_test(f"依赖 {module_name}", False, f"{description} 导入失败: {e}")
        
        # 测试 AgenticX 模块
        for module_name, description in agenticx_modules:
            try:
                __import__(module_name)
                self.log_test(f"模块 {module_name}", True, description)
            except ImportError as e:
                self.log_test(f"模块 {module_name}", False, f"{description} 导入失败: {e}")
    
    async def test_external_services(self):
        """测试外部服务连接"""
        print("\n🔗 测试外部服务连接...")
        
        # 测试 Neo4j 连接
        await self._test_neo4j_connection()
        
        # 测试 Redis 连接
        await self._test_redis_connection()
        
        # 测试 OpenAI API
        await self._test_openai_api()
    
    async def _test_neo4j_connection(self):
        """测试 Neo4j 连接"""
        try:
            from neo4j import GraphDatabase
            
            uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
            username = os.getenv('NEO4J_USERNAME', 'neo4j')
            password = os.getenv('NEO4J_PASSWORD')
            
            if not password:
                self.log_test("Neo4j 连接", False, "密码未设置")
                return
            
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # 测试连接
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                if test_value == 1:
                    self.log_test("Neo4j 连接", True, f"连接成功 ({uri})")
                else:
                    self.log_test("Neo4j 连接", False, "连接测试失败")
            
            driver.close()
            
        except ImportError:
            self.log_test("Neo4j 连接", False, "neo4j 库未安装")
        except Exception as e:
            self.log_test("Neo4j 连接", False, f"连接失败: {e}")
    
    async def _test_redis_connection(self):
        """测试 Redis 连接"""
        try:
            import redis
            
            host = os.getenv('REDIS_HOST', 'localhost')
            port = int(os.getenv('REDIS_PORT', '6379'))
            password = os.getenv('REDIS_PASSWORD')
            
            r = redis.Redis(
                host=host,
                port=port,
                password=password,
                decode_responses=True,
                socket_timeout=5
            )
            
            # 测试连接
            r.ping()
            self.log_test("Redis 连接", True, f"连接成功 ({host}:{port})")
            
        except ImportError:
            self.log_test("Redis 连接", False, "redis 库未安装")
        except Exception as e:
            self.log_test("Redis 连接", False, f"连接失败: {e}")
    
    async def _test_openai_api(self):
        """测试 OpenAI API"""
        try:
            import openai
            
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                self.log_test("OpenAI API", False, "API 密钥未设置")
                return
            
            client = openai.OpenAI(api_key=api_key)
            
            # 测试简单的 API 调用
            response = await asyncio.wait_for(
                asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=5
                ),
                timeout=10
            )
            
            if response.choices:
                self.log_test("OpenAI API", True, "API 调用成功")
            else:
                self.log_test("OpenAI API", False, "API 响应异常")
                
        except ImportError:
            self.log_test("OpenAI API", False, "openai 库未安装")
        except asyncio.TimeoutError:
            self.log_test("OpenAI API", False, "API 调用超时")
        except Exception as e:
            self.log_test("OpenAI API", False, f"API 调用失败: {e}")
    
    def test_workspace_directories(self):
        """测试工作空间目录"""
        print("\n📂 测试工作空间目录...")
        
        workspace_dirs = [
            "./workspace",
            "./workspace/cache",
            "./workspace/logs",
            "./workspace/exports"
        ]
        
        for dir_path in workspace_dirs:
            path = Path(dir_path)
            if path.exists():
                self.log_test(f"目录 {dir_path}", True, "目录存在")
            else:
                try:
                    path.mkdir(parents=True, exist_ok=True)
                    self.log_test(f"目录 {dir_path}", True, "目录已创建")
                except Exception as e:
                    self.log_test(f"目录 {dir_path}", False, f"创建失败: {e}")
    
    def print_summary(self):
        """打印测试总结"""
        print("\n" + "="*60)
        print("📊 测试总结")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for _, success, _ in self.test_results if success)
        failed_tests = total_tests - passed_tests
        
        print(f"总测试数: {total_tests}")
        print(f"通过: {passed_tests} ✅")
        print(f"失败: {failed_tests} ❌")
        print(f"成功率: {passed_tests/total_tests*100:.1f}%")
        
        if failed_tests > 0:
            print("\n❌ 失败的测试:")
            for test_name, success, message in self.test_results:
                if not success:
                    print(f"  • {test_name}: {message}")
            
            print("\n💡 建议:")
            print("  1. 检查环境变量配置")
            print("  2. 确保外部服务正在运行")
            print("  3. 安装缺失的依赖包")
            print("  4. 查看详细错误信息")
        else:
            print("\n🎉 所有测试都通过了！系统已准备就绪。")
        
        print("="*60)
    
    async def run_all_tests(self):
        """运行所有测试"""
        print("🧪 AgenticX GraphRAG 系统测试")
        print("="*60)
        
        # 运行各项测试
        self.test_environment_variables()
        self.test_config_file()
        self.test_data_directory()
        self.test_python_dependencies()
        self.test_workspace_directories()
        await self.test_external_services()
        
        # 打印总结
        self.print_summary()


async def main():
    """主函数"""
    tester = SystemTester()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())