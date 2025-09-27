"""提示词管理器

用于加载和管理YAML格式的提示词文件
"""

import os
import yaml
from typing import Dict, Any, Optional
from loguru import logger


class PromptManager:
    """提示词管理器"""
    
    def __init__(self, prompts_dir: str = "prompts"):
        self.prompts_dir = prompts_dir
        self.prompts_cache = {}
        
        # 确保prompts目录存在
        if not os.path.exists(self.prompts_dir):
            os.makedirs(self.prompts_dir)
            logger.info(f"📁 创建提示词目录: {self.prompts_dir}")
        
        logger.info(f"🔧 提示词管理器初始化完成，目录: {self.prompts_dir}")
    
    def load_prompt(self, prompt_name: str) -> Dict[str, Any]:
        """加载提示词文件
        
        Args:
            prompt_name: 提示词文件名（不含.yml扩展名）
            
        Returns:
            提示词配置字典
        """
        if prompt_name in self.prompts_cache:
            return self.prompts_cache[prompt_name]
        
        prompt_file = os.path.join(self.prompts_dir, f"{prompt_name}.yml")
        
        if not os.path.exists(prompt_file):
            logger.error(f"❌ 提示词文件不存在: {prompt_file}")
            return {}
        
        try:
            with open(prompt_file, 'r', encoding='utf-8') as f:
                prompt_config = yaml.safe_load(f)
            
            self.prompts_cache[prompt_name] = prompt_config
            logger.debug(f"📄 加载提示词文件: {prompt_name}")
            
            return prompt_config
            
        except Exception as e:
            logger.error(f"❌ 加载提示词文件失败 {prompt_file}: {e}")
            return {}
    
    def get_prompt_template(self, prompt_name: str, template_key: str = "template") -> str:
        """获取提示词模板，支持嵌套路径访问
        
        Args:
            prompt_name: 提示词文件名
            template_key: 模板键名，支持点号分隔的嵌套路径，如"domain_templates.technology.template"
            
        Returns:
            提示词模板字符串
        """
        prompt_config = self.load_prompt(prompt_name)
        
        if not prompt_config:
            return ""
        
        # 支持嵌套路径访问
        template = self._get_nested_value(prompt_config, template_key)
        
        if not template:
            logger.warning(f"⚠️ 提示词模板为空: {prompt_name}.{template_key}")
        
        return template
    
    def _get_nested_value(self, data: dict, key_path: str):
        """获取嵌套字典中的值
        
        Args:
            data: 字典数据
            key_path: 点号分隔的键路径，如"domain_templates.technology.template"
            
        Returns:
            嵌套值或空字符串
        """
        keys = key_path.split('.')
        current = data
        
        try:
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return ""
            return current if isinstance(current, str) else ""
        except Exception:
            return ""
    
    def format_prompt(self, prompt_name: str, template_key: str = "template", **kwargs) -> str:
        """格式化提示词
        
        Args:
            prompt_name: 提示词文件名
            template_key: 模板键名
            **kwargs: 模板变量
            
        Returns:
            格式化后的提示词
        """
        template = self.get_prompt_template(prompt_name, template_key)
        
        if not template:
            return ""
        
        try:
            formatted_prompt = template.format(**kwargs)
            logger.debug(f"✅ 提示词格式化完成: {prompt_name}.{template_key}")
            return formatted_prompt
            
        except KeyError as e:
            logger.error(f"❌ 提示词格式化失败，缺少变量 {e}: {prompt_name}.{template_key}")
            return template
        except Exception as e:
            logger.error(f"❌ 提示词格式化失败: {e}")
            return template
    
    def list_prompts(self) -> list:
        """列出所有可用的提示词文件"""
        if not os.path.exists(self.prompts_dir):
            return []
        
        prompt_files = []
        for file in os.listdir(self.prompts_dir):
            if file.endswith('.yml') or file.endswith('.yaml'):
                prompt_files.append(file[:-4])  # 移除扩展名
        
        return prompt_files
    
    def reload_prompts(self):
        """重新加载所有提示词（清除缓存）"""
        self.prompts_cache.clear()
        logger.info("🔄 提示词缓存已清除")