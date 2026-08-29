"""Embedding 工厂类 - 根据配置创建对应的 Embedding 实例

支持的类型：
- ollama: Ollama 本地模型（默认，推荐）— 零费用，中文效果好
- local_hf: HuggingFace 本地模型
- api: API 调用（OpenAI、DashScope 等）
"""
import yaml
from pathlib import Path
from typing import Optional, Union
from langchain_core.embeddings import Embeddings


class EmbeddingFactory:
    """Embedding 工厂类

    根据配置文件创建对应的 Embedding 实例。
    """

    @staticmethod
    def create(config_path: Optional[Union[str, Path]] = None):
        """创建 Embedding 实例

        Args:
            config_path: 配置文件路径，默认使用项目根目录下的 config/config.yaml

        Returns:
            LangChain Embeddings 实例
        """
        if config_path is None:
            config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"

        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        embed_config = config['embedding']
        embed_type = embed_config.get('type', 'ollama')

        print(f"📦 创建 Embedding: type={embed_type}")

        if embed_type == "ollama":
            return EmbeddingFactory._create_ollama(embed_config)
        elif embed_type == "api":
            return EmbeddingFactory._create_api(embed_config)
        else:
            return EmbeddingFactory._create_local_hf(embed_config)

    @staticmethod
    def _create_ollama(config: dict) -> Embeddings:
        """创建 Ollama Embedding 实例

        使用 Ollama 本地运行的 Embedding 模型（如 bge-m3）。
        优点：零 API 费用、数据不出本地、中文效果好。
        """
        from langchain_ollama import OllamaEmbeddings

        ollama_config = config.get('ollama', {})
        model_name = ollama_config.get('model_name', 'bge-m3')
        base_url = ollama_config.get('base_url', 'http://localhost:11434')

        print(f"  模型: {model_name}")
        print(f"  地址: {base_url}")

        return OllamaEmbeddings(
            model=model_name,
            base_url=base_url
        )

    @staticmethod
    def _create_local_hf(config: dict) -> Embeddings:
        """创建 HuggingFace 本地 Embedding 实例"""
        from langchain_huggingface import HuggingFaceEmbeddings

        local_config = config.get('local_hf', {})
        model_name = local_config.get('model_name', 'BAAI/bge-base-zh-v1.5')

        print(f"  模型: {model_name}")

        return HuggingFaceEmbeddings(model_name=model_name)

    @staticmethod
    def _create_api(config: dict) -> Embeddings:
        """创建 API Embedding 实例"""
        from langchain_openai import OpenAIEmbeddings

        api_config = config.get('api', {})
        print(f"  提供商: {api_config.get('provider', 'openai')}")
        print(f"  模型: {api_config.get('model_name')}")

        return OpenAIEmbeddings(
            model=api_config.get('model_name'),
            api_key=api_config.get('api_key', ''),
            base_url=api_config.get('base_url', '')
        )


if __name__ == "__main__":
    # 测试
    embeddings = EmbeddingFactory.create()
    # 简单测试向量化
    result = embeddings.embed_query("测试文本")
    print(f"✅ Embedding 创建成功，向量维度: {len(result)}")
