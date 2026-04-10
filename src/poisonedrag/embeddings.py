"""
嵌入模型模块

支持多种 Embedding Provider：
- DashScope（阿里云 API）：text-embedding-v3 等
- Ollama（本地）：qwen3-embedding:0.6b 等

通过配置中的 embedding_provider 选择使用哪个 Provider。
"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings

from .config import get_config


class EmbeddingModel:
    """
    嵌入模型封装类

    支持 DashScope（API）和 Ollama（本地）两种 Provider。
    通过配置中的 embedding_provider 自动选择。
    """

    def __init__(
        self,
        provider: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化嵌入模型

        Args:
            provider: 嵌入模型提供商（dashscope / ollama），默认从配置读取
            base_url: Ollama 服务地址（仅 ollama 需要），默认从配置读取
            model: 嵌入模型名称，默认从配置读取
            api_key: DashScope API Key（仅 dashscope 需要），默认从配置读取
        """
        config = get_config()
        self.provider = (provider or config.embedding_provider).lower()
        self.model_name = model or (
            config.dashscope_embedding_model
            if self.provider == "dashscope"
            else config.ollama_embedding_model
        )

        if self.provider == "dashscope":
            self._init_dashscope(api_key)
        elif self.provider == "ollama":
            self._init_ollama(base_url)
        else:
            raise ValueError(
                f"不支持的 embedding provider: {self.provider}，支持: dashscope, ollama"
            )

    def _init_dashscope(self, api_key: Optional[str] = None):
        """初始化 DashScope 嵌入模型"""
        from langchain_community.embeddings import DashScopeEmbeddings

        key = api_key or self._get_config_value("dashscope_api_key", "")
        if not key:
            raise ValueError(
                "DashScope API Key 未设置。请在 .env 文件中配置 DASH_SCOPE_API_KEY，"
                "或在配置中传入 api_key 参数。"
            )

        self._embeddings = DashScopeEmbeddings(
            model=self.model_name,
            dashscope_api_key=key,
        )
        self.base_url = "https://dashscope.aliyuncs.com"

    def _init_ollama(self, base_url: Optional[str] = None):
        """初始化 Ollama 本地嵌入模型"""
        from langchain_ollama import OllamaEmbeddings

        url = base_url or self._get_config_value("ollama_base_url", "http://localhost:11434")

        self._embeddings = OllamaEmbeddings(
            base_url=url,
            model=self.model_name,
        )
        self.base_url = url

    def _get_config_value(self, key: str, default=None):
        """从全局配置获取指定值"""
        config = get_config()
        return getattr(config, key, default)

    @property
    def embeddings(self) -> Embeddings:
        """获取 LangChain 嵌入模型实例"""
        return self._embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        将查询文本转换为向量

        Args:
            text: 查询文本

        Returns:
            文本向量（浮点数列表）
        """
        return self._embeddings.embed_query(text)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        将多个文档转换为向量

        Args:
            texts: 文档文本列表

        Returns:
            文档向量列表
        """
        return self._embeddings.embed_documents(texts)

    def __repr__(self) -> str:
        return f"EmbeddingModel(provider='{self.provider}', model='{self.model_name}')"


def get_embedding_model() -> EmbeddingModel:
    """获取嵌入模型实例（使用全局配置）"""
    return EmbeddingModel()
