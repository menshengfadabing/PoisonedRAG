"""
嵌入模型模块

使用 OpenAI 兼容接口，支持任意 base_url：
- DashScope（阿里云）：https://dashscope.aliyuncs.com/compatible-mode/v1
- Ollama 本地：http://localhost:11434/v1（自动使用原生 API）
- vLLM / LM Studio 等任意 OpenAI 兼容服务
"""

from typing import List, Optional

from langchain_core.embeddings import Embeddings

from .config import get_config


class EmbeddingModel:
    """
    嵌入模型封装类

    自动检测 Ollama 本地服务并使用原生 API，
    其他情况使用 OpenAI 兼容接口（DashScope / vLLM 等）。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        """
        初始化嵌入模型

        Args:
            base_url: API 基础地址，默认从配置读取
            model: 模型名称，默认从配置读取
            api_key: API Key，默认从配置读取
        """
        config = get_config()
        self.base_url = base_url or config.embedding_base_url
        self.model_name = model or config.embedding_model
        self.api_key = api_key or config.embedding_api_key

        # 自动检测服务类型，使用对应的原生 API
        if "localhost:11434" in self.base_url or "127.0.0.1:11434" in self.base_url:
            self._init_ollama()
        elif "dashscope" in self.base_url:
            self._init_dashscope()
        else:
            self._init_openai()

    def _init_ollama(self):
        """初始化 Ollama 本地嵌入模型（使用原生 API）"""
        from langchain_ollama import OllamaEmbeddings

        # 提取 host:port 部分
        url = self.base_url.replace("/v1", "").replace("/api", "")

        self._embeddings = OllamaEmbeddings(
            base_url=url,
            model=self.model_name,
        )

    def _init_dashscope(self):
        """初始化 DashScope 嵌入模型（使用原生 API）"""
        from langchain_community.embeddings import DashScopeEmbeddings

        self._embeddings = DashScopeEmbeddings(
            model=self.model_name,
            dashscope_api_key=self.api_key,
        )

    def _init_openai(self):
        """初始化 OpenAI 兼容的嵌入模型"""
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(
            model=self.model_name,
            openai_api_key=self.api_key,
            openai_api_base=self.base_url,
        )

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
        return f"EmbeddingModel(base_url='{self.base_url}', model='{self.model_name}')"


def get_embedding_model() -> EmbeddingModel:
    """获取嵌入模型实例（使用全局配置）"""
    return EmbeddingModel()
