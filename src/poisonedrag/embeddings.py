"""
嵌入模型模块

封装 Ollama 本地嵌入模型，提供文本向量化功能。
使用 langchain_ollama.OllamaEmbeddings 实现。
"""

from typing import List, Optional

from langchain_ollama import OllamaEmbeddings

from .config import get_config


class EmbeddingModel:
    """
    嵌入模型封装类

    使用 Ollama 本地部署的嵌入模型进行文本向量化。
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        """
        初始化嵌入模型

        Args:
            base_url: Ollama 服务地址，默认从配置读取
            model: 嵌入模型名称，默认从配置读取
        """
        config = get_config()
        self.base_url = base_url or config.ollama_base_url
        self.model_name = model or config.ollama_embedding_model

        # 初始化 Ollama 嵌入模型
        self._embeddings = OllamaEmbeddings(
            base_url=self.base_url,
            model=self.model_name,
        )

    @property
    def embeddings(self) -> OllamaEmbeddings:
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