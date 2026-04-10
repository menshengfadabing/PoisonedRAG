"""
向量数据库模块

使用 ChromaDB 实现本地向量存储和检索。
支持文档的添加、删除、相似度搜索等功能。
"""

import os
from typing import List, Optional, Dict, Any

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma

from .config import get_config


class VectorStore:
    """
    向量数据库封装类

    使用 ChromaDB 进行文档的向量存储和检索。
    """

    def __init__(
        self,
        embedding_function: Optional[Embeddings] = None,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
    ):
        """
        初始化向量数据库

        Args:
            embedding_function: 嵌入函数，用于将文本转换为向量
            persist_directory: 持久化目录
            collection_name: 集合名称
        """
        config = get_config()
        self.persist_directory = persist_directory or config.chroma_persist_directory
        self.collection_name = collection_name or config.chroma_collection_name
        self.embedding_function = embedding_function

        # 确保目录存在
        os.makedirs(self.persist_directory, exist_ok=True)

        # 初始化向量存储
        self._vectorstore: Optional[Chroma] = None

    def _get_or_create_vectorstore(self) -> Chroma:
        """获取或创建向量存储实例"""
        if self._vectorstore is None:
            if self.embedding_function is None:
                raise ValueError("必须提供 embedding_function 才能初始化向量存储")

            self._vectorstore = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embedding_function,
                collection_name=self.collection_name,
            )
        return self._vectorstore

    @property
    def vectorstore(self) -> Chroma:
        """获取向量存储实例"""
        return self._get_or_create_vectorstore()

    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文档到向量存储

        Args:
            documents: 文档列表
            ids: 文档 ID 列表（可选）

        Returns:
            添加的文档 ID 列表
        """
        return self.vectorstore.add_documents(documents, ids=ids)

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表（可选）
            ids: 文档 ID 列表（可选）

        Returns:
            添加的文档 ID 列表
        """
        return self.vectorstore.add_texts(texts, metadatas=metadatas, ids=ids)

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            相似文档列表
        """
        return self.vectorstore.similarity_search(query, k=k, filter=filter)

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """
        带分数的相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量
            filter: 元数据过滤条件

        Returns:
            (文档, 分数) 元组列表，分数越小越相似
        """
        return self.vectorstore.similarity_search_with_score(query, k=k, filter=filter)

    def delete(self, ids: Optional[List[str]] = None) -> None:
        """
        删除文档

        Args:
            ids: 要删除的文档 ID 列表
        """
        self.vectorstore.delete(ids=ids)

    def get(self, ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        获取文档

        Args:
            ids: 文档 ID 列表

        Returns:
            包含文档信息的字典
        """
        return self.vectorstore.get(ids=ids)

    def as_retriever(self, **kwargs: Any):
        """
        转换为检索器

        Args:
            **kwargs: 检索器参数

        Returns:
            LangChain 检索器实例
        """
        return self.vectorstore.as_retriever(**kwargs)

    def count(self) -> int:
        """获取文档数量"""
        return self.vectorstore._collection.count()

    def clear(self) -> None:
        """清空向量存储"""
        # 获取所有文档 ID
        all_docs = self.vectorstore.get()
        if all_docs and all_docs.get("ids"):
            self.vectorstore.delete(ids=all_docs["ids"])

    def __repr__(self) -> str:
        return f"VectorStore(collection='{self.collection_name}', persist_dir='{self.persist_directory}')"


def get_vectorstore(
    embedding_function: Optional[Embeddings] = None,
    collection_name: Optional[str] = None,
) -> VectorStore:
    """
    获取向量存储实例

    Args:
        embedding_function: 嵌入函数
        collection_name: 集合名称

    Returns:
        VectorStore 实例
    """
    return VectorStore(
        embedding_function=embedding_function,
        collection_name=collection_name,
    )