"""
知识库数据管理模块

管理正常知识库，支持从文件加载文档、添加到向量存储。
支持格式：PDF, Word(.docx), PowerPoint(.pptx), Markdown, JSON, TXT
"""

import os
import json
from typing import List, Optional, Dict, Any
from pathlib import Path

from langchain_core.documents import Document

from ..vectorstore import VectorStore
from ..embeddings import EmbeddingModel, get_embedding_model
from .document_loader import (
    load_file,
    load_directory as _load_directory,
    load_json,
    load_markdown,
    load_text,
    SUPPORTED_EXTENSIONS,
)


class KnowledgeBase:
    """
    知识库管理类

    负责管理正常知识库，包括：
    - 从文件加载文档
    - 文档预处理
    - 添加到向量存储
    """

    def __init__(
        self,
        vectorstore: Optional[VectorStore] = None,
        data_dir: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        reviewer: Optional[Any] = None,
    ):
        """
        初始化知识库

        Args:
            vectorstore: 向量存储实例
            data_dir: 知识库数据目录
            embedding_model: 嵌入模型实例
            reviewer: 文档审查器实例（可选），设置后文档入库前必须通过审查
        """
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model or get_embedding_model()
        self.data_dir = data_dir or self._get_default_data_dir()
        self.reviewer = reviewer  # 可选的审查器

        # 文档元数据
        self._documents: List[Document] = []

    def _get_default_data_dir(self) -> str:
        """获取默认数据目录"""
        # 相对于项目根目录
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "data" / "knowledge")

    def load_from_json(self, file_path: str) -> List[Document]:
        """
        从 JSON 文件加载文档

        JSON 格式示例:
        [
            {
                "content": "文档内容",
                "metadata": {"source": "来源", "category": "分类"}
            }
        ]

        Args:
            file_path: JSON 文件路径

        Returns:
            文档列表
        """
        documents = []

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            content = item.get("content", "")
            metadata = item.get("metadata", {})
            metadata["source"] = metadata.get("source", file_path)
            metadata["type"] = "knowledge"

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def load_from_markdown(self, file_path: str) -> List[Document]:
        """
        从 Markdown 文件加载文档

        按 H1/H2 标题分割文档

        Args:
            file_path: Markdown 文件路径

        Returns:
            文档列表
        """
        import re

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # 按标题分割
        sections = re.split(r'\n(?=#{1,2}\s)', content)
        documents = []

        for section in sections:
            section = section.strip()
            if not section:
                continue

            # 提取标题作为元数据
            title_match = re.match(r'^(#{1,2})\s+(.+)$', section)
            if title_match:
                level = len(title_match.group(1))
                title = title_match.group(2).strip()
            else:
                title = "未命名"
                level = 0

            doc = Document(
                page_content=section,
                metadata={
                    "source": file_path,
                    "title": title,
                    "level": level,
                    "type": "knowledge",
                }
            )
            documents.append(doc)

        return documents

    def load_from_text(self, file_path: str, chunk_size: int = 500) -> List[Document]:
        """
        从纯文本文件加载文档

        按段落分割，或按指定大小分块

        Args:
            file_path: 文本文件路径
            chunk_size: 分块大小（字符数）

        Returns:
            文档列表
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        documents = []

        # 尝试按段落分割
        paragraphs = content.split('\n\n')

        current_chunk = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            if len(current_chunk) + len(para) <= chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    doc = Document(
                        page_content=current_chunk.strip(),
                        metadata={
                            "source": file_path,
                            "type": "knowledge",
                        }
                    )
                    documents.append(doc)
                current_chunk = para + "\n\n"

        # 添加最后一个块
        if current_chunk:
            doc = Document(
                page_content=current_chunk.strip(),
                metadata={
                    "source": file_path,
                    "type": "knowledge",
                }
            )
            documents.append(doc)

        return documents

    def load_from_directory(
        self,
        directory: Optional[str] = None,
        recursive: bool = True,
    ) -> List[Document]:
        """
        从目录加载所有支持的文档（PDF/DOCX/PPTX/MD/JSON/TXT）

        Args:
            directory: 目录路径，默认使用 data_dir
            recursive: 是否递归加载子目录

        Returns:
            文档列表
        """
        directory = directory or self.data_dir

        if not os.path.exists(directory):
            return []

        return _load_directory(directory, recursive=recursive)

    def add_document(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        添加单个文档

        Args:
            content: 文档内容
            metadata: 元数据

        Returns:
            文档 ID
        """
        metadata = metadata or {}
        metadata["type"] = "knowledge"

        doc = Document(page_content=content, metadata=metadata)
        self._documents.append(doc)

        if self.vectorstore:
            return self.vectorstore.add_documents([doc])[0]

        return ""

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        添加多个文档

        Args:
            documents: 文档列表

        Returns:
            文档 ID 列表

        Raises:
            ValueError: 如果配置了审查器但文档未通过审查
        """
        # 如果设置了审查器，入库前进行安全检查
        if self.reviewer:
            documents = self._review_documents(documents)

        # 标记为知识库文档
        for doc in documents:
            doc.metadata["type"] = "knowledge"

        self._documents.extend(documents)

        if self.vectorstore:
            return self.vectorstore.add_documents(documents)

        return []

    def index_to_vectorstore(self, documents: Optional[List[Document]] = None) -> int:
        """
        将文档索引到向量存储

        Args:
            documents: 要索引的文档，默认使用已加载的文档

        Returns:
            索引的文档数量

        Raises:
            ValueError: 如果配置了审查器但文档未通过审查
        """
        if not self.vectorstore:
            raise ValueError("未设置向量存储")

        docs = documents or self._documents
        if not docs:
            return 0

        # 如果设置了审查器，入库前进行安全检查
        if self.reviewer and documents is not None:
            docs = self._review_documents(documents)

        self.vectorstore.add_documents(docs)
        return len(docs)

    def _review_documents(self, documents: List[Document]) -> List[Document]:
        """
        对文档进行安全审查

        Args:
            documents: 待审查的文档列表

        Returns:
            通过审查的文档列表

        Raises:
            ValueError: 如果有文档未通过审查
        """
        if not self.reviewer:
            return documents

        risky_docs = []
        safe_docs = []

        for doc in documents:
            result = self.reviewer.quick_review(doc.page_content)
            if result and result.results:
                chunk_result = result.results[0]
                if chunk_result.is_safe:
                    safe_docs.append(doc)
                else:
                    risky_docs.append(
                        f"来源={doc.metadata.get('source', '未知')}, "
                        f"风险={chunk_result.risk_type or '未知'}, "
                        f"分数={chunk_result.risk_score:.2f}, "
                        f"原因={chunk_result.reason or '无'}"
                    )
            else:
                # 审查失败，保守起见拒绝
                risky_docs.append(f"来源={doc.metadata.get('source', '未知')}, 审查失败")

        if risky_docs:
            details = "\n".join(f"  - {d}" for d in risky_docs)
            raise ValueError(
                f"{len(risky_docs)} 篇文档未通过安全审查，已拒绝入库：\n{details}"
            )

        return safe_docs

    def clear(self):
        """清空知识库"""
        self._documents = []
        if self.vectorstore:
            self.vectorstore.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取知识库统计信息

        Returns:
            统计信息字典
        """
        stats = {
            "document_count": len(self._documents),
            "data_dir": self.data_dir,
            "vectorstore_count": self.vectorstore.count() if self.vectorstore else 0,
        }
        return stats

    def __len__(self) -> int:
        return len(self._documents)

    def __repr__(self) -> str:
        return f"KnowledgeBase(documents={len(self._documents)}, data_dir='{self.data_dir}')"


def create_knowledge_base(
    vectorstore: Optional[VectorStore] = None,
    data_dir: Optional[str] = None,
) -> KnowledgeBase:
    """
    创建知识库实例

    Args:
        vectorstore: 向量存储实例
        data_dir: 数据目录

    Returns:
        KnowledgeBase 实例
    """
    return KnowledgeBase(
        vectorstore=vectorstore,
        data_dir=data_dir,
    )