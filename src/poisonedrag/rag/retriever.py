"""
RAG 检索器模块

实现文档检索功能，支持相似度搜索和元数据过滤。
集成安全过滤器进行语料过滤。
"""

from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from ..vectorstore import VectorStore
from ..config import get_config


@dataclass
class RetrievalResult:
    """
    检索结果数据类

    包含检索到的文档及其相关元信息。
    """
    documents: List[Document]
    scores: List[float]
    filtered_count: int = 0  # 被过滤器拦截的文档数
    warnings: List[str] = None  # 安全警告信息

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

    @property
    def is_safe(self) -> bool:
        """检查结果是否安全"""
        return self.filtered_count == 0 and len(self.warnings) == 0


class Retriever(BaseRetriever):
    """
    自定义检索器类

    继承自 LangChain BaseRetriever，支持向量检索和安全过滤。
    """

    # Pydantic 字段定义
    vectorstore: VectorStore = None
    top_k: int = 5
    score_threshold: float = 0.5
    filter: Optional[Dict[str, Any]] = None
    content_filter: Any = None  # ContentFilter 实例，避免循环导入

    class Config:
        """Pydantic 配置"""
        arbitrary_types_allowed = True

    def __init__(
        self,
        vectorstore: VectorStore,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        filter: Optional[Dict[str, Any]] = None,
        content_filter: Optional[Any] = None,
    ):
        """
        初始化检索器

        Args:
            vectorstore: 向量存储实例
            top_k: 返回结果数量
            score_threshold: 相似度阈值（分数小于此值才返回）
            filter: 元数据过滤条件
            content_filter: 内容过滤器实例
        """
        config = get_config()
        top_k = top_k or config.retriever_top_k
        score_threshold = score_threshold or config.retriever_score_threshold

        super().__init__(
            vectorstore=vectorstore,
            top_k=top_k,
            score_threshold=score_threshold,
            filter=filter,
            content_filter=content_filter,
        )

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        """
        获取相关文档（LangChain 接口）

        Args:
            query: 查询文本
            run_manager: 回调管理器

        Returns:
            相关文档列表
        """
        return self.retrieve(query).documents

    def retrieve(self, query: str) -> RetrievalResult:
        """
        执行检索

        Args:
            query: 查询文本

        Returns:
            RetrievalResult 包含文档和元信息
        """
        # 执行向量相似度搜索
        results = self.vectorstore.similarity_search_with_score(
            query=query,
            k=self.top_k,
            filter=self.filter,
        )

        # 分离文档和分数
        documents = []
        scores = []
        for doc, score in results:
            documents.append(doc)
            scores.append(score)

        # 应用内容过滤
        filtered_docs = []
        filtered_scores = []
        filtered_count = 0
        warnings = []

        for doc, score in zip(documents, scores):
            # 注意：ChromaDB 返回的是距离分数，越小越好（0=完全匹配）
            # score_threshold 配置为 0.5，对于距离分数来说这个阈值太低
            # 正常的相似文档距离分数通常在 0.5~1.5 之间
            # 暂时禁用分数阈值过滤，依赖内容过滤器进行安全检查
            # if score > self.score_threshold:
            #     continue

            # 应用内容过滤器
            if self.content_filter:
                filter_result = self.content_filter.filter_document(doc)
                if not filter_result.is_safe:
                    filtered_count += 1
                    warnings.extend(filter_result.warnings)
                    continue

            filtered_docs.append(doc)
            filtered_scores.append(score)

        return RetrievalResult(
            documents=filtered_docs,
            scores=filtered_scores,
            filtered_count=filtered_count,
            warnings=warnings,
        )

    def retrieve_with_context(
        self,
        query: str,
    ) -> Tuple[str, List[Document]]:
        """
        检索并格式化上下文

        Args:
            query: 查询文本

        Returns:
            (格式化的上下文字符串, 文档列表)
        """
        result = self.retrieve(query)
        context = self._format_documents(result.documents)
        return context, result.documents

    def _format_documents(self, documents: List[Document]) -> str:
        """
        格式化文档列表为上下文字符串

        Args:
            documents: 文档列表

        Returns:
            格式化的上下文字符串
        """
        if not documents:
            return ""

        formatted = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content
            formatted.append(f"[文档 {i}] 来源: {source}\n{content}")

        return "\n\n".join(formatted)


def create_retriever(
    vectorstore: VectorStore,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    content_filter: Optional[Any] = None,
) -> Retriever:
    """
    创建检索器实例

    Args:
        vectorstore: 向量存储实例
        top_k: 返回结果数量
        score_threshold: 相似度阈值
        content_filter: 内容过滤器实例

    Returns:
        Retriever 实例
    """
    return Retriever(
        vectorstore=vectorstore,
        top_k=top_k,
        score_threshold=score_threshold,
        content_filter=content_filter,
    )