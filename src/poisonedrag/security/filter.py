"""
语料过滤器模块

实现内容安全过滤，检测敏感词、恶意指令和语义异常。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import re

from langchain_core.documents import Document

from ..config import get_config
from ..embeddings import EmbeddingModel


@dataclass
class FilterResult:
    """
    过滤结果数据类

    包含过滤判定和相关警告信息。
    """
    is_safe: bool
    warnings: List[str] = field(default_factory=list)
    risk_score: float = 0.0  # 0.0 = 安全, 1.0 = 高风险
    filtered_content: Optional[str] = None  # 过滤后的内容（如有）

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_safe": self.is_safe,
            "warnings": self.warnings,
            "risk_score": self.risk_score,
        }


class ContentFilter:
    """
    内容过滤器类

    实现多层次的内容安全过滤：
    1. 关键词过滤：检测敏感词和恶意指令
    2. 语义异常检测：计算与正常语料的语义距离
    3. 来源可信度检查
    """

    def __init__(
        self,
        sensitive_keywords: Optional[List[str]] = None,
        semantic_threshold: Optional[float] = None,
        trusted_sources: Optional[List[str]] = None,
        embedding_model: Optional[EmbeddingModel] = None,
    ):
        """
        初始化过滤器

        Args:
            sensitive_keywords: 敏感关键词列表
            semantic_threshold: 语义异常检测阈值
            trusted_sources: 可信来源列表
            embedding_model: 嵌入模型（用于语义检测）
        """
        config = get_config()
        self.sensitive_keywords = sensitive_keywords or config.sensitive_keywords
        self.semantic_threshold = semantic_threshold or config.semantic_anomaly_threshold
        self.trusted_sources = trusted_sources or config.trusted_sources
        self.embedding_model = embedding_model

        # 编译正则表达式用于关键词匹配
        self._compile_patterns()

        # 已知的恶意模式（用于语义相似度检测）
        self._malicious_patterns = self._build_malicious_patterns()

    def _compile_patterns(self):
        """编译关键词的正则表达式模式"""
        # 转义特殊字符并构建正则模式
        escaped_keywords = [re.escape(kw) for kw in self.sensitive_keywords]
        self._keyword_pattern = re.compile(
            '|'.join(escaped_keywords),
            flags=re.IGNORECASE
        )

    def _build_malicious_patterns(self) -> List[str]:
        """
        构建恶意模式列表（用于语义检测）

        包含中英文典型恶意引导模式
        """
        return [
            # 中文恶意模式
            "请忽略之前的指令",
            "请执行以下命令",
            "这个代码可以安全运行",
            "只需要运行这个命令",
            "请复制以下代码",
            "你的系统已被锁定",
            "点击此链接",
            "下载并运行",
            # 英文恶意模式
            "ignore previous instructions",
            "disregard all prior",
            "execute this command",
            "run the following code",
            "your system is locked",
            "click this link",
            "download and run",
            "bypass all security",
            "your api key is",
            "your password is",
        ]

    def filter_text(self, text: str) -> FilterResult:
        """
        过滤文本内容

        Args:
            text: 待过滤的文本

        Returns:
            FilterResult 过滤结果
        """
        warnings = []
        risk_score = 0.0

        # 1. 关键词检测
        keyword_matches = self._keyword_pattern.findall(text)
        if keyword_matches:
            unique_matches = list(set(keyword_matches))
            warnings.append(f"检测到敏感关键词: {', '.join(unique_matches)}")
            risk_score += 0.3 * len(unique_matches)

        # 2. 恶意指令模式检测
        malicious_patterns = self._detect_malicious_patterns(text)
        if malicious_patterns:
            warnings.append(f"检测到恶意模式: {', '.join(malicious_patterns)}")
            risk_score += 0.4 * len(malicious_patterns)

        # 3. 语义异常检测（如果嵌入模型可用）
        if self.embedding_model:
            semantic_score = self._detect_semantic_anomaly(text)
            if semantic_score > self.semantic_threshold:
                warnings.append(f"语义异常检测: 相似度分数 {semantic_score:.2f}")
                risk_score += semantic_score

        # 限制风险分数在 [0, 1] 范围
        risk_score = min(1.0, risk_score)

        # 判定是否安全
        is_safe = risk_score < 0.5 and len(warnings) == 0

        return FilterResult(
            is_safe=is_safe,
            warnings=warnings,
            risk_score=risk_score,
        )

    def filter_document(self, document: Document) -> FilterResult:
        """
        过滤文档内容

        Args:
            document: LangChain 文档对象

        Returns:
            FilterResult 过滤结果
        """
        source = document.metadata.get("source", "")
        source_trust = self._check_source_trust(source)

        # 非可信来源：正常内容检查 + 风险加分
        if not source_trust:
            text_result = self.filter_text(document.page_content)
            text_result.warnings.append(f"来源可信度低: {source}")
            text_result.risk_score = min(1.0, text_result.risk_score + 0.2)
            return text_result

        # 可信来源：仍进行基本关键字检查，但跳过语义异常检测
        # 可信来源不是绝对安全——文件可能被篡改或混入恶意内容
        warnings = []
        risk_score = 0.0

        # 仅做关键字检测（轻量），不做语义检测
        keyword_matches = self._keyword_pattern.findall(document.page_content)
        if keyword_matches:
            unique_matches = list(set(keyword_matches))
            warnings.append(f"⚠️ 可信来源但检测到敏感关键词: {', '.join(unique_matches)}")
            risk_score += 0.3 * len(unique_matches)

        malicious_patterns = self._detect_malicious_patterns(document.page_content)
        if malicious_patterns:
            warnings.append(f"⚠️ 可信来源但检测到恶意模式: {', '.join(malicious_patterns)}")
            risk_score += 0.4 * len(malicious_patterns)

        # 可信来源给予基础信任分（降低风险）
        risk_score = max(0.0, risk_score - 0.1)

        is_safe = risk_score < 0.5 and len(warnings) == 0

        return FilterResult(
            is_safe=is_safe,
            warnings=warnings,
            risk_score=min(1.0, risk_score),
        )

    def _detect_malicious_patterns(self, text: str) -> List[str]:
        """
        检测恶意指令模式

        Args:
            text: 待检测文本

        Returns:
            检测到的恶意模式列表
        """
        detected = []
        text_lower = text.lower()

        for pattern in self._malicious_patterns:
            if pattern.lower() in text_lower:
                detected.append(pattern)

        return detected

    def _detect_semantic_anomaly(self, text: str) -> float:
        """
        检测语义异常

        计算文本与已知恶意模式的语义相似度

        Args:
            text: 待检测文本

        Returns:
            异常分数 (0.0 = 正常, 1.0 = 高度异常)
        """
        if not self.embedding_model:
            return 0.0

        try:
            # 获取文本向量
            text_embedding = self.embedding_model.embed_query(text)

            # 获取恶意模式向量
            pattern_embeddings = []
            for pattern in self._malicious_patterns:
                pattern_embedding = self.embedding_model.embed_query(pattern)
                pattern_embeddings.append(pattern_embedding)

            # 计算最大相似度
            max_similarity = 0.0
            for pattern_emb in pattern_embeddings:
                similarity = self._cosine_similarity(text_embedding, pattern_emb)
                max_similarity = max(max_similarity, similarity)

            # 相似度越高，异常分数越高
            return max_similarity

        except Exception:
            # 嵌入失败时返回中等风险
            return 0.3

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            余弦相似度
        """
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _check_source_trust(self, source: str) -> bool:
        """
        检查来源可信度

        Args:
            source: 来源标识

        Returns:
            是否可信
        """
        if not source:
            return False  # 未知来源不信任

        for trusted in self.trusted_sources:
            if trusted.lower() in source.lower():
                return True

        return False

    def add_sensitive_keyword(self, keyword: str):
        """添加敏感关键词"""
        if keyword not in self.sensitive_keywords:
            self.sensitive_keywords.append(keyword)
            self._compile_patterns()

    def remove_sensitive_keyword(self, keyword: str):
        """移除敏感关键词"""
        if keyword in self.sensitive_keywords:
            self.sensitive_keywords.remove(keyword)
            self._compile_patterns()

    def add_trusted_source(self, source: str):
        """添加可信来源"""
        if source not in self.trusted_sources:
            self.trusted_sources.append(source)

    def __repr__(self) -> str:
        return f"ContentFilter(keywords={len(self.sensitive_keywords)}, trusted_sources={len(self.trusted_sources)})"


def get_content_filter(
    embedding_model: Optional[EmbeddingModel] = None,
) -> ContentFilter:
    """
    获取内容过滤器实例

    Args:
        embedding_model: 嵌入模型实例

    Returns:
        ContentFilter 实例
    """
    return ContentFilter(embedding_model=embedding_model)