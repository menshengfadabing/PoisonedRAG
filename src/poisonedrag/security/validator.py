"""
生成结果校验器模块

实现响应安全校验，包括事实一致性检查和安全规则匹配。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
import re

from langchain_core.documents import Document

from ..config import get_config
from ..embeddings import EmbeddingModel


@dataclass
class ValidationResult:
    """
    校验结果数据类

    包含校验判定和相关警告信息。
    """
    is_safe: bool = True
    is_consistent: bool = True  # 事实一致性
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 置信度 (0.0-1.0)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "is_safe": self.is_safe,
            "is_consistent": self.is_consistent,
            "warnings": self.warnings,
            "confidence": self.confidence,
            "details": self.details,
        }


class ResponseValidator:
    """
    响应校验器类

    实现多层次的安全校验：
    1. 事实一致性检查：答案与检索内容的一致性
    2. 安全规则匹配：检测潜在有害输出
    3. 置信度评估：输出可信度评分
    """

    # 安全规则 - 禁止输出的内容模式
    UNSAFE_PATTERNS = [
        # 敏感信息泄露
        (r"password\s*[=:]\s*\S+", "检测到密码泄露"),
        (r"api[_-]?key\s*[=:]\s*\S+", "检测到 API 密钥泄露"),
        (r"secret[_-]?key\s*[=:]\s*\S+", "检测到密钥泄露"),
        (r"token\s*[=:]\s*\S+", "检测到令牌泄露"),

        # 危险操作指令
        (r"rm\s+-rf", "检测到危险删除命令"),
        (r"format\s+[a-zA-Z]:", "检测到格式化命令"),
        (r"DROP\s+TABLE", "检测到危险 SQL 操作"),

        # 恶意链接
        (r"https?://[^\s]*\.(exe|bat|cmd|sh|ps1)", "检测到可疑下载链接"),
        (r"点击.*下载", "检测到可疑下载引导"),

        # 绕过指令
        (r"ignore\s+(all\s+)?(previous|above)", "检测到绕过指令"),
        (r"disregard\s+(all\s+)?(previous|above)", "检测到绕过指令"),
    ]

    # 安全关键词（响应中不应出现）—— 中英文
    UNSAFE_KEYWORDS = [
        # 中文
        "执行此命令",
        "运行以下代码",
        "你的密码是",
        "你的密钥是",
        "银行账户",
        "信用卡号",
        # 英文
        "execute this command",
        "run the following code",
        "your password is",
        "your api key is",
        "your secret key is",
        "ignore all previous instructions",
        "disregard previous instructions",
    ]

    def __init__(
        self,
        embedding_model: Optional[EmbeddingModel] = None,
        consistency_threshold: float = 0.5,
    ):
        """
        初始化校验器

        Args:
            embedding_model: 嵌入模型（用于一致性检查）
            consistency_threshold: 一致性阈值
        """
        self.embedding_model = embedding_model
        self.consistency_threshold = consistency_threshold

        # 编译不安全模式正则表达式
        self._compiled_patterns = [
            (re.compile(pattern, re.IGNORECASE), message)
            for pattern, message in self.UNSAFE_PATTERNS
        ]

    def validate(
        self,
        query: str,
        response: str,
        documents: List[Document],
    ) -> ValidationResult:
        """
        验证响应内容

        Args:
            query: 用户查询
            response: 生成的响应
            documents: 检索到的文档（作为事实依据）

        Returns:
            ValidationResult 校验结果
        """
        result = ValidationResult()

        # 1. 安全规则匹配
        safety_result = self._check_safety_rules(response)
        result.warnings.extend(safety_result["warnings"])
        if not safety_result["is_safe"]:
            result.is_safe = False
            result.details["safety_violations"] = safety_result["violations"]

        # 2. 事实一致性检查
        if documents:
            consistency_result = self._check_consistency(response, documents)
            result.is_consistent = consistency_result["is_consistent"]
            if not result.is_consistent:
                result.warnings.append("响应内容与检索文档的一致性较低")
            result.details["consistency_score"] = consistency_result["score"]

        # 3. 置信度评估
        result.confidence = self._calculate_confidence(
            response=response,
            documents=documents,
            is_safe=result.is_safe,
            is_consistent=result.is_consistent,
        )

        return result

    def _check_safety_rules(self, response: str) -> Dict[str, Any]:
        """
        检查安全规则

        Args:
            response: 待检查的响应

        Returns:
            安全检查结果字典
        """
        warnings = []
        violations = []
        is_safe = True

        # 检查不安全模式
        for pattern, message in self._compiled_patterns:
            if pattern.search(response):
                warnings.append(message)
                violations.append(message)
                is_safe = False

        # 检查不安全关键词
        response_lower = response.lower()
        for keyword in self.UNSAFE_KEYWORDS:
            if keyword.lower() in response_lower:
                warning = f"检测到不安全关键词: {keyword}"
                warnings.append(warning)
                violations.append(warning)
                is_safe = False

        return {
            "is_safe": is_safe,
            "warnings": warnings,
            "violations": violations,
        }

    def _check_consistency(
        self,
        response: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """
        检查事实一致性

        Args:
            response: 生成的响应
            documents: 检索到的文档

        Returns:
            一致性检查结果字典
        """
        if not self.embedding_model:
            # 没有嵌入模型时使用简单关键词匹配
            return self._simple_consistency_check(response, documents)

        try:
            # 获取响应向量
            response_embedding = self.embedding_model.embed_query(response)

            # 获取文档向量并计算平均相似度
            similarities = []
            for doc in documents:
                doc_embedding = self.embedding_model.embed_query(doc.page_content)
                similarity = self._cosine_similarity(response_embedding, doc_embedding)
                similarities.append(similarity)

            # 平均相似度作为一致性分数
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0

            return {
                "is_consistent": avg_similarity >= self.consistency_threshold,
                "score": avg_similarity,
            }

        except Exception:
            return self._simple_consistency_check(response, documents)

    def _simple_consistency_check(
        self,
        response: str,
        documents: List[Document],
    ) -> Dict[str, Any]:
        """
        简单的一致性检查（基于关键词重叠）

        Args:
            response: 生成的响应
            documents: 检索到的文档

        Returns:
            一致性检查结果字典
        """
        # 提取响应关键词
        response_words = set(re.findall(r'\w+', response.lower()))

        # 提取文档关键词
        doc_words = set()
        for doc in documents:
            doc_words.update(re.findall(r'\w+', doc.page_content.lower()))

        # 计算 Jaccard 相似度
        if not response_words or not doc_words:
            return {"is_consistent": True, "score": 0.5}

        intersection = response_words & doc_words
        union = response_words | doc_words
        jaccard = len(intersection) / len(union) if union else 0.0

        # Jaccard 相似度通常较低，使用调整后的阈值
        adjusted_threshold = 0.1

        return {
            "is_consistent": jaccard >= adjusted_threshold,
            "score": jaccard,
        }

    def _calculate_confidence(
        self,
        response: str,
        documents: List[Document],
        is_safe: bool,
        is_consistent: bool,
    ) -> float:
        """
        计算置信度

        Args:
            response: 生成的响应
            documents: 检索到的文档
            is_safe: 是否安全
            is_consistent: 是否一致

        Returns:
            置信度分数 (0.0-1.0)
        """
        confidence = 1.0

        # 安全性影响
        if not is_safe:
            confidence *= 0.3

        # 一致性影响
        if not is_consistent:
            confidence *= 0.5

        # 响应长度影响（太短或太长降低置信度）
        if len(response) < 10:
            confidence *= 0.7
        elif len(response) > 2000:
            confidence *= 0.9

        # 文档数量影响
        if documents:
            confidence *= min(1.0, 0.7 + 0.1 * len(documents))
        else:
            confidence *= 0.6  # 没有检索文档时降低置信度

        return round(confidence, 2)

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

    def add_unsafe_pattern(self, pattern: str, message: str):
        """
        添加不安全模式

        Args:
            pattern: 正则表达式模式
            message: 匹配时的警告消息
        """
        compiled = re.compile(pattern, re.IGNORECASE)
        self._compiled_patterns.append((compiled, message))

    def add_unsafe_keyword(self, keyword: str):
        """
        添加不安全关键词

        Args:
            keyword: 关键词
        """
        if keyword not in self.UNSAFE_KEYWORDS:
            self.UNSAFE_KEYWORDS.append(keyword)

    def __repr__(self) -> str:
        return f"ResponseValidator(patterns={len(self._compiled_patterns)}, keywords={len(self.UNSAFE_KEYWORDS)})"


def get_response_validator(
    embedding_model: Optional[EmbeddingModel] = None,
) -> ResponseValidator:
    """
    获取响应校验器实例

    Args:
        embedding_model: 嵌入模型实例

    Returns:
        ResponseValidator 实例
    """
    return ResponseValidator(embedding_model=embedding_model)