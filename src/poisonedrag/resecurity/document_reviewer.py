"""
文档审查器模块

使用 LLM 对入库文档进行安全审查，识别潜在风险内容。
支持批量审查和结构化输出。

优化版本：利用大模型长上下文窗口实现高效批量审查。
- 支持动态批次大小计算
- 配置参数可从 config.py 自定义
"""

import json
import re
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.messages import SystemMessage, HumanMessage

from ..config import get_config
from .rule_checker import RuleChecker, RuleCheckResult


class RiskType(Enum):
    """风险类型枚举"""
    MISINFORMATION = "虚假信息"
    MALICIOUS_CODE = "恶意代码"
    SENSITIVE_INFO = "敏感信息"
    PHISHING = "钓鱼欺诈"
    MISLEADING_ADVICE = "误导建议"
    DEPENDENCY_POISONING = "依赖投毒"
    FACT_INJECTION = "错误事实注入"
    HIDDEN_BACKDOOR = "隐蔽后门"
    OTHER = "其他风险"


@dataclass
class ChunkReviewResult:
    """单个文档块的审查结果"""
    chunk_id: int
    content: str
    risk_score: float = 0.0
    risk_type: Optional[str] = None
    reason: Optional[str] = None

    def is_safe(self, threshold: float = 0.5) -> bool:
        return self.risk_score < threshold


@dataclass
class BatchReviewResult:
    """批量审查结果"""
    results: List[ChunkReviewResult] = field(default_factory=list)
    raw_response: str = ""
    success: bool = True
    error_message: str = ""

    def get_safe_chunks(self, threshold: float = 0.5) -> List[ChunkReviewResult]:
        return [r for r in self.results if r.is_safe(threshold)]

    def get_risky_chunks(self, threshold: float = 0.5) -> List[ChunkReviewResult]:
        return [r for r in self.results if not r.is_safe(threshold)]


class DocumentReviewer:
    """
    文档审查器（优化版）

    利用大模型长上下文窗口实现高效批量审查。
    所有参数从配置文件读取，支持动态批次计算。
    """

    # 批量审查的系统提示（精简版）
    BATCH_SYSTEM_PROMPT = """你是文档安全审查员。批量审查文档，识别风险。

风险类型: misinformation/malicious_code/sensitive_info/phishing/misleading_advice/dependency_poisoning/fact_injection/hidden_backdoor

评分标准: 0.0=安全, ≥0.5=有风险, 0.8+=高风险

重点检查:
1. 可疑包名 (typosquatting)
2. 硬编码凭证
3. 危险命令
4. 认证旁路

返回 JSON 格式，每个文档一个结果。"""

    def __init__(
        self,
        llm=None,
        config=None,
        enable_rule_checker: bool = True,
    ):
        """
        初始化文档审查器

        Args:
            llm: LangChain LLM 实例
            config: 配置对象，为 None 时从全局配置获取
            enable_rule_checker: 是否启用规则检查器
        """
        self.llm = llm
        self.config = config or get_config()
        self.enable_rule_checker = enable_rule_checker
        self.rule_checker = RuleChecker() if enable_rule_checker else None

        # 从配置读取参数
        self._load_config()

    def _load_config(self):
        """从配置文件加载参数"""
        # 文档处理参数
        self.doc_max_length = self.config.review_doc_max_length
        self.doc_chunk_size = self.config.review_doc_chunk_size
        self.doc_chunk_overlap = self.config.review_doc_chunk_overlap

        # 批量处理参数
        self.min_batch_size = self.config.review_min_batch_size
        self.max_batch_size = self.config.review_max_batch_size
        self.max_workers = self.config.review_max_workers
        self.dynamic_batch = self.config.review_dynamic_batch

        # 计算批次大小
        if self.dynamic_batch:
            self.batch_size = self.config.calculate_max_batch_size(self.doc_max_length)
        else:
            self.batch_size = min(50, self.max_batch_size)

    def refresh_config(self):
        """刷新配置（当配置变更时调用）"""
        self._load_config()

    def get_batch_info(self) -> dict:
        """获取当前批次配置信息"""
        return self.config.get_review_batch_info(self.doc_max_length)

    def batch_quick_review(
        self,
        texts: List[str],
        use_parallel: bool = True,
    ) -> List[ChunkReviewResult]:
        """
        批量快速审查多个文本（高性能版本）

        利用长上下文窗口一次审查多个文档，大幅减少 API 调用次数。

        Args:
            texts: 待审查文本列表
            use_parallel: 是否使用并行处理

        Returns:
            ChunkReviewResult 列表
        """
        if not texts:
            return []

        results = [None] * len(texts)

        # 第一阶段：规则引擎预筛选
        rule_results = []
        if self.rule_checker:
            rule_results = [self.rule_checker.check(text) for text in texts]

        # 规则引擎直接判定为高风险的，跳过 LLM 审查
        llm_needed_indices = []
        for i, text in enumerate(texts):
            if rule_results and rule_results[i].risk_level == "high":
                # 规则引擎高风险，直接标记
                results[i] = ChunkReviewResult(
                    chunk_id=i,
                    content=text,
                    risk_score=0.75,
                    risk_type="依赖投毒" if rule_results[i].suspicious_packages else "高风险",
                    reason=f"规则引擎检测: {', '.join(rule_results[i].risk_indicators[:3])}"
                )
            else:
                llm_needed_indices.append(i)

        if not llm_needed_indices:
            return results

        # 第二阶段：批量 LLM 审查
        texts_to_review = [texts[i] for i in llm_needed_indices]

        if use_parallel and len(texts_to_review) > self.batch_size:
            # 并行处理多个批次
            batch_results = self._parallel_batch_review(texts_to_review)
        else:
            # 单线程批量处理
            batch_results = self._batch_review_single_thread(texts_to_review)

        # 合并结果
        for idx, result in zip(llm_needed_indices, batch_results):
            result.chunk_id = idx
            results[idx] = result

        # 填充 None（不应该发生，但做保护）
        for i, r in enumerate(results):
            if r is None:
                results[i] = ChunkReviewResult(
                    chunk_id=i,
                    content=texts[i],
                    risk_score=0.5,
                    risk_type="审查失败",
                    reason="结果缺失"
                )

        return results

    def _parallel_batch_review(self, texts: List[str]) -> List[ChunkReviewResult]:
        """
        并行批量审查

        将文本分成多个批次，并行调用 LLM。
        """
        batches = []
        for i in range(0, len(texts), self.batch_size):
            batches.append((i, texts[i:i + self.batch_size]))

        all_results = [None] * len(texts)

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._review_batch, start_idx, batch): start_idx
                for start_idx, batch in batches
            }

            for future in as_completed(futures):
                start_idx = futures[future]
                try:
                    batch_results = future.result()
                    for j, result in enumerate(batch_results):
                        all_results[start_idx + j] = result
                except Exception as e:
                    # 批次失败，标记为需要人工审核
                    batch = batches[[b[0] for b in batches].index(start_idx)][1]
                    for j, text in enumerate(batch):
                        all_results[start_idx + j] = ChunkReviewResult(
                            chunk_id=start_idx + j,
                            content=text,
                            risk_score=0.5,
                            risk_type="审查失败",
                            reason=str(e)
                        )

        return all_results

    def _batch_review_single_thread(self, texts: List[str]) -> List[ChunkReviewResult]:
        """单线程批量审查"""
        all_results = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._review_batch(i, batch)
            all_results.extend(batch_results)

        return all_results

    def _review_batch(
        self,
        start_id: int,
        chunks: List[str],
    ) -> List[ChunkReviewResult]:
        """
        审查单个批次（利用长上下文）
        """
        if not self.llm:
            return [
                ChunkReviewResult(
                    chunk_id=start_id + i,
                    content=chunk,
                    risk_score=0.5,
                    risk_type="LLM未初始化",
                    reason="LLM 未配置"
                )
                for i, chunk in enumerate(chunks)
            ]

        # 构建紧凑的批量提示，使用配置的文档最大长度
        chunks_text = "\n---\n".join([
            f"[{i+1}] {chunk[:self.doc_max_length]}{'...' if len(chunk) > self.doc_max_length else ''}"
            for i, chunk in enumerate(chunks)
        ])

        user_prompt = f"""审查以下 {len(chunks)} 个文档，返回 JSON 数组:

{chunks_text}

返回格式:
{{"results": [
  {{"id": 1, "score": 0.0, "type": null, "reason": null}},
  {{"id": 2, "score": 0.8, "type": "dependency_poisoning", "reason": "可疑包名"}}
]}}

只返回 JSON，无其他内容。"""

        messages = [
            SystemMessage(content=self.BATCH_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        try:
            response = self.llm.invoke(messages)
            return self._parse_batch_response(response.content, chunks, start_id)
        except Exception as e:
            return [
                ChunkReviewResult(
                    chunk_id=start_id + i,
                    content=chunk,
                    risk_score=0.5,
                    risk_type="LLM调用失败",
                    reason=str(e)
                )
                for i, chunk in enumerate(chunks)
            ]

    def _parse_batch_response(
        self,
        raw_content: str,
        chunks: List[str],
        start_id: int,
    ) -> List[ChunkReviewResult]:
        """解析批量审查响应"""
        # 提取 JSON
        json_match = re.search(r'\{[\s\S]*\}', raw_content)
        if not json_match:
            return [
                ChunkReviewResult(
                    chunk_id=start_id + i,
                    content=chunk,
                    risk_score=0.5,
                    risk_type="解析失败",
                    reason="JSON 格式异常"
                )
                for i, chunk in enumerate(chunks)
            ]

        try:
            data = json.loads(json_match.group())
            results_list = data.get("results", [])

            # 构建 ID 到结果的映射
            result_map = {r.get("id"): r for r in results_list}

            parsed = []
            for i, chunk in enumerate(chunks):
                doc_id = i + 1
                if doc_id in result_map:
                    r = result_map[doc_id]
                    parsed.append(ChunkReviewResult(
                        chunk_id=start_id + i,
                        content=chunk,
                        risk_score=float(r.get("score", 0.5)),
                        risk_type=r.get("type"),
                        reason=r.get("reason"),
                    ))
                else:
                    parsed.append(ChunkReviewResult(
                        chunk_id=start_id + i,
                        content=chunk,
                        risk_score=0.5,
                        risk_type="未审查",
                        reason="LLM 未返回结果"
                    ))

            return parsed

        except (json.JSONDecodeError, ValueError):
            return [
                ChunkReviewResult(
                    chunk_id=start_id + i,
                    content=chunk,
                    risk_score=0.5,
                    risk_type="解析失败",
                    reason="JSON 解析错误"
                )
                for i, chunk in enumerate(chunks)
            ]

    def quick_review(self, text: str) -> ChunkReviewResult:
        """
        快速审查单个文本（兼容旧接口）

        注意：批量审查请使用 batch_quick_review 方法以获得更好性能。
        """
        results = self.batch_quick_review([text], use_parallel=False)
        return results[0] if results else ChunkReviewResult(
            chunk_id=0,
            content=text,
            risk_score=0.5,
            risk_type="审查失败",
            reason="无结果"
        )

    # 保留旧方法以兼容
    def review_chunks(
        self,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> BatchReviewResult:
        """
        批量审查文档块（兼容旧接口）
        """
        results = self.batch_quick_review(chunks, use_parallel=True)
        return BatchReviewResult(
            results=results,
            success=True
        )


def create_document_reviewer(llm=None) -> DocumentReviewer:
    """创建文档审查器实例"""
    return DocumentReviewer(llm=llm)