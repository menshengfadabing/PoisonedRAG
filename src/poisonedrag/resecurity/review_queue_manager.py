"""
审核队列管理器模块

管理文档审查队列、人工审核流程和审查日志。
持久化存储待审核文档和审查历史。
"""

import json
import os
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

from langchain_core.documents import Document

from .document_reviewer import ChunkReviewResult


class ReviewStatus(Enum):
    """审核状态"""
    PENDING = "pending"           # 待审核
    APPROVED = "approved"         # 已通过
    REJECTED = "rejected"         # 已拒绝
    AUTO_APPROVED = "auto_approved"  # 自动通过


@dataclass
class QueuedDocument:
    """队列中的待审核文档"""
    id: str                              # 唯一标识
    content: str                         # 文档内容
    risk_score: float                    # 风险分数
    risk_type: Optional[str] = None      # 风险类型
    reason: Optional[str] = None         # 风险原因
    source: str = ""                     # 来源文件
    created_at: str = ""                 # 入队时间
    status: ReviewStatus = ReviewStatus.PENDING

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "risk_score": self.risk_score,
            "risk_type": self.risk_type,
            "reason": self.reason,
            "source": self.source,
            "created_at": self.created_at,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QueuedDocument":
        """从字典创建"""
        return cls(
            id=data["id"],
            content=data["content"],
            risk_score=data["risk_score"],
            risk_type=data.get("risk_type"),
            reason=data.get("reason"),
            source=data.get("source", ""),
            created_at=data.get("created_at", ""),
            status=ReviewStatus(data.get("status", "pending")),
        )


@dataclass
class ReviewLog:
    """审查日志记录"""
    timestamp: str
    doc_id: str
    action: str                    # auto_approved, approved, rejected
    risk_score: float
    risk_type: Optional[str] = None
    reason: Optional[str] = None
    operator: str = "system"       # system 或用户标识
    comment: str = ""              # 人工审核时的备注

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ReviewQueueManager:
    """
    审核队列管理器

    负责：
    - 管理待审核文档队列
    - 记录审查日志
    - 持久化存储
    """

    def __init__(self, data_dir: Optional[str] = None):
        """
        初始化队列管理器

        Args:
            data_dir: 数据存储目录
        """
        self.data_dir = data_dir or self._get_default_data_dir()
        self.queue_file = os.path.join(self.data_dir, "review_queue.json")
        self.log_file = os.path.join(self.data_dir, "review_logs.json")

        # 确保目录存在
        os.makedirs(self.data_dir, exist_ok=True)

        # 加载持久化数据
        self._queue: List[QueuedDocument] = self._load_queue()
        self._logs: List[ReviewLog] = self._load_logs()

    def _get_default_data_dir(self) -> str:
        """获取默认数据目录"""
        project_root = Path(__file__).parent.parent.parent.parent
        return str(project_root / "data" / "review")

    def _load_queue(self) -> List[QueuedDocument]:
        """加载待审核队列"""
        if not os.path.exists(self.queue_file):
            return []

        try:
            with open(self.queue_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [QueuedDocument.from_dict(item) for item in data]
        except Exception:
            return []

    def _save_queue(self):
        """保存待审核队列"""
        with open(self.queue_file, 'w', encoding='utf-8') as f:
            json.dump(
                [doc.to_dict() for doc in self._queue],
                f,
                ensure_ascii=False,
                indent=2
            )

    def _load_logs(self) -> List[ReviewLog]:
        """加载审查日志"""
        if not os.path.exists(self.log_file):
            return []

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return [ReviewLog(**item) for item in data]
        except Exception:
            return []

    def _save_logs(self):
        """保存审查日志"""
        with open(self.log_file, 'w', encoding='utf-8') as f:
            json.dump(
                [log.to_dict() for log in self._logs],
                f,
                ensure_ascii=False,
                indent=2
            )

    def add_to_queue(
        self,
        review_result: ChunkReviewResult,
        source: str = "",
    ) -> QueuedDocument:
        """
        将审查结果添加到待审核队列

        Args:
            review_result: 审查结果
            source: 来源文件名

        Returns:
            QueuedDocument 队列文档
        """
        doc_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{review_result.chunk_id}"

        queued_doc = QueuedDocument(
            id=doc_id,
            content=review_result.content,
            risk_score=review_result.risk_score,
            risk_type=review_result.risk_type,
            reason=review_result.reason,
            source=source,
            created_at=datetime.now().isoformat(),
            status=ReviewStatus.PENDING,
        )

        self._queue.append(queued_doc)
        self._save_queue()

        return queued_doc

    def add_batch_to_queue(
        self,
        review_results: List[ChunkReviewResult],
        source: str = "",
    ) -> List[QueuedDocument]:
        """
        批量添加审查结果到队列

        Args:
            review_results: 审查结果列表
            source: 来源文件名

        Returns:
            QueuedDocument 列表
        """
        queued_docs = []
        for result in review_results:
            doc = self.add_to_queue(result, source)
            queued_docs.append(doc)
        return queued_docs

    def get_pending_queue(self) -> List[QueuedDocument]:
        """获取待审核队列"""
        return [doc for doc in self._queue if doc.status == ReviewStatus.PENDING]

    def get_all_queue(self) -> List[QueuedDocument]:
        """获取所有队列文档"""
        return self._queue.copy()

    def get_by_id(self, doc_id: str) -> Optional[QueuedDocument]:
        """根据 ID 获取文档"""
        for doc in self._queue:
            if doc.id == doc_id:
                return doc
        return None

    def approve(self, doc_id: str, operator: str = "user", comment: str = "") -> bool:
        """
        批准入库

        Args:
            doc_id: 文档 ID
            operator: 操作者
            comment: 备注

        Returns:
            是否成功
        """
        doc = self.get_by_id(doc_id)
        if not doc:
            return False

        doc.status = ReviewStatus.APPROVED
        self._save_queue()

        # 记录日志
        self._add_log(ReviewLog(
            timestamp=datetime.now().isoformat(),
            doc_id=doc_id,
            action="approved",
            risk_score=doc.risk_score,
            risk_type=doc.risk_type,
            reason=doc.reason,
            operator=operator,
            comment=comment,
        ))

        return True

    def reject(self, doc_id: str, operator: str = "user", comment: str = "") -> bool:
        """
        拒绝入库

        Args:
            doc_id: 文档 ID
            operator: 操作者
            comment: 拒绝原因

        Returns:
            是否成功
        """
        doc = self.get_by_id(doc_id)
        if not doc:
            return False

        doc.status = ReviewStatus.REJECTED
        self._save_queue()

        # 记录日志
        self._add_log(ReviewLog(
            timestamp=datetime.now().isoformat(),
            doc_id=doc_id,
            action="rejected",
            risk_score=doc.risk_score,
            risk_type=doc.risk_type,
            reason=doc.reason,
            operator=operator,
            comment=comment,
        ))

        return True

    def auto_approve(self, doc_id: str) -> bool:
        """
        自动批准（低风险文档）

        Args:
            doc_id: 文档 ID

        Returns:
            是否成功
        """
        doc = self.get_by_id(doc_id)
        if not doc:
            return False

        doc.status = ReviewStatus.AUTO_APPROVED
        self._save_queue()

        # 记录日志
        self._add_log(ReviewLog(
            timestamp=datetime.now().isoformat(),
            doc_id=doc_id,
            action="auto_approved",
            risk_score=doc.risk_score,
            risk_type=doc.risk_type,
            reason=doc.reason,
            operator="system",
            comment="风险分数低于阈值，自动通过",
        ))

        return True

    def _add_log(self, log: ReviewLog):
        """添加日志记录"""
        self._logs.append(log)
        self._save_logs()

    def get_logs(
        self,
        action_filter: Optional[str] = None,
        limit: int = 100,
    ) -> List[ReviewLog]:
        """
        获取审查日志

        Args:
            action_filter: 动作过滤 (auto_approved, approved, rejected)
            limit: 最大数量

        Returns:
            日志列表
        """
        logs = self._logs.copy()

        if action_filter:
            logs = [log for log in logs if log.action == action_filter]

        # 按时间倒序
        logs.sort(key=lambda x: x.timestamp, reverse=True)

        return logs[:limit]

    def get_approved_documents(self) -> List[QueuedDocument]:
        """获取所有已批准的文档（包括自动批准）"""
        return [
            doc for doc in self._queue
            if doc.status in (ReviewStatus.APPROVED, ReviewStatus.AUTO_APPROVED)
        ]

    def clear_processed(self):
        """清除已处理的文档（已批准或已拒绝）"""
        self._queue = [
            doc for doc in self._queue
            if doc.status == ReviewStatus.PENDING
        ]
        self._save_queue()

    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        total = len(self._queue)
        pending = len([d for d in self._queue if d.status == ReviewStatus.PENDING])
        approved = len([d for d in self._queue if d.status == ReviewStatus.APPROVED])
        auto_approved = len([d for d in self._queue if d.status == ReviewStatus.AUTO_APPROVED])
        rejected = len([d for d in self._queue if d.status == ReviewStatus.REJECTED])

        return {
            "total": total,
            "pending": pending,
            "approved": approved,
            "auto_approved": auto_approved,
            "rejected": rejected,
            "logs_count": len(self._logs),
        }

    def to_documents(self, doc_ids: List[str]) -> List[Document]:
        """
        将批准的文档转换为 LangChain Document 对象

        Args:
            doc_ids: 文档 ID 列表

        Returns:
            Document 列表
        """
        documents = []
        for doc_id in doc_ids:
            doc = self.get_by_id(doc_id)
            if doc and doc.status in (ReviewStatus.APPROVED, ReviewStatus.AUTO_APPROVED):
                documents.append(Document(
                    page_content=doc.content,
                    metadata={
                        "source": doc.source,
                        "type": "knowledge",
                        "reviewed": True,
                        "risk_score": doc.risk_score,
                    }
                ))
        return documents


def create_review_queue_manager(data_dir: Optional[str] = None) -> ReviewQueueManager:
    """创建审核队列管理器实例"""
    return ReviewQueueManager(data_dir=data_dir)