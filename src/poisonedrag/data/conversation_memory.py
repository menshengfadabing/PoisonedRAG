"""
对话记忆管理模块

将会话历史记录持久化到本地文件系统，支持多会话管理。
存储路径: data/conversations/
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional


# 对话存储根目录
_conversations_dir = Path(__file__).resolve().parent.parent.parent.parent / "data" / "conversations"


def _ensure_dir():
    """确保存储目录存在"""
    _conversations_dir.mkdir(parents=True, exist_ok=True)


def _session_file(session_id: str) -> Path:
    """获取会话文件路径"""
    return _conversations_dir / f"{session_id}.json"


def create_session(title: str = "新对话") -> str:
    """
    创建新会话

    Args:
        title: 会话标题

    Returns:
        会话 ID
    """
    _ensure_dir()
    session_id = str(uuid.uuid4())[:8]
    session_data = {
        "id": session_id,
        "title": title,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": [],
    }
    with open(_session_file(session_id), "w", encoding="utf-8") as f:
        json.dump(session_data, f, ensure_ascii=False, indent=2)
    return session_id


def load_session(session_id: str) -> Optional[Dict[str, Any]]:
    """
    加载会话

    Args:
        session_id: 会话 ID

    Returns:
        会话数据字典，不存在则返回 None
    """
    filepath = _session_file(session_id)
    if not filepath.exists():
        return None
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def save_session(session_id: str, data: Dict[str, Any]) -> bool:
    """
    保存会话

    Args:
        session_id: 会话 ID
        data: 会话数据

    Returns:
        是否保存成功
    """
    try:
        data["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        filepath = _session_file(session_id)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception:
        return False


def add_message(session_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> bool:
    """
    添加消息到会话

    Args:
        session_id: 会话 ID
        role: 消息角色 (user / assistant)
        content: 消息内容
        metadata: 附加元数据

    Returns:
        是否添加成功
    """
    session = load_session(session_id)
    if session is None:
        return False

    msg = {
        "role": role,
        "content": content,
        "metadata": metadata or {},
    }
    session["messages"].append(msg)
    session["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return save_session(session_id, session)


def list_sessions(limit: int = 50) -> List[Dict[str, Any]]:
    """
    列出所有会话（按更新时间倒序）

    Args:
        limit: 最大返回数量

    Returns:
        会话摘要列表
    """
    _ensure_dir()
    sessions = []
    for filepath in sorted(_conversations_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            sessions.append({
                "id": data.get("id", filepath.stem),
                "title": data.get("title", "新对话"),
                "created_at": data.get("created_at", ""),
                "updated_at": data.get("updated_at", ""),
                "message_count": len(data.get("messages", [])),
            })
        except Exception:
            continue
        if len(sessions) >= limit:
            break
    return sessions


def delete_session(session_id: str) -> bool:
    """删除会话"""
    filepath = _session_file(session_id)
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def clear_all_sessions() -> int:
    """清空所有会话，返回删除数量"""
    _ensure_dir()
    count = 0
    for filepath in _conversations_dir.glob("*.json"):
        filepath.unlink()
        count += 1
    return count
