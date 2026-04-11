"""
文本分割器模块

封装 LangChain 的文本分割功能，支持多种分割策略。
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    CharacterTextSplitter,
)


@dataclass
class SplitConfig:
    """分割配置"""
    chunk_size: int = 500           # 块大小（字符数）
    chunk_overlap: int = 50         # 块重叠
    separator: str = "\n\n"         # 分隔符
    split_method: str = "recursive"  # 分割方法: recursive, character, markdown


class TextSplitter:
    """
    文本分割器

    支持多种分割策略：
    - recursive: 递归字符分割（默认）
    - character: 简单字符分割
    - markdown: Markdown 标题分割
    """

    def __init__(self, config: Optional[SplitConfig] = None):
        """
        初始化分割器

        Args:
            config: 分割配置
        """
        self.config = config or SplitConfig()
        self._splitter = self._create_splitter()

    def _create_splitter(self):
        """创建 LangChain 分割器实例"""
        if self.config.split_method == "recursive":
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
                length_function=len,
            )
        elif self.config.split_method == "character":
            return CharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separator=self.config.separator,
            )
        elif self.config.split_method == "markdown":
            # Markdown 分割器需要配合其他分割器使用
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " ", ""],
            )
        else:
            return RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
            )

    def split_text(self, text: str) -> List[str]:
        """
        分割文本

        Args:
            text: 待分割文本

        Returns:
            文本块列表
        """
        if not text or not text.strip():
            return []

        # 清理空白：将连续空白（包括换行、制表符）压缩为单个空格
        # 避免 PDF/DOCX 提取时的多余换行导致异常分割
        import re
        text = re.sub(r'\s+', ' ', text).strip()

        if not text:
            return []

        chunks = self._splitter.split_text(text)

        # 过滤过小的块（少于 5 字符的纯标题/标签通常无语义价值）
        chunks = [c for c in chunks if len(c.strip()) >= 5]

        return chunks

    def split_texts(self, texts: List[str]) -> List[str]:
        """
        批量分割文本

        Args:
            texts: 文本列表

        Returns:
            所有文本块的列表
        """
        all_chunks = []
        for text in texts:
            chunks = self.split_text(text)
            all_chunks.extend(chunks)
        return all_chunks

    def split_file(
        self,
        file_path: str,
        encoding: str = "utf-8",
    ) -> List[str]:
        """
        分割文件

        Args:
            file_path: 文件路径
            encoding: 文件编码

        Returns:
            文本块列表
        """
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            return self.split_text(content)
        except Exception as e:
            print(f"读取文件失败: {e}")
            return []

    def split_markdown_by_headers(
        self,
        text: str,
        headers_to_split_on: Optional[List[tuple]] = None,
    ) -> List[Dict[str, Any]]:
        """
        按 Markdown 标题分割

        Args:
            text: Markdown 文本
            headers_to_split_on: 要分割的标题级别

        Returns:
            包含内容和元数据的字典列表
        """
        if headers_to_split_on is None:
            headers_to_split_on = [
                ("#", "header1"),
                ("##", "header2"),
                ("###", "header3"),
            ]

        try:
            md_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=headers_to_split_on
            )
            documents = md_splitter.split_text(text)
            return [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in documents
            ]
        except Exception:
            # Markdown 分割失败时回退到普通分割
            return [{"content": chunk, "metadata": {}} for chunk in self.split_text(text)]


def create_text_splitter(
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    split_method: str = "recursive",
) -> TextSplitter:
    """
    创建文本分割器

    Args:
        chunk_size: 块大小
        chunk_overlap: 块重叠
        split_method: 分割方法

    Returns:
        TextSplitter 实例
    """
    config = SplitConfig(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        split_method=split_method,
    )
    return TextSplitter(config)