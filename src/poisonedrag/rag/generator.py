"""
RAG 生成器模块

基于 LangGraph 构建对话链，实现检索增强生成。
集成安全校验器对生成结果进行验证。
"""

from typing import List, Optional, Dict, Any, TypedDict, Annotated
from dataclasses import dataclass, field
import operator

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END

from ..llm import LLMModel, get_llm
from .retriever import Retriever, RetrievalResult


# 定义状态类型
class RAGState(TypedDict):
    """RAG 状态定义"""
    query: str  # 用户查询
    documents: List[Document]  # 检索到的文档
    context: str  # 格式化的上下文
    response: str  # 生成的响应
    history: List[Dict[str, str]]  # 对话历史
    is_safe: bool  # 是否通过安全检查
    warnings: List[str]  # 安全警告
    confidence: float  # 置信度


@dataclass
class GenerationResult:
    """
    生成结果数据类

    包含生成的响应及相关元信息。
    """
    response: str
    documents: List[Document]
    is_safe: bool = True
    warnings: List[str] = field(default_factory=list)
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "response": self.response,
            "documents": [
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in self.documents
            ],
            "is_safe": self.is_safe,
            "warnings": self.warnings,
            "confidence": self.confidence,
        }


class Generator:
    """
    RAG 生成器类

    使用 LangGraph 构建对话链，实现检索增强生成。
    """

    # 系统提示词模板
    SYSTEM_PROMPT = """你是一个有帮助的AI助手。请基于提供的上下文信息回答用户问题。
如果上下文中没有相关信息，请诚实地说你不知道，不要编造答案。
回答要简洁、准确、有帮助。

上下文信息：
{context}

请回答用户的问题。"""

    def __init__(
        self,
        llm: Optional[LLMModel] = None,
        retriever: Optional[Retriever] = None,
        response_validator: Optional[Any] = None,  # ResponseValidator 实例
        system_prompt: Optional[str] = None,
    ):
        """
        初始化生成器

        Args:
            llm: LLM 模型实例
            retriever: 检索器实例
            response_validator: 响应校验器实例
            system_prompt: 自定义系统提示词
        """
        self.llm = llm or get_llm()
        self.retriever = retriever
        self.response_validator = response_validator
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT

        # 构建 LangGraph 工作流
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """构建 LangGraph 工作流"""
        # 定义节点函数
        def retrieve_node(state: RAGState) -> Dict[str, Any]:
            """检索节点"""
            if not self.retriever:
                return {"documents": [], "context": ""}

            result = self.retriever.retrieve(state["query"])
            context = self.retriever._format_documents(result.documents)

            return {
                "documents": result.documents,
                "context": context,
                "warnings": result.warnings,
            }

        def generate_node(state: RAGState) -> Dict[str, Any]:
            """生成节点"""
            # 构建系统提示
            system_content = self.system_prompt.format(context=state["context"])

            # 构建消息
            messages = self.llm.create_messages(
                query=state["query"],
                system_prompt=system_content,
                history=state.get("history", []),
            )

            # 调用 LLM
            response = self.llm.invoke(messages)

            return {"response": response.content}

        def validate_node(state: RAGState) -> Dict[str, Any]:
            """校验节点"""
            if not self.response_validator:
                return {"is_safe": True, "confidence": 1.0}

            # 执行校验
            validation_result = self.response_validator.validate(
                query=state["query"],
                response=state["response"],
                documents=state["documents"],
            )

            return {
                "is_safe": validation_result.is_safe,
                "warnings": state.get("warnings", []) + validation_result.warnings,
                "confidence": validation_result.confidence,
            }

        def safe_fallback_node(state: RAGState) -> Dict[str, Any]:
            """安全兜底节点：当校验失败时替换响应"""
            fallback = (
                "⚠️ 检测到该回答可能存在安全风险或事实不一致，已为您拦截原始响应。"
                "建议结合其他可靠来源验证该问题的答案。"
            )
            return {
                "response": fallback,
                "warnings": state.get("warnings", []) + [
                    "原始响应未通过安全校验，已替换为兜底提示。"
                ],
            }

        # 条件路由：校验失败时走安全兜底
        def route_after_validate(state: RAGState) -> str:
            if state.get("is_safe", True):
                return "end"
            return "block"

        # 构建图
        workflow = StateGraph(RAGState)

        # 添加节点
        workflow.add_node("retrieve", retrieve_node)
        workflow.add_node("generate", generate_node)
        workflow.add_node("validate", validate_node)
        workflow.add_node("safe_fallback", safe_fallback_node)

        # 定义边
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "validate")
        workflow.add_conditional_edges(
            "validate",
            route_after_validate,
            {"end": END, "block": "safe_fallback"},
        )
        workflow.add_edge("safe_fallback", END)

        return workflow.compile()

    def generate(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> GenerationResult:
        """
        执行 RAG 生成

        Args:
            query: 用户查询
            history: 对话历史

        Returns:
            GenerationResult 包含响应和元信息
        """
        # 初始状态
        initial_state: RAGState = {
            "query": query,
            "documents": [],
            "context": "",
            "response": "",
            "history": history or [],
            "is_safe": True,
            "warnings": [],
            "confidence": 1.0,
        }

        # 执行工作流
        final_state = self._graph.invoke(initial_state)

        return GenerationResult(
            response=final_state["response"],
            documents=final_state["documents"],
            is_safe=final_state["is_safe"],
            warnings=final_state["warnings"],
            confidence=final_state["confidence"],
        )

    def chat(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        简化版对话接口，直接返回响应文本

        Args:
            query: 用户查询
            history: 对话历史

        Returns:
            生成的响应文本
        """
        result = self.generate(query, history)
        return result.response

    def stream(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        流式生成（⚠️ 不经过安全校验，仅用于受信任场景）

        ⚠️ 安全警告：此方法跳过 validate_node，不安全的响应会被直接输出。
        如需安全的流式输出，请使用 `stream_safe()` 方法。

        Args:
            query: 用户查询
            history: 对话历史

        Yields:
            生成的文本片段
        """
        if not self.retriever:
            context = ""
        else:
            result = self.retriever.retrieve(query)
            context = self.retriever._format_documents(result.documents)

        # 构建系统提示
        system_content = self.system_prompt.format(context=context)

        # 构建消息
        messages = self.llm.create_messages(
            query=query,
            system_prompt=system_content,
            history=history,
        )

        # 流式调用 LLM
        for chunk in self.llm.llm.stream(messages):
            yield chunk.content

    def stream_safe(
        self,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
    ):
        """
        安全的流式生成：先非流式校验，通过后在流式输出

        流程：
        1. 非流式生成 + 安全校验
        2. 若校验通过，流式输出完整响应
        3. 若校验失败，流式输出安全警告

        Args:
            query: 用户查询
            history: 对话历史

        Yields:
            生成的文本片段（带安全保证）
        """
        # Step 1: 非流式生成 + 校验
        result = self.generate(query, history)

        # Step 2: 根据校验结果决定是否流式输出
        if result.is_safe:
            # 安全：逐字流式输出
            for char in result.response:
                yield char
        else:
            # 不安全：流式输出安全警告
            warning = (
                "⚠️ 该回答未通过安全校验，原始响应已被拦截。"
                "建议结合其他可靠来源验证答案。"
            )
            for char in warning:
                yield char


def create_generator(
    llm: Optional[LLMModel] = None,
    retriever: Optional[Retriever] = None,
    response_validator: Optional[Any] = None,
) -> Generator:
    """
    创建生成器实例

    Args:
        llm: LLM 模型实例
        retriever: 检索器实例
        response_validator: 响应校验器实例

    Returns:
        Generator 实例
    """
    return Generator(
        llm=llm,
        retriever=retriever,
        response_validator=response_validator,
    )