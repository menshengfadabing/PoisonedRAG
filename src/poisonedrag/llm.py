"""
对话模型模块

封装 DeepSeek API，提供对话生成功能。
使用 langchain_openai.ChatOpenAI 实现。
"""

from typing import Optional, List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage

from .config import get_config


class LLMModel:
    """
    对话模型封装类

    使用 DeepSeek API 进行对话生成。
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        初始化对话模型

        Args:
            api_key: DeepSeek API 密钥，默认从配置读取
            base_url: API 基础地址，默认从配置读取
            model: 模型名称，默认从配置读取
            temperature: 生成温度，控制随机性
            max_tokens: 最大生成 token 数
        """
        config = get_config()
        self.api_key = api_key or config.deepseek_api_key
        self.base_url = base_url or config.deepseek_base_url
        self.model_name = model or config.deepseek_model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 初始化 DeepSeek LLM（通过 OpenAI 兼容接口）
        self._llm = ChatOpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

    @property
    def llm(self) -> ChatOpenAI:
        """获取 LangChain LLM 实例"""
        return self._llm

    def invoke(
        self,
        messages: List[BaseMessage],
        **kwargs: Any
    ) -> BaseMessage:
        """
        调用模型生成回复

        Args:
            messages: 消息列表
            **kwargs: 其他参数

        Returns:
            模型回复消息
        """
        return self._llm.invoke(messages, **kwargs)

    def create_messages(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> List[BaseMessage]:
        """
        创建消息列表

        Args:
            query: 用户查询
            system_prompt: 系统提示词
            history: 对话历史 [{"role": "user/assistant", "content": "..."}]

        Returns:
            消息列表
        """
        messages = []

        # 添加系统提示
        if system_prompt:
            messages.append(SystemMessage(content=system_prompt))

        # 添加对话历史
        if history:
            for msg in history:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if role == "user":
                    messages.append(HumanMessage(content=content))
                elif role == "assistant":
                    messages.append(AIMessage(content=content))

        # 添加当前查询
        messages.append(HumanMessage(content=query))

        return messages

    def chat(
        self,
        query: str,
        system_prompt: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        简化版对话接口

        Args:
            query: 用户查询
            system_prompt: 系统提示词
            history: 对话历史

        Returns:
            模型回复文本
        """
        messages = self.create_messages(query, system_prompt, history)
        response = self.invoke(messages)
        return response.content

    def __repr__(self) -> str:
        return f"LLMModel(model='{self.model_name}', base_url='{self.base_url}')"


def get_llm() -> LLMModel:
    """获取 LLM 实例（使用全局配置）"""
    return LLMModel()


def get_review_llm() -> LLMModel:
    """
    获取文档审查专用的 LLM 实例

    使用独立的 API Key，避免与对话模型混用

    Returns:
        LLMModel 实例（配置为审查模式）
    """
    config = get_config()
    return LLMModel(
        api_key=config.review_api_key,
        base_url=config.deepseek_base_url,
        model=config.review_model,
        temperature=0.1,  # 低温度，更确定的输出
        max_tokens=4096,  # 更大的输出限制（批量审查需要）
    )