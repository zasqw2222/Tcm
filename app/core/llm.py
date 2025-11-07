import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Iterator, Any, Callable, Optional
load_dotenv(find_dotenv(), override=True)


class LLMConfig(BaseModel):
    """LLM配置类，用于定义语言模型的配置参数"""
    model_name: str = Field(default=os.getenv(
        "MODEL_NAME"), description="模型名称，指定要使用的语言模型")
    temperature: float = Field(
        default=0.7, ge=0.0, le=2.0, description="温度参数，控制生成文本的随机性，0表示完全确定性")
    max_tokens: int = Field(default=4096, ge=1, le=32768,
                            description="最大token长度，控制生成文本的最大长度")
    base_url: Optional[str] = Field(default=os.getenv(
        "MODEL_BASE_URL"), description="基础URL")
    api_key: Optional[str] = Field(default=os.getenv(
        "MODEL_KEY"), description="API密钥")
    top_p: Optional[float] = Field(
        default=0.8, ge=0.0, le=1.0, description="核采样参数，控制生成文本的多样性")
    presence_penalty: Optional[float] = Field(
        default=0.0, ge=0.0, le=2.0, description="存在惩罚参数，控制生成文本的新颖性")
    extra_body: Optional[dict] = Field(default={
        "chat_template_kwargs": {"enable_thinking": False}
    }, description="额外请求体")


class LLM:
    def __init__(self, config: LLMConfig):
        """
        初始化LLM类，加载配置，并实例化底层的ChatOpenAI模型。
        :param config: LLMConfig，语言模型的相关配置
        """
        self.config = config  # 保存传入的配置对象
        # 创建一个ChatOpenAI实例，设置相关参数
        self.model = ChatOpenAI(
            model=self.config.model_name,  # 模型名称
            temperature=self.config.temperature,  # 生成文本的随机性
            max_tokens=self.config.max_tokens,  # 响应最大 token 数
            openai_api_base=self.config.base_url,  # API基础URL
            openai_api_key=self.config.api_key,  # API密钥
            extra_body=self.config.extra_body,  # 其他body参数如模板参数
            top_p=self.config.top_p,  # nucleus采样参数
            presence_penalty=self.config.presence_penalty,  # 存在惩罚参数
        )

    def generate(self, prompt: str) -> str:
        """
        以同步方式返回模型生成的结果。
        :param prompt: 输入的文本提示
        :return: 生成的文本字符串
        """
        return self.model.invoke(prompt)

    def stream(self, prompt: str) -> Iterator[str]:
        """
        以流式（增量）方式返回模型生成的内容迭代器。
        :param prompt: 输入的文本提示
        :return: 生成文本的迭代器，每次迭代返回部分字符串
        """
        return self.model.stream(prompt)
