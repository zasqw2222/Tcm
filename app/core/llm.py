import os
from dotenv import load_dotenv, find_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Iterator, Any, Callable, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Optional
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


llm = ChatOpenAI(
    model=os.getenv('MODEL_NAME'),
    openai_api_base=os.getenv('MODEL_BASE_URL'),
    openai_api_key=os.getenv('MODEL_KEY'),
    temperature=os.getenv('MODEL_TEMPERATURE'),
    max_tokens=os.getenv('MODEL_MAX_TOKEN'),
    presence_penalty=os.getenv('MODEL_PRESENCE_PENALTY'),
    top_p=os.getenv('MODEL_TOP_P'),
    extra_body={
        "top_k": os.getenv('MODEL_TOP_K'),
        "chat_template_kwargs": {"enable_thinking": os.getenv('MODEL_IS_THINK')},
    },

)


class MedicalConsultation:
    def __init__(self, llm, system_prompt):
        """"""
        self.llm = llm
        self.prompt_template = PromptTemplate.from_template(system_prompt)
        self.messages = []
        self.patient_info = None
        self.round_count = 0

    def set_patient_info(self, disease: str, name: str, age: str, sex: str,
                         tongue: str = "未查", face: str = "未查",
                         left_pulse: str = "未查", right_pulse: str = "未查"):
        """
        设置患者基本信息

        Args:
            disease: 主诉疾病（如"头痛"、"胃痛"）
            name: 患者姓名
            age: 年龄
            sex: 性别（男/女）
            tongue: 舌象（可选）
            face: 面象（可选）
            left_pulse: 左手脉象（可选）
            right_pulse: 右手脉象（可选）
        """
        self.patient_info = {
            "disease": disease,
            "name": name,
            "age": age,
            "sex": sex,
            "tongueFront": tongue,
            "face": face,
            "leftPulse": left_pulse,
            "rightPulse": right_pulse
        }
        return self

    def _format_system_prompt(self) -> str:
        """格式化系统提示词"""
        if not self.patient_info:
            raise ValueError("请先使用 set_patient_info() 设置患者信息")

        return self.prompt_template.format(**self.patient_info)

    def reset(self):
        """重置对话（开始新的问诊）"""
        self.messages = []
        self.round_count = 0
        return self

    def get_status(self) -> Dict:
        """获取当前状态"""
        return {
            "patient": self.patient_info.get("name") if self.patient_info else None,
            "round": self.round_count,
            "message_count": len(self.messages)
        }

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        获取格式化的对话历史

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        history = []
        for msg in self.messages:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                history.append({"role": "assistant", "content": msg.content})
        return history

    def invoke(self, user_message: str) -> str:
        """
        发送用户消息并获取AI回复

        Args:
            user_message: 患者的回答或提问

        Returns:
            医生的回复
        """

        # 添加用户消息到历史
        self.messages.append(HumanMessage(content=f'{user_message}'))
        self.round_count += 1

        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._format_system_prompt()),
            MessagesPlaceholder(variable_name="history")
        ])

        # 调用模型
        try:
            chain = chat_template | self.llm
            response = chain.invoke({"history": self.messages})

        except Exception as e:
            print(f"模型调用失败: {e}")
            return "抱歉，模型暂时无法响应。"

        content = response.content if hasattr(response, 'content') else ""
        response = AIMessage(content)

        self.messages.append(response)

        # 返回字符串内容
        return response.content

    async def stream(self, user_message: str):
        """
        发送用户消息并流式获取AI回复（异步生成器）

        Args:
            user_message: 患者的回答或提问

        Yields:
            AI回复的文本片段
        """
        # 添加用户消息到历史
        self.messages.append(HumanMessage(content=f'{user_message}'))
        self.round_count += 1

        # 创建聊天模板
        chat_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self._format_system_prompt()),
            MessagesPlaceholder(variable_name="history")
        ])

        # 调用模型（流式）
        try:
            chain = chat_template | self.llm
            full_response = ""

            # 流式输出
            async for chunk in chain.astream({"history": self.messages}):
                if hasattr(chunk, 'content'):
                    content = chunk.content
                    if content:
                        full_response += content
                        yield content

            # 添加完整回复到历史
            self.messages.append(AIMessage(content=full_response))

        except Exception as e:
            error_msg = "抱歉，模型暂时无法响应。"
            self.messages.append(AIMessage(content=error_msg))
            yield error_msg
