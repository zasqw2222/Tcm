from pydantic import BaseModel, Field
from typing import Optional
from app.core.llm import MedicalConsultation, llm
from langchain_core.messages import HumanMessage, AIMessage
from app.prompt.m_prompt import prompt
import uuid

from typing import Dict, List

# prod 替换数据库
# sessions: Dict[str, MedicalConsultation] = {}


class PatientInfo(BaseModel):
    """患者信息"""
    disease: str = Field(..., description="主诉疾病，如：头痛、胃痛")
    name: str = Field(..., description="患者姓名")
    age: str = Field(..., description="年龄")
    sex: str = Field(..., description="性别（男/女）")
    tongue: str = Field(default="未查", description="舌象")
    face: str = Field(default="未查", description="面象")
    left_pulse: str = Field(default="未查", description="左手脉象")
    right_pulse: str = Field(default="未查", description="右手脉象")

    class Config:
        json_schema_extra = {
            "example": {
                "disease": "头痛",
                "name": "张三",
                "age": "35",
                "sex": "男",
                "tongue": "舌红苔薄白",
                "face": "面色正常",
                "left_pulse": "脉弦",
                "right_pulse": "脉弦"
            }
        }


class StatusResponse(BaseModel):
    """状态响应"""
    session_id: str = Field(..., description="会话ID")
    patient: Optional[str] = Field(None, description="患者姓名")
    round: int = Field(..., description="对话轮次")
    message_count: int = Field(..., description="消息数量")


class Sessions:
    def __init__(self):
        self.sessions: Dict[str, MedicalConsultation] = {}

    def create_session(self, patient: PatientInfo) -> str:
        """
        创建一个新的会话。

        参数:
            patient (PatientInfo): 患者的基本信息。

        返回:
            str: 新创建的会话ID。

        异常:
            ValueError: 当患者信息为空时抛出。
            RuntimeError: 创建会话过程中发生异常时抛出。
        """
        if patient is None:
            raise ValueError("患者信息不能为空")

        try:
            # 生成唯一会话ID
            session_id = str(uuid.uuid4())

            # 创建问诊实例
            consultation = MedicalConsultation(llm, prompt)

            # 设置患者信息
            consultation.set_patient_info(
                disease=patient.disease,
                name=patient.name,
                age=patient.age,
                sex=patient.sex,
                tongue=patient.tongue,
                face=patient.face,
                left_pulse=patient.left_pulse,
                right_pulse=patient.right_pulse
            )
            # 保存会话
            self.sessions[session_id] = consultation

            return session_id

        except Exception as e:
            raise RuntimeError(f"创建会话失败: {str(e)}") from e

    def get_session_status(self, session_id: str) -> StatusResponse:
        """
        获取指定 session_id 的会话状态。

        参数:
            session_id (str): 会话ID。

        返回:
            StatusResponse: 包含会话状态的响应对象。

        异常:
            ValueError: 当会话ID不存在时抛出。
            RuntimeError: 获取状态过程中发生异常时抛出。
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        try:
            consultation = self.sessions[session_id]
            status_info = consultation.get_status()

            return StatusResponse(
                session_id=session_id,
                **status_info
            )
        except Exception as e:
            raise RuntimeError(f"获取会话状态失败: {str(e)}") from e

    def reset_session(self, session_id: str) -> str:
        """
        重置指定 session_id 的会话（即清空对话历史并重置轮次）。

        参数:
            session_id (str): 会话ID。

        返回:
            dict: 包含重置结果信息和会话ID。

        异常:
            ValueError: 如果会话不存在或重置失败时抛出。
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")

        try:
            consultation = self.sessions[session_id]
            consultation.reset()

            return session_id

        except Exception as e:
            raise ValueError(f"重置对话失败: {str(e)}") from e

    def get_medicalConsultation(self, session_id: str) -> MedicalConsultation:
        """
        获取指定 session_id 的会话对象。

        参数:
            session_id (str): 会话ID。

        返回:
            MedicalConsultation: 会话对象。

        异常:
            ValueError: 如果会话不存在时抛出。
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        try:
            return self.sessions[session_id]
        except Exception as e:
            raise ValueError(f"获取会话对象失败: {str(e)}") from e

    def get_conversation_history(self, session_id: str) -> List[Dict[str, str]]:
        """
        获取格式化的对话历史

        Returns:
            [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
        """
        if session_id not in self.sessions:
            raise ValueError(f"会话不存在: {session_id}")
        try:
            consultation = self.sessions[session_id]
            return consultation.get_conversation_history()
        except Exception as e:
            raise ValueError(f"获取对话历史失败: {str(e)}") from e
