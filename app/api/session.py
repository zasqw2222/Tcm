
from fastapi import HTTPException, APIRouter, status
from pydantic import BaseModel, Field
from typing import Optional
from app.core.llm import LLMConfig, MedicalConsultation, llm
from app.prompt.m_prompt import prompt
import uuid
from datetime import datetime
from typing import Dict


# prod 替换数据库
sessions: Dict[str, MedicalConsultation] = {}


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


class SessionCreate(BaseModel):
    """创建会话请求"""
    patient_info: PatientInfo
    llm_config: Optional[LLMConfig] = Field(
        default=None, description="LLM 配置参数")


class SessionResponse(BaseModel):
    """会话响应"""
    session_id: str = Field(..., description="会话ID")
    created_at: str = Field(..., description="创建时间")
    patient_name: str = Field(..., description="患者姓名")
    message: str = Field(default="会话创建成功")


session_router = APIRouter(tags=["Session路由"])


@session_router.post("/session", response_model=SessionResponse)
async def create_session(request: SessionCreate):
    """
    创建新的问诊会话

    - **patient_info**: 患者基本信息
    - 返回会话ID，后续请求需要使用此ID
    """
    try:
        # 生成唯一会话ID
        session_id = str(uuid.uuid4())

        # 创建问诊实例
        consultation = MedicalConsultation(llm, prompt)

        # 设置患者信息
        consultation.set_patient_info(
            disease=request.patient_info.disease,
            name=request.patient_info.name,
            age=request.patient_info.age,
            sex=request.patient_info.sex,
            tongue=request.patient_info.tongue,
            face=request.patient_info.face,
            left_pulse=request.patient_info.left_pulse,
            right_pulse=request.patient_info.right_pulse
        )

        # 应用 LLM 配置（如果提供）
        if request.llm_config:
            consultation.update_llm_config(
                temperature=request.llm_config.temperature,
                max_tokens=request.llm_config.max_tokens,
                presence_penalty=request.llm_config.presence_penalty,
                top_p=request.llm_config.top_p,
                top_k=request.llm_config.top_k
            )

        # 保存会话
        sessions[session_id] = consultation

        return SessionResponse(
            session_id=session_id,
            created_at=datetime.now().isoformat(),
            patient_name=request.patient_info.name,
            message="会话创建成功"
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"创建会话失败: {str(e)}"
        )


@session_router.get("/sessions/{session_id}/status", response_model=StatusResponse)
def get_session_status(session_id: str):
    """
    获取会话状态

    - **session_id**: 会话ID
    - 返回会话的当前状态信息
    """
    if session_id not in sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {session_id}"
        )

    try:
        consultation = sessions[session_id]
        status_info = consultation.get_status()

        return StatusResponse(
            session_id=session_id,
            **status_info
        )

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"获取状态失败: {str(e)}"
        )
