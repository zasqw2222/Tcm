
import logging
import json
import asyncio
from fastapi import HTTPException, APIRouter, status, FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from app.core.session import Sessions, PatientInfo
from contextlib import asynccontextmanager
from typing import Optional
from app.tools.res import UnifiedResponse, success, error
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


chat_router = APIRouter()
global_sessions = None


class StatusResponse(BaseModel):
    """状态响应"""
    session_id: str = Field(..., description="会话ID")
    patient: Optional[str] = Field(None, description="患者姓名")
    round: int = Field(..., description="对话轮次")
    message_count: int = Field(..., description="消息数量")


class StreamRequest(BaseModel):
    """流式请求"""
    session_id: str = Field(..., description="会话ID")
    user_message: str = Field(..., description="用户消息")


class InvokeRequest(BaseModel):
    """调用请求"""
    session_id: str = Field(..., description="会话ID")
    user_message: str = Field(..., description="用户消息")


class InvokeResponse(BaseModel):
    """调用响应"""
    session_id: str = Field(..., description="会话ID")
    user_message: str = Field(..., description="用户消息")
    response: str = Field(..., description="回复")
    round: int = Field(..., description="对话轮次")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：启动时初始化会话管理器 Sessions，关闭时清理资源

    该 lifespan 上下文管理器在 FastAPI 启动和关闭时自动调用，用于：
    - 应用启动时初始化全局 Sessions 实例
    - 应用关闭时自动处理 Session 相关资源释放

    如果初始化失败，将记录错误日志并抛出异常以终止服务。
    """
    print('应用生命周期管理：启动时初始化会话管理器 Sessions，关闭时清理资源')
    global global_sessions
    try:
        global_sessions = Sessions()
    except Exception as e:
        logger.error(f"应用生命周期管理失败: {str(e)}")
        raise
    yield  # 应用运行期间


@chat_router.post("/create_session", response_model=UnifiedResponse)
async def create_session(patient: PatientInfo) -> UnifiedResponse:
    """
    创建一个新的会话。

    Args:
        patient (PatientInfo): 患者的基本信息。

    Returns:
        str: 新创建的会话ID字符串。

    Raises:
        HTTPException: 创建会话失败时抛出，状态码500。
    """
    try:
        return success(global_sessions.create_session(patient))
    except Exception as e:
        logger.error(f"创建会话失败: {str(e)}")
        return error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"创建会话失败: {str(e)}")


@chat_router.get("/get_session_status", response_model=UnifiedResponse)
async def get_session_status(session_id: str) -> UnifiedResponse:
    """
    获取指定会话的状态。

    Args:
        session_id (str): 会话ID。

    Returns:
        StatusResponse: 包含会话状态的响应对象。
    """
    try:
        return success(global_sessions.get_session_status(session_id))
    except Exception as e:
        logger.error(f"获取会话状态失败: {str(e)}")
        return error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"获取会话状态失败: {str(e)}")


@chat_router.post("/reset_session", response_model=UnifiedResponse)
async def reset_session(session_id: str) -> UnifiedResponse:
    """
    重置指定会话。

    Args:
        session_id (str): 会话ID。

    Returns:
        str: 重置会话后的会话ID。

    Raises:
        HTTPException: 重置会话失败时抛出，状态码500。
    """
    try:
        return success(global_sessions.reset_session(session_id))
    except Exception as e:
        logger.error(f"重置会话失败: {str(e)}")
        return error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"重置会话失败: {str(e)}")


@chat_router.get("/get_conversation_history", response_model=UnifiedResponse)
async def get_conversation_history(session_id: str) -> UnifiedResponse:
    """
    获取指定会话的对话历史。

    Args:
        session_id (str): 会话ID。

    Returns:
        List[Dict[str, str]]: 对话历史列表。
    """

    if session_id not in global_sessions.sessions:
        return error(status.HTTP_404_NOT_FOUND, f"会话不存在: {session_id}")

    try:
        return success(global_sessions.get_conversation_history(session_id))
    except Exception as e:
        logger.error(f"获取对话历史失败: {str(e)}")
        return error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"获取对话历史失败: {str(e)}")


@chat_router.post("/stream", response_model=StreamRequest)
async def stream(request: StreamRequest) -> StreamingResponse:
    """
    发送消息进行问诊（流式输出）

    - **session_id**: 会话ID
    - **message**: 患者的回答或提问
    - 返回 Server-Sent Events 流式数据
    """
    # 检查会话是否存在
    try:
        consultation = global_sessions.get_medicalConsultation(
            request.session_id)
    except Exception as e:
        return error(status.HTTP_500_INTERNAL_SERVER_ERROR, f"获取会话失败: {str(e)}")

    async def generate():
        try:
            # 使用流式方式获取回复
            async for chunk in consultation.stream(request.message):
                # 发送 SSE 格式的数据
                data = json.dumps(
                    {"content": chunk, "done": False}, ensure_ascii=False)
                yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # 让出控制权

            # 发送完成信号
            final_data = json.dumps({
                "content": "",
                "done": True,
                "round": consultation.round_count
            }, ensure_ascii=False)
            yield f"data: {final_data}\n\n"

        except Exception as e:
            error_data = json.dumps({"error": str(e)}, ensure_ascii=False)
            yield f"data: {error_data}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@chat_router.post("/invoke", response_model=UnifiedResponse)
async def invoke(request: InvokeRequest) -> UnifiedResponse:
    """
    发送消息进行问诊（非流式）

    - **session_id**: 会话ID
    - **message**: 患者的回答或提问
    - 返回AI的回复

    注意：建议前端先调用 /validate 接口验证语义相关性，验证通过后再调用此接口
    """
    # 检查会话是否存在
    if request.session_id not in global_sessions.sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"会话不存在: {request.session_id}"
        )

    try:
        consultation = global_sessions.get_medicalConsultation(
            request.session_id)

        # 调用模型获取回复
        ai_response = consultation.invoke(request.user_message)

        return success({
            "session_id": request.session_id,
            "user_message": request.user_message,
            "response": ai_response,
            "round": consultation.round_count
        })

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"处理消息失败: {str(e)}"
        )
