from fastapi import status
from pydantic import BaseModel
from typing import Optional, Any


class UnifiedResponse(BaseModel):
    code: int
    message: str
    data: Optional[Any] = None


def success(data: Any) -> UnifiedResponse:
    """
    成功响应函数
    """
    return UnifiedResponse(code=status.HTTP_200_OK, message='success', data=data)

def error(code: int, message: str) -> UnifiedResponse:
    """
    错误响应函数
    """
    return UnifiedResponse(code=code, message=message, data=None)