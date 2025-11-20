from fastapi import HTTPException, APIRouter
from pydantic import BaseModel
from typing import List, Union
from app.core.embedding import Embedding
import torch

embedding = Embedding()  # 初始化 Embedding 实例
embedding_model = embedding.remote_embedding()  # 加载模型，只加载一次

embedding_router = APIRouter(tags=["Embedding路由"])  # 路由定义


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-large-zh-v15"  # 默认模型名


class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str = "local"
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0}


@embedding_router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    """
    创建 Embedding

    - **input**: 输入的文本或文本列表（str 或 List[str]）
    - **model**: 使用的模型名称（默认为 "bge-large-zh-v15"）

    返回值格式为:
    - object: 返回对象类型 "list"
    - data: 每个输入文本的 embedding 结果列表
        - object: "embedding"
        - embedding: embedding 向量
        - index: 输入的索引
    - model: 使用的embedding模型名
    - usage: token信息 (未统计，返回0)
    """
    inputs = request.input
    if isinstance(inputs, str):  # 转为列表统一处理
        inputs = [inputs]

    try:
        # 确保 inputs 非空
        if not inputs:
            return EmbeddingResponse(data=[], model=request.model)

        embeddings = embedding_model.encode(inputs, normalize_embeddings=True)

        # 处理 Tensor 转换为 numpy
        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()
        elif not hasattr(embeddings, '__iter__') and not isinstance(embeddings, list):
             # 防御性编程：如果 encode 返回单个非列表对象（不太可能，但以防万一）
             embeddings = [embeddings]

        data = [
            EmbeddingObject(
                embedding=emb.tolist() if hasattr(emb, 'tolist') else list(emb),
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        
        # 简单估算 usage (字符数作为 token 数的近似，仅供参考)
        total_tokens = sum(len(s) for s in inputs)
        
        return EmbeddingResponse(
            data=data, 
            model=request.model,
            usage={"prompt_tokens": total_tokens, "total_tokens": total_tokens}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {str(e)}")
