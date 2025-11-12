# 引入 FastAPI、HTTPException 和 APIRouter，用于创建API路由与异常处理
from fastapi import HTTPException, APIRouter
# 引入 pydantic 的 BaseModel，用于数据模型校验
from pydantic import BaseModel
# 引入类型提示 List 和 Union
from typing import List, Union
# 引入自定义的 Embedding 类
from app.core.embedding import Embedding

# 创建 Embedding 实例，用于后续处理
embedding = Embedding()
# 预加载模型，仅加载一次以提升性能
embedding_model = embedding.remote_embedding()

# 创建用于 Embedding 接口的 APIRouter
embedding_router = APIRouter(tags=["Embedding路由"])

# 定义请求体的数据模型，包括输入内容（可以为字符串或字符串列表）和模型名（默认为 "bge-large-zh-v15"）
class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: str = "bge-large-zh-v15"  # 默认模型名称

# 定义 Embedding 对象的数据结构，每个对象包括类型说明（object）、向量结果（embedding）、及索引（index）
class EmbeddingObject(BaseModel):
    object: str = "embedding"
    embedding: List[float]
    index: int

# 定义响应体的数据结构，包括类型（object）、数据（data）、模型名（model）、以及用量信息（usage）
class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: List[EmbeddingObject]
    model: str = "local"
    usage: dict = {"prompt_tokens": 0, "total_tokens": 0}

# 定义 API 路由/接口，用于生成文本的 embedding
@embedding_router.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(request: EmbeddingRequest):
    inputs = request.input
    # 如果输入为字符串，则转为列表，统一处理
    if isinstance(inputs, str):
        inputs = [inputs]

    try:
        # 调用 embedding_model 的 encode 方法，对输入文本进行编码，生成 embedding 向量
        embeddings = embedding_model.encode(inputs, normalize_embeddings=True)

        # 组装返回的数据，embedding 向量转换为 list
        data = [
            EmbeddingObject(
                embedding=emb.tolist() if hasattr(emb, 'tolist') else list(emb),
                index=i
            )
            for i, emb in enumerate(embeddings)
        ]
        # 返回 embedding 结果
        return EmbeddingResponse(data=data, model=request.model)
    except Exception as e:
        # 捕捉异常，返回 500 错误
        raise HTTPException(
            status_code=500, detail=f"Embedding failed: {str(e)}")
