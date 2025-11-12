from fastapi import APIRouter
from app.api.embedding import embedding_router

router = APIRouter()

router.include_router(
    embedding_router, prefix='/api/v1', tags=['Embedding路由'])
