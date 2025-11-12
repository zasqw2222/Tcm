from fastapi import APIRouter
from app.api.embedding import embedding_router
from app.api.tts import tts_router

router = APIRouter()

router.include_router(
    embedding_router, prefix='/api/v1', tags=['Embedding路由'])
router.include_router(
    tts_router, prefix='/api/v1', tags=['TTS路由'])