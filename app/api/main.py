import os
from dotenv import load_dotenv, find_dotenv
from fastapi import APIRouter
from app.api.embedding import embedding_router
from app.api.tts import tts_router
from app.api.session import session_router

load_dotenv(find_dotenv(), override=True)

router = APIRouter()
prefix = os.getenv('ROUTER_PREFIX')

router.include_router(
    embedding_router, prefix=prefix, tags=['Embedding路由'])
router.include_router(
    tts_router, prefix=prefix, tags=['TTS路由'])

router.include_router(
    session_router, prefix=prefix, tags=['Session路由'])
