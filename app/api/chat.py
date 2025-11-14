import os
import logging
import numpy as np
from dotenv import load_dotenv, find_dotenv
from fastapi import HTTPException, APIRouter, status

from app.core.llm import llm, MedicalConsultation, LLMConfig
from contextlib import asynccontextmanager
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(find_dotenv(), override=True)

chat_router = APIRouter()





