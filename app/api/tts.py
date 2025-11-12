import os
import logging
import numpy as np
from dotenv import load_dotenv, find_dotenv
from funasr import AutoModel

from fastapi import HTTPException, APIRouter, Body
from contextlib import asynccontextmanager

load_dotenv(find_dotenv(), override=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TTS_MODEL_PATH = os.getenv("ASR_MODEL")
# 流式识别配置（参考 app_steam.py 示例）
CHUNK_SIZE = [0, 10, 5]  # [0, 10, 5] 600ms, [0, 8, 4] 480ms
ENCODER_CHUNK_LOOK_BACK = 4  # number of chunks to lookback for encoder self-attention
DECODER_CHUNK_LOOK_BACK = 1  # number of encoder chunks to lookback for decoder cross-attention
CHUNK_STRIDE = CHUNK_SIZE[1] * 960  # 600ms，每个chunk的采样点数
SAMPLE_RATE = 16000

tts_router = APIRouter(tags=["TTS路由"])


@asynccontextmanager
async def lifespan(app: tts_router):
    """应用生命周期管理：启动时加载模型，关闭时清理资源"""
    global model
    # 启动时执行
    try:
        logger.info("正在加载流式语音识别模型...")
        logger.info("模型: paraformer-zh-streaming")

        # model = AutoModel(model="paraformer-zh-streaming")
        model = AutoModel(model=TTS_MODEL_PATH)

        logger.info("流式模型加载成功！")

    except Exception as e:
        logger.error(f"模型加载失败: {str(e)}")
        raise

    yield  # 应用运行期间


@tts_router.post("/api/recognize_audio_data")
async def recognize_audio_data(audio_data: bytes = Body(...)):
    """
    接收完整的音频二进制数据并识别

    使用场景：浏览器端录音时，点击录音后不发送任何数据，点击结束录音时一次性将所有录音数据发送给此接口

    请求格式:
    - Content-Type: application/octet-stream 或 audio/pcm
    - Body: PCM音频二进制数据 (16kHz, 16bit, mono)

    返回格式:
    - success: 是否成功
    - text: 识别结果文本
    - details: 详细识别信息
    - audio_length_seconds: 音频长度（秒）
    """
    if model is None:
        return {"success": False, "error": "模型未加载", "code": 500}

    try:
        # 接收二进制音频数据
        raw_bytes = audio_data

        if len(raw_bytes) == 0:
            return {
                "success": False,
                "error": "未接收到音频数据",
                "code": 400
            }

        logger.info(f"接收到音频数据，大小: {len(raw_bytes)} bytes")

        audio_data = np.frombuffer(raw_bytes, dtype=np.int16)
        
 
        audio_data = audio_data.astype(np.float32) / 32768.0


        if len(audio_data) < SAMPLE_RATE * 0.1:  # 小于0.1秒
            return {
                "success": False,
                "error": "音频数据太短（至少需要0.1秒）",
                "code": 400
            }

        logger.info(
            f"开始识别完整音频，长度: {len(audio_data)} samples ({len(audio_data)/SAMPLE_RATE:.2f}秒)")

        result = model.generate(input=audio_data)

        if result and isinstance(result, list) and len(result) > 0:
            text = result[0].get("text", "") if isinstance(
                result[0], dict) else ""
        elif result and isinstance(result, dict):
            text = result.get("text", "")
        else:
            text = ""

        audio_length = len(audio_data) / SAMPLE_RATE

        logger.info(f"识别完成，结果: {text[:100]}...，音频长度: {audio_length:.2f}秒")

        return {
            "success": True,
            "text": text,
            "details": result,
            "audio_length_seconds": round(audio_length, 2),
            "audio_samples": len(audio_data)
        }

    except Exception as e:
        logger.error(f"识别失败: {str(e)}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "code": 500
        }
