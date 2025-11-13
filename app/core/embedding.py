import os
from dotenv import load_dotenv, find_dotenv
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch

# 加载.env配置文件中的环境变量
load_dotenv(find_dotenv(), override=True)

# 获取本地嵌入模型路径环境变量
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")


class Embedding:
    def __init__(self):
        pass

    def local_embedding(self) -> HuggingFaceEmbeddings:
        """
        使用本地 HuggingFaceEmbeddings 模型进行文本编码
        在langchain内部使用
        返回已初始化的嵌入模型实例
        """
        return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    def remote_embedding(self) -> SentenceTransformer:
        """
        使用 SentenceTransformer 进行文本嵌入
        为了api兼容，返回 SentenceTransformer 对象
        返回已初始化的嵌入模型实例
        """
        model = SentenceTransformer(
            EMBEDDING_MODEL_PATH, device="cuda" if torch.cuda.is_available() else "cpu")

        model.eval()
        return model

