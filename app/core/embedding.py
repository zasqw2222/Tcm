import os
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field
from typing import Literal
from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
import torch

# 加载.env配置文件中的环境变量
load_dotenv(find_dotenv(), override=True)

# 获取本地嵌入模型路径环境变量
EMBEDDING_MODEL_PATH = os.getenv("EMBEDDING_MODEL_PATH")


# class EmbeddingConfig(BaseModel):
#     # 嵌入模型类型：可以是 'local' 或 'remote'
#     type: Literal["local", "remote"] = Field(
#         default="local", description="嵌入模型类型"
#     )


class Embedding:
    def __init__(self):
        pass
        # 保存配置对象
        # self.config = config
        # 根据配置类型初始化对应的嵌入模型，并保存为实例属性
        # if self.config.type == "local":
        #     self._embedding_model = self.local_embedding()
        # elif self.config.type == "remote":
        #     self._embedding_model = self.remote_embedding()
        # else:
        #     raise ValueError(f"不支持的嵌入模型类型: {self.config.type}")

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

