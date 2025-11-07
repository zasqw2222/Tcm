import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
from typing import Optional
from pydantic import Field
from app.core.embedding import Embedding
from langchain_chroma import Chroma
from app.vectorstores.config import VectorStoreConfig, VectorStoreBase



# 初始化Embedding对象
embedding = Embedding()


class VSChromaConfig(VectorStoreConfig):
    """Chroma向量存储配置类"""
    db_path: str = Field(description="数据库连接路径")  # 数据库文件保存路径
    index_type: str = Field(default="FLAT", description="索引类型")  # 向量索引类型
    metric_type: str = Field(default="L2", description="距离度量类型")  # 度量方式
    collection_name: Optional[str] = Field(
        default=None, description="集合名称")  # Chroma集合名称


class VSChroma(VectorStoreBase):
    def __init__(self, config: VSChromaConfig):
        # 初始化父类，并传递配置参数
        super().__init__(config)

    def create_vector_store(self) -> Chroma:
        # 创建Chroma向量存储对象
        return Chroma(
            persist_directory=self.config.db_path,     # 指定数据存储路径
            embedding_function=embedding.local_embedding(),  # 指定embedding函数
            collection_name=self.config.collection_name      # 指定集合名称
        )
