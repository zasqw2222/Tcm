# Milvus 向量存储相关模块引入
import sys
from pathlib import Path

# 将项目根目录添加到 Python 路径中，确保可以导入 app 模块
# 这样无论文件是作为模块导入还是直接运行都能正常工作
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from app.vectorstores.config import VectorStoreConfig, VectorStoreBase
from langchain_milvus import Milvus
from app.core.embedding import Embedding
from pydantic import Field
from typing import Optional

# 初始化嵌入模型
embedding = Embedding()

class VSMilvusConfig(VectorStoreConfig):
    """Milvus 向量存储配置类"""
    db_path: str = Field(description="数据库连接路径")
    index_type: str = Field(default="FLAT", description="索引类型")
    metric_type: str = Field(default="L2", description="距离度量类型")
    collection_name: Optional[str] = Field(default=None, description="集合名称")

class VSMilvus(VectorStoreBase):
    def __init__(self, config: VSMilvusConfig):
        # 初始化基类
        super().__init__(config)

    def create_vector_store(self) -> Milvus:
        """
        创建 Milvus 向量存储实例
        """
        return Milvus(
            # 使用远端嵌入模型函数
            embedding_function=embedding.local_embedding(),
            # 连接参数，db_path 作为 Milvus 的 URI
            connection_args={"uri": self.config.db_path},
            # 索引参数，包括索引类型和距离度量类型
            index_params={
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type
            },
            # 指定集合名称
            collection_name=self.config.collection_name
        )
