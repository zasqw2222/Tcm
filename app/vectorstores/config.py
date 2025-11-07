from abc import ABC, abstractmethod
from pydantic import BaseModel, Field
from typing import List, Optional, Any
from langchain_core.documents import Document


class VectorStoreConfig(BaseModel):
    """向量存储配置类"""
    db_path: str = Field(description="数据库连接路径")
    index_type: str = Field(default="FLAT", description="索引类型")
    metric_type: str = Field(default="L2", description="距离度量类型")
    collection_name: Optional[str] = Field(default=None, description="集合名称")


class VectorStoreBase(ABC):
    def __init__(self, config: VectorStoreConfig):
        self.config = config
        self.vector_store = self.create_vector_store()

    @abstractmethod
    def create_vector_store(self) -> Any:
        """
        创建向量存储实例
        """
        pass

    def add_documents(self, documents: List[Document]) -> None:
        """
        添加文档到向量存储

        Args:
            documents: 要添加的文档列表
        """
        if not documents:
            return

        try:
            self.vector_store.add_documents(
                documents, collection_name=self.config.collection_name)
        except Exception as e:
            raise

    def query(self, query: str, k: int = 1) -> List[Document]:
        """
        相似性搜索查询

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            相似文档列表
        """
        if not query.strip():
            return []

        try:
            results = self.vector_store.similarity_search(query, k)
            return results
        except Exception as e:
            raise

    def query_with_score(self, query: str, k: int = 1) -> List[tuple]:
        """
        带分数的相似性搜索查询

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            (文档, 分数) 元组列表
        """
        if not query.strip():
            return []

        try:
            results = self.vector_store.similarity_search_with_score(query, k)
            return results
        except Exception as e:
            raise

    def get_retriever(self, search_type: str = 'similarity', k: int = 1) -> Any:
        """
        获取检索器

        Args:
            search_type: 搜索类型
            k: 返回结果数量

        Returns:
            检索器对象
        """
        try:
            retriever = self.vector_store.as_retriever(
                search_type=search_type,
                search_kwargs={"k": k}
            )
            return retriever
        except Exception as e:
            raise

    def get_all_documents(self, limit: int = 1000) -> List[Document]:
        """
        获取所有文档数据

        Args:
            limit: 返回的最大文档数量，默认1000

        Returns:
            所有文档列表
        """
        try:
            # 使用一个通用的查询来获取所有数据
            # 这里使用一个空字符串或通用查询来获取所有数据
            results = self.vector_store.similarity_search("", k=limit)
            return results
        except Exception as e:
            print(f"获取所有文档时出错: {e}")
            return []

    def get_collection_info(self) -> dict:
        """
        获取集合信息

        Returns:
            集合信息字典
        """
        try:
            collection = self.vector_store._collection

            # 获取集合基本信息
            collection_info = {
                "collection_name": self.config.collection_name,
            }

            try:
                if hasattr(collection, 'num_entities'):
                    collection_info["total_entities"] = collection.num_entities
                elif hasattr(collection, 'count'):
                    collection_info["total_entities"] = collection.count()
                else:
                    all_docs = self.get_all_documents(limit=10000)
                    collection_info["total_entities"] = len(all_docs)
            except Exception as count_error:
                print(f"获取文档数量时出错: {count_error}")
                collection_info["total_entities"] = 0

            return collection_info
        except Exception as e:
            print(f"获取集合信息时出错: {e}")
            return {}

    def search_by_ids(self, ids: List[str]) -> List[Document]:
        """
        根据ID列表获取文档

        Args:
            ids: 文档ID列表

        Returns:
            对应的文档列表
        """
        try:
            # 这里需要根据实际的Milvus API来实现
            # 由于langchain_milvus可能不直接支持按ID查询，我们使用相似性搜索作为替代
            results = []
            for doc_id in ids:
                # 使用ID作为查询文本进行搜索
                docs = self.vector_store.similarity_search(doc_id, k=1)
                if docs:
                    results.extend(docs)
            return results
        except Exception as e:
            print(f"根据ID搜索文档时出错: {e}")
            return []

    def get_documents_count(self) -> int:
        """
        获取文档总数

        Returns:
            文档总数
        """
        try:
            collection = self.vector_store._collection
            if hasattr(collection, 'num_entities'):
                return collection.num_entities
            else:
                # 如果无法直接获取数量，通过搜索来估算
                all_docs = self.get_all_documents(limit=10000)
                return len(all_docs)
        except Exception as e:
            print(f"获取文档数量时出错: {e}")
            return 0

    def delete_collection(self, ids: List[str] = None) -> None:
        """
        删除集合中的指定文档

        Args:
            ids: 要删除的文档ID列表
        """
        try:
            self.vector_store.delete(ids)
        except Exception as e:
            print(f"删除文档时出错: {e}")
            raise

    def clear_collection(self) -> None:
        pass
