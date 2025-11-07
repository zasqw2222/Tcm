
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Callable
from pydantic import BaseModel

class SplitConfig(BaseModel):
    """文本分割配置类"""
    chunk_size: int = 200  # 分块大小
    chunk_overlap: int = 20  # 分块重叠大小
    length_function: Callable = len  # 长度计算函数
    is_separator_regex: bool = False  # 分隔符是否为正则表达式
    separator: str = "\n\n"  # 分隔符


class SplitRecursiveConfig(BaseModel):
    """递归文本分割配置类"""
    chunk_size: int = 200  # 分块大小
    chunk_overlap: int = 20  # 分块重叠大小
    length_function: Callable = len  # 长度计算函数
    is_separator_regex: bool = False  # 分隔符是否为正则表达式


def split_from_character(
        documents: List[Document], config: SplitConfig) -> List[Document]:
    """
    使用字符分割器分割文档

    Args:
        documents: 要分割的文档列表
        config: 分割配置

    Returns:
        分割后的文档列表
    """
    text_splitter = CharacterTextSplitter(**config.model_dump())
    documents = text_splitter.split_documents(documents)
    return documents


def split_from_recursive(documents: List[Document], config: SplitRecursiveConfig) -> List[Document]:
    """
    使用递归字符分割器分割文档

    Args:
        documents: 要分割的文档列表
        config: 递归分割配置

    Returns:
        分割后的文档列表
    """
    text_splitter = RecursiveCharacterTextSplitter(**config.model_dump())
    documents = text_splitter.split_documents(documents)
    return documents
