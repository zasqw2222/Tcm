import os
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader, BSHTMLLoader, DirectoryLoader, JSONLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_core.documents import Document
from typing import List


def load_document(file_path: str) -> List[Document]:
    """
    根据文件扩展名从文件路径加载文档。

    对 .txt 文件自动尝试 utf-8 和 gbk 编码。

    Args:
        file_path: 文档文件的路径

    Returns:
        文档对象列表

    Raises:
        ValueError: 如果文件扩展名不受支持
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {os.path.abspath(file_path)}")

    _, extension = os.path.splitext(file_path.lower())

    loader_map = {
        '.pdf': PyPDFLoader,
        '.docx': Docx2txtLoader,
        '.csv': CSVLoader,
        '.html': BSHTMLLoader,
        '.md': UnstructuredMarkdownLoader,
    }

    if extension == '.txt':
        for encoding in ['utf-8', 'gbk', 'gb2312']:
            try:
                loader = TextLoader(file_path, encoding=encoding)
                docs = loader.load()
                return docs
            except UnicodeDecodeError:
                continue
            except Exception as e:
                continue
        raise RuntimeError(f"所有编码尝试失败，无法读取文本文件: {file_path}")

    elif extension in loader_map:
        loader_class = loader_map[extension]
        try:
            loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            raise RuntimeError(f"加载 {file_path} 时出错: {e}") from e
    else:
        raise ValueError(f"不支持的文件扩展名: {extension}")


def load_documents_from_directory(directory_path: str, glob: str = "**/*.{pdf,docx,txt,csv}") -> List[Document]:
    """
    从指定目录加载所有文档。

    Args:
        directory_path: 文档目录的路径
        glob: 文件匹配模式，默认为 "**/*.{pdf,docx,txt,csv}"

    Returns:
        文档对象列表

    Raises:
        FileNotFoundError: 如果目录不存在
    """
    if not os.path.exists(directory_path):
        raise FileNotFoundError(f"目录未找到: {directory_path}")

    loader = DirectoryLoader(directory_path, glob=glob,
                             use_multithreading=True)
    return loader.load()


def load_documents_from_json(file_path: str, jq_schema: str = '.', is_text_content: bool = True) -> List[Document]:
    """
    从JSON文件加载文档。

    Args:
        file_path: JSON文件的路径
        jq_schema: jq查询模式，默认为 '.'
        is_text_content: 是否为文本内容，默认为 True

    Returns:
        文档对象列表

    Raises:
        FileNotFoundError: 如果文件不存在
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")

    _, extension = os.path.splitext(file_path.lower())
    if extension == '.json':
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=jq_schema,
            text_content=is_text_content)
    elif extension == '.jsonl':
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=jq_schema,
            text_content=is_text_content,
            json_lines=True,
        )
    else:
        raise ValueError(f"不支持的文件扩展名: {extension}")

    return loader.load()
