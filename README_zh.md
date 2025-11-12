# tcm 项目文档

## 项目简介

`tcm` 是一个基于 Python 的智能体（Agent）集成项目，集成了多种主流的向量存储（如 Chroma、Milvus）与大语言模型 (LLM) 能力，适用于智能搜索、知识库问答等场景。

## 主要特性

- 支持多种向量存储后端（Chroma、Milvus等）
- 基于 LangChain 框架，易于扩展
- 支持多种大模型（OpenAI、Huggingface 等）
- 便捷的文档管理与检索接口

## 依赖环境

本项目推荐使用 [uv](https://github.com/astral-sh/uv) 管理和安装依赖。

- Python >= 3.13
- 依赖已在 `pyproject.toml` 中声明

> 安装 uv（如未安装）：
>
> ```bash
> pip install uv
> ```
>
> 使用 uv 安装依赖：
>
> ```bash
> uv sync
> ```
`

## 快速开始

1. **克隆仓库并安装依赖**

   ```bash
   git clone <your-repo-url>
   cd tcm
   uv pip install -r pyproject.toml
   ```

2. **运行主程序**

   ```bash
   python main.py
   ```

   示例输出：

   ```
   你好！（来自大语言模型的回复）
   ```

3. **修改配置**

   - 向量存储配置在 `app/vectorstores/config.py`，可按需修改数据库路径、集合名、索引类型等。
   - LLM 配置可在 `app/core/llm.py` 等处修改。

## 目录结构说明

```
tcm/
├── app/
│   ├── core/              # 大模型及核心逻辑
│   └── vectorstores/      # 向量存储相关
├── main.py                # 程序入口
├── pyproject.toml         # 项目依赖声明
└── README.md              # 项目说明文档
```

## 常见用法

- **添加文档**：使用 `VectorStoreBase.add_documents()` 接口。
- **检索文档**：通过 `query()` 方法进行相似性搜索。
- **扩展大模型或向量存储**：实现对应基类并注册即可。

## 联系讨论

如有问题或建议，欢迎提 Issue 或 PR！

