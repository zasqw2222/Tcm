

[中文](./README_zh.md)



## Introduction

tcm is a Python-based agent integration project that combines various mainstream vector stores (such as Chroma, Milvus) and large language model (LLM) capabilities, suitable for scenarios such as intelligent search and knowledge base Q&A.

## Features
Support for multiple vector store backends (Chroma, Milvus, etc.)
Based on the LangChain framework, easy to extend
Support for multiple large models (OpenAI, Huggingface, etc.)
Convenient document management and retrieval interfaces

## dependent


It is recommended to use [uv](https://github.com/astral-sh/uv) manage and install dependencies

- Python >= 3.13
- The dependency has already been declared in `pyproject.toml`.

> 


> Install uv (if not installed)：
> ```bash
> pip install uv
> ```
>Install dependencies using uv for the selected text：
> 


>
> ```bash
> uv sync
> ```
`

## Quick Start

1. **Clone the repository and install dependencies**

   ```bash
   git clone <your-repo-url>
   cd tcm
   uv pip install -r pyproject.toml
   ```

2. **Run the main program**

   ```bash
   python main.py
   ```


3. **Modify configuration**

   - The user-selected text vector storage configuration is located in `app/vectorstores/config.py`, where the database path, collection name, index type, and more can be modified as needed.
   - The LLM configuration for the selected text can be modified in `app/core/llm.py` and other locations.

## description

```
tcm/
├── app/
│   ├── core/              
│   └── vectorstores/      
├── main.py                
├── pyproject.toml         
└── README.md              
```

## Usage

- Add the selected text to the document: use the `VectorStoreBase.add_documents()` interface.
- Search documents: perform similarity search using the `query()` method
- Extend the selected text **large model or vector storage**: implement the corresponding base class and register it.

## Contact to discuss
If you have any questions or suggestions, feel free to open an issue or PR!

