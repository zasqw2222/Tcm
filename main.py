from app.core.llm import LLM, LLMConfig


def main():
    llm = LLM(LLMConfig())
    res = llm.generate('你好')
    print(res)


if __name__ == "__main__":
    main()
