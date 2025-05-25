from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from weaviate_search import weaviate_search
import os
from langchain_community.llms.tongyi import Tongyi


def char_qa(query: str,collection_name: str = "docs123", limit: int = 2):
    
    prompt = hub.pull("rlm/rag-prompt")
    api_key=os.getenv("DASHSCOPE_API_KEY")

    llm = Tongyi(
        model="qwen-plus",
        api_key=api_key,
    )
    
    contents = weaviate_search(query=query,collection_name=collection_name,limit=limit)

    qa_chain = (
        {
            "context": lambda x: contents,
            "question": lambda x: query,
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    result = qa_chain.invoke(query)
    return result


if __name__ == "__main__":
    query = """
    I would like to know about vaccinating cats.
    """
    
    result = char_qa(query=query,collection_name="Test_collection003", limit=2)
    print(f"result: {result}")

