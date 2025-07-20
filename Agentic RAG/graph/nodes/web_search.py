from typing import Any, Dict

from langchain.schema import Document
from langchain_tavily import TavilySearch

from graph.state import GraphState
from dotenv import load_dotenv

load_dotenv()
web_search_tool = TavilySearch(max_results=3)

def web_search(state: GraphState) -> Dict[str, Any]:
    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    tavily_results = web_search_tool.invoke({"query": question})

    # Safe handling: list of strings or dicts
    results_as_text = []
    for res in tavily_results:
        if isinstance(res, dict) and "content" in res:
            results_as_text.append(res["content"])
        elif isinstance(res, str):
            results_as_text.append(res)
        else:
            print("Unexpected Tavily result format:", res)
    joined_tavily_result = "\n".join(results_as_text)

    web_results = Document(page_content=joined_tavily_result)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}

if __name__ == "__main__":
    web_search(state={"question": "agent memory", "documents": None})
