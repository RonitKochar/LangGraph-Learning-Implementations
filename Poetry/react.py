import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from tavily import TavilyClient  # Assuming you have the Tavily Python SDK

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

def triple(num: float) -> float:
    """
    Param num: a number to triple
    returns: the triple of the input number
    """
    return float(num) * 3

def tavily_search(query: str) -> str:
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=tavily_api_key)
    result = client.search(query=query, max_results=1)
    # Extract the relevant snippet or summary
    return result['results'][0]['content'] if result['results'] else "No results found."

llm = ChatOpenAI(model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0)
