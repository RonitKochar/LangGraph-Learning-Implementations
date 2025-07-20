from langchain_openai import ChatOpenAI
from schemas import ReviseAnswer
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(model="mistralai/mistral-small-3.2-24b-instruct:free")
payload = {
    "answer": "Test answer.",
    "reflection": {"missing": "None", "superfluous": "None"},
    "search_queries": ["AI-powered SOC startups"],
    "references": ["https://example.com"]
}
messages = [HumanMessage(content="Write about AI-Powered SOC / autonomous soc problem domain, list startups that do that and raised capital.")]
response = llm.invoke(messages)
print(response)