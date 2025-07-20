import os
from dotenv import load_dotenv
from langgraph.graph import MessagesState
from react import llm, triple, tavily_search

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

SYSTEM_MESSAGE = """
You are a helpful assistant.
"""

def run_agent_reasoning(state: MessagesState) -> MessagesState:
    """
    Run the agent reasoning node.
    """
    user_message = state["messages"][-1].content
    # Check if the user wants a search
    if "weather in Tokyo" in user_message.lower():
        search_result = tavily_search("weather in Tokyo")
        # Try to extract a number from the result (simplified)
        import re
        match = re.search(r'(\d+)', search_result)
        if match:
            number = float(match.group(1))
            tripled = triple(number)
            answer = f"{search_result}\nTriple of the main number ({number}) is {tripled}."
        else:
            answer = f"{search_result}\nCould not find a number to triple."
        response = {"role": "assistant", "content": answer}
    else:
        # Fallback: just use the LLM
        response = llm.invoke([{"role": "system", "content": SYSTEM_MESSAGE}, *state["messages"]])
    return {"messages": [response]}
