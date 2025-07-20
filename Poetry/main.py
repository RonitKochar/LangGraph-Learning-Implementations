from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langgraph.graph import MessagesState, StateGraph
from nodes import run_agent_reasoning
import time

load_dotenv()

START = "__start__"
END = "__end__"
AGENT_REASON = "agent_reason"
LAST = -1

flow = StateGraph(MessagesState)
flow.add_node(AGENT_REASON, run_agent_reasoning)
flow.set_entry_point(AGENT_REASON)
flow.add_edge(AGENT_REASON, AGENT_REASON)  # Loop for simplicity

app = flow.compile()
app.get_graph().draw_mermaid_png(output_file_path="flow.png")

if __name__ == "__main__":
    print(f"[{time.strftime('%H:%M:%S')}] Hello ReAct LangGraph with manual Tavily search")
    print(f"[{time.strftime('%H:%M:%S')}] Invoking app with first message...")
    res = app.invoke({"messages": [HumanMessage(content="What is the weather in Tokyo? List it and then triple it.")]})
    print(f"[{time.strftime('%H:%M:%S')}] App invocation complete.")
    print(f"[{time.strftime('%H:%M:%S')}] Final response:")
    print(res["messages"][LAST]["content"])
