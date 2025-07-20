import os
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

reflection_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a virtual twitter influencer grading a tweet. Generate critique and recommendations for the user's tweet."
            "Always provide detailed recommendations, including requests for length, virality, style, etc.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

generation_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a twitter techie influencer assistant tasked with writing excellent twitter posts."
            "Generate the best twitter post possible for the user's request."
            "If the user provides critique, respond with a revised version of your previous attempts.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

llm = ChatOpenAI(model="mistralai/mistral-small-3.2-24b-instruct:free", temperature=0)
generate_chain = generation_prompt | llm
reflect_chain = reflection_prompt | llm