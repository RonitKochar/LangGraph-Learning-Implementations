import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI

class GradeAnswer(BaseModel):
    
    binary_score: str = Field(
        description="Answer addresses the question, 'yes' or 'no'"
    )
    
llm = ChatOpenAI(temperature=0, model="mistralai/mistral-small-3.2-24b-instruct:free")

system = """You are a grader assessing whether an answer addresses / resolves a question \n
    Respond ONLY in this JSON format (no other text):

    {{ "binary_score": "yes" }}
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer resolves the question."""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

def answer_grader(question: str, generation: str) -> GradeAnswer:
    prompt_obj = answer_prompt.format_prompt(question=question, generation=generation)
    messages = prompt_obj.to_messages()
    response = llm.invoke(messages)
    content = response.content.strip() if response and response.content else ""

    # Extract JSON block between '{' and '}'
    start = content.find("{")
    end = content.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"Grader did not return a JSON object: {content}")

    json_text = content[start:end+1]
    try:
        return GradeAnswer.parse_raw(json_text)
    except ValidationError as ve:
        raise ValueError(f"Could not parse grader output: {json_text}") from ve
