import os
from dotenv import load_dotenv

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENROUTER_MISTRAL_SMALL_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field, ValidationError

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(temperature=0, model="mistralai/mistral-small-3.2-24b-instruct:free")

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    
    binary_score: str = Field(
        description="Answer is grounded in the facts, 'yes' or 'no'"
    )

system = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n
    Respond ONLY in this JSON format (no other text):
    
    {{ "binary_score": "yes" }}
    Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in / supported by the set of facts."""
hallucination_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ]
)

def hallucination_grader(documents: str, generation: str) -> GradeHallucinations:
    prompt_obj = hallucination_prompt.format_prompt(documents=documents, generation=generation)  # ✅
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
        return GradeHallucinations.parse_raw(json_text)
    except ValidationError as ve:
        raise ValueError(f"Could not parse grader output: {json_text}") from ve
