import os

from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# Define working LLM parameters
temperature = 0.3
api_key = os.getenv("GROQ_API_KEY")

groq_llm = ChatGroq(
    api_key=api_key,
    model="openai/gpt-oss-120b",
    temperature=temperature
)

response = groq_llm.invoke(
    "Explain what is an LLM."
)

print(response.content)