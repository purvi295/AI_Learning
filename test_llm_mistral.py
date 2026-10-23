import os

from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

# Define working LLM parameters
temperature = 0.3
api_key = os.getenv("MISTRAL_API_KEY")

mistral_llm = ChatMistralAI(
    api_key=api_key,
    model="codestral-2508",
    temperature=temperature,
)

response = mistral_llm.invoke(
    "Explain what is an LLM."
)

print(response.content)
