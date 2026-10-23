import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: LangChain Model Integration
# LangChain wraps provider SDKs so they all share the same interface
# You import from langchain_groq / langchain_mistralai / langchain_google_genai
# Then call .invoke(), .stream(), .ainvoke() — same method, any provider

message = "What is 2 + 2? Answer in one word."

# --- Groq ---
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0)
print("=== Groq ===")
response = groq_llm.invoke(message)
print(response.content)

# --- Mistral ---
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0)
print("\n=== Mistral ===")
response = mistral_llm.invoke(message)
print(response.content)

# --- Google Gemini ---
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.8-flash", temperature=0)
print("\n=== Google Gemini ===")
response = google_llm.invoke(message)
print(response.content)
