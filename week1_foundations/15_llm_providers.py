import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: LLM Providers — API vs SDK vs LangChain
# Provider API:  raw HTTP POST to the provider's server (lowest level)
# Provider SDK:  provider's Python library (easier than raw HTTP)
# LangChain:     abstraction layer that works with ANY provider the same way
#
# You are now using LangChain — meaning you can swap Groq ↔ Mistral ↔ Google
# by changing just ONE line of code!

question = "What is Python? Answer in one sentence."

# --- Same question, 3 different providers, but IDENTICAL code pattern ---

# Groq via LangChain
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
print("=== Groq via LangChain ===")
print(groq_llm.invoke(question).content)

# Mistral via LangChain
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral via LangChain ===")
print(mistral_llm.invoke(question).content)

# Google via LangChain
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini via LangChain ===")
print(google_llm.invoke(question).content)

print("\n---")
print("Notice: .invoke(question) is IDENTICAL for all 3 providers.")
print("That is the power of LangChain — one interface, any provider.")
