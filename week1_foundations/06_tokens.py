import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Tokens
# A token is the basic unit an LLM reads and writes — roughly 3/4 of a word
# "LangChain is great!" = ["Lang", "Chain", " is", " great", "!"] = 5 tokens
# Why it matters: models have a token LIMIT per request (context window)
# Pricing is also based on tokens (input tokens + output tokens)

print("=== Token Examples (no API call needed) ===")
print("'Hello'              → 1 token")
print("'Hello World'        → 2 tokens")
print("'LangChain'          → 2 tokens  (Lang + Chain)")
print("'Unbelievably'       → 4 tokens  (Un + belie + vably or similar)")
print("'2024'               → 1 token")
print("'1 token ≈ 0.75 words'")
print("'1000 tokens ≈ 750 words ≈ ~1.5 pages of text'")

# Ask the LLM to explain tokens
question = "Explain what a 'token' is in an LLM in 2-3 simple sentences. Use a real word example."

# --- Groq ---
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
print("\n=== Groq ===")
print(groq_llm.invoke(question).content)

# --- Mistral ---
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral ===")
print(mistral_llm.invoke(question).content)

# --- Google Gemini ---
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini ===")
print(google_llm.invoke(question).content)
