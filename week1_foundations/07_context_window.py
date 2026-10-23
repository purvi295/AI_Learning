import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Context Window
# Context window = the maximum number of tokens the LLM can "see" at once
# It includes: your system prompt + all conversation history + the current question
# If context fills up → older messages get dropped or you get an error
#
# Real-life analogy: a person who can only hold 10 pages in their memory at once
# GPT-4o:    128,000 tokens
# Gemini 1.5 Pro: 1,000,000 tokens
# Mistral Large: 128,000 tokens

print("=== Context Window Sizes ===")
print("GPT-4o (OpenAI):        128,000 tokens  (~100,000 words)")
print("Gemini 1.5 Pro (Google): 1,000,000 tokens (~750,000 words!)")
print("Mistral Large:           128,000 tokens  (~100,000 words)")
print("openai/gpt-oss-120b (Groq): check your Groq dashboard")
print()
print("Rule of thumb: 1 token ≈ 0.75 words, 1 page ≈ 700 tokens")

question = "In 2 sentences, explain what the LLM 'context window' is. Use an analogy."

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
