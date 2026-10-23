import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Agentic AI
# Agentic AI = systems designed to PLAN and COMPLETE complex multi-step goals
# More advanced than a single AI agent — can use multiple agents, tools, memory
# Examples: AutoGPT, LangGraph agents, coding agents

question = "In 2 sentences, what is Agentic AI and how does it differ from a single AI Agent?"

# --- Groq ---
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
print("=== Groq ===")
print(groq_llm.invoke(question).content)

# --- Mistral ---
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral ===")
print(mistral_llm.invoke(question).content)

# --- Google Gemini ---
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini ===")
print(google_llm.invoke(question).content)
