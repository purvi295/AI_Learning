import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Hallucination
# Hallucination = when the LLM confidently generates WRONG or MADE-UP information
# It does not lie on purpose — it predicts the next token based on patterns
# A "plausible-sounding" but false answer is hallucination
#
# Risk: medical, legal, financial facts — ALWAYS verify with real sources

print("=== Hallucination Demo ===")
print("Asking the model about a FAKE class that does not exist...")
print("Watch how confidently it may respond!\n")

fake_question = "Explain the Laravel QuantumMiddlewareOptimizer class and its methods."

# --- Groq ---
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
print("=== Groq ===")
print(groq_llm.invoke(fake_question).content)

# --- Mistral ---
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== Mistral ===")
print(mistral_llm.invoke(fake_question).content)

# --- Google Gemini ---
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== Google Gemini ===")
print(google_llm.invoke(fake_question).content)

print("\n---")
print("Note: 'QuantumMiddlewareOptimizer' does NOT exist in Laravel.")
print("If any model answered confidently with fake details = hallucination.")
print("If a model said it doesn't exist = good grounding behavior.")
