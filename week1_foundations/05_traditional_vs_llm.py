import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Traditional Software vs LLM Software
# Traditional Software: deterministic, rule-based, same input = same output
# LLM Software:         probabilistic, flexible, same input can give different output
#
# Traditional: if user_said == "hello": reply = "Hi!"
# LLM:         llm.invoke("User said hello, how should I reply?")

print("=== Traditional Software (no LLM needed) ===")
user_input = "hello"
if user_input == "hello":
    print("Reply: Hi! How can I help you?")
else:
    print("Reply: I don't understand.")

# Now the same thing with LLM — it handles ANY input, not just "hello"
question = "User said: 'hello'. Reply in a warm and professional way. One sentence only."

# --- Groq ---
groq_llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
print("\n=== LLM Software — Groq ===")
print(groq_llm.invoke(question).content)

# --- Mistral ---
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
print("\n=== LLM Software — Mistral ===")
print(mistral_llm.invoke(question).content)

# --- Google Gemini ---
google_llm = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)
print("\n=== LLM Software — Google Gemini ===")
print(google_llm.invoke(question).content)
