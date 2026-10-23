import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Reasoning Models vs Chat Models
# Chat Models:     fast, conversational, good for Q&A, summarization, coding
# Reasoning Models: slower, think step-by-step before answering, better for math/logic
#
# Examples:
# Chat:      GPT-4o, Gemini Flash, Mistral Large, Claude 3 Haiku
# Reasoning: o1, o3, Gemini 2.0 Flash Thinking, DeepSeek-R1

question = "In 2 sentences, what is the difference between a chat model and a reasoning model in AI?"

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

print("=== Groq ===")
print(groq_llm.invoke(question).content)

print("\n=== Mistral ===")
print(mistral_llm.invoke(question).content)

print("\n=== Google Gemini ===")
print(google_llm.invoke(question).content)
