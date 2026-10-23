import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: Transformer — High Level
# Transformer = the neural network architecture behind modern LLMs
# Key mechanism: "Attention" — the model learns which words matter for each other
# "The cat sat on the mat" → to understand "sat", attention looks at "cat"
# You don't need to know the math — just know it enables understanding of context

question = "In 2 simple sentences, what is a Transformer in AI and what problem does it solve?"

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

print("=== Groq ===")
print(groq_llm.invoke(question).content)

print("\n=== Mistral ===")
print(mistral_llm.invoke(question).content)

print("\n=== Google Gemini ===")
print(google_llm.invoke(question).content)
