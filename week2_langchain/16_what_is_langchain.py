import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: What is LangChain?
# LangChain = a Python framework for building LLM-powered applications
# It provides ready-made building blocks:
#   - Model integrations (ChatGroq, ChatMistralAI, ChatGoogleGenerativeAI)
#   - Prompt templates (ChatPromptTemplate)
#   - Output parsers (StrOutputParser, PydanticOutputParser)
#   - Chains (connect components with |)
#   - Agents, tools, memory, retrievers...
#
# Key idea: prompt | model | parser = a CHAIN

prompt = ChatPromptTemplate.from_template("Explain {topic} in one sentence.")

groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

parser = StrOutputParser()

# Build 3 chains — one per provider, same prompt and parser
chain_groq    = prompt | groq_llm    | parser
chain_mistral = prompt | mistral_llm | parser
chain_google  = prompt | google_llm  | parser

topic = {"topic": "LangChain"}

print("=== What is LangChain? — Groq ===")
print(chain_groq.invoke(topic))

print("\n=== What is LangChain? — Mistral ===")
print(chain_mistral.invoke(topic))

print("\n=== What is LangChain? — Google Gemini ===")
print(chain_google.invoke(topic))

print("\n---")
print("Notice: prompt | llm | parser is the LangChain LCEL chain syntax.")
print("The | (pipe) operator composes components into a pipeline.")
