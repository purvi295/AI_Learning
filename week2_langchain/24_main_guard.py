import os
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Topic: if __name__ == '__main__'
# This guard ensures a block of code ONLY runs when the file is executed directly
# It does NOT run when the file is imported by another file
#
# Example:
#   python 24_main_guard.py  → block runs  ✅
#   import 24_main_guard     → block skips ✅

def greet(name: str) -> str:
    return f"Hello, {name}!"

def ask_llm(question: str) -> None:
    groq_llm    = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="openai/gpt-oss-120b", temperature=0.3)
    mistral_llm = ChatMistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="codestral-2508", temperature=0.3)
    google_llm  = ChatGoogleGenerativeAI(api_key=os.getenv("GOOGLE_API_KEY"), model="gemini-3.6-flash", temperature=0.3)

    print("Groq:   ", groq_llm.invoke(question).content)
    print("Mistral:", mistral_llm.invoke(question).content)
    print("Google: ", google_llm.invoke(question).content)

if __name__ == "__main__":
    # This block ONLY runs when you do: python 24_main_guard.py
    print("=== if __name__ == '__main__' demo ===")
    print(greet("AI Engineer"))

    print("\n=== Asking LLMs: what does if __name__ == main mean? ===")
    ask_llm("In one sentence, what does 'if __name__ == __main__' mean in Python?")
