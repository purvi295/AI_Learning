import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# define working llm's params
model = "gemini-3.6-flash"
temperature = 0.3
api_key = os.getenv("GOOGLE_API_KEY")

geminiLlm = ChatGoogleGenerativeAI(api_key=api_key, model=model, temperature=temperature)

print(geminiLlm)
response = geminiLlm.invoke(
    "Explain the Laravel QuantumMiddlewareOptimizer class"
)

print(response.content)