import os
import asyncio
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_mistralai import ChatMistralAI
from langchain_groq import ChatGroq

load_dotenv()

# 1. Initialize the LLM
# used for the mistral model
api_key = os.getenv("MISTRAL_API_KEY")
llm = ChatMistralAI(
    api_key=api_key,    
    model="codestral-2508",
    temperature=0.3,
)
 
#used for the groq model
# api_key = os.getenv("GROQ_API_KEY")
# llm = ChatGroq(
#     api_key=api_key,    
#     model="openai/gpt-oss-120b",
#     temperature=0.3,
# )

# 2. Define reusable ChatPromptTemplate with placeholders
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert tutor. Explain concepts clearly and concisely for a {audience} audience."),
    ("human", "Explain what {topic} is in under {word_limit} words.")
])

# 3. Create a chain using LangChain Expression Language (LCEL)
chain = prompt_template | llm | StrOutputParser()

# 4. Interactive loop allowing user to enter prompts
async def main():
    print("=" * 50)
    print("  Interactive Prompt Template with Mistral / Groq")
    print("  (Type 'exit' or 'q' to quit)")
    print("=" * 50)

    while True:
        try:
            topic = input("\nEnter a topic (or 'exit'): ").strip()
            if not topic or topic.lower() in ("exit", "q"):
                print("Exiting. Goodbye!")
                break

            audience = input("Target audience [press Enter for 'beginner']: ").strip() or "beginner"
            word_limit = input("Word limit [press Enter for '50']: ").strip() or "50"

            print("\nThinking...")
            # Note: ainvoke is asynchronous, so it must be awaited!
            response = await chain.ainvoke({
                "topic": topic,
                "audience": audience,
                "word_limit": word_limit,
            })
            # #this is used as the stream + async 
            # async for chunk in llm.astream(
            #     "Explain LangChain."
            # ):
            #  print(chunk.content, end="", flush=True)
            print("\n" + "-" * 50)
            print(response)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\nInterrupted. Exiting...")
            break

if __name__ == "__main__":
    asyncio.run(main())
