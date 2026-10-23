import os
from typing import Optional, List, Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, ValidationError
from langchain_mistralai import ChatMistralAI

load_dotenv()

# ==============================================================================
# SECTION 1: WHAT IS A PYDANTIC SCHEMA?
# ==============================================================================
# Pydantic is a data validation and parsing library for Python.
# A "Schema" is a class inheriting from `BaseModel` where you define:
#  1. Field names and their Python types (str, int, list, etc.)
#  2. Default values or optional fields
#  3. Constraints (min/max length, number bounds, regex)
#  4. Descriptions (crucial for LLMs to understand what to extract)
#  5. Custom validation logic via @field_validator
# ==============================================================================

class Address(BaseModel):
    city: str = Field(description="City name")
    country: str = Field(default="USA", description="Country name")

class UserProfile(BaseModel):
    # 1. Type annotations & Field constraints
    name: str = Field(..., min_length=2, max_length=50, description="Full name of the user")
    age: int = Field(..., ge=0, le=120, description="Age in years (between 0 and 120)")
    email: str = Field(..., description="User's email address")
    role: Literal["admin", "user", "guest"] = Field(default="user", description="Account role")
    skills: List[str] = Field(default_factory=list, description="List of technical skills")
    address: Optional[Address] = Field(default=None, description="Optional nested address")

    # 2. Custom validation logic
    @field_validator("email")
    @classmethod
    def validate_email_domain(cls, value: str) -> str:
        if "@" not in value or "." not in value.split("@")[-1]:
            raise ValueError("Invalid email format! Must contain '@' and a domain dot.")
        return value.lower()


# ==============================================================================
# SECTION 2: PYDANTIC IN LANGCHAIN / LLMs (STRUCTURED OUTPUT)
# ==============================================================================
# In LLM workflows, language models normally return unstructured text/markdown.
# By passing a Pydantic schema to `llm.with_structured_output(Schema)`, the LLM:
#  1. Receives the schema converted to JSON Schema.
#  2. Uses function/tool calling to guarantee the response matches this structure.
#  3. LangChain automatically deserializes the JSON response into a Pydantic instance!
# ==============================================================================

class SentimentAnalysis(BaseModel):
    product_name: str = Field(description="Name of the product mentioned")
    sentiment: Literal["positive", "neutral", "negative"] = Field(
        description="Overall sentiment of the review"
    )
    rating: int = Field(
        ge=1, le=5, 
        description="Estimated star rating out of 5 based on customer tone"
    )
    summary: str = Field(
        max_length=300, 
        description="A concise one-sentence summary of the review"
    )
    pros: List[str] = Field(
        description="Key positive aspects mentioned by the user"
    )
    cons: List[str] = Field(
        description="Key complaints or drawbacks mentioned by the user"
    )


# ==============================================================================
# DEMONSTRATION & EXECUTION
# ==============================================================================
def demo_pure_pydantic():
    print("=" * 60)
    print("1. PURE PYDANTIC VALIDATION DEMO")
    print("=" * 60)

    # A. Valid Data Example
    valid_data = {
        "name": "Alex Mercer",
        "age": 28,
        "email": "Alex.Mercer@Example.COM",
        "role": "admin",
        "skills": ["Python", "FastAPI", "LangChain"],
        "address": {"city": "San Francisco", "country": "USA"}
    }

    user = UserProfile(**valid_data)
    print(" Successfully created UserProfile object:")
    print(f"  Name: {user.name}")
    print(f"  Age: {user.age}")
    print(f"  Normalized Email: {user.email}")
    print(f"  Skills: {', '.join(user.skills)}")
    print(f"  Address: {user.address.city}, {user.address.country}")

    # Export to Dict and JSON
    print("\n Exporting to standard formats:")
    print("  .model_dump() (Python Dict):", user.model_dump())
    print("  .model_dump_json() (JSON String):", user.model_dump_json(indent=2))

    # B. Invalid Data Handling Example
    print("\n Testing Validation Failure (Invalid Age & Invalid Email):")
    invalid_data = {
        "name": "B",               # Too short (< 2 chars)
        "age": 150,               # Exceeds max age (le=120)
        "email": "not-an-email",  # Fails validator
    }
    try:
        UserProfile(**invalid_data)
    except ValidationError as e:
        print(" Caught ValidationError as expected!")
        for error in e.errors():
            field_name = error['loc'][0]
            message = error['msg']
            print(f"   - Field '{field_name}': {message}")


def demo_llm_structured_output():
    print("\n" + "=" * 60)
    print("2. LANGCHAIN + PYDANTIC STRUCTURED OUTPUT DEMO")
    print("=" * 60)

    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        print(" [Warning] MISTRAL_API_KEY not set in .env. Skipping LLM demo.")
        return

    # Initialize LLM
    llm = ChatMistralAI(
        api_key=api_key,
        model="codestral-2508",
        temperature=0.0,
    )

    # Bind the Pydantic schema to the model
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    unstructured_review = """
    I bought the SoundWave Pro Headphones last week. The sound quality and active
    noise cancellation are truly phenomenal, and battery life easily lasts over 30 hours.
    However, the headband feels slightly tight after 2 hours of continuous wear, and the
    charging cable included in the box is way too short. Overall, pretty satisfied with the purchase!
    """

    print(" Unstructured Text Input:")
    print(unstructured_review.strip())
    print("\n Extracting structured data using LLM...")

    # Invoke returns a validated SentimentAnalysis instance directly
    result: SentimentAnalysis = structured_llm.invoke(unstructured_review)

    print("\n Extracted Pydantic Object:")
    print(f"  Type: {type(result)}")
    print(f"  Product Name : {result.product_name}")
    print(f"  Sentiment    : {result.sentiment}")
    print(f"  Rating       : {result.rating} / 5")
    print(f"  Summary      : {result.summary}")
    print(f"  Pros         : {result.pros}")
    print(f"  Cons         : {result.cons}")

    print("\n Convert directly to JSON for database or API response:")
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    demo_pure_pydantic()
    demo_llm_structured_output()
