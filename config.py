"""
config.py — Shared LLM Factory for All Providers
==================================================
Import this from any topic file to get a ready-to-use LangChain model.

Usage:
    from config import get_llm, print_header, print_section

    # Get Groq model
    llm = get_llm("groq")

    # Get Mistral model
    llm = get_llm("mistral")

    # Get Google Gemini model
    llm = get_llm("google")
"""

import os
import sys
from typing import Optional, Any
from dotenv import load_dotenv

# Ensure root dir is always in sys.path (so any sub-folder script can import config)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Load .env from root (works regardless of which subfolder runs the script)
load_dotenv(os.path.join(ROOT_DIR, ".env"))

# ──────────────────────────────────────────────────────────────────────────────
# DEFAULT MODELS — update these when provider model names change
# ──────────────────────────────────────────────────────────────────────────────
DEFAULT_MODELS = {
    "groq":    "openai/gpt-oss-120b",
    "mistral": "codestral-2508",
    "google":  "gemini-3.6-flash",
}

PROVIDER_NAMES = {
    "groq":    "Groq (openai/gpt-oss-120b)",
    "mistral": "Mistral (codestral-2508)",
    "google":  "Google Gemini (gemini-3.6-flash)",
}


def get_api_key(provider: str) -> str:
    """Return the API key for the given provider from .env"""
    key_map = {
        "groq":    "GROQ_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "google":  "GOOGLE_API_KEY",
    }
    env_var = key_map.get(provider.lower())
    if not env_var:
        raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(key_map)}")
    key = os.getenv(env_var)
    if not key:
        raise ValueError(f"'{env_var}' is not set in your .env file.")
    return key


def get_llm(provider: str = "mistral", temperature: float = 0.3,
            model: Optional[str] = None, **kwargs: Any):
    """
    Factory: returns a ready-to-use LangChain chat model for the given provider.

    Args:
        provider:    'groq', 'mistral', or 'google'
        temperature: randomness 0.0 (deterministic) → 1.0+ (creative)
        model:       override model name; defaults to DEFAULT_MODELS[provider]
        **kwargs:    any extra kwargs forwarded to the model constructor
    """
    provider = provider.lower()
    api_key  = get_api_key(provider)
    model    = model or DEFAULT_MODELS[provider]

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=api_key, model=model, temperature=temperature, **kwargs)

    elif provider == "mistral":
        from langchain_mistralai import ChatMistralAI
        return ChatMistralAI(api_key=api_key, model=model, temperature=temperature, **kwargs)

    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(api_key=api_key, model=model,
                                      temperature=temperature, **kwargs)
    else:
        raise ValueError(f"Unknown provider '{provider}'.")


# ──────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS — used by every topic file for consistent, pretty output
# ──────────────────────────────────────────────────────────────────────────────
ALL_PROVIDERS = ["groq", "mistral", "google"]


def print_header(title: str, week: int, topic_num: int) -> None:
    width = 60
    print("\n" + "=" * width)
    print(f"  Week {week} | Topic {topic_num}")
    print(f"  {title}")
    print("=" * width)


def print_section(label: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  {label}")
    print(f"{'─' * 50}")


def print_provider_result(provider: str, result: str) -> None:
    name = PROVIDER_NAMES.get(provider, provider)
    print(f"\n  🤖 [{name}]")
    print(f"  {result.strip()}")


def run_all_providers(prompt: str, temperature: float = 0.3) -> None:
    """Run the same prompt through all 3 providers and print results side-by-side."""
    for provider in ALL_PROVIDERS:
        try:
            llm = get_llm(provider, temperature=temperature)
            response = llm.invoke(prompt)
            print_provider_result(provider, response.content)
        except Exception as e:
            print(f"\n  ⚠️  [{PROVIDER_NAMES.get(provider, provider)}] Error: {e}")
