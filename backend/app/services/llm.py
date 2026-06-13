"""
LLM service — Mistral AI (cloud) via langchain_mistralai.
"""
import logging

from langchain_mistralai import ChatMistralAI

logger = logging.getLogger("rag.llm")


def build_llm(
    api_key: str,
    model_name: str = "mistral-small-latest",
    max_new_tokens: int = 512,
    temperature: float = 0.3,
) -> ChatMistralAI:
    logger.info("Initialising Mistral AI LLM: %s", model_name)
    llm = ChatMistralAI(
        mistral_api_key=api_key,
        model=model_name,
        max_tokens=max_new_tokens,
        temperature=temperature,
    )
    logger.info("Mistral AI LLM ready (%s)", model_name)
    return llm
