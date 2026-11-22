"""Test rapide du LLM local (Ollama) — python test.py"""
from langchain_ollama import ChatOllama

llm = ChatOllama(model="qwen2.5:7b", temperature=0)
print(llm.invoke("Qui est le meilleur peintre français ? Réponds en une phrase.").content)
