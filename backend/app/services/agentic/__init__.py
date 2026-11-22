"""
RAG agentique multi-agents (LangGraph).

Inspiré des architectures vues dans les cours IBM / Coursera
« Agentic AI with LangChain and LangGraph » et
« Multi-Agent Systems and Agentic RAG with LangGraph » :
  • routage des requêtes (Router / Planner agent)
  • RAG correctif — CRAG (Document Grader + Query Rewriter)
  • Reflection / Reflexion (Critic agent → Reviser agent, avec recherche
    complémentaire pilotée par l'auto-critique)
  • orchestration de plusieurs agents spécialisés par un graphe d'état partagé.
"""
from app.services.agentic.graph import AgenticRAGPipeline

__all__ = ["AgenticRAGPipeline"]
