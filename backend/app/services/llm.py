"""
LLM service — fabrique de modèles de chat.

Fournisseurs supportés :
  • Ollama  (langchain_ollama)        — LOCAL, par défaut (aucune clé API, données qui
                                         ne quittent pas la machine)
  • Mistral AI (langchain_mistralai)  — fournisseur historique du projet (cloud)
  • Google Gemini (langchain_google_genai) — optionnel (cloud)

`build_llm()` construit un modèle à partir de paramètres explicites ;
`build_llm_from_settings()` lit la configuration (.env) ;
`build_judge_llm()` construit le juge RAGAS (température 0, sortie longue).
"""
import logging

logger = logging.getLogger("rag.llm")


def build_llm(
    api_key: str = "",
    model_name: str = "qwen2.5:7b",
    max_new_tokens: int = 1024,
    temperature: float = 0.3,
    provider: str = "ollama",
    max_retries: int = 6,
    base_url: str = "http://localhost:11434",
    num_ctx: int = 16384,
    keep_alive: str = "30m",
):
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        logger.info("Initialising Ollama LLM (local): %s @ %s", model_name, base_url)
        kwargs = dict(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_predict=max_new_tokens,
            num_ctx=num_ctx,
            keep_alive=keep_alive,
        )
        try:
            # Désactive le mode « thinking » des modèles de raisonnement
            # (qwen3, gpt-oss…) : on veut des réponses / JSON directs.
            return ChatOllama(reasoning=False, **kwargs)
        except TypeError:                       # vieilles versions de langchain-ollama
            return ChatOllama(**kwargs)

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        logger.info("Initialising Google Gemini LLM: %s", model_name)
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=temperature,
            max_output_tokens=max_new_tokens,
            max_retries=max_retries,
            timeout=120,
        )

    from langchain_mistralai import ChatMistralAI

    logger.info("Initialising Mistral AI LLM: %s", model_name)
    return ChatMistralAI(
        mistral_api_key=api_key,
        model=model_name,
        max_tokens=max_new_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )


def _api_key(settings, provider: str) -> str:
    if provider == "ollama":
        return ""
    key = settings.GOOGLE_API_KEY if provider == "google" else settings.MISTRAL_API_KEY
    if not key:
        raise RuntimeError(
            f"Fournisseur '{provider}' choisi mais aucune clé API trouvée dans .env "
            "(MISTRAL_API_KEY / GOOGLE_API_KEY). Utiliser LLM_PROVIDER=ollama pour du 100 % local."
        )
    return key


def build_llm_from_settings(settings, *, temperature: float | None = None,
                            max_new_tokens: int | None = None):
    """Construit le LLM applicatif décrit par la configuration."""
    provider = settings.resolved_provider
    return build_llm(
        api_key=_api_key(settings, provider),
        model_name=settings.resolved_model,
        max_new_tokens=max_new_tokens or settings.llm_max_new_tokens,
        temperature=settings.llm_temperature if temperature is None else temperature,
        provider=provider,
        base_url=settings.ollama_base_url,
        num_ctx=settings.ollama_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )


def build_judge_llm(settings, model_name: str | None = None):
    """Juge RAGAS : température 0 (sorties déterministes) et sortie longue
    (le JSON structuré demandé par RAGAS peut être volumineux)."""
    provider = settings.judge_provider.lower().strip()
    if provider == "auto":
        provider = settings.resolved_provider
    default = {
        "ollama": settings.judge_model,
        "mistral": settings.mistral_default_model,
        "google": settings.google_default_model,
    }[provider]
    return build_llm(
        api_key=_api_key(settings, provider),
        model_name=model_name or default,
        max_new_tokens=settings.judge_max_new_tokens,
        temperature=0.0,
        provider=provider,
        base_url=settings.ollama_base_url,
        num_ctx=settings.judge_num_ctx,
        keep_alive=settings.ollama_keep_alive,
    )


def llm_text(result) -> str:
    """Contenu texte d'une réponse LLM (gère les contenus multi-parties)."""
    content = result.content if hasattr(result, "content") else result
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, str):
                parts.append(p)
            elif isinstance(p, dict) and p.get("type", "text") == "text":
                parts.append(p.get("text", ""))
        return "".join(parts)
    text = str(content)
    # Certains modèles locaux laissent passer un bloc de raisonnement.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text.strip()
