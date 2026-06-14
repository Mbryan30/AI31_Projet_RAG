"""
Notebook cells 2 & 3 — parsing RGPD HTML + building parent/child docs.
Extracted into a standalone service so it can be called once at startup
(or re-triggered via POST /index).
"""
import re
import json
import uuid
import logging
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger("rag.indexing")


# ── HTML → structured JSON ────────────────────────────────────────────

def parse_rgpd_html(html_path: str) -> list[dict]:
    """Return a list of chapters, each with a list of article dicts."""
    path = Path(html_path)
    if not path.exists():
        raise FileNotFoundError(f"RGPD HTML not found: {html_path}")

    logger.info("Parsing RGPD HTML: %s", html_path)
    soup = BeautifulSoup(path.read_text(encoding="utf-8"), "html.parser")

    def clean(el) -> str:
        return el.get_text(" ", strip=True) if el else ""

    result: list[dict] = []
    for chapitre_div in soup.find_all("div", id=re.compile(r"^cpt_[IVXLCDM]+$")):
        nom    = clean(chapitre_div.find("p", class_="oj-ti-section-1"))
        titre  = clean(chapitre_div.find("p", class_="oj-ti-section-2"))
        if not nom.startswith("CHAPITRE"):
            continue

        chapter: dict = {"chapitre": nom, "titre_chapitre": titre, "contenu": []}
        seen: set[str] = set()

        for art_div in chapitre_div.find_all("div", id=re.compile(r"^art_\d+$")):
            art_id = art_div.get("id")
            if art_id in seen:
                continue
            seen.add(art_id)

            numero = clean(art_div.find("p", class_="oj-ti-art"))
            titre_art = clean(art_div.find("p", class_="oj-sti-art"))
            if not numero.startswith("Article"):
                continue

            contenu = "\n".join(
                clean(p) for p in art_div.find_all("p", class_="oj-normal") if clean(p)
            )
            chapter["contenu"].append({
                "article": numero,
                "titre_article": titre_art,
                "contenu_article": contenu,
            })

        result.append(chapter)

    logger.info(
        "Parsed %d chapters, %d articles",
        len(result),
        sum(len(c["contenu"]) for c in result),
    )
    return result


# ── structured JSON → LangChain Documents ────────────────────────────

def build_documents(
    data: list[dict],
    chunk_size: int = 300,
    chunk_overlap: int = 50,
) -> tuple[list[Document], list[Document], dict[str, Document]]:
    """
    Returns:
        parent_documents  — full article texts
        child_documents   — small chunks for embedding
        parent_index      — {parent_id: parent_doc}
    """
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    parent_documents: list[Document] = []
    child_documents:  list[Document] = []

    for chapter in data:
        chap_nom   = chapter.get("chapitre", "")
        chap_titre = chapter.get("titre_chapitre", "")

        for article in chapter.get("contenu", []):
            numero   = article.get("article", "")
            titre    = article.get("titre_article", "")
            contenu  = article.get("contenu_article", "")
            if not contenu.strip():
                continue

            # ID DÉTERMINISTE : ainsi le parent_id stocké dans ChromaDB
            # (build initial) reste identique au parent_id reconstruit en
            # mémoire à chaque démarrage. Sinon les chunks de la DB pointent
            # vers des parents introuvables et retrieve_parents renvoie [].
            parent_id = f"{chap_nom}::{numero}"
            parent_text = (
                f"{chap_nom} – {chap_titre}\n"
                f"{numero} – {titre}\n\n"
                f"{contenu}"
            )
            meta_base = {
                "chapitre":       chap_nom,
                "titre_chapitre": chap_titre,
                "article":        numero,
                "titre_article":  titre,
                "source":         "RGPD",
                "parent_id":      parent_id,
            }

            parent_documents.append(Document(
                page_content=parent_text,
                metadata={**meta_base, "type": "parent"},
            ))

            for i, chunk in enumerate(child_splitter.split_text(contenu)):
                child_documents.append(Document(
                    page_content=chunk,
                    metadata={**meta_base, "type": "child_chunk", "chunk_index": i},
                ))

    parent_index = {doc.metadata["parent_id"]: doc for doc in parent_documents}
    logger.info(
        "Built %d parents, %d child chunks",
        len(parent_documents),
        len(child_documents),
    )
    return parent_documents, child_documents, parent_index


# ── Convenience: parse + build in one call ───────────────────────────

def build_from_html(html_path: str) -> tuple[list[Document], list[Document], dict[str, Document]]:
    data = parse_rgpd_html(html_path)
    return build_documents(data)