#!/usr/bin/env python
"""
Run once to parse the RGPD HTML and persist the ChromaDB index to disk.
After this, the server loads from disk on every start (no re-indexing needed).

Usage:
    python scripts/build_index.py
    python scripts/build_index.py --html data/L_2016119FR.01000101.html
"""
import argparse
import logging
import sys
from pathlib import Path

# Make sure app/ is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.core.config   import get_settings
from app.core.logging  import setup_logging
from app.services.indexing    import build_from_html
from app.services.vectorstore import build_embedding_model, load_or_build_vectorstore

setup_logging()
logger = logging.getLogger("rag.build_index")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--html", default=None, help="Path to RGPD HTML file")
    args = parser.parse_args()

    settings  = get_settings()
    html_path = args.html or settings.rgpd_html_path

    logger.info("Building index from: %s", html_path)
    parent_docs, child_docs, _ = build_from_html(html_path)

    logger.info("Loading embedding model…")
    embedding = build_embedding_model(settings.embedding_model)

    # Force rebuild
    import shutil
    persist = Path(settings.chroma_persist_dir)
    if persist.exists():
        logger.warning("Removing existing index at %s", persist)
        shutil.rmtree(persist)

    load_or_build_vectorstore(
        child_documents=child_docs,
        embedding_model=embedding,
        persist_dir=settings.chroma_persist_dir,
    )
    logger.info("Index built ✓  (%d parent docs)", len(parent_docs))


if __name__ == "__main__":
    main()