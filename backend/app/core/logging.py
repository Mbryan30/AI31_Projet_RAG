import logging
import sys
from app.core.config import get_settings


def setup_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Console Windows en UTF-8 : sinon les caractères spéciaux des réponses
    # (emoji ⚠️, etc.) font planter les logs avec UnicodeEncodeError (cp1252).
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]


logger = logging.getLogger("rag")