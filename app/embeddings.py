import logging

from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading sentence-transformers semantic search model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Semantic search model loaded")
    return _model
