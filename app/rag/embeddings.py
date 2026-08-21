from sentence_transformers import SentenceTransformer

_model = None


def get_embedding_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def embed_text(text: str):
    """Returns a 384-dim embedding as a plain Python list (for pgvector)."""
    model = get_embedding_model()
    return model.encode(text).tolist()


def embed_batch(texts: list):
    """Batch version -- faster for ingesting many chunks at once."""
    model = get_embedding_model()
    return model.encode(texts).tolist()