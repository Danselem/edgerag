from pathlib import Path

import edgeparse
from sentence_transformers import SentenceTransformer

from config import settings


def chunk_text(text: str, chunk_size: int = 1024, overlap: int = 0) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if overlap < 0:
        raise ValueError(f"overlap must be non-negative, got {overlap}")
    if overlap >= chunk_size:
        raise ValueError(f"overlap ({overlap}) must be less than chunk_size ({chunk_size})")

    return [
        chunk
        for start in range(0, len(text), chunk_size - overlap)
        if (chunk := text[start : start + chunk_size].strip())
    ]


def load_embedder(model_name: str | None = None) -> SentenceTransformer:
    return SentenceTransformer(str(model_name or settings.embedding.model_name))


def parse_document(document_path: str | Path) -> str:
    path = Path(document_path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"Expected a PDF file, got: {path.suffix}")
    return edgeparse.convert(str(path), format="markdown")
