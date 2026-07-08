import logging
from pathlib import Path

import edgeparse
from qdrant_edge import (
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeVectorParams,
    Point,
    Query,
    QueryRequest,
    TurboQuantBitSize,
    TurboQuantQuantizationConfig,
    UpdateOperation,
)

from base import chunk_text, load_embedder
from config import settings

logger = logging.getLogger(__name__)


class QuantizedIndexer:
    def __init__(self, shard_path: str | None = None) -> None:
        self.shard_path = Path(shard_path or settings.shard.quantized_path)
        self.model = load_embedder()
        self.shard = self._load_or_create_shard()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors=EdgeVectorParams(
                size=settings.embedding.model_dim,
                distance=Distance.Cosine,
            ),
            quantization_config=TurboQuantQuantizationConfig(
                always_ram=True,
                bits=TurboQuantBitSize.Bits4,
            ),
        )
        self.shard_path.mkdir(parents=True, exist_ok=True)
        if any(self.shard_path.iterdir()):
            return EdgeShard.load(str(self.shard_path))
        return EdgeShard.create(str(self.shard_path), config)

    def index(self, document_path: str | None = None) -> int:
        doc_path = document_path or settings.document.document_path
        markdown = edgeparse.convert(str(doc_path), format="markdown")
        chunks = chunk_text(
            markdown,
            chunk_size=settings.document.chunk_size,
            overlap=settings.document.overlap,
        )
        embeddings = self.model.encode_document(chunks)

        self.shard.update(
            UpdateOperation.upsert_points(
                [
                    Point(
                        index,
                        embeddings[index - 1].tolist(),
                        {
                            "text": chunk,
                            "source": Path(doc_path).name,
                            "chunk_index": index - 1,
                        },
                    )
                    for index, chunk in enumerate(chunks, start=1)
                ]
            )
        )
        self.shard.optimize()

        info = self.shard.info()
        logger.info("Points: %d", info.points_count)
        logger.info("Indexed vectors: %d", info.indexed_vectors_count)
        return len(chunks)

    def retrieve(self, user_query: str, limit: int = 2) -> str:
        query_vector = self.model.encode_query(user_query).tolist()
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)


def main() -> None:
    indexer = QuantizedIndexer()
    count = indexer.index()
    logger.info("Indexed %d chunks", count)

    context = indexer.retrieve("what are the key risks to the global economy?")
    logger.debug("Context:\n%s", context)


if __name__ == "__main__":
    main()
