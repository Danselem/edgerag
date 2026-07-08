import logging
from pathlib import Path

import edgeparse
from qdrant_edge import (
    Bm25,
    Bm25Config,
    Distance,
    EdgeConfig,
    EdgeShard,
    EdgeSparseVectorParams,
    EdgeVectorParams,
    Fusion,
    Modifier,
    Point,
    Prefetch,
    Query,
    QueryRequest,
    UpdateOperation,
)

from base import chunk_text, load_embedder
from config import settings

logger = logging.getLogger(__name__)


class HybridIndexer:
    def __init__(self, shard_path: str | None = None) -> None:
        self.shard_path = Path(shard_path or settings.shard.hybrid_path)
        self.model = load_embedder()
        self.bm25 = Bm25(Bm25Config(language="english"))
        self.shard = self._load_or_create_shard()

    def _load_or_create_shard(self) -> EdgeShard:
        config = EdgeConfig(
            vectors={
                "dense": EdgeVectorParams(
                    size=settings.embedding.model_dim,
                    distance=Distance.Cosine,
                )
            },
            sparse_vectors={
                "sparse": EdgeSparseVectorParams(modifier=Modifier.Idf)
            },
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
                        {
                            "dense": embeddings[index - 1].tolist(),
                            "sparse": self.bm25.embed_document(chunk),
                        },
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
        return len(chunks)

    def retrieve(self, user_query: str, limit: int = 2) -> str:
        dense_query = self.model.encode_query(user_query).tolist()
        sparse_query = self.bm25.embed_query(user_query)

        results = self.shard.query(
            QueryRequest(
                prefetches=[
                    Prefetch(
                        query=Query.Nearest(dense_query, using="dense"),
                        limit=limit,
                    ),
                    Prefetch(
                        query=Query.Nearest(sparse_query, using="sparse"),
                        limit=limit,
                    ),
                ],
                query=Fusion.Rrf(k=60),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)


def main() -> None:
    indexer = HybridIndexer()
    count = indexer.index()
    logger.info("Indexed %d chunks", count)

    context = indexer.retrieve("what are the key risks to the global economy?")
    logger.debug("Context:\n%s", context)


if __name__ == "__main__":
    main()
