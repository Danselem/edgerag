import logging
import os

from fastapi import APIRouter, HTTPException

os.environ["HF_HUB_OFFLINE"] = "1"

from api.models import ConfigResponse, HealthResponse, QueryRequest, QueryResponse
from config import settings
from prompts import SYSTEM_PROMPT

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


def _retrieve_context(query: str, shard_type: str, limit: int) -> str:
    from qdrant_edge import (
        Bm25,
        Bm25Config,
        EdgeShard,
        Fusion,
        Prefetch,
        Query,
        QueryRequest as QdQueryRequest,
    )
    from sentence_transformers import SentenceTransformer

    embedder = SentenceTransformer(str(settings.embedding.model_name))

    path_map = {
        "dense": settings.shard.dense_path,
        "hybrid": settings.shard.hybrid_path,
        "quantized": settings.shard.quantized_path,
    }
    shard = EdgeShard.load(str(path_map[shard_type]))

    if shard_type == "hybrid":
        bm25 = Bm25(Bm25Config(language="english"))
        dense_query = embedder.encode_query(query).tolist()
        sparse_query = bm25.embed_query(query)
        results = shard.query(
            QdQueryRequest(
                prefetches=[
                    Prefetch(query=Query.Nearest(dense_query, using="dense"), limit=limit),
                    Prefetch(query=Query.Nearest(sparse_query, using="sparse"), limit=limit),
                ],
                query=Fusion.Rrf(k=60),
                limit=limit,
                with_payload=True,
            )
        )
    else:
        query_vector = embedder.encode_query(query).tolist()
        results = shard.query(
            QdQueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )

    return "".join(point.payload["text"] for point in results)


def _generate_answer(query: str, context: str) -> str:
    import litert_lm

    engine = litert_lm.Engine(
        str(settings.llm.model_path),
        backend=litert_lm.Backend.CPU(),
    )
    messages = [litert_lm.Message.system(SYSTEM_PROMPT)]
    prompt = f"Context:\n{context}\n\nQuestion:\n{query}"

    with engine.create_conversation(messages=messages) as conversation:
        response = conversation.send_message(prompt)
        return response["content"][0]["text"]


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@router.get("/config", response_model=ConfigResponse)
def config() -> ConfigResponse:
    return ConfigResponse(
        embedding_model=settings.embedding.repo_id,
        llm_model=settings.llm.repo_id,
        document=settings.document.document_path.name,
        shard_types=["dense", "hybrid", "quantized"],
    )


@router.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    try:
        context = _retrieve_context(req.query, req.shard_type, req.limit)
        answer = _generate_answer(req.query, context)
        return QueryResponse(
            answer=answer,
            context=[context],
            metadata={
                "shard_type": req.shard_type,
                "limit": req.limit,
                "document": settings.document.document_path.name,
            },
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))
