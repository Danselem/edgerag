import pytest
from pydantic import ValidationError

from api.models import (
    ConfigResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
)


class TestQueryRequest:
    def test_defaults(self):
        req = QueryRequest(query="What is AI?")
        assert req.query == "What is AI?"
        assert req.shard_type == "dense"
        assert req.limit == 2

    def test_custom_values(self):
        req = QueryRequest(query="test", shard_type="hybrid", limit=5)
        assert req.shard_type == "hybrid"
        assert req.limit == 5

    def test_quantized_shard_type(self):
        req = QueryRequest(query="test", shard_type="quantized")
        assert req.shard_type == "quantized"

    def test_invalid_shard_type_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", shard_type="invalid")

    def test_limit_zero_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=0)

    def test_limit_eleven_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="test", limit=11)

    def test_empty_query_rejected(self):
        with pytest.raises(ValidationError):
            QueryRequest(query="")

    def test_whitespace_only_query_accepted(self):
        req = QueryRequest(query="   ")
        assert req.query == "   "


class TestQueryResponse:
    def test_serialization_roundtrip(self):
        resp = QueryResponse(
            answer="AI is artificial intelligence.",
            context=["AI context chunk 1"],
            metadata={"shard_type": "dense", "limit": 2},
        )
        data = resp.model_dump()
        assert data["answer"] == "AI is artificial intelligence."
        assert len(data["context"]) == 1
        assert data["metadata"]["shard_type"] == "dense"


class TestHealthResponse:
    def test_default_values(self):
        resp = HealthResponse()
        assert resp.status == "ok"
        assert resp.version == "0.1.0"


class TestConfigResponse:
    def test_fields(self):
        resp = ConfigResponse(
            embedding_model="Qwen/Qwen3-Embedding-0.6B",
            llm_model="google/gemma-4-2b",
            document="report.pdf",
            shard_types=["dense", "hybrid", "quantized"],
        )
        assert resp.embedding_model == "Qwen/Qwen3-Embedding-0.6B"
        assert len(resp.shard_types) == 3
