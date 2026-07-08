from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a FastAPI TestClient with mocked dependencies."""
    with patch("api.routes._retrieve_context") as mock_retrieve, \
         patch("api.routes._generate_answer") as mock_generate:
        mock_retrieve.return_value = "Test context from the document."
        mock_generate.return_value = "Test answer based on context."
        from api.app import app
        yield TestClient(app), mock_retrieve, mock_generate


class TestRootEndpoint:
    def test_root_returns_welcome(self, client):
        client, _, _ = client
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Edge RAG API"
        assert data["docs"] == "/docs"


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        client, _, _ = client
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"


class TestConfigEndpoint:
    def test_config_returns_metadata(self, client):
        client, _, _ = client
        response = client.get("/api/v1/config")
        assert response.status_code == 200
        data = response.json()
        assert "embedding_model" in data
        assert "llm_model" in data
        assert "document" in data
        assert data["shard_types"] == ["dense", "hybrid", "quantized"]


class TestQueryEndpoint:
    def test_query_happy_path(self, client):
        client, mock_retrieve, mock_generate = client
        response = client.post(
            "/api/v1/query",
            json={"query": "What is AI?", "shard_type": "dense", "limit": 2},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["answer"] == "Test answer based on context."
        assert data["context"] == ["Test context from the document."]
        assert data["metadata"]["shard_type"] == "dense"
        mock_retrieve.assert_called_once_with("What is AI?", "dense", 2)

    def test_query_invalid_shard_type(self, client):
        client, _, _ = client
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "shard_type": "invalid"},
        )
        assert response.status_code == 422

    def test_query_empty_body(self, client):
        client, _, _ = client
        response = client.post("/api/v1/query", json={})
        assert response.status_code == 422

    def test_query_retrieve_exception_returns_500(self, client):
        client, mock_retrieve, _ = client
        mock_retrieve.side_effect = RuntimeError("Shard not found")
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "shard_type": "dense"},
        )
        assert response.status_code == 500
        assert "Shard not found" in response.json()["detail"]

    def test_query_generate_exception_returns_500(self, client):
        client, _, mock_generate = client
        mock_generate.side_effect = RuntimeError("LLM load failed")
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "shard_type": "dense"},
        )
        assert response.status_code == 500
        assert "LLM load failed" in response.json()["detail"]

    def test_query_hybrid_shard_type(self, client):
        client, mock_retrieve, mock_generate = client
        response = client.post(
            "/api/v1/query",
            json={"query": "test", "shard_type": "hybrid", "limit": 3},
        )
        assert response.status_code == 200
        mock_retrieve.assert_called_once_with("test", "hybrid", 3)
