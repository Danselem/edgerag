import pytest
from pathlib import Path

from config import (
    DocumentConfig,
    EmbeddingConfig,
    LLMConfig,
    Settings,
    ShardConfig,
)


class TestEmbeddingConfig:
    def test_valid_data(self):
        cfg = EmbeddingConfig(
            repo_id="Qwen/Qwen3-Embedding-0.6B",
            model_name="models/qwen3_embed",
            model_dim=1024,
        )
        assert cfg.repo_id == "Qwen/Qwen3-Embedding-0.6B"
        assert cfg.model_dim == 1024

    def test_model_dim_zero_rejected(self):
        with pytest.raises(Exception):
            EmbeddingConfig(
                repo_id="test",
                model_name="models/test",
                model_dim=0,
            )

    def test_model_dim_negative_rejected(self):
        with pytest.raises(Exception):
            EmbeddingConfig(
                repo_id="test",
                model_name="models/test",
                model_dim=-1,
            )

    def test_resolve_joins_root(self, tmp_path):
        cfg = EmbeddingConfig(
            repo_id="test",
            model_name="models/embed",
            model_dim=512,
        )
        cfg.resolve(tmp_path)
        assert cfg.model_name == (tmp_path / "models/embed").resolve()


class TestLLMConfig:
    def test_valid_data(self):
        cfg = LLMConfig(
            repo_id="google/gemma-4-2b",
            filename="gemma4-2b.tflite",
            model_path="models/gemma4",
        )
        assert cfg.filename == "gemma4-2b.tflite"

    def test_resolve_joins_root(self, tmp_path):
        cfg = LLMConfig(
            repo_id="test",
            filename="model.tflite",
            model_path="models/llm",
        )
        cfg.resolve(tmp_path)
        assert cfg.model_path == (tmp_path / "models/llm").resolve()


class TestDocumentConfig:
    def test_valid_data(self):
        cfg = DocumentConfig(
            document_path="pdf/test.pdf",
            chunk_size=1024,
            overlap=100,
        )
        assert cfg.chunk_size == 1024
        assert cfg.overlap == 100

    def test_chunk_size_zero_rejected(self):
        with pytest.raises(Exception):
            DocumentConfig(
                document_path="pdf/test.pdf",
                chunk_size=0,
                overlap=0,
            )

    def test_overlap_negative_rejected(self):
        with pytest.raises(Exception):
            DocumentConfig(
                document_path="pdf/test.pdf",
                chunk_size=100,
                overlap=-1,
            )

    def test_resolve_joins_root(self, tmp_path):
        cfg = DocumentConfig(
            document_path="pdf/doc.pdf",
            chunk_size=512,
            overlap=50,
        )
        cfg.resolve(tmp_path)
        assert cfg.document_path == (tmp_path / "pdf/doc.pdf").resolve()


class TestShardConfig:
    def test_valid_data(self):
        cfg = ShardConfig(
            dense_path="shards/dense",
            hybrid_path="shards/hybrid",
            quantized_path="shards/quantized",
        )
        assert cfg.dense_path == Path("shards/dense")

    def test_resolve_joins_all_paths(self, tmp_path):
        cfg = ShardConfig(
            dense_path="shards/dense",
            hybrid_path="shards/hybrid",
            quantized_path="shards/quantized",
        )
        cfg.resolve(tmp_path)
        assert cfg.dense_path == (tmp_path / "shards/dense").resolve()
        assert cfg.hybrid_path == (tmp_path / "shards/hybrid").resolve()
        assert cfg.quantized_path == (tmp_path / "shards/quantized").resolve()


class TestSettings:
    def test_assembles_nested_configs(self):
        settings = Settings(
            embedding=EmbeddingConfig(
                repo_id="test/embed",
                model_name="models/embed",
                model_dim=512,
            ),
            llm=LLMConfig(
                repo_id="test/llm",
                filename="model.tflite",
                model_path="models/llm",
            ),
            document=DocumentConfig(
                document_path="pdf/doc.pdf",
                chunk_size=256,
                overlap=32,
            ),
            shard=ShardConfig(
                dense_path="shards/dense",
                hybrid_path="shards/hybrid",
                quantized_path="shards/quantized",
            ),
        )
        assert settings.embedding.model_dim == 512
        assert settings.document.chunk_size == 256

    def test_resolve_delegates_to_all(self, tmp_path):
        settings = Settings(
            embedding=EmbeddingConfig(
                repo_id="test",
                model_name="embed",
                model_dim=128,
            ),
            llm=LLMConfig(
                repo_id="test",
                filename="m.tflite",
                model_path="llm",
            ),
            document=DocumentConfig(
                document_path="doc.pdf",
                chunk_size=64,
                overlap=8,
            ),
            shard=ShardConfig(
                dense_path="d",
                hybrid_path="h",
                quantized_path="q",
            ),
        )
        settings.resolve(tmp_path)
        assert settings.embedding.model_name == (tmp_path / "embed").resolve()
        assert settings.llm.model_path == (tmp_path / "llm").resolve()
        assert settings.document.document_path == (tmp_path / "doc.pdf").resolve()
        assert settings.shard.dense_path == (tmp_path / "d").resolve()
