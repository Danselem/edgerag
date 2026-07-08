from pathlib import Path
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sample_text() -> str:
    """Return a 2000-char repeating text for chunk_text tests."""
    return ("abcdefghij" * 200)[:2000]


@pytest.fixture
def sample_settings():
    """Return a Settings-like object with test values (no disk I/O)."""
    from config import DocumentConfig, EmbeddingConfig, LLMConfig, Settings, ShardConfig

    return Settings(
        embedding=EmbeddingConfig(
            repo_id="test/embedding",
            model_name="models/test-embed",
            model_dim=1024,
        ),
        llm=LLMConfig(
            repo_id="test/llm",
            filename="model.tflite",
            model_path="models/test-llm/model.tflite",
        ),
        document=DocumentConfig(
            document_path="pdf/test.pdf",
            chunk_size=512,
            overlap=50,
        ),
        shard=ShardConfig(
            dense_path="shards/dense",
            hybrid_path="shards/hybrid",
            quantized_path="shards/quantized",
        ),
    )


@pytest.fixture
def mock_settings(monkeypatch):
    """Monkeypatch config.settings with controlled test values."""
    from config import DocumentConfig, EmbeddingConfig, LLMConfig, Settings, ShardConfig

    test_settings = Settings(
        embedding=EmbeddingConfig(
            repo_id="test/embedding",
            model_name=Path("models/test-embed"),
            model_dim=1024,
        ),
        llm=LLMConfig(
            repo_id="test/llm",
            filename="model.tflite",
            model_path=Path("models/test-llm/model.tflite"),
        ),
        document=DocumentConfig(
            document_path=Path("pdf/test.pdf"),
            chunk_size=512,
            overlap=50,
        ),
        shard=ShardConfig(
            dense_path=Path("shards/dense"),
            hybrid_path=Path("shards/hybrid"),
            quantized_path=Path("shards/quantized"),
        ),
    )
    monkeypatch.setattr("config.settings", test_settings)
    return test_settings


@pytest.fixture
def tmp_pdf(tmp_path) -> Path:
    """Create a minimal valid PDF in a temp directory."""
    pdf_path = tmp_path / "test.pdf"
    # Minimal valid PDF content
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
        b"2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
        b"3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
        b"/Resources<</Font<</F1 4 0 R>>>>>>endobj\n"
        b"4 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
        b"xref\n0 5\n"
        b"0000000000 65535 f \n"
        b"0000000009 00000 n \n"
        b"0000000058 00000 n \n"
        b"0000000115 00000 n \n"
        b"0000000266 00000 n \n"
        b"trailer<</Size 5/Root 1 0 R>>\n"
        b"startxref\n345\n%%EOF\n"
    )
    pdf_path.write_bytes(pdf_content)
    return pdf_path
