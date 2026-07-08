from unittest.mock import MagicMock, patch

import pytest


class TestDownloadEmbeddingModel:
    @patch("download_models.snapshot_download")
    def test_calls_snapshot_download(self, mock_snapshot):
        from download_models import download_embedding_model
        download_embedding_model()
        mock_snapshot.assert_called_once()
        call_kwargs = mock_snapshot.call_args
        assert "Qwen" in str(call_kwargs)


class TestDownloadGemmaModel:
    @patch("download_models.hf_hub_download")
    def test_calls_hf_hub_download(self, mock_hf):
        from download_models import download_gemma_model
        download_gemma_model()
        mock_hf.assert_called_once()
        call_kwargs = mock_hf.call_args
        assert "gemma" in str(call_kwargs).lower() or "google" in str(call_kwargs)
