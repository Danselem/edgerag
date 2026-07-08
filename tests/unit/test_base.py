import pytest

from base import chunk_text


class TestChunkTextValidation:
    def test_empty_string_returns_empty_list(self):
        assert chunk_text("") == []

    def test_chunk_size_zero_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=0)

    def test_chunk_size_negative_raises(self):
        with pytest.raises(ValueError, match="chunk_size must be positive"):
            chunk_text("hello", chunk_size=-1)

    def test_overlap_negative_raises(self):
        with pytest.raises(ValueError, match="overlap must be non-negative"):
            chunk_text("hello", overlap=-1)

    def test_overlap_equals_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            chunk_text("hello", chunk_size=5, overlap=5)

    def test_overlap_greater_than_chunk_size_raises(self):
        with pytest.raises(ValueError, match="overlap.*must be less than chunk_size"):
            chunk_text("hello", chunk_size=3, overlap=5)


class TestChunkTextBehavior:
    def test_text_shorter_than_chunk_size(self, sample_text):
        result = chunk_text("hello", chunk_size=1024)
        assert result == ["hello"]

    def test_text_exactly_chunk_size(self):
        text = "a" * 100
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert len(result) == 1
        assert result[0] == text

    def test_text_multiple_of_chunk_size(self):
        text = "a" * 200
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert result[0] == "a" * 100
        assert result[1] == "a" * 100

    def test_text_not_multiple_of_chunk_size(self):
        text = "a" * 150
        result = chunk_text(text, chunk_size=100, overlap=0)
        assert len(result) == 2
        assert result[0] == "a" * 100
        assert result[1] == "a" * 50

    def test_overlap_creates_overlapping_chunks(self):
        text = "a" * 150
        result = chunk_text(text, chunk_size=100, overlap=20)
        assert len(result) == 2
        assert result[0] == "a" * 100
        assert result[1] == "a" * 70

    def test_chunk_size_one(self):
        text = "abc"
        result = chunk_text(text, chunk_size=1, overlap=0)
        assert result == ["a", "b", "c"]

    def test_whitespace_stripped_from_chunks(self):
        text = "  hello   world  "
        result = chunk_text(text, chunk_size=20, overlap=0)
        assert all(chunk == chunk.strip() for chunk in result)

    def test_empty_chunks_filtered_out(self):
        text = "   "
        result = chunk_text(text, chunk_size=10, overlap=0)
        assert result == []

    def test_realistic_text_chunking(self, sample_text):
        result = chunk_text(sample_text, chunk_size=500, overlap=50)
        assert len(result) > 1
        for chunk in result:
            assert len(chunk) <= 500
            assert chunk == chunk.strip()
