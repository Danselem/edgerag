import pytest


@pytest.mark.e2e
class TestDenseRAGPipeline:
    def test_answer_returns_non_empty_string(self):
        from rag_dense import DenseRAG
        rag = DenseRAG()
        answer = rag.answer("What are the key themes?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_retrieve_returns_context(self):
        from rag_dense import DenseRAG
        rag = DenseRAG()
        context = rag.retrieve("What is the AI supercycle?")
        assert isinstance(context, str)
        assert len(context) > 0


@pytest.mark.e2e
class TestHybridRAGPipeline:
    def test_answer_returns_non_empty_string(self):
        from rag_hybrid import HybridRAG
        rag = HybridRAG()
        answer = rag.answer("What are the key themes?")
        assert isinstance(answer, str)
        assert len(answer) > 0

    def test_retrieve_returns_context(self):
        from rag_hybrid import HybridRAG
        rag = HybridRAG()
        context = rag.retrieve("What is inflation dynamics?")
        assert isinstance(context, str)
        assert len(context) > 0
