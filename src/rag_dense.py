import logging
import os

import litert_lm
from qdrant_edge import EdgeShard, Query, QueryRequest
from sentence_transformers import SentenceTransformer

os.environ["HF_HUB_OFFLINE"] = "1"

from config import settings
from prompts import SYSTEM_PROMPT
from utils import silence_stderr

logger = logging.getLogger(__name__)
silence_stderr()


class DenseRAG:
    def __init__(self, shard_path: str | None = None) -> None:
        logger.info("Loading embedding model...")
        self.model = SentenceTransformer(str(settings.embedding.model_name))
        logger.info("Loading shard...")
        self.shard = EdgeShard.load(str(shard_path or settings.shard.dense_path))
        self.model_path = str(settings.llm.model_path)
        logger.info("DenseRAG initialized")

    def retrieve(self, user_query: str, limit: int = 2) -> str:
        logger.info("Retrieving context for: %s", user_query[:50])
        query_vector = self.model.encode_query(user_query).tolist()
        results = self.shard.query(
            QueryRequest(
                query=Query.Nearest(query_vector),
                limit=limit,
                with_payload=True,
            )
        )
        return "".join(point.payload["text"] for point in results)

    def answer(self, user_query: str, limit: int = 3) -> str:
        context = self.retrieve(user_query, limit=limit)
        messages = [litert_lm.Message.system(SYSTEM_PROMPT)]

        logger.info("Loading LLM engine (this may take a moment)...")
        with litert_lm.Engine(
            self.model_path,
            backend=litert_lm.Backend.CPU(),
        ) as engine:
            logger.info("LLM engine loaded, generating response...")
            with engine.create_conversation(messages=messages) as conversation:
                response = conversation.send_message(
                    f"Context:\n{context}\n\nQuestion:\n{user_query}"
                )
        logger.info("Response generated")
        return response["content"][0]["text"]


def main() -> None:
    rag = DenseRAG()
    answer = rag.answer("what are the key risks to the global economy?")
    logger.info("Answer:\n%s", answer)


if __name__ == "__main__":
    main()
