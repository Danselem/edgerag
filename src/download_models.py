import logging

from huggingface_hub import hf_hub_download, snapshot_download

from config import settings

logger = logging.getLogger(__name__)


def download_embedding_model() -> None:
    logger.info("Downloading embedding model: %s", settings.embedding.repo_id)
    snapshot_download(
        repo_id=settings.embedding.repo_id,
        local_dir=str(settings.embedding.model_name),
        local_dir_use_symlinks=False,
    )
    logger.info("Embedding model saved at: %s", settings.embedding.model_name)


def download_gemma_model() -> None:
    logger.info("Downloading LLM model: %s", settings.llm.repo_id)
    model_path = hf_hub_download(
        repo_id=settings.llm.repo_id,
        filename=settings.llm.filename,
        local_dir=str(settings.llm.model_path.parent),
    )
    logger.info("LLM model saved at: %s", model_path)


def main() -> None:
    download_embedding_model()
    download_gemma_model()


if __name__ == "__main__":
    main()
