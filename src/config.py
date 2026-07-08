import logging
import sys
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


class EmbeddingConfig(BaseModel):
    repo_id: str
    model_name: Path
    model_dim: int = Field(gt=0)

    def resolve(self, root: Path) -> None:
        self.model_name = (root / self.model_name).resolve()


class LLMConfig(BaseModel):
    repo_id: str
    filename: str
    model_path: Path

    def resolve(self, root: Path) -> None:
        self.model_path = (root / self.model_path).resolve()


class DocumentConfig(BaseModel):
    document_path: Path
    chunk_size: int = Field(gt=0)
    overlap: int = Field(ge=0)

    def resolve(self, root: Path) -> None:
        self.document_path = (root / self.document_path).resolve()


class ShardConfig(BaseModel):
    dense_path: Path
    hybrid_path: Path
    quantized_path: Path

    def resolve(self, root: Path) -> None:
        self.dense_path = (root / self.dense_path).resolve()
        self.hybrid_path = (root / self.hybrid_path).resolve()
        self.quantized_path = (root / self.quantized_path).resolve()


class Settings(BaseModel):
    embedding: EmbeddingConfig
    llm: LLMConfig
    document: DocumentConfig
    shard: ShardConfig

    def resolve(self, root: Path) -> None:
        self.embedding.resolve(root)
        self.llm.resolve(root)
        self.document.resolve(root)
        self.shard.resolve(root)


with open(_CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)

settings = Settings(**_cfg)
settings.resolve(_PROJECT_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
