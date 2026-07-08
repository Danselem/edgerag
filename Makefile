.DEFAULT_GOAL := help
SRC := src

## ──────────────────────────────────────────────
## Setup
## ──────────────────────────────────────────────

.PHONY: init
init: ## Create venv, pyproject.toml, and install all deps
	uv venv --python 3.12
	uv init && rm main.py

.PHONY: bootstrap
bootstrap: ## Install deps into existing venv
	uv add --requirements requirements.txt
	uv add --group dev --requirements requirements-dev.txt

.PHONY: install
install: ## Install deps into existing venv
	uv sync

## ──────────────────────────────────────────────
## Models
## ──────────────────────────────────────────────

.PHONY: download
download: ## Download embedding + LLM models
	cd $(SRC) && uv run python download_models.py

## ──────────────────────────────────────────────
## Indexing
## ──────────────────────────────────────────────

.PHONY: index-dense
index-dense: ## Build dense vector index
	cd $(SRC) && uv run python index_dense.py

.PHONY: index-hybrid
index-hybrid: ## Build hybrid (dense+sparse) index
	cd $(SRC) && uv run python index_hybrid.py

.PHONY: index-quantized
index-quantized: ## Build 4-bit quantized index
	cd $(SRC) && uv run python index_quantized.py

.PHONY: index-all
index-all: index-dense index-hybrid index-quantized ## Build all index variants

## ──────────────────────────────────────────────
## Inference
## ──────────────────────────────────────────────

.PHONY: rag-dense
rag-dense: ## Run RAG inference (dense)
	cd $(SRC) && uv run python rag_dense.py

.PHONY: rag-hybrid
rag-hybrid: ## Run RAG inference (hybrid dense+sparse)
	cd $(SRC) && uv run python rag_hybrid.py

.PHONY: rag-quantized
rag-quantized: ## Run RAG inference (quantized)
	cd $(SRC) && uv run python rag_quantized.py

## ──────────────────────────────────────────────
## API + UI
## ──────────────────────────────────────────────

.PHONY: start
start: ## Launch FastAPI + Streamlit together
	cd $(SRC) && uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 &
	cd $(SRC) && uv run streamlit run ui.py

## ──────────────────────────────────────────────
## API
## ──────────────────────────────────────────────

.PHONY: api
api: ## Launch FastAPI server only (port 8000)
	cd $(SRC) && uv run uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

## ──────────────────────────────────────────────
## UI
## ──────────────────────────────────────────────

.PHONY: ui
ui: ## Launch Streamlit only (requires FastAPI running)
	cd $(SRC) && uv run streamlit run ui.py

## ──────────────────────────────────────────────
## Dev
## ──────────────────────────────────────────────

.PHONY: lint
lint: ## Run linter
	uv run ruff check $(SRC)/

.PHONY: fmt
fmt: ## Format code
	uv run ruff format $(SRC)/

## ──────────────────────────────────────────────
## Tests
## ──────────────────────────────────────────────

.PHONY: test
test: ## Run unit + integration tests
	uv run pytest tests/unit/ tests/integration/ -v

.PHONY: test-unit
test-unit: ## Run unit tests only
	uv run pytest tests/unit/ -v

.PHONY: test-integration
test-integration: ## Run integration tests only
	uv run pytest tests/integration/ -v

.PHONY: test-e2e
test-e2e: ## Run e2e tests (requires models on disk)
	uv run pytest tests/e2e/ -v -m e2e

.PHONY: test-cov
test-cov: ## Run tests with coverage report
	uv run pytest tests/unit/ tests/integration/ --cov=src --cov-report=term-missing --cov-report=html

## ──────────────────────────────────────────────
## Clean
## ──────────────────────────────────────────────

.PHONY: clean
clean: ## Remove venv, caches, and shard data
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf shards/

## ──────────────────────────────────────────────
## Help
## ──────────────────────────────────────────────

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}'
