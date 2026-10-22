# Shortcuts for the common tasks of the project. Run `make help` to list them.

DATA ?= data/abalone.csv
IMAGE ?= abalone-api
CONTAINER ?= abalone-api
# Host ports, bound to the container ports 8001 (API) and 4201 (Prefect)
API_PORT ?= 8000
PREFECT_PORT ?= 4200

# The modelling modules import each other by name (e.g. `from utils import ...`)
export PYTHONPATH := src/modelling

.DEFAULT_GOAL := help
.PHONY: help install lint train prefect-config prefect-server deploy api \
	docker-build docker-run docker-stop clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

install: ## Create the environment and install the pre-commit hooks
	uv sync
	uv run pre-commit install

lint: ## Run the pre-commit hooks (ruff linting and formatting) on all files
	uv run pre-commit run --all-files

train: ## Train the model once on DATA and save the encoder and model
	uv run python src/modelling/main.py $(DATA)

prefect-config: ## Point Prefect to the local server (once per machine)
	uv run prefect config set PREFECT_API_URL=http://127.0.0.1:$(PREFECT_PORT)/api

prefect-server: ## Start the Prefect server (UI on http://127.0.0.1:4200)
	uv run prefect server start

deploy: ## Serve the deployment that retrains the model every day at 1 pm
	uv run python src/modelling/deployment.py

api: ## Run the prediction API locally (docs on http://127.0.0.1:8001/docs)
	uv run uvicorn src.web_service.main:app --host 127.0.0.1 --port 8001 --reload

docker-build: ## Build the Docker image of the API and the Prefect server
	docker build -f Dockerfile.app -t $(IMAGE) .

docker-run: ## Run the container (API on port 8000, Prefect UI on port 4200)
	docker run --rm --name $(CONTAINER) \
		-p 0.0.0.0:$(API_PORT):8001 -p 0.0.0.0:$(PREFECT_PORT):4201 $(IMAGE)

docker-stop: ## Stop the running container
	docker stop $(CONTAINER)

clean: ## Remove the Python caches
	find . -type d -name "__pycache__" -not -path "./.venv/*" -prune -exec rm -rf {} +
	rm -rf .ruff_cache

