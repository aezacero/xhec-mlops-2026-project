<div align="center">

# MLOps Project: Abalone Age Prediction

[![Python Version](https://img.shields.io/badge/python-3.10%20or%203.11-blue.svg)]()
[![Linting: ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-informational?logo=pre-commit&logoColor=white)](https://github.com/aezacero/xhec-mlops-2026-project/blob/main/.pre-commit-config.yaml)
</div>

## Project Overview

Abalone are sea snails whose age is traditionally found by cutting the shell, staining it and counting its rings under a microscope, a slow and tedious process. This project predicts the number of rings (and so the age, roughly rings + 1.5 years) from simple physical measurements such as the length, the diameter and the weights of the abalone.

The project turns a notebook model into a small production system:

- a **training pipeline** written as Python modules and orchestrated with **Prefect**,
- a **Prefect deployment** that retrains the model every day,
- a **FastAPI** web service that validates its inputs with **Pydantic** and serves predictions,
- a **Docker** image that runs the API and the Prefect server together.

The data is the [Abalone dataset](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) from Kaggle.

## Team

| Name | GitHub |
|------|--------|
| Arnolfo Acero | [@aezacero](https://github.com/aezacero) |
| Litong Hou | [@melo0430](https://github.com/melo0430) |
| Martino Orioli | [@Martinoor](https://github.com/Martinoor) |
| Chloe Vacca | [@chloe363](https://github.com/chloe363) |

## How the Project Was Built

Each step was developed on its own branch and merged into `main` through a pull request.

| Branch | Step |
|---|---|
| `0/environment_setup` | uv environment, pre-commit hooks (ruff) and GitHub Actions CI |
| `1/eda_and_modelling_notebooks` | Exploratory analysis and a first model tracked with MLflow, in notebooks |
| `2/from_notebooks_to_modules` | Training and prediction code moved from the notebooks to Python modules |
| `3/use_prefect` | Training run as a Prefect flow, with a daily retraining deployment |
| `4/deploy_model_api` | FastAPI prediction service, Pydantic validation and Docker image |

## Repository Structure

```
├── bin/run_services.sh       # Starts the Prefect server and the API inside the container
├── Dockerfile.app            # Docker image of the API and the Prefect server
├── Makefile                  # Shortcuts for the commands below (`make help`)
├── notebooks/                # EDA and modelling notebooks
├── src/
│   ├── modelling/            # Training pipeline (Prefect flow and deployment)
│   └── web_service/          # FastAPI app
│       ├── lib/              # Pydantic schemas and inference
│       └── local_objects/    # Trained encoder.pkl and model.pkl used by the API
├── pyproject.toml            # Dependencies, managed with uv
└── uv.lock                   # Locked versions of the dependencies
```

---

# Instructions: How to Run the Project

**Start here.** Follow the steps in order. Run every command from the root of the repository.

| Step | What it does | Required? |
|---|---|---|
| [1. Install the prerequisites](#step-1-install-the-prerequisites) | Tools needed on your machine | Yes |
| [2. Clone the repository](#step-2-clone-the-repository) | Gets the code | Yes |
| [3. Create the environment](#step-3-create-the-environment) | Installs the dependencies | Yes |
| [4. Download the data](#step-4-download-the-data) | Gets the training set | Only to train |
| [5. Train the model](#step-5-train-the-model) | Trains and saves the model | Optional, a trained model is committed |
| [6. Schedule the retraining with Prefect](#step-6-schedule-the-retraining-with-prefect) | Retrains every day at 1 pm | Optional |
| [7. Run the API](#step-7-run-the-api) | Starts the prediction service (Docker or local) | Yes |
| [8. Make a prediction](#step-8-make-a-prediction) | Sends new data to the API | Yes |
| [9. Stop everything](#step-9-stop-everything) | Shuts the services down | Yes |

Every step shows a `make` shortcut and the equivalent plain command. Use whichever you prefer: `make` is optional (run `make help` to list all shortcuts).

## Step 1: Install the prerequisites

- [Git](https://git-scm.com/downloads)
- [uv](https://docs.astral.sh/uv/getting-started/installation/), which also installs Python 3.10 or 3.11 for you if it is missing
- [Docker](https://docs.docker.com/get-docker/), to run the API in a container (Docker Desktop must be open and running)
- A [Kaggle account](https://www.kaggle.com/account/login), to download the dataset (only needed to train)
- `make` (optional): preinstalled on macOS and Linux. On Windows, use the plain commands instead.

Check the installation:
```bash
git --version
uv --version
docker --version
```

## Step 2: Clone the repository

```bash
git clone https://github.com/aezacero/xhec-mlops-2026-project.git
cd xhec-mlops-2026-project
```

## Step 3: Create the environment

```bash
make install
```
or, without `make`:
```bash
uv sync                        # installs the app and dev dependencies from uv.lock into .venv/
uv run pre-commit install      # runs ruff linting and formatting on every commit
```

To check that the code passes the hooks (optional):
```bash
make lint        # or: uv run pre-commit run --all-files
```

Every command in this guide starts with `uv run`, so you never need to activate the environment. To activate it anyway: `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`).

## Step 4: Download the data

Download `abalone.csv` from the [Kaggle page](https://www.kaggle.com/datasets/rodolfomendes/abalone-dataset) and save it as `data/abalone.csv`. Create the `data/` folder first if it does not exist:

```bash
mkdir -p data
```

The `data/` folder is git-ignored, so the dataset is never committed.

## Step 5: Train the model

A trained encoder and model are already committed in `src/web_service/local_objects/`, so you can skip to [Step 7](#step-7-run-the-api) if you only want to use the API.

```bash
make train       # or: uv run python src/modelling/main.py data/abalone.csv
```

To train on another file: `make train DATA=path/to/file.csv`.

The training cleans the data and fits a linear regression on 80% of it. It prints the RMSE, MAE and R² on the remaining 20%, next to a baseline that always predicts the mean, then saves the fitted encoder and model to `src/web_service/local_objects/` (`encoder.pkl` and `model.pkl`), where the API reads them.

> **Note**: once you have run `make prefect-config` (Step 6), Prefect expects its server to be running. Start it with `make prefect-server` before training, otherwise the training fails to connect.

## Step 6: Schedule the retraining with Prefect

The training is a [Prefect](https://docs.prefect.io/) flow (`train_model_workflow` in `src/modelling/main.py`) made of five tasks: `prepare_data`, `split_data`, `fit_and_encode`, `train_and_evaluate` and `save_objects`. A deployment retrains the model **every day at 1 pm (Europe/Paris)**. This step needs the data from Step 4 and two terminals.

1. **Point Prefect to a local server** (once per machine):
   ```bash
   make prefect-config    # or: uv run prefect config set PREFECT_API_URL=http://127.0.0.1:4200/api
   ```

2. **Start the Prefect server** in a first terminal and leave it running. The UI is at http://127.0.0.1:4200.
   ```bash
   make prefect-server    # or: uv run prefect server start
   ```

3. **Run the training flow once** in a second terminal:
   ```bash
   make train
   ```
   The run and its five tasks appear in the UI under *Runs*, with the metrics in the logs.

4. **Start the scheduled deployment** in the second terminal and leave it running:
   ```bash
   make deploy            # or: uv run python src/modelling/deployment.py
   ```
   The `train-model` deployment appears in the UI under *Deployments* with its schedule. Click *Quick run* to trigger a training immediately instead of waiting for 1 pm. Scheduled runs only execute while this command is running.

Stop the deployment and the server with `Ctrl+C` in each terminal before going to Step 7 with Docker, since the container also uses port 4200.

## Step 7: Run the API

Choose **option A (Docker, recommended)** or **option B (local)**.

### Option A: in Docker

The image runs both the API and the Prefect server (see `bin/run_services.sh`). Make sure Docker is running and that nothing else uses ports 8000 and 4200 (stop any local Prefect server from Step 6).

1. **Build the image**:
   ```bash
   make docker-build      # or: docker build -f Dockerfile.app -t abalone-api .
   ```

2. **Run the container**:
   ```bash
   make docker-run
   ```
   or, without `make`:
   ```bash
   docker run --rm --name abalone-api -p 0.0.0.0:8000:8001 -p 0.0.0.0:4200:4201 abalone-api
   ```

3. **Check that it works** by opening:
   - http://localhost:8000, which returns `{"health_check": "App up and running!"}`
   - http://localhost:8000/docs, the interactive API documentation
   - http://localhost:4200, the Prefect UI

Inside the container, the API listens on port 8001 and Prefect on port 4201; the `-p` options map them to ports 8000 and 4200 on your machine. The image contains the encoder and model present when it was built: after retraining, run `make docker-build` again to serve the new model.

### Option B: locally, without Docker

```bash
make api
```
or, without `make`:
```bash
PYTHONPATH=src/modelling uv run uvicorn src.web_service.main:app --port 8001 --reload
```
On Windows (PowerShell), set the variable first:
```powershell
$env:PYTHONPATH="src/modelling"; uv run uvicorn src.web_service.main:app --port 8001 --reload
```

The API is then at http://127.0.0.1:8001 (docs at http://127.0.0.1:8001/docs). `PYTHONPATH` is needed because the API reuses the modelling modules, which import each other by name. The model is loaded once at startup, so restart the API after a retraining.

## Step 8: Make a prediction

**From the browser**: open http://localhost:8000/docs (or http://127.0.0.1:8001/docs for option B), click `POST /predict`, then *Try it out*, fill in the request body and click *Execute*.

**From the terminal**, in a second terminal (use port `8001` instead of `8000` for option B):

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "sex": "M",
        "length": 0.455,
        "diameter": 0.365,
        "height": 0.095,
        "whole_weight": 0.514,
        "shucked_weight": 0.2245,
        "viscera_weight": 0.101,
        "shell_weight": 0.15
      }'
```

Response (`201 Created`):

```json
{"rings": 8.86}
```

The age of the abalone is about `rings + 1.5` years. Predictions are clipped to at least 1 ring, since a linear model can go below that for the smallest infants.

The request body is validated: `sex` must be `M`, `F` or `I` (infant), and the seven measurements must be positive numbers. An invalid request gets a `422` response that says which field is wrong.

## Step 9: Stop everything

- **Docker**: press `Ctrl+C` in the terminal running the container, or run `make docker-stop` (`docker stop abalone-api`) from another terminal. The container is removed automatically.
- **Local API, Prefect server or deployment**: press `Ctrl+C` in their terminals.

---

# Reference

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Health check |
| `POST` | `/predict` | Predicts the number of rings of one abalone |

The schemas are in `src/web_service/lib/models.py` and the inference in `src/web_service/lib/inference.py`.

## Training modules

| Module (`src/modelling/`) | Role |
|---|---|
| `main.py` | Runs the full training as a Prefect flow, from the command line |
| `deployment.py` | Serves the training flow on a daily schedule |
| `preprocessing.py` | Loads and cleans the data, encodes the sex, builds the features |
| `training.py` | Fits and evaluates the model |
| `predicting.py` | Predicts the number of rings of new abalones with the saved encoder and model |
| `utils.py` | Saves and loads the pickled objects |

## Makefile commands

| Command | Description |
|---|---|
| `make help` | List all commands |
| `make install` | Create the environment and install the pre-commit hooks |
| `make lint` | Run the pre-commit hooks on all files |
| `make train` | Train the model on `data/abalone.csv` (or `DATA=...`) |
| `make prefect-config` | Point Prefect to the local server |
| `make prefect-server` | Start the Prefect server |
| `make deploy` | Serve the daily retraining deployment |
| `make api` | Run the API locally on port 8001 |
| `make docker-build` | Build the Docker image |
| `make docker-run` | Run the container (API on 8000, Prefect UI on 4200) |
| `make docker-stop` | Stop the container |
| `make clean` | Remove the Python caches |

## Development

- **Dependencies** are managed with uv. Add or update a package with `uv add <package>==<version>`, which updates `pyproject.toml` and `uv.lock`; notebook-only and tooling packages go in the dev group with `uv add --dev`.
- **Code quality**: the pre-commit hooks run ruff (linting and formatting) and check that `uv.lock` is up to date. Run them on everything with `make lint`.
- **CI**: GitHub Actions installs the locked environment and runs the hooks with Python 3.10 and 3.11 on every push and pull request (`.github/workflows/ci.yaml`).