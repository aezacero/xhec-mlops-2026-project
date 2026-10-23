#!/bin/bash

# Prefect server in the background, the API in the foreground to keep the container running
prefect server start --host 0.0.0.0 --port 4201 &
exec uvicorn src.web_service.main:app --host 0.0.0.0 --port 8001
