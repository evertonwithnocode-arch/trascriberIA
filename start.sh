#!/bin/bash

# Ativar o virtual environment
source venv/bin/activate

# Rodar o servidor FastAPI na porta 5000
uvicorn main:app --host 0.0.0.0 --port 5000
