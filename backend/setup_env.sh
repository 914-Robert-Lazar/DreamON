#!/bin/bash

# Create a new Conda environment
conda create -n dreamon_env python=3.9 -y

# Activate the environment
conda activate dreamon_env

# Install FastAPI and Uvicorn for the backend
pip install fastapi uvicorn

# Install LangChain and its dependencies
pip install langchain faiss-cpu pdfminer.six

# Install Hugging Face Transformers and related libraries
pip install transformers bitsandbytes accelerate

# Install additional dependencies for keyword extraction
pip install spacy

# Download SpaCy language model (if needed)
python -m spacy download en_core_web_sm

# Install langchain_openai for pipeline functionality
pip install langchain-openai

# Install any other dependencies from requirements.txt
pip install -r backend/requirements.txt

echo "Conda environment 'dreamon_env' created and all dependencies installed."