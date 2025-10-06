"""
Configuration file for LangChain Retrieval Comparison Project
Update these paths according to your setup
"""

import os

# Base paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# Model paths (update these with your actual paths)
# LLM_MODEL_PATH = "meta-llama/Llama-3.2-3B"  # e.g., "microsoft/DialoGPT-medium"
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # e.g., "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL_CACHE = "/path/to/your/embedding_model_cache"
# Data paths (update these with your actual paths)
PDF_PATH = "path/to/your/pdf_file.pdf"  # e.g., "https://huggingface.co/datasets/example-dataset"

# ChromaDB settings
CHROMA_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")

# Retrieval settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K_RESULTS = 5

# Model settings
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
MAX_LENGTH = 512
