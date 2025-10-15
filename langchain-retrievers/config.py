"""
Configuration file for LangChain Retrieval Comparison Project
Update these paths according to your setup
"""

import os
from pathlib import Path

# Base paths
# PROJECT_ROOT is langchain-retrievers/ directory (where this config.py lives)
PROJECT_ROOT = Path(__file__).parent.resolve()

# RAG_ROOT is the parent directory (one level up)
RAG_ROOT = PROJECT_ROOT.parent

# Shared resources - UPDATE THESE PATHS TO YOUR ACTUAL LOCATIONS
MODELS_DIR = "path/to/your/models"                    # e.g., RAG_ROOT / "models"
CHROMA_BASE_DIR = "path/to/your/chroma_db"            # e.g., RAG_ROOT / "chroma_db"

# Project-specific data
DATA_DIR = PROJECT_ROOT / "data"                      # Relative to config.py location

# Model paths
EMBEDDING_MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"  # HuggingFace model name

# Data file paths (project-specific)
TRANSFORMED_PAPERS_PATH = DATA_DIR / "transformed_papers_llm_metadata.json"
FAQ_DATA_URL = "https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/refs/heads/main/03-evaluation/search_evaluation/documents-with-ids.json"

# ChromaDB settings - UPDATE THESE PATHS
CHROMA_PERSIST_DIRECTORY = "path/to/your/chroma_db"              # Base directory
CHROMA_SQ_DIR = "path/to/your/chroma_db/sq_hf"                   # Self-Query directory
CHROMA_MQ_DIR = "path/to/your/chroma_db/mq_faq"                  # Multi-Query directory

# Retrieval settings
CHUNK_SIZE = 200
CHUNK_OVERLAP = 20
TOP_K_RESULTS = 5

# Model settings
DEVICE = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
MAX_LENGTH = 512