from pydantic import BaseModel
from typing import List
import os

class Config(BaseModel):
    # Paths
    project_folder: str = "."
    vector_db_path: str = "./vector_db"

    # File processing
    allowed_extensions: List[str] = [
        ".py", ".js", ".ts", ".html", ".css",
        ".md", ".json", ".yaml", ".yml", ".txt"
    ]

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Ollama settings
    ollama_base_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    chat_model: str = "llama3.2"
    max_tokens: int = 4000

    # Retrieval
    top_k_chunks: int = 5
    similarity_threshold: float = 0.1  # Very low for testing

config = Config()