import os
from typing import List, Dict, Any
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import tiktoken
from config import config

class DocumentEmbedder:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")

    def _read_file(self, file_path: str) -> str:
        """Read file content with encoding handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                return ""

    def process_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Process a file and return chunked documents with embeddings."""
        content = self._read_file(file_path)
        if not content.strip():
            return []

        # Split into chunks
        chunks = self.text_splitter.split_text(content)

        documents = []
        for i, chunk in enumerate(chunks):
            if chunk.strip():
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk)

                doc = {
                    "content": chunk,
                    "embedding": embedding,
                    "metadata": {
                        "file_path": file_path,
                        "chunk_index": i,
                        "chunk_size": len(chunk)
                    }
                }
                documents.append(doc)

        return documents