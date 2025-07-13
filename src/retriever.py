import faiss
import numpy as np
import pickle
import os
from typing import List, Dict, Any, Tuple
from langchain_ollama import OllamaEmbeddings
from config import config

class VectorRetriever:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model=config.embedding_model,
            base_url=config.ollama_base_url
        )
        self.index = None
        self.documents = []
        self.dimension = None

        # Create vector DB directory if it doesn't exist
        os.makedirs(config.vector_db_path, exist_ok=True)

        # Load existing index if available
        self.load_index()

    def _get_embedding_dimension(self):
        """Get the embedding dimension from the model."""
        if self.dimension is None:
            test_embedding = self.embeddings.embed_query("test")
            self.dimension = len(test_embedding)
        return self.dimension

    def create_index(self):
        """Create a new FAISS index."""
        dim = self._get_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        self.documents = []

    def add_documents(self, documents: List[Dict[str, Any]]):
        """Add documents to the vector index."""
        if not documents:
            return

        if self.index is None:
            self.create_index()

        embeddings = np.array([doc["embedding"] for doc in documents]).astype('float32')
        self.index.add(embeddings)

        # Store document metadata (without embeddings to save space)
        for doc in documents:
            doc_copy = doc.copy()
            del doc_copy["embedding"]
            self.documents.append(doc_copy)

    def search(self, query: str, top_k: int = None) -> List[Tuple[Dict[str, Any], float]]:
        """Search for similar documents."""
        if self.index is None or self.index.ntotal == 0:
            return []

        if top_k is None:
            top_k = config.top_k_chunks

        # Embed query
        query_embedding = self.embeddings.embed_query(query)
        query_vector = np.array([query_embedding]).astype('float32')

        # Search
        distances, indices = self.index.search(query_vector, min(top_k, self.index.ntotal))

        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if idx < len(self.documents):
                similarity = float(1 / (1 + distance))  # Convert to Python float
                if similarity >= config.similarity_threshold:
                    results.append((self.documents[idx], similarity))

        return results

    def remove_file_documents(self, file_path: str):
        """Remove all documents for a specific file."""
        if self.index is None:
            return

        # Find indices of documents to remove
        indices_to_remove = []
        for i, doc in enumerate(self.documents):
            if doc["metadata"]["file_path"] == file_path:
                indices_to_remove.append(i)

        # Remove from documents list (reverse order to maintain indices)
        for i in reversed(indices_to_remove):
            del self.documents[i]

        # Rebuild index (FAISS doesn't support efficient removal)
        if self.documents:
            self.rebuild_index()
        else:
            self.create_index()

    def rebuild_index(self):
        """Rebuild the FAISS index from current documents."""
        if not self.documents:
            self.create_index()
            return

        # Re-embed documents
        texts = [doc["content"] for doc in self.documents]
        embeddings = self.embeddings.embed_documents(texts)

        dim = self._get_embedding_dimension()
        self.index = faiss.IndexFlatL2(dim)
        embeddings_array = np.array(embeddings).astype('float32')
        self.index.add(embeddings_array)

    def save_index(self):
        """Save index and documents to disk."""
        if self.index is not None:
            faiss.write_index(self.index, os.path.join(config.vector_db_path, "index.faiss"))

        # Save dimension info
        index_meta = {
            "dimension": self.dimension,
            "model": config.embedding_model
        }

        with open(os.path.join(config.vector_db_path, "documents.pkl"), "wb") as f:
            pickle.dump(self.documents, f)

        with open(os.path.join(config.vector_db_path, "meta.pkl"), "wb") as f:
            pickle.dump(index_meta, f)

    def load_index(self):
        """Load index and documents from disk."""
        index_path = os.path.join(config.vector_db_path, "index.faiss")
        docs_path = os.path.join(config.vector_db_path, "documents.pkl")
        meta_path = os.path.join(config.vector_db_path, "meta.pkl")

        if os.path.exists(index_path) and os.path.exists(docs_path):
            try:
                self.index = faiss.read_index(index_path)
                with open(docs_path, "rb") as f:
                    self.documents = pickle.load(f)

                # Load dimension info
                if os.path.exists(meta_path):
                    with open(meta_path, "rb") as f:
                        meta = pickle.load(f)
                        self.dimension = meta.get("dimension")

                print(f"Loaded {len(self.documents)} documents from vector DB")
            except Exception as e:
                print(f"Error loading index: {e}")
                self.create_index()
        else:
            self.create_index()