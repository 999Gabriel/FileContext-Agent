import os
from typing import List, Dict, Any
from langchain_ollama import ChatOllama
from embedder import DocumentEmbedder
from retriever import VectorRetriever
from file_watcher import FileWatcher
from config import config

class SmartFileContextAgent:
    def __init__(self):
        # Initialize Ollama chat model
        self.llm = ChatOllama(
            model=config.chat_model,
            base_url=config.ollama_base_url,
            temperature=0.1
        )

        # Initialize components
        self.embedder = DocumentEmbedder()
        self.retriever = VectorRetriever()
        self.file_watcher = FileWatcher(self.embedder, self.retriever)

        # System prompt
        self.system_prompt = (
            "You are a helpful coding assistant. Use only the provided file context "
            "to answer the user's question. Be specific, concise, and include file names "
            "and line references when possible."
        )

    def start(self, folder_path: str = None):
        """Initialize the agent and start watching files."""
        if folder_path:
            config.project_folder = folder_path

        print("Starting Smart File Context Agent...")

        # Initial scan and indexing
        self.file_watcher.scan_and_index_all()

        # Start file watching
        self.file_watcher.start_watching()

        print("Agent is ready! You can now ask questions about your codebase.")

    def stop(self):
        """Stop the agent."""
        self.file_watcher.stop_watching()
        print("Agent stopped.")

    def _build_context(self, retrieved_docs: List[Dict[str, Any]], max_chars: int = 8000) -> str:
        """Build context from retrieved documents, respecting character limits."""
        context_parts = []
        total_chars = 0

        for doc, similarity in retrieved_docs:
            file_path = doc["metadata"]["file_path"]
            content = doc["content"]

            # Format the context piece
            context_piece = f"\n--- File: {file_path} ---\n{content}\n"

            if total_chars + len(context_piece) <= max_chars:
                context_parts.append(context_piece)
                total_chars += len(context_piece)
            else:
                break

        return "\n".join(context_parts)

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about the codebase."""
        print(f"Question: {question}")

        # Retrieve relevant documents
        retrieved_docs = self.retriever.search(question)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant context in the codebase to answer your question.",
                "sources": [],
                "context_used": ""
            }

        # Build context
        context = self._build_context(retrieved_docs)

        # Create prompt
        prompt = f"""Context:
{context}

Question: {question}

Instructions: {self.system_prompt}

Answer:"""

        # Query Ollama
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # Extract sources
            sources = []
            for doc, similarity in retrieved_docs:
                sources.append({
                    "file": doc["metadata"]["file_path"],
                    "similarity": round(similarity, 3),
                    "chunk_index": doc["metadata"]["chunk_index"]
                })

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context
            }

        except Exception as e:
            return {
                "answer": f"Error querying LLM: {str(e)}",
                "sources": [],
                "context_used": context
            }

    def status(self) -> Dict[str, Any]:
        """Get agent status."""
        return {
            "is_watching": self.file_watcher.is_running,
            "documents_indexed": len(self.retriever.documents),
            "project_folder": config.project_folder,
            "allowed_extensions": config.allowed_extensions,
            "embedding_model": config.embedding_model,
            "chat_model": config.chat_model
        }

    def ask(self, question: str) -> Dict[str, Any]:
        """Ask a question about the codebase."""
        print(f"Question: {question}")

        # Retrieve relevant documents
        retrieved_docs = self.retriever.search(question)

        if not retrieved_docs:
            return {
                "answer": "I couldn't find any relevant context in the codebase to answer your question.",
                "sources": [],
                "context_used": ""
            }

        # Build context
        context = self._build_context(retrieved_docs)

        # Create prompt
        prompt = f"""Context:
    {context}

    Question: {question}

    Instructions: {self.system_prompt}

    Answer:"""

        # Query Ollama
        try:
            response = self.llm.invoke(prompt)
            answer = response.content.strip()

            # Extract sources - convert numpy floats to Python floats
            sources = []
            for doc, similarity in retrieved_docs:
                sources.append({
                    "file": doc["metadata"]["file_path"],
                    "similarity": float(similarity),  # Convert numpy float32 to Python float
                    "chunk_index": doc["metadata"]["chunk_index"]
                })

            return {
                "answer": answer,
                "sources": sources,
                "context_used": context
            }

        except Exception as e:
            return {
                "answer": f"Error querying LLM: {str(e)}",
                "sources": [],
                "context_used": context
            }