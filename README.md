# File Context Agent

A smart AI-powered code analysis tool that indexes your codebase and answers questions using local LLMs via Ollama. Each project gets its own isolated vector database for accurate, context-aware responses.

## Features

- 🤖 **Local AI**: Uses Ollama for embeddings and chat (no external API calls)
- 📁 **Project Isolation**: Each project gets its own vector database
- 🔄 **Smart Indexing**: Automatically detects file changes and updates index
- 📊 **Multiple Formats**: Supports Python, JavaScript, HTML, CSS, Markdown, and more
- ⚡ **Fast Retrieval**: FAISS-powered vector search for relevant code context
- 🧹 **Database Management**: Clear, rebuild, or manage multiple project databases

## Prerequisites

1. **Python 3.11+**
2. **Ollama** with required models:
   ```bash
   # Install Ollama
   curl -fsSL https://ollama.ai/install.sh | sh

   # Pull required models
   ollama pull llama3.2
   ollama pull nomic-embed-text
nstallation
Clone the repository:  
git clone https://github.com/999Gabriel/FileContext-Agent.git
cd FileContext-Agent
Install dependencies:  
pip install -r requirements.txt
Verify Ollama is running:  
curl http://localhost:11434/api/tags
Usage
Basic Commands
Ask questions about your codebase:
python src/cli.py ask -q "What HTML pages are in this project?" -f /path/to/project
python src/cli.py ask -q "How is authentication implemented?" -f /path/to/project
python src/cli.py ask -q "What JavaScript libraries are used?" -f /path/to/project
Start file watching mode:
python src/cli.py start -f /path/to/project
Check project status:
python src/cli.py status -f /path/to/project
Example Queries
# Analyze a web project
python src/cli.py ask -q "What is the main functionality of this application?" -f ./my-web-app

# Find specific implementations
python src/cli.py ask -q "How is user registration handled?" -f ./my-backend

# Understand project structure
python src/cli.py ask -q "What are the main components and their relationships?" -f ./my-frontend
Configuration
The system uses intelligent defaults but can be customized via src/config.py:
# Supported file types
allowed_extensions = [".py", ".js", ".ts", ".html", ".css", ".md", ".json", ".yaml", ".yml", ".txt"]

# Chunking parameters
chunk_size = 1000
chunk_overlap = 200

# Retrieval settings
top_k_chunks = 5
similarity_threshold = 0.1

# Ollama settings
ollama_base_url = "http://localhost:11434"
embedding_model = "nomic-embed-text"
chat_model = "llama3.2"
How It Works
File Scanning: Recursively scans your project for supported file types
Chunking: Splits large files into overlapping chunks for better context
Embedding: Uses nomic-embed-text to create vector embeddings
Indexing: Stores embeddings in FAISS for fast similarity search
Querying: Finds relevant code chunks and uses llama3.2 to generate answers
Project Structure
FileContext-Agent/
├── src/
│   ├── cli.py              # Command-line interface
│   ├── agent.py            # Main agent orchestration
│   ├── retriever.py        # Vector database and search
│   ├── indexer.py          # File processing and chunking
│   └── config.py           # Configuration settings
├── vector_dbs/             # Project-specific databases (auto-created)
│   ├── project1_abc123/    # Vector DB for project1
│   └── project2_def456/    # Vector DB for project2
└── requirements.txt
Database Isolation
Each project gets its own vector database based on:  
Project name (from folder name)
Path hash (to handle projects with same name)
Example database paths:
vector_dbs/
├── GlamCV_a1b2c3d4/        # /Users/john/projects/GlamCV
├── MyApp_e5f6g7h8/         # /Users/john/other/MyApp
└── GlamCV_i9j0k1l2/        # /Users/jane/work/GlamCV
Supported File Types
Code: .py, .js, .ts
Web: .html, .css
Config: .json, .yaml, .yml
Documentation: .md, .txt
Performance Tips
Large Projects: The initial indexing may take time for large codebases
Incremental Updates: Only changed files are re-indexed automatically
Memory Usage: FAISS keeps embeddings in memory for fast search
Model Selection: nomic-embed-text provides good balance of speed/quality
Troubleshooting
Ollama Connection Issues:
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama if needed
ollama serve
Model Not Found:
# Pull missing models
ollama pull llama3.2
ollama pull nomic-embed-text
Database Issues:
# Clear corrupted database
python src/cli.py clear -f /path/to/project

# Or clear all databases
python src/cli.py clear --all
Memory Issues:  
Reduce chunk_size in config for large files
Clear unused project databases
Use reindex command to rebuild corrupted indexes
Examples
Web Development Project:
python src/cli.py ask -q "What APIs are exposed by this backend?" -f ./my-api
python src/cli.py ask -q "How is the frontend routing configured?" -f ./my-frontend
Python Project:
python src/cli.py ask -q "What are the main classes and their methods?" -f ./my-python-app
python src/cli.py ask -q "How is error handling implemented?" -f ./my-library
Documentation:
python src/cli.py ask -q "What is the installation process?" -f ./my-project
python src/cli.py ask -q "What are the configuration options?" -f ./my-tool
Contributing
Fork the repository
Create a feature branch
Make your changes
Test with different project types
Submit a pull request
Acknowledgments
Ollama for local LLM serving
LangChain for LLM abstractions
FAISS for vector similarity search
Nomic for the embedding model
