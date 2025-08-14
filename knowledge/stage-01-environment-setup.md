# Stage 1: Environment Setup and Project Structure

## Overview
This stage focuses on setting up the development environment and creating the foundational project structure for the research agent.

## Objectives
- Create Python virtual environment
- Install required dependencies
- Set up project directory structure
- Configure development environment

## Requirements Installation

### Core Dependencies
```txt
langchain>=0.2
langgraph>=0.2
langchain-openai>=0.1
chromadb>=0.5
sentence-transformers>=3.0
pypdf>=4.2
unstructured>=0.15
scipy>=1.12
numpy>=1.26
pydantic>=2.7
typer>=0.12
fastapi>=0.111
uvicorn>=0.30
structlog>=24.1
python-dotenv>=1.0
```

## Project Structure
```
research-agent/
├─ .venv/
├─ .env                      # API keys and settings
├─ requirements.txt
├─ data/
│  ├─ inbox/                 # New PDF files (auto-processed)
│  ├─ processed/             # Processed files
│  └─ examples/              # Sample files
├─ storage/
│  ├─ vectors/               # Chroma database
│  └─ checkpoints/           # LangGraph state (SQLite)
├─ src/
│  ├─ config.py
│  ├─ schema.py              # State and Models definitions
│  ├─ loaders.py             # PDF text extraction
│  ├─ chunking.py            # Text chunking
│  ├─ embeddings.py          # Vector creation/indexing
│  ├─ tools_scipy.py         # SciPy analytical functions
│  ├─ nodes.py               # LangGraph nodes
│  ├─ graph.py               # Graph building and linking
│  ├─ reporter.py            # Report generation
│  ├─ run_cli.py             # Command line interface
│  └─ api.py                 # REST interface (FastAPI)
└─ README.md
```

## Implementation Steps

### 1. Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Directory Creation
```bash
mkdir -p data/{inbox,processed,examples}
mkdir -p storage/{vectors,checkpoints}
mkdir -p src
```

## Status
- [x] Project structure defined
- [ ] Virtual environment created
- [ ] Dependencies installed
- [ ] Directories created
- [ ] Environment configuration ready

## Next Stage
Stage 2: Configuration and Environment Variables Setup