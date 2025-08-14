# Stage 2: Configuration and Environment Variables

## Overview
This stage configures the environment variables and API keys needed for the research agent to function properly.

## Environment Configuration

### .env File Setup
```env
OPENAI_API_KEY=sk-...              # OpenAI API key or alternative provider
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini              # Change to your preferred/available model
CHROMA_DIR=./storage/vectors
CHECKPOINT_DB=./storage/checkpoints/graph_state.sqlite
```

### Alternative for Local Models
For local model usage (Ollama):
```env
LLM_MODEL=llama3
USE_LOCAL_MODEL=true
```

## Configuration Module

The config.py module centralizes all configuration management and provides default values.

## Implementation Details

### Environment Variables
- **OPENAI_API_KEY**: Required for OpenAI API access
- **EMBEDDING_MODEL**: Model for text embeddings (default: text-embedding-3-small)
- **LLM_MODEL**: Language model for analysis (default: gpt-4o-mini)
- **CHROMA_DIR**: Directory for vector storage
- **CHECKPOINT_DB**: SQLite database for LangGraph state management

### Security Considerations
- Never commit .env files to version control
- Use environment-specific configurations
- Rotate API keys regularly
- Validate all configuration on startup

## Status
- [x] Configuration structure defined
- [ ] .env file created
- [ ] config.py module implemented
- [ ] Environment validation added

## Next Stage
Stage 3: Data Models and Schemas Implementation