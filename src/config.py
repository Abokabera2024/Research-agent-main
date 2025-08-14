import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # منفصل عن OpenRouter
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-4o-mini:free")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1")
EMBEDDING_API_BASE = os.getenv("EMBEDDING_API_BASE", "https://openrouter.ai/api/v1")

# Storage Configuration
CHROMA_DIR = os.getenv("CHROMA_DIR", "./storage/vectors")
CHECKPOINT_DB = os.getenv("CHECKPOINT_DB", "./storage/checkpoints/graph_state.sqlite")

# Processing Configuration
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "1200"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "150"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "6"))

# Validation
def validate_config():
    """Validate that required configuration is present."""
    if OPENROUTER_API_KEY and OPENROUTER_API_KEY.startswith("sk-or-v1"):
        print(f"OpenRouter API key configured: {OPENROUTER_API_KEY[:10]}...{OPENROUTER_API_KEY[-4:]}")
        return True
    elif OPENAI_API_KEY and OPENAI_API_KEY.startswith("sk-"):
        print(f"OpenAI API key configured: {OPENAI_API_KEY[:10]}...{OPENAI_API_KEY[-4:]}")
        return True
    else:
        print("Warning: No valid API key found. Using mock responses.")
        return False

if __name__ == "__main__":
    validate_config()
    print("Configuration validated successfully")