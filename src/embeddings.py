from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from schema import DocChunk
import os
import structlog
from pathlib import Path

logger = structlog.get_logger()

def get_embedding_model():
    """
    Get an embeddings model instance.
    
    Returns:
        OpenAIEmbeddings or HuggingFaceEmbeddings instance
    """
    try:
        # Try OpenRouter first if available
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key.startswith("sk-or-v1"):
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            api_base = os.getenv("EMBEDDING_API_BASE", "https://openrouter.ai/api/v1")
            logger.debug("Creating OpenRouter embeddings model", model=model)
            try:
                return OpenAIEmbeddings(
                    model=model,
                    openai_api_key=openrouter_key,
                    openai_api_base=api_base
                )
            except Exception as e:
                logger.warning("OpenRouter embeddings failed, trying regular OpenAI", error=str(e))
        
        # Try regular OpenAI if available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.startswith("sk-") and not openai_key.startswith("sk-or-v1"):
            model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            logger.debug("Creating OpenAI embeddings model", model=model)
            try:
                return OpenAIEmbeddings(model=model)
            except Exception as e:
                logger.warning("OpenAI embeddings failed, falling back to HuggingFace", error=str(e))
        
        # Use local HuggingFace model as fallback (more reliable)
        logger.info("Using local HuggingFace embeddings")
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
    except Exception as e:
        logger.warning("Failed to create embeddings, using HuggingFace", error=str(e))
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )

def build_or_load_vectorstore(chunks: List[DocChunk], persist_dir: str):
    """
    Build or load a vector store from document chunks.
    
    Args:
        chunks: List of document chunks to embed
        persist_dir: Directory to persist the vector store
        
    Returns:
        Chroma vectorstore instance
    """
    try:
        logger.info("Building vector store", 
                   chunks_count=len(chunks), 
                   persist_dir=persist_dir)
        
        # Validate input
        if not chunks:
            raise ValueError("No chunks provided for vector store")
        
        # Create persist directory if it doesn't exist
        Path(persist_dir).mkdir(parents=True, exist_ok=True)
        
        embeddings = get_embedding_model()
        texts = [c.text for c in chunks if c.text.strip()]
        
        if not texts:
            raise ValueError("No valid text content found in chunks")
            
        metadatas = [{
            "doc_id": c.doc_id, 
            "chunk_id": c.chunk_id,
            **c.meta
        } for c in chunks if c.text.strip()]
        
        logger.info("Preparing texts for embedding", 
                   text_count=len(texts),
                   sample_text=texts[0][:100] if texts else "")
        
        # Create vector store
        vs = Chroma.from_texts(
            texts=texts, 
            embedding=embeddings, 
            metadatas=metadatas, 
            persist_directory=persist_dir
        )
        
        logger.info("Vector store created successfully", 
                   chunks_embedded=len(chunks))
        
        return vs
        
    except Exception as e:
        logger.error("Failed to build vector store", 
                    error=str(e), 
                    persist_dir=persist_dir)
        raise

def as_retriever(persist_dir: str, k: int = 5):
    """
    Create a retriever from an existing vector store.
    
    Args:
        persist_dir: Directory where vector store is persisted
        k: Number of documents to retrieve
        
    Returns:
        Retriever instance
    """
    try:
        logger.debug("Creating retriever", persist_dir=persist_dir, k=k)
        
        embeddings = get_embedding_model()
        vs = Chroma(
            embedding_function=embeddings, 
            persist_directory=persist_dir
        )
        
        retriever = vs.as_retriever(search_kwargs={"k": k})
        logger.debug("Retriever created successfully")
        
        return retriever
        
    except Exception as e:
        logger.error("Failed to create retriever", 
                    error=str(e), 
                    persist_dir=persist_dir)
        raise

def search_similar(query: str, persist_dir: str, k: int = 5):
    """
    Search for similar documents using the vector store.
    
    Args:
        query: Search query
        persist_dir: Directory where vector store is persisted
        k: Number of similar documents to return
        
    Returns:
        List of similar documents
    """
    try:
        logger.debug("Searching similar documents", 
                    query=query[:100], k=k)
        
        embeddings = get_embedding_model()
        vs = Chroma(
            embedding_function=embeddings, 
            persist_directory=persist_dir
        )
        
        docs = vs.similarity_search(query, k=k)
        
        logger.info("Similar documents found", 
                   query=query[:50], 
                   results_count=len(docs))
        
        return docs
        
    except Exception as e:
        logger.error("Failed to search similar documents", 
                    error=str(e), 
                    query=query[:50])
        raise

def get_collection_info(persist_dir: str) -> dict:
    """
    Get information about the vector store collection.
    
    Args:
        persist_dir: Directory where vector store is persisted
        
    Returns:
        Dictionary with collection information
    """
    try:
        embeddings = get_embedding_model()
        vs = Chroma(
            embedding_function=embeddings, 
            persist_directory=persist_dir
        )
        
        # Get collection statistics
        collection = vs._collection
        count = collection.count()
        
        info = {
            "document_count": count,
            "persist_dir": persist_dir,
            "status": "healthy" if count > 0 else "empty"
        }
        
        logger.debug("Collection info retrieved", **info)
        return info
        
    except Exception as e:
        logger.error("Failed to get collection info", 
                    error=str(e), 
                    persist_dir=persist_dir)
        return {
            "document_count": 0,
            "persist_dir": persist_dir,
            "status": "error",
            "error": str(e)
        }