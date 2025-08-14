from typing import List
from schema import DocChunk
import textwrap
import uuid
import structlog
import re

logger = structlog.get_logger()

def simple_chunk(text: str, doc_id: str, chunk_size: int = 1200, overlap: int = 150) -> List[DocChunk]:
    """
    Break text into overlapping chunks.
    
    Args:
        text: Text to chunk
        doc_id: Document identifier
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of DocChunk objects
    """
    # Validate input
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for chunking", doc_id=doc_id)
        return []
    
    if not doc_id:
        logger.error("No document ID provided for chunking")
        return []
    
    chunks = []
    start = 0
    chunk_num = 0
    
    logger.info("Starting text chunking", 
               doc_id=doc_id, 
               text_length=len(text), 
               chunk_size=chunk_size, 
               overlap=overlap,
               text_preview=text[:200])
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        segment = text[start:end]
        
        # Try to break at sentence boundaries when possible
        if end < len(text) and not segment.endswith('.'):
            # Look for the last sentence boundary
            last_period = segment.rfind('.')
            last_newline = segment.rfind('\n')
            boundary = max(last_period, last_newline)
            
            if boundary > start + chunk_size // 2:  # Only if we're not losing too much text
                end = start + boundary + 1
                segment = text[start:end]
        
        chunk_id = f"{doc_id}-chunk-{chunk_num:04d}-{uuid.uuid4().hex[:8]}"
        
        chunk = DocChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=segment.strip(),
            meta={
                "chunk_number": chunk_num,
                "start_pos": start,
                "end_pos": end,
                "char_count": len(segment)
            }
        )
        
        chunks.append(chunk)
        chunk_num += 1
        
        # Move start position for next chunk
        if end >= len(text):
            break
            
        start = end - overlap
        if start < 0:
            start = 0
    
    logger.info("Text chunking completed", 
               doc_id=doc_id, 
               total_chunks=len(chunks))
    
    return chunks

def smart_chunk(text: str, doc_id: str, chunk_size: int = 1200, overlap: int = 150) -> List[DocChunk]:
    """
    Advanced chunking that respects paragraph and section boundaries.
    
    Args:
        text: Text to chunk
        doc_id: Document identifier
        chunk_size: Target size of each chunk
        overlap: Number of characters to overlap
        
    Returns:
        List of DocChunk objects with smart boundaries
    """
    # Validate input
    if not text or not text.strip():
        logger.error("Empty or invalid text provided for smart chunking", doc_id=doc_id)
        return []
    
    # Split into paragraphs - try multiple strategies
    paragraphs = re.split(r'\n\s*\n', text)
    
    # If we only got one paragraph, try other splitting strategies
    if len(paragraphs) == 1:
        # Try splitting by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        if len(sentences) > 1:
            # Group sentences into paragraph-like chunks
            paragraphs = []
            current_para = ""
            for sentence in sentences:
                if len(current_para) + len(sentence) > chunk_size // 2 and current_para:
                    paragraphs.append(current_para.strip())
                    current_para = sentence
                else:
                    current_para += " " + sentence if current_para else sentence
            if current_para.strip():
                paragraphs.append(current_para.strip())
        else:
            # Fallback: split by double newlines or single newlines
            paragraphs = re.split(r'\n+', text)
            paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_chunk = ""
    chunk_num = 0
    
    logger.info("Starting smart chunking", 
               doc_id=doc_id, 
               paragraphs=len(paragraphs),
               text_length=len(text))
    
    for paragraph in paragraphs:
        # Skip empty paragraphs
        if not paragraph.strip():
            continue
            
        # If adding this paragraph would exceed chunk size, finalize current chunk
        if len(current_chunk) + len(paragraph) > chunk_size and current_chunk:
            chunk_id = f"{doc_id}-smart-{chunk_num:04d}-{uuid.uuid4().hex[:8]}"
            
            chunk = DocChunk(
                doc_id=doc_id,
                chunk_id=chunk_id,
                text=current_chunk.strip(),
                meta={
                    "chunk_number": chunk_num,
                    "char_count": len(current_chunk),
                    "chunk_type": "smart"
                }
            )
            
            chunks.append(chunk)
            chunk_num += 1
            
            # Start new chunk with overlap
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph
        else:
            if current_chunk:
                current_chunk += "\n\n" + paragraph
            else:
                current_chunk = paragraph
    
    # Add final chunk if there's remaining text
    if current_chunk.strip():
        chunk_id = f"{doc_id}-smart-{chunk_num:04d}-{uuid.uuid4().hex[:8]}"
        
        chunk = DocChunk(
            doc_id=doc_id,
            chunk_id=chunk_id,
            text=current_chunk.strip(),
            meta={
                "chunk_number": chunk_num,
                "char_count": len(current_chunk),
                "chunk_type": "smart"
            }
        )
        
        chunks.append(chunk)
    
    # If we still only have one chunk and it's very large, force split it
    if len(chunks) == 1 and len(chunks[0].text) > chunk_size * 2:
        logger.info("Large single chunk detected, force splitting", doc_id=doc_id, size=len(chunks[0].text))
        # Use simple chunking as fallback
        return simple_chunk(text, doc_id, chunk_size, overlap)
    
    logger.info("Smart chunking completed", 
               doc_id=doc_id, 
               total_chunks=len(chunks))
    
    return chunks