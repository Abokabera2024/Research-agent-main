from langchain_community.document_loaders import PyPDFLoader
from pathlib import Path
import structlog
from typing import Optional

logger = structlog.get_logger()

def load_pdf_text(pdf_path: str) -> str:
    """
    Load and extract text from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Combined text content from all pages
        
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: If PDF loading fails
    """
    try:
        logger.info("Loading PDF", path=pdf_path)
        
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        if not pages:
            raise ValueError(f"No pages found in PDF: {pdf_path}")
        
        # Combine text from all pages
        text = "\n".join([p.page_content for p in pages if p.page_content.strip()])
        
        if not text.strip():
            raise ValueError(f"No text content extracted from PDF: {pdf_path}")
        
        logger.info("PDF loaded successfully", 
                   path=pdf_path, 
                   pages=len(pages), 
                   text_length=len(text),
                   text_preview=text[:200])
        
        return text
        
    except Exception as e:
        logger.error("Failed to load PDF", path=pdf_path, error=str(e))
        raise

def assign_doc_id(pdf_path: str) -> str:
    """
    Generate a document ID from the PDF file path.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Document ID based on the file name
    """
    p = Path(pdf_path)
    doc_id = f"{p.stem}"
    logger.debug("Assigned document ID", path=pdf_path, doc_id=doc_id)
    return doc_id

def validate_pdf_file(pdf_path: str) -> bool:
    """
    Validate that the file exists and is a PDF.
    
    Args:
        pdf_path: Path to validate
        
    Returns:
        True if valid PDF file, False otherwise
    """
    try:
        path = Path(pdf_path)
        return path.exists() and path.suffix.lower() == '.pdf'
    except Exception:
        return False