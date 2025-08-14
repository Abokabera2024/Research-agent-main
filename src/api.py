from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import shutil
import os
import uuid
import json
from datetime import datetime
import structlog

from graph import build_graph, run_graph
from reporter import save_complete_analysis, generate_comprehensive_report
from config import validate_config
from schema import Decision

# Configure logging
logger = structlog.get_logger()

# FastAPI app
app = FastAPI(
    title="Research Agent API",
    description="AI-powered scientific document analysis with statistical validation",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global graph instance
graph = None

# Data models
class AnalysisRequest(BaseModel):
    query: Optional[str] = None
    include_debug: bool = False

class AnalysisResponse(BaseModel):
    job_id: str
    doc_id: str
    status: str
    message: str
    report_paths: Optional[Dict[str, str]] = None
    decision: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    components: Dict[str, str]

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    processed_at: str
    status: str
    report_available: bool

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application."""
    global graph
    try:
        validate_config()
        graph = build_graph()
        logger.info("API startup completed", status="ready")
    except Exception as e:
        logger.error("API startup failed", error=str(e))
        raise

# API key dependency (simplified for demo)
async def get_api_key(api_key: Optional[str] = None):
    """Simple API key validation."""
    # In production, implement proper authentication
    return api_key

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Test graph availability
        graph_status = "ok" if graph else "error"
        
        # Test storage directories
        storage_dirs = ["./storage/vectors", "./storage/reports", "./data/inbox"]
        storage_status = "ok" if all(Path(d).exists() for d in storage_dirs) else "error"
        
        return HealthResponse(
            status="healthy" if graph_status == "ok" and storage_status == "ok" else "degraded",
            timestamp=datetime.now().isoformat(),
            version="1.0.0",
            components={
                "graph": graph_status,
                "storage": storage_status,
                "api": "ok"
            }
        )
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    query: str = Form(None),
    include_debug: bool = Form(False),
    api_key: Optional[str] = Depends(get_api_key)
):
    """
    Analyze a single PDF document.
    
    Upload a PDF file and get comprehensive analysis including:
    - Content extraction and chunking
    - LLM-powered analysis
    - Statistical validation with SciPy
    - Relevance decision with confidence score
    """
    try:
        # Validate file type
        if not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        # Generate job and document IDs
        job_id = str(uuid.uuid4())
        doc_id = Path(file.filename).stem
        
        logger.info("Starting document analysis", 
                   job_id=job_id, 
                   filename=file.filename)
        
        # Save uploaded file
        inbox_dir = Path("./data/inbox")
        inbox_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = inbox_dir / f"{job_id}_{file.filename}"
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Start background analysis
        background_tasks.add_task(
            process_document,
            str(file_path),
            job_id,
            query,
            include_debug
        )
        
        return AnalysisResponse(
            job_id=job_id,
            doc_id=doc_id,
            status="processing",
            message="Document analysis started"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Document analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

async def process_document(file_path: str, job_id: str, query: Optional[str], include_debug: bool):
    """Background task to process document."""
    try:
        logger.info("Processing document", job_id=job_id, file_path=file_path)
        
        # Run analysis
        final_state = run_graph(graph, file_path, query, thread_id=job_id)
        
        # Save reports
        output_dir = "./storage/reports"
        report_paths = save_complete_analysis(final_state, output_dir)
        
        # Create job result file
        result = {
            "job_id": job_id,
            "doc_id": final_state.get("doc_id"),
            "status": "completed" if not final_state.get("error") else "failed",
            "processed_at": datetime.now().isoformat(),
            "report_paths": report_paths,
            "decision": final_state.get("decision").model_dump() if final_state.get("decision") else None,
            "error": final_state.get("error")
        }
        
        # Save job result
        jobs_dir = Path("./storage/jobs")
        jobs_dir.mkdir(parents=True, exist_ok=True)
        
        job_file = jobs_dir / f"{job_id}.json"
        with job_file.open("w") as f:
            json.dump(result, f, indent=2)
        
        logger.info("Document processing completed", 
                   job_id=job_id, 
                   status=result["status"])
        
    except Exception as e:
        logger.error("Document processing failed", 
                    job_id=job_id, 
                    error=str(e))
        
        # Save error result
        error_result = {
            "job_id": job_id,
            "status": "failed",
            "processed_at": datetime.now().isoformat(),
            "error": str(e)
        }
        
        jobs_dir = Path("./storage/jobs")
        jobs_dir.mkdir(parents=True, exist_ok=True)
        
        job_file = jobs_dir / f"{job_id}.json"
        with job_file.open("w") as f:
            json.dump(error_result, f, indent=2)

@app.get("/status/{job_id}", response_model=AnalysisResponse)
async def get_job_status(job_id: str):
    """Get the status of an analysis job."""
    try:
        job_file = Path(f"./storage/jobs/{job_id}.json")
        
        if not job_file.exists():
            return AnalysisResponse(
                job_id=job_id,
                doc_id="unknown",
                status="not_found",
                message="Job not found or still processing"
            )
        
        with job_file.open("r") as f:
            result = json.load(f)
        
        return AnalysisResponse(**result, message="Job completed")
        
    except Exception as e:
        logger.error("Status check failed", job_id=job_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/report/{doc_id}")
async def get_report(doc_id: str, format: str = "json"):
    """Get analysis report for a document."""
    try:
        reports_dir = Path("./storage/reports")
        
        if format == "json":
            # Find JSON report
            json_files = list(reports_dir.glob(f"{doc_id}-*.json"))
            if not json_files:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return FileResponse(
                json_files[0],
                media_type="application/json",
                filename=f"{doc_id}_report.json"
            )
        
        elif format == "markdown":
            # Find Markdown report
            md_files = list(reports_dir.glob(f"{doc_id}-*.md"))
            if not md_files:
                raise HTTPException(status_code=404, detail="Report not found")
            
            return FileResponse(
                md_files[0],
                media_type="text/markdown",
                filename=f"{doc_id}_report.md"
            )
        
        else:
            raise HTTPException(status_code=400, detail="Format must be 'json' or 'markdown'")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Report retrieval failed", doc_id=doc_id, error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents", response_model=List[DocumentInfo])
async def list_documents():
    """List all processed documents."""
    try:
        jobs_dir = Path("./storage/jobs")
        documents = []
        
        if jobs_dir.exists():
            for job_file in jobs_dir.glob("*.json"):
                try:
                    with job_file.open("r") as f:
                        job_data = json.load(f)
                    
                    documents.append(DocumentInfo(
                        doc_id=job_data.get("doc_id", "unknown"),
                        filename=job_file.stem,
                        processed_at=job_data.get("processed_at", "unknown"),
                        status=job_data.get("status", "unknown"),
                        report_available=bool(job_data.get("report_paths"))
                    ))
                except Exception as e:
                    logger.warning("Failed to read job file", file=str(job_file), error=str(e))
                    continue
        
        return documents
        
    except Exception as e:
        logger.error("Document listing failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch")
async def batch_analyze(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    query: str = Form(None),
    api_key: Optional[str] = Depends(get_api_key)
):
    """Analyze multiple documents in batch."""
    try:
        batch_id = str(uuid.uuid4())
        job_ids = []
        
        logger.info("Starting batch analysis", 
                   batch_id=batch_id, 
                   file_count=len(files))
        
        for file in files:
            if not file.filename.lower().endswith('.pdf'):
                continue  # Skip non-PDF files
            
            job_id = str(uuid.uuid4())
            job_ids.append(job_id)
            
            # Save file
            inbox_dir = Path("./data/inbox")
            inbox_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = inbox_dir / f"{job_id}_{file.filename}"
            
            with file_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Start background processing
            background_tasks.add_task(
                process_document,
                str(file_path),
                job_id,
                query,
                False
            )
        
        return {
            "batch_id": batch_id,
            "job_ids": job_ids,
            "status": "processing",
            "message": f"Started processing {len(job_ids)} documents"
        }
        
    except Exception as e:
        logger.error("Batch analysis failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/cleanup")
async def cleanup():
    """Clean temporary files and cache."""
    try:
        cleanup_paths = [
            "./data/inbox",
            "./storage/vectors",
            "./storage/jobs"
        ]
        
        cleaned = []
        for path in cleanup_paths:
            dir_path = Path(path)
            if dir_path.exists():
                for file in dir_path.iterdir():
                    if file.is_file():
                        file.unlink()
                        cleaned.append(str(file))
        
        logger.info("Cleanup completed", files_removed=len(cleaned))
        
        return {
            "status": "completed",
            "files_removed": len(cleaned),
            "message": "Cleanup completed successfully"
        }
        
    except Exception as e:
        logger.error("Cleanup failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error("Unhandled exception", 
                path=request.url.path, 
                error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "path": request.url.path
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    # Run the API server
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )