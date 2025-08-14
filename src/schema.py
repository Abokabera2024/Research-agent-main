from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field

class DocChunk(BaseModel):
    """Represents a chunk of text from a document."""
    doc_id: str = Field(description="Document identifier")
    chunk_id: str = Field(description="Unique chunk identifier")
    text: str = Field(description="Text content of the chunk")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class AnalysisResult(BaseModel):
    """Contains analysis findings and statistics."""
    findings: List[str] = Field(default_factory=list, description="List of key findings")
    stats: Dict[str, Any] = Field(default_factory=dict, description="Statistical information")
    needs_scipy: bool = Field(default=False, description="Whether SciPy analysis is needed")
    rationale: str = Field(default="", description="Reasoning for the analysis")

class Decision(BaseModel):
    """Represents the agent's decision about document relevance."""
    label: str = Field(description="Decision label (e.g., 'relevant', 'irrelevant', 'uncertain')")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1")
    criteria: List[str] = Field(default_factory=list, description="Decision criteria used")

class Report(BaseModel):
    """Final report structure."""
    summary: str = Field(description="Summary of the analysis")
    methods: List[str] = Field(default_factory=list, description="Methods used in analysis")
    decisions: Decision = Field(description="Final decision about the document")
    attachments: Dict[str, Any] = Field(default_factory=dict, description="Additional attachments")

# Graph State Definition
class GraphState(TypedDict, total=False):
    """State that flows through the LangGraph pipeline."""
    doc_path: str
    doc_id: str
    raw_text: str
    chunks: List[DocChunk]
    query: str
    retrieved: List[DocChunk]
    analysis: AnalysisResult
    scipy_out: Dict[str, Any]
    decision: Decision
    report: Report
    human_feedback: Optional[str]
    error: Optional[str]