"""Simple test script to verify basic functionality."""
import sys
import os
sys.path.append('/home/runner/work/Research-agent/Research-agent/src')

# Test imports
try:
    from config import validate_config
    from schema import DocChunk, AnalysisResult, Decision, Report, GraphState
    from loaders import assign_doc_id, validate_pdf_file
    from chunking import simple_chunk
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test basic functionality
def test_basic_functionality():
    print("\n=== Testing Basic Functionality ===")
    
    # Test document ID assignment
    doc_id = assign_doc_id("/path/to/test.pdf")
    print(f"✓ Document ID: {doc_id}")
    
    # Test schema validation
    chunk = DocChunk(
        doc_id="test",
        chunk_id="test-001",
        text="This is a test chunk."
    )
    print(f"✓ DocChunk created: {chunk.chunk_id}")
    
    # Test text chunking
    test_text = "This is a test document. " * 100  # Create longer text
    chunks = simple_chunk(test_text, "test-doc")
    print(f"✓ Created {len(chunks)} chunks from test text")
    
    # Test analysis result
    analysis = AnalysisResult(
        findings=["Test finding"],
        needs_scipy=False,
        rationale="Test rationale"
    )
    print(f"✓ AnalysisResult created: {len(analysis.findings)} findings")
    
    # Test decision
    decision = Decision(
        label="relevant",
        confidence=0.8,
        criteria=["test criterion"]
    )
    print(f"✓ Decision created: {decision.label} ({decision.confidence})")
    
    print("\n✓ All basic tests passed!")

if __name__ == "__main__":
    test_basic_functionality()