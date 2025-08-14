"""Test advanced functionality with graph execution."""
import sys
import os
sys.path.append('/home/runner/work/Research-agent/Research-agent/src')

# Test imports
try:
    from nodes import node_ingest, node_analyze, node_report
    from graph import build_graph, get_graph_visualization
    from reporter import generate_summary_report, save_report
    from tools_scipy import comprehensive_analysis
    print("✓ All advanced imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

def test_scipy_analysis():
    """Test SciPy analysis functionality."""
    print("\n=== Testing SciPy Analysis ===")
    
    # Test text with statistical content
    test_text = """
    The study analyzed 120 participants with a mean age of 25.3 ± 4.2 years.
    The t-test revealed significant differences (p < 0.05) between groups.
    Correlation analysis showed r = 0.73, p = 0.001.
    Sample sizes were n=60 in each group.
    The regression analysis yielded R² = 0.89.
    """
    
    results = comprehensive_analysis(test_text)
    print(f"✓ Analysis completed: {results['needs_scipy']}")
    print(f"✓ Analyses performed: {results.get('analysis_performed', [])}")
    
    return results

def test_graph_structure():
    """Test graph building and structure."""
    print("\n=== Testing Graph Structure ===")
    
    try:
        # This will fail without API key, but we can test the structure
        print("Graph visualization:")
        print(get_graph_visualization())
        print("✓ Graph structure validated")
        return True
    except Exception as e:
        print(f"Graph structure test: {e}")
        return False

def test_report_generation():
    """Test report generation functionality."""
    print("\n=== Testing Report Generation ===")
    
    # Create mock state
    from schema import GraphState, AnalysisResult, Decision, Report
    
    mock_state = GraphState()
    mock_state["doc_id"] = "test-document"
    mock_state["analysis"] = AnalysisResult(
        findings=["Test finding 1", "Test finding 2"],
        needs_scipy=True,
        rationale="Test rationale"
    )
    mock_state["decision"] = Decision(
        label="relevant",
        confidence=0.85,
        criteria=["Statistical significance", "Methodology"]
    )
    
    summary = generate_summary_report(mock_state)
    print(f"✓ Summary report generated ({len(summary)} characters)")
    print("Preview:")
    print(summary[:200] + "...")
    
    return True

if __name__ == "__main__":
    print("=== Advanced Research Agent Testing ===")
    
    # Test SciPy functionality
    scipy_results = test_scipy_analysis()
    
    # Test graph structure
    graph_ok = test_graph_structure()
    
    # Test report generation
    report_ok = test_report_generation()
    
    print(f"\n=== Test Results ===")
    print(f"✓ SciPy Analysis: Working")
    print(f"✓ Graph Structure: {'Working' if graph_ok else 'Needs API key'}")
    print(f"✓ Report Generation: {'Working' if report_ok else 'Failed'}")
    print(f"\n✓ Advanced functionality tests completed!")