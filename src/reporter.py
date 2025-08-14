from pathlib import Path
from datetime import datetime
import json
import structlog
from typing import Dict, Any, Optional
from schema import Report, GraphState

logger = structlog.get_logger()

def save_report(markdown_text: str, out_dir: str, doc_id: str) -> str:
    """
    Save a markdown report to file.
    
    Args:
        markdown_text: Report content in markdown format
        out_dir: Output directory for the report
        doc_id: Document identifier for filename
        
    Returns:
        Path to saved report file
    """
    try:
        # Create output directory
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{doc_id}-{timestamp}.md"
        report_path = output_path / filename
        
        # Write report to file
        report_path.write_text(markdown_text, encoding="utf-8")
        
        logger.info("Report saved successfully", 
                   path=str(report_path),
                   doc_id=doc_id)
        
        return str(report_path)
        
    except Exception as e:
        logger.error("Failed to save report", 
                    error=str(e),
                    doc_id=doc_id)
        raise

def save_report_json(report_data: Dict[str, Any], out_dir: str, doc_id: str) -> str:
    """
    Save report data as JSON file.
    
    Args:
        report_data: Report data dictionary
        out_dir: Output directory
        doc_id: Document identifier
        
    Returns:
        Path to saved JSON file
    """
    try:
        output_path = Path(out_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{doc_id}-{timestamp}.json"
        json_path = output_path / filename
        
        # Write JSON data
        with json_path.open('w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info("JSON report saved", 
                   path=str(json_path),
                   doc_id=doc_id)
        
        return str(json_path)
        
    except Exception as e:
        logger.error("Failed to save JSON report", 
                    error=str(e),
                    doc_id=doc_id)
        raise

def generate_comprehensive_report(state: GraphState, include_debug: bool = False) -> Dict[str, Any]:
    """
    Generate comprehensive report from graph state.
    
    Args:
        state: Final graph state
        include_debug: Include debug information
        
    Returns:
        Complete report data dictionary
    """
    try:
        logger.info("Generating comprehensive report", 
                   doc_id=state.get("doc_id"))
        
        timestamp = datetime.now().isoformat()
        
        # Base report structure
        report_data = {
            "metadata": {
                "doc_id": state.get("doc_id"),
                "doc_path": state.get("doc_path"),
                "generated_at": timestamp,
                "version": "1.0"
            },
            "processing": {
                "chunks_created": len(state.get("chunks", [])),
                "chunks_retrieved": len(state.get("retrieved", [])),
                "text_length": len(state.get("raw_text", "")),
                "query_used": state.get("query"),
                "error_occurred": bool(state.get("error"))
            }
        }
        
        # Add analysis results
        if state.get("analysis"):
            analysis = state["analysis"]
            report_data["analysis"] = {
                "findings": analysis.findings,
                "stats": analysis.stats,
                "needs_scipy": analysis.needs_scipy,
                "rationale": analysis.rationale
            }
        
        # Add SciPy results
        if state.get("scipy_out"):
            report_data["statistical_analysis"] = state["scipy_out"]
        
        # Add decision
        if state.get("decision"):
            decision = state["decision"]
            report_data["decision"] = {
                "label": decision.label,
                "confidence": decision.confidence,
                "criteria": decision.criteria
            }
        
        # Add final report
        if state.get("report"):
            report = state["report"]
            report_data["final_report"] = {
                "summary": report.summary,
                "methods": report.methods,
                "attachments": report.attachments
            }
        
        # Add error information if present
        if state.get("error"):
            report_data["error"] = {
                "message": state["error"],
                "occurred_at": timestamp
            }
        
        # Debug information (optional)
        if include_debug:
            report_data["debug"] = {
                "state_keys": list(state.keys()),
                "chunks_sample": [c.chunk_id for c in state.get("chunks", [])[:3]],
                "retrieved_sample": [c.chunk_id for c in state.get("retrieved", [])[:3]]
            }
        
        logger.info("Comprehensive report generated", 
                   doc_id=state.get("doc_id"),
                   sections=len(report_data))
        
        return report_data
        
    except Exception as e:
        logger.error("Failed to generate comprehensive report", error=str(e))
        return {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "error": str(e)
            },
            "error": str(e)
        }

def generate_summary_report(state: GraphState) -> str:
    """
    Generate a concise summary report.
    
    Args:
        state: Final graph state
        
    Returns:
        Summary report as markdown string
    """
    try:
        doc_id = state.get("doc_id", "unknown")
        
        summary_lines = [
            f"# Research Agent Summary: {doc_id}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ""
        ]
        
        # Processing status
        if state.get("error"):
            summary_lines.extend([
                "## Status: ERROR",
                f"Error: {state['error']}",
                ""
            ])
        else:
            summary_lines.extend([
                "## Status: COMPLETED",
                ""
            ])
        
        # Quick stats
        chunks = len(state.get("chunks", []))
        retrieved = len(state.get("retrieved", []))
        
        summary_lines.extend([
            "## Processing Statistics",
            f"- Text chunks created: {chunks}",
            f"- Relevant chunks retrieved: {retrieved}",
            ""
        ])
        
        # Decision
        if state.get("decision"):
            decision = state["decision"]
            summary_lines.extend([
                "## Decision",
                f"- **Relevance:** {decision.label}",
                f"- **Confidence:** {decision.confidence:.2f}",
                ""
            ])
        
        # Key findings
        if state.get("analysis"):
            findings = state["analysis"].findings[:3]  # Top 3 findings
            if findings:
                summary_lines.extend([
                    "## Key Findings",
                    *[f"- {finding}" for finding in findings],
                    ""
                ])
        
        # Statistical analysis
        if state.get("scipy_out"):
            scipy_results = state["scipy_out"]
            analyses = scipy_results.get("analysis_performed", [])
            if analyses:
                summary_lines.extend([
                    "## Statistical Analysis",
                    f"- Analyses performed: {', '.join(analyses)}",
                    ""
                ])
        
        summary_text = "\n".join(summary_lines)
        
        logger.info("Summary report generated", 
                   doc_id=doc_id,
                   length=len(summary_text))
        
        return summary_text
        
    except Exception as e:
        logger.error("Failed to generate summary report", error=str(e))
        return f"# Error generating summary\n\nError: {str(e)}"

def save_complete_analysis(state: GraphState, out_dir: str) -> Dict[str, str]:
    """
    Save complete analysis in multiple formats.
    
    Args:
        state: Final graph state
        out_dir: Output directory
        
    Returns:
        Dictionary with paths to saved files
    """
    try:
        doc_id = state.get("doc_id", "unknown")
        paths = {}
        
        # Save markdown report
        if state.get("report"):
            md_path = save_report(state["report"].summary, out_dir, doc_id)
            paths["markdown"] = md_path
        
        # Save JSON data
        comprehensive_data = generate_comprehensive_report(state, include_debug=True)
        json_path = save_report_json(comprehensive_data, out_dir, doc_id)
        paths["json"] = json_path
        
        # Save summary
        summary_text = generate_summary_report(state)
        summary_path = save_report(summary_text, out_dir, f"{doc_id}-summary")
        paths["summary"] = summary_path
        
        logger.info("Complete analysis saved", 
                   doc_id=doc_id,
                   formats=list(paths.keys()))
        
        return paths
        
    except Exception as e:
        logger.error("Failed to save complete analysis", error=str(e))
        raise

def create_report_index(reports_dir: str) -> str:
    """
    Create an index of all reports in a directory.
    
    Args:
        reports_dir: Directory containing reports
        
    Returns:
        Path to index file
    """
    try:
        reports_path = Path(reports_dir)
        
        if not reports_path.exists():
            return ""
        
        # Find all report files
        md_files = list(reports_path.glob("*.md"))
        json_files = list(reports_path.glob("*.json"))
        
        # Create index
        index_lines = [
            "# Research Agent Reports Index",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            f"## Summary",
            f"- Total markdown reports: {len(md_files)}",
            f"- Total JSON reports: {len(json_files)}",
            "",
            "## Markdown Reports"
        ]
        
        for md_file in sorted(md_files):
            index_lines.append(f"- [{md_file.name}]({md_file.name})")
        
        index_lines.extend([
            "",
            "## JSON Reports"
        ])
        
        for json_file in sorted(json_files):
            index_lines.append(f"- [{json_file.name}]({json_file.name})")
        
        index_text = "\n".join(index_lines)
        index_path = reports_path / "index.md"
        index_path.write_text(index_text, encoding="utf-8")
        
        logger.info("Report index created", 
                   path=str(index_path),
                   reports_count=len(md_files) + len(json_files))
        
        return str(index_path)
        
    except Exception as e:
        logger.error("Failed to create report index", error=str(e))
        return ""