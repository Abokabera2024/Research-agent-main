import os
import typer
from pathlib import Path
from dotenv import load_dotenv
from graph import build_graph, run_graph, stream_graph, get_graph_visualization
from reporter import save_complete_analysis, create_report_index
from config import validate_config
import structlog
from typing import Optional
import json

# Configure structured logging
structlog.configure(
    processors=[
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.make_filtering_bound_logger(20),  # INFO level
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()
app = typer.Typer(
    name="research-agent",
    help="Research Agent CLI - Analyze scientific documents with AI and SciPy",
    add_completion=False
)

def setup_environment():
    """Setup and validate environment."""
    try:
        load_dotenv()
        validate_config()
        return True
    except Exception as e:
        typer.echo(f"❌ Environment setup failed: {e}", err=True)
        return False

@app.command()
def run(
    pdf_path: str = typer.Argument(..., help="Path to PDF file to analyze"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Optional retrieval query"),
    output_dir: str = typer.Option("./storage/reports", "--output", "-o", help="Output directory for reports"),
    stream: bool = typer.Option(False, "--stream", "-s", help="Stream progress in real-time"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging")
):
    """
    Analyze a single PDF document.
    
    Example:
        research-agent run paper.pdf --query "statistical significance and p-values"
    """
    if not setup_environment():
        raise typer.Exit(1)
    
    # Set logging level
    if verbose:
        structlog.configure(
            processors=[structlog.dev.ConsoleRenderer()],
            wrapper_class=structlog.make_filtering_bound_logger(10),  # DEBUG level
            logger_factory=structlog.WriteLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    
    # Validate input file
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        typer.echo(f"❌ PDF file not found: {pdf_path}", err=True)
        raise typer.Exit(1)
    
    if not pdf_file.suffix.lower() == '.pdf':
        typer.echo(f"❌ File must be a PDF: {pdf_path}", err=True)
        raise typer.Exit(1)
    
    try:
        typer.echo(f"🔬 Analyzing document: {pdf_file.name}")
        
        # Build graph
        typer.echo("🔧 Building analysis pipeline...")
        graph = build_graph()
        
        # Run analysis
        if stream:
            typer.echo("📊 Streaming analysis progress...")
            for step in stream_graph(graph, str(pdf_file), query):
                node = step['node']
                state = step['state']
                
                if state.get('error'):
                    typer.echo(f"❌ Error in {node}: {state['error']}")
                else:
                    typer.echo(f"✅ Completed: {node}")
            
            # Get final state (this is a simplification)
            final_state = step['state']  # Last state
        else:
            typer.echo("⚙️ Running analysis...")
            final_state = run_graph(graph, str(pdf_file), query)
        
        # Check for errors
        if final_state.get('error'):
            typer.echo(f"❌ Analysis failed: {final_state['error']}", err=True)
            raise typer.Exit(1)
        
        # Save results
        typer.echo("💾 Saving reports...")
        paths = save_complete_analysis(final_state, output_dir)
        
        # Display results
        typer.echo("✅ Analysis completed successfully!")
        typer.echo(f"📄 Document: {final_state.get('doc_id', 'unknown')}")
        
        if final_state.get('decision'):
            decision = final_state['decision']
            typer.echo(f"🎯 Decision: {decision.label} (confidence: {decision.confidence:.2f})")
        
        typer.echo("📁 Reports saved:")
        for format_type, path in paths.items():
            typer.echo(f"  - {format_type}: {path}")
        
    except Exception as e:
        logger.error("CLI execution failed", error=str(e))
        typer.echo(f"❌ Analysis failed: {e}", err=True)
        
        # Add more detailed error information
        import traceback
        logger.error("Full traceback", traceback=traceback.format_exc())
        
        if verbose:
            typer.echo(f"Full error details: {traceback.format_exc()}", err=True)
        
        raise typer.Exit(1)

@app.command()
def batch(
    input_dir: str = typer.Argument(..., help="Directory containing PDF files"),
    output_dir: str = typer.Option("./storage/reports", "--output", "-o", help="Output directory"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Query for all documents"),
    max_files: int = typer.Option(10, "--max", "-m", help="Maximum number of files to process")
):
    """
    Process multiple PDF files in batch.
    
    Example:
        research-agent batch ./papers --query "methodology and results"
    """
    if not setup_environment():
        raise typer.Exit(1)
    
    input_path = Path(input_dir)
    if not input_path.exists():
        typer.echo(f"❌ Input directory not found: {input_dir}", err=True)
        raise typer.Exit(1)
    
    # Find PDF files
    pdf_files = list(input_path.glob("*.pdf"))[:max_files]
    
    if not pdf_files:
        typer.echo(f"❌ No PDF files found in: {input_dir}", err=True)
        raise typer.Exit(1)
    
    typer.echo(f"📚 Found {len(pdf_files)} PDF files to process")
    
    # Build graph once
    graph = build_graph()
    processed = 0
    failed = 0
    
    # Process each file
    with typer.progressbar(pdf_files, label="Processing documents") as progress:
        for pdf_file in progress:
            try:
                typer.echo(f"\n🔬 Processing: {pdf_file.name}")
                
                final_state = run_graph(graph, str(pdf_file), query)
                
                if final_state.get('error'):
                    typer.echo(f"❌ Failed: {final_state['error']}")
                    failed += 1
                else:
                    save_complete_analysis(final_state, output_dir)
                    processed += 1
                    
                    if final_state.get('decision'):
                        decision = final_state['decision']
                        typer.echo(f"✅ {decision.label} (confidence: {decision.confidence:.2f})")
                
            except Exception as e:
                typer.echo(f"❌ Error processing {pdf_file.name}: {e}")
                failed += 1
    
    # Create index
    create_report_index(output_dir)
    
    # Summary
    typer.echo(f"\n📊 Batch processing complete:")
    typer.echo(f"  - Processed: {processed}")
    typer.echo(f"  - Failed: {failed}")
    typer.echo(f"  - Reports: {output_dir}")

@app.command()
def status():
    """Check system status and configuration."""
    typer.echo("🔍 Research Agent Status Check")
    typer.echo("=" * 40)
    
    # Environment check
    try:
        load_dotenv()
        validate_config()
        typer.echo("✅ Environment: OK")
    except Exception as e:
        typer.echo(f"❌ Environment: {e}")
        return
    
    # Check directories
    directories = [
        "./data/inbox",
        "./data/processed", 
        "./data/examples",
        "./storage/vectors",
        "./storage/checkpoints",
        "./storage/reports"
    ]
    
    typer.echo("\n📁 Directory Structure:")
    for dir_path in directories:
        path = Path(dir_path)
        if path.exists():
            typer.echo(f"  ✅ {dir_path}")
        else:
            typer.echo(f"  ❌ {dir_path} (missing)")
    
    # Check configuration
    typer.echo("\n⚙️  Configuration:")
    config_vars = ["OPENAI_API_KEY", "LLM_MODEL", "EMBEDDING_MODEL"]
    for var in config_vars:
        value = os.getenv(var)
        if value:
            display_value = value[:10] + "..." if len(value) > 10 else value
            typer.echo(f"  ✅ {var}: {display_value}")
        else:
            typer.echo(f"  ❌ {var}: Not set")
    
    # Graph test
    typer.echo("\n🔗 Graph Structure:")
    try:
        graph = build_graph()
        typer.echo("  ✅ Graph builds successfully")
    except Exception as e:
        typer.echo(f"  ❌ Graph build failed: {e}")

@app.command()
def init():
    """Initialize the research agent environment."""
    typer.echo("🚀 Initializing Research Agent")
    
    # Create directories
    directories = [
        "./data/inbox",
        "./data/processed", 
        "./data/examples",
        "./storage/vectors",
        "./storage/checkpoints",
        "./storage/reports"
    ]
    
    for dir_path in directories:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        typer.echo(f"📁 Created: {dir_path}")
    
    # Check for .env file
    env_file = Path(".env")
    if not env_file.exists():
        typer.echo("📝 Creating .env file template...")
        env_template = """# Research Agent Configuration
OPENAI_API_KEY=your-openai-api-key-here
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o-mini
CHROMA_DIR=./storage/vectors
CHECKPOINT_DB=./storage/checkpoints/graph_state.sqlite
"""
        env_file.write_text(env_template)
        typer.echo("✅ .env file created - please add your API keys!")
    else:
        typer.echo("✅ .env file already exists")
    
    typer.echo("\n🎉 Initialization complete!")
    typer.echo("📋 Next steps:")
    typer.echo("  1. Add your OpenAI API key to .env file")
    typer.echo("  2. Run 'research-agent status' to verify setup")
    typer.echo("  3. Test with 'research-agent run your-document.pdf'")

@app.command()
def info():
    """Show information about the research agent."""
    typer.echo("🤖 Research Agent Information")
    typer.echo("=" * 40)
    
    typer.echo("📖 Description:")
    typer.echo("  Scientific document analysis with AI and statistical validation")
    
    typer.echo("\n🔧 Capabilities:")
    typer.echo("  - PDF document ingestion and processing")
    typer.echo("  - Semantic text chunking and embedding")
    typer.echo("  - LLM-powered content analysis")
    typer.echo("  - Statistical analysis with SciPy")
    typer.echo("  - Automated relevance decisions")
    typer.echo("  - Comprehensive report generation")
    
    typer.echo("\n🏗️  Architecture:")
    print(get_graph_visualization())

if __name__ == "__main__":
    app()