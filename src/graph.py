from langgraph.graph import StateGraph, END
from schema import GraphState
from nodes import (
    node_ingest, node_split_embed, node_retrieve,
    node_analyze, node_scipy_compute, node_decide, node_report
)
import os
import structlog
from pathlib import Path

logger = structlog.get_logger()

def build_graph():
    """
    Build and configure the LangGraph workflow.
    
    Returns:
        Compiled graph with checkpointer
    """
    try:
        logger.info("Building LangGraph workflow")
        
        # Create the graph builder
        builder = StateGraph(GraphState)
        
        # Add all nodes
        builder.add_node("ingest", node_ingest)
        builder.add_node("split_embed", node_split_embed)
        builder.add_node("retrieve", node_retrieve)
        builder.add_node("analyze", node_analyze)
        builder.add_node("scipy_compute", node_scipy_compute)
        builder.add_node("decide", node_decide)
        builder.add_node("report", node_report)
        
        # Set entry point
        builder.set_entry_point("ingest")
        
        # Add sequential edges
        builder.add_edge("ingest", "split_embed")
        builder.add_edge("split_embed", "retrieve")
        builder.add_edge("retrieve", "analyze")
        
        # Add conditional routing for SciPy analysis
        def route_scipy(state: GraphState):
            """
            Route to SciPy computation if needed, otherwise go to decision.
            
            Args:
                state: Current graph state
                
            Returns:
                Next node name
            """
            try:
                # Check if there's an error first
                if state.get("error"):
                    logger.warning("Error detected, skipping to decision", 
                                 error=state["error"])
                    return "decide"
                
                # Check if analysis indicates SciPy is needed
                analysis = state.get("analysis")
                if analysis and analysis.needs_scipy:
                    logger.info("Routing to SciPy computation")
                    return "scipy_compute"
                else:
                    logger.info("Skipping SciPy computation")
                    return "decide"
                    
            except Exception as e:
                logger.error("Routing error, defaulting to decide", error=str(e))
                return "decide"
        
        builder.add_conditional_edges("analyze", route_scipy, {
            "scipy_compute": "scipy_compute",
            "decide": "decide"
        })
        
        # Complete the workflow
        builder.add_edge("scipy_compute", "decide")
        builder.add_edge("decide", "report")
        builder.add_edge("report", END)
        
        # Setup checkpoint persistence (simplified for now)
        checkpoint_db = os.getenv("CHECKPOINT_DB", "./storage/checkpoints/graph_state.sqlite")
        
        # Ensure checkpoint directory exists
        checkpoint_path = Path(checkpoint_db)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create checkpointer - using basic memory checkpointer for now
        # TODO: Implement SqliteSaver when available
        # checkpointer = SqliteSaver.from_conn_string(checkpoint_db)
        
        # Compile the graph (without checkpointer for now)
        graph = builder.compile()  # checkpointer=checkpointer
        
        logger.info("LangGraph workflow built successfully", 
                   checkpoint_db=checkpoint_db)
        
        return graph
        
    except Exception as e:
        logger.error("Failed to build graph", error=str(e))
        raise

def get_graph_visualization():
    """
    Get a text representation of the graph structure.
    
    Returns:
        String representation of the workflow
    """
    return """
Research Agent Workflow:

START → ingest → split_embed → retrieve → analyze
                                              ↓
                               needs_scipy? ─┐
                                  ↓         │
                              scipy_compute │
                                  ↓         │
                               decide ←─────┘
                                  ↓
                               report → END

Nodes:
- ingest: Load and validate PDF document
- split_embed: Chunk text and create embeddings
- retrieve: Search for relevant content
- analyze: LLM analysis of retrieved content
- scipy_compute: Statistical analysis (conditional)
- decide: Make relevance decision
- report: Generate final report
"""

def run_graph(graph, doc_path: str, query: str = None, thread_id: str = None):
    """
    Run the graph on a document.
    
    Args:
        graph: Compiled LangGraph instance
        doc_path: Path to PDF document
        query: Optional retrieval query
        thread_id: Optional thread ID for checkpointing
        
    Returns:
        Final state after processing
    """
    try:
        logger.info("Starting graph execution", 
                   doc_path=doc_path, 
                   query=query,
                   thread_id=thread_id)
        
        # Prepare initial state
        initial_state = {"doc_path": doc_path}
        if query:
            initial_state["query"] = query
        
        # Configure execution
        config = {
            "configurable": {
                "thread_id": thread_id or f"run-{os.path.basename(doc_path)}"
            }
        }
        
        # Execute the graph
        final_state = graph.invoke(initial_state, config)
        
        logger.info("Graph execution completed", 
                   doc_id=final_state.get("doc_id"),
                   has_error=bool(final_state.get("error")))
        
        return final_state
        
    except Exception as e:
        logger.error("Graph execution failed", 
                    error=str(e), 
                    doc_path=doc_path)
        raise

def stream_graph(graph, doc_path: str, query: str = None, thread_id: str = None):
    """
    Stream graph execution with real-time updates.
    
    Args:
        graph: Compiled LangGraph instance
        doc_path: Path to PDF document
        query: Optional retrieval query
        thread_id: Optional thread ID for checkpointing
        
    Yields:
        State updates during execution
    """
    try:
        logger.info("Starting streaming graph execution", 
                   doc_path=doc_path)
        
        # Prepare initial state
        initial_state = {"doc_path": doc_path}
        if query:
            initial_state["query"] = query
        
        # Configure execution
        config = {
            "configurable": {
                "thread_id": thread_id or f"stream-{os.path.basename(doc_path)}"
            }
        }
        
        # Stream execution
        for step in graph.stream(initial_state, config):
            node_name = list(step.keys())[0]
            node_output = step[node_name]
            
            logger.debug("Graph step completed", 
                        node=node_name,
                        has_error=bool(node_output.get("error")))
            
            yield {
                "node": node_name,
                "state": node_output
            }
        
        logger.info("Streaming graph execution completed")
        
    except Exception as e:
        logger.error("Streaming graph execution failed", 
                    error=str(e), 
                    doc_path=doc_path)
        raise

def get_graph_state(graph, thread_id: str):
    """
    Get the current state for a thread.
    
    Args:
        graph: Compiled LangGraph instance
        thread_id: Thread identifier
        
    Returns:
        Current state or None if not found
    """
    try:
        config = {"configurable": {"thread_id": thread_id}}
        state = graph.get_state(config)
        return state.values if state else None
        
    except Exception as e:
        logger.error("Failed to get graph state", 
                    error=str(e), 
                    thread_id=thread_id)
        return None

def list_checkpoints(graph, thread_id: str = None):
    """
    List available checkpoints.
    
    Args:
        graph: Compiled LangGraph instance
        thread_id: Optional thread ID filter
        
    Returns:
        List of checkpoint information
    """
    try:
        # This would require accessing the checkpointer directly
        # Implementation depends on LangGraph checkpoint interface
        logger.debug("Listing checkpoints", thread_id=thread_id)
        # Placeholder implementation
        return []
        
    except Exception as e:
        logger.error("Failed to list checkpoints", error=str(e))
        return []