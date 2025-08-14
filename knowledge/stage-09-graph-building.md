# Stage 9: Graph Building and Routing

## Overview
This stage implements the LangGraph workflow that orchestrates the entire research agent pipeline with conditional routing and state management.

## Graph Architecture

### Sequential Nodes
1. **ingest** → Document loading and validation
2. **split_embed** → Text chunking and vector embedding
3. **retrieve** → Semantic search and chunk retrieval
4. **analyze** → LLM-based content analysis

### Conditional Routing
- **analyze** → **scipy_compute** (if analysis.needs_scipy is True)
- **analyze** → **decide** (if SciPy analysis not needed)
- **scipy_compute** → **decide** (after statistical analysis)

### Final Processing
- **decide** → **report** → END

## Implementation Details

### State Management
- Uses SqliteSaver for checkpoint persistence
- Enables workflow resumption and debugging
- Thread-based isolation for concurrent processing

### Routing Logic
- Conditional edges based on analysis requirements
- Fallback paths for error scenarios
- Skip unnecessary processing steps

### Error Handling
- Graceful degradation with partial results
- Error state preservation in checkpoints
- Recovery mechanisms for failed nodes

## Features
- **Persistence**: SQLite checkpoints for reliability
- **Concurrency**: Thread-safe state management
- **Monitoring**: Full workflow observability
- **Recovery**: Resume from any checkpoint

## Configuration
- Checkpoint database location configurable
- Node timeout and retry settings
- Custom routing conditions

## Status
- [x] Graph architecture designed
- [ ] graph.py implemented
- [ ] Conditional routing tested
- [ ] Checkpoint functionality validated

## Next Stage
Stage 10: Report Generation Implementation