# Stage 3: Data Models and Schemas

## Overview
This stage defines the data models and schemas that will be used throughout the research agent pipeline using Pydantic for data validation.

## Core Data Models

### Document Processing Models
- **DocChunk**: Represents a chunk of text from a document
- **AnalysisResult**: Contains analysis findings and statistics
- **Decision**: Represents the agent's decision about document relevance
- **Report**: Final report structure

### Graph State
The GraphState TypedDict defines the state that flows through the LangGraph pipeline, containing all intermediate and final results.

## Implementation Details

### DocChunk Model
- Stores document ID, chunk ID, text content, and metadata
- Used for text retrieval and vector storage

### AnalysisResult Model
- Contains extracted findings, statistical data, and SciPy requirements
- Includes rationale for analysis decisions

### Decision Model
- Stores relevance label, confidence score, and decision criteria
- Used for final document classification

### Report Model
- Comprehensive summary of the entire analysis process
- Includes methods used, decisions made, and attachments

### GraphState
- Central state object that flows through all processing nodes
- Contains all intermediate and final processing results
- Enables state persistence and recovery

## Validation Features
- Type safety with Pydantic
- Automatic data validation
- Serialization for storage and API responses

## Status
- [x] Data models designed
- [ ] schema.py implemented
- [ ] Validation tests added
- [ ] Integration with other modules

## Next Stage
Stage 4: Document Loading and Processing Implementation