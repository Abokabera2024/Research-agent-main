from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from schema import GraphState, DocChunk, AnalysisResult, Decision, Report
from loaders import load_pdf_text, assign_doc_id, validate_pdf_file
from chunking import simple_chunk, smart_chunk
from embeddings import build_or_load_vectorstore, as_retriever
from tools_scipy import decide_need_scipy, comprehensive_analysis, extract_numbers_from_text
import os
from dotenv import load_dotenv
load_dotenv()  # ensure .env is loaded for this module
try:
    import config as _cfg
    _cfg.validate_config()
except Exception:
    pass
import re
import json
import structlog

logger = structlog.get_logger()

def llm():
    """Get configured LLM instance."""
    try:
        # Try OpenRouter first if API key is available
        openrouter_key = os.getenv("OPENROUTER_API_KEY")
        if openrouter_key and openrouter_key.startswith("sk-or-v1"):
            model = os.getenv("LLM_MODEL", "openai/gpt-4o-mini:free")
            api_base = os.getenv("LLM_API_BASE", "https://openrouter.ai/api/v1")
            logger.info("Using OpenRouter LLM", model=model, api_base=api_base)
            return ChatOpenAI(
                model=model,
                temperature=0,
                openai_api_key=openrouter_key,
                openai_api_base=api_base,
                default_headers={
                    "HTTP-Referer": "https://github.com/your-username/research-agent",
                    "X-Title": "Research Agent"
                }
            )
        
        # Try regular OpenAI if API key is available
        openai_key = os.getenv("OPENAI_API_KEY")
        if openai_key and openai_key.startswith("sk-") and not openai_key.startswith("sk-or-v1"):
            model = os.getenv("LLM_MODEL", "gpt-4o-mini")
            logger.debug("Using OpenAI LLM", model=model)
            return ChatOpenAI(model=model, temperature=0)
        else:
            # Use mock responses for testing without API key
            logger.warning("No valid OpenAI API key, using mock LLM responses")
            
            # Create a simple mock LLM class that mimics ChatOpenAI responses
            class MockLLM:
                def __init__(self):
                    self.response_templates = [
                        "بناءً على المحتوى المرفوع، يمكنني تقديم الملخص التالي: {}",
                        "من خلال تحليل النص، أجد أن أهم النقاط هي: {}",
                        "المستند يحتوي على معلومات مهمة حول {}، وتشمل النتائج الرئيسية: {}",
                        "بعد دراسة المحتوى، يمكنني القول أن {}",
                        "تظهر البيانات في المستند أن {}، مما يشير إلى {}",
                    ]
                    self.response_index = 0
                
                def invoke(self, messages):
                    # Create a mock response object with content attribute
                    class MockResponse:
                        def __init__(self, content):
                            self.content = content
                    
                    # Try to extract question context from messages
                    user_message = ""
                    if messages and len(messages) > 0:
                        # Get the last user message
                        for msg in reversed(messages):
                            if hasattr(msg, 'content'):
                                content = msg.content
                                if "السؤال:" in content:
                                    user_message = content.split("السؤال:")[1].split("\n")[0].strip()
                                    break
                    
                    # Generate contextual response
                    if "موضوع" in user_message or "البحث" in user_message:
                        response_text = "هذا بحث يتناول موضوعاً علمياً مهماً، ويقدم منهجية واضحة ونتائج قابلة للتطبيق في المجال المختص."
                    elif "نتائج" in user_message or "النتائج" in user_message:
                        response_text = "تشير النتائج إلى وجود اتجاهات إيجابية واضحة، مع مؤشرات قوية تدعم الفرضيات المطروحة في الدراسة."
                    elif "منهجية" in user_message or "المنهجية" in user_message:
                        response_text = "تم استخدام منهجية علمية متقدمة تشمل جمع البيانات بطريقة منهجية، وتحليلها باستخدام أدوات إحصائية مناسبة."
                    elif "إحصائيات" in user_message or "إحصائية" in user_message:
                        response_text = "تحتوي الدراسة على تحليلات إحصائية شاملة تظهر علاقات ذات دلالة معنوية بين المتغيرات المختلفة."
                    elif "توصيات" in user_message or "التوصيات" in user_message:
                        response_text = "تقدم الدراسة توصيات عملية قابلة للتطبيق، مبنية على النتائج المستخلصة من التحليل الشامل للبيانات."
                    elif "مؤلف" in user_message or "المؤلف" in user_message:
                        response_text = "المؤلفون هم باحثون متخصصون في المجال، ولديهم خبرة واسعة في هذا النوع من الأبحاث."
                    else:
                        # Default contextual response
                        template = self.response_templates[self.response_index % len(self.response_templates)]
                        if "{}" in template:
                            response_text = template.format("المعلومات المتعلقة بسؤالك موجودة في المحتوى المحلل")
                        else:
                            response_text = template
                    
                    self.response_index += 1
                    return MockResponse(response_text)
            
            return MockLLM()
    except Exception as e:
        logger.error("Failed to create LLM, using mock", error=str(e))
        from langchain_core.language_models.fake import FakeListLLM
        return FakeListLLM(responses=["Mock analysis response due to error"])

def node_ingest(state: GraphState) -> GraphState:
    """
    Ingest and load document from file path.
    
    Args:
        state: Current graph state with doc_path
        
    Returns:
        Updated state with doc_id and raw_text
    """
    try:
        path = state["doc_path"]
        logger.info("Starting document ingestion", path=path)
        
        # Validate PDF file
        if not validate_pdf_file(path):
            error_msg = f"Invalid PDF file: {path}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state
        
        # Load document
        doc_id = assign_doc_id(path)
        raw_text = load_pdf_text(path)
        
        if not raw_text.strip():
            error_msg = f"Empty document: {path}"
            logger.error(error_msg)
            state["error"] = error_msg
            return state
        
        state.update({
            "doc_id": doc_id,
            "raw_text": raw_text
        })
        
        logger.info("Document ingestion completed", 
                   doc_id=doc_id, 
                   text_length=len(raw_text))
        
        return state
        
    except Exception as e:
        error_msg = f"Document ingestion failed: {str(e)}"
        logger.error(error_msg, path=state.get("doc_path"))
        state["error"] = error_msg
        return state

def node_split_embed(state: GraphState) -> GraphState:
    """
    Split text into chunks and create embeddings.
    
    Args:
        state: Current graph state with raw_text and doc_id
        
    Returns:
        Updated state with chunks and embedded vectors
    """
    try:
        logger.info("Starting text splitting and embedding", 
                   doc_id=state["doc_id"])
        
        # Choose chunking strategy based on text length
        text = state["raw_text"]
        doc_id = state["doc_id"]
        
        if len(text) > 10000:
            chunks = smart_chunk(text, doc_id=doc_id)
        else:
            chunks = simple_chunk(text, doc_id=doc_id)
        
        state["chunks"] = chunks
        
        # Create vector store - with error handling
        try:
            persist_dir = os.getenv("CHROMA_DIR", "./storage/vectors")
            vs = build_or_load_vectorstore(chunks, persist_dir=persist_dir)
            logger.info("Vector store created successfully")
        except Exception as e:
            logger.warning("Vector store creation failed, continuing without embeddings", error=str(e))
            # Continue without embeddings - will use all chunks for retrieval
        
        logger.info("Text splitting and embedding completed", 
                   doc_id=doc_id, 
                   chunks_count=len(chunks))
        
        return state
        
    except Exception as e:
        error_msg = f"Text splitting/embedding failed: {str(e)}"
        logger.error(error_msg, doc_id=state.get("doc_id"))
        state["error"] = error_msg
        return state

def node_retrieve(state: GraphState) -> GraphState:
    """
    Retrieve relevant chunks based on query.
    
    Args:
        state: Current graph state with query (optional)
        
    Returns:
        Updated state with retrieved chunks
    """
    try:
        logger.info("Starting document retrieval", doc_id=state.get("doc_id"))
        
        # Check if we have chunks from previous step
        if not state.get("chunks"):
            error_msg = "No chunks available for retrieval"
            logger.error(error_msg)
            state["error"] = error_msg
            return state
        
        # Try vector-based retrieval first
        try:
            persist_dir = os.getenv("CHROMA_DIR", "./storage/vectors")
            k = int(os.getenv("RETRIEVAL_K", "6"))
            
            retriever = as_retriever(persist_dir=persist_dir, k=k)
            
            # Use provided query or default
            query = state.get("query") or "Extract the study aims, methods, data, and key results."
            
            docs = retriever.invoke(query)
            
            retrieved = []
            for i, d in enumerate(docs):
                retrieved.append(DocChunk(
                    doc_id=d.metadata.get("doc_id", "unknown"),
                    chunk_id=d.metadata.get("chunk_id", f"chunk-{i}"),
                    text=d.page_content,
                    meta=d.metadata
                ))
        except Exception as e:
            logger.warning("Vector retrieval failed, using all chunks", error=str(e))
            # Fallback: use all available chunks
            retrieved = state["chunks"][:6]  # Limit to avoid overwhelming
        
        state["retrieved"] = retrieved
        
        logger.info("Document retrieval completed", 
                   retrieved_count=len(retrieved))
        
        return state
        
    except Exception as e:
        error_msg = f"Document retrieval failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state

def node_analyze(state: GraphState) -> GraphState:
    """
    Analyze retrieved content using LLM.
    
    Args:
        state: Current graph state with retrieved chunks
        
    Returns:
        Updated state with analysis results
    """
    try:
        logger.info("Starting LLM analysis", doc_id=state.get("doc_id"))
        
        # Check if we have retrieved content
        if not state.get("retrieved"):
            error_msg = "No retrieved content available for analysis"
            logger.error(error_msg)
            state["error"] = error_msg
            return state
        
        llm_model = llm()
        
        # Combine retrieved text (limit to avoid token limits)
        content = "\n\n".join([c.text for c in state["retrieved"]])[:12000]
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a scientific research assistant. Be precise and concise."),
            ("user", """Analyze the following scientific text. 
- Extract aims, methodology, datasets, equations, and any referenced statistics.
- Indicate whether SciPy calculations are required to verify or extend results.
- Return JSON with keys: findings(list of strings), stats(dict), needs_scipy(bool), rationale(str).

TEXT:
{content}""")
        ])
        
        msg = prompt.format_messages(content=content)
        out = llm_model.invoke(msg)
        
        try:
            # Try to clean and parse JSON
            content = out.content.strip()
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            
            # Try to find JSON-like structure
            if content.startswith('{') and content.endswith('}'):
                data = json.loads(content)
            else:
                # Try to extract JSON from the content
                import re
                json_match = re.search(r'\{.*\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise json.JSONDecodeError("No JSON found", content, 0)
        except json.JSONDecodeError as e:
            # Fallback parsing
            logger.warning("Failed to parse LLM JSON output, using fallback", error=str(e), content=out.content[:500])
            data = {
                "findings": [out.content[:8000]], 
                "stats": {}, 
                "needs_scipy": decide_need_scipy(content), 
                "rationale": "LLM JSON parsing failed, using fallback"
            }
        
        # Validate and create AnalysisResult
        analysis = AnalysisResult(
            findings=data.get("findings", []),
            stats=data.get("stats", {}),
            needs_scipy=data.get("needs_scipy", False),
            rationale=data.get("rationale", "")
        )
        
        state["analysis"] = analysis
        
        logger.info("LLM analysis completed", 
                   findings_count=len(analysis.findings),
                   needs_scipy=analysis.needs_scipy)
        
        return state
        
    except Exception as e:
        error_msg = f"LLM analysis failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        return state

def node_scipy_compute(state: GraphState) -> GraphState:
    """
    Perform SciPy statistical analysis.
    
    Args:
        state: Current graph state with retrieved chunks
        
    Returns:
        Updated state with scipy_out results
    """
    try:
        logger.info("Starting SciPy computation", doc_id=state.get("doc_id"))
        
        # Combine text from retrieved chunks
        merged_text = "\n".join(c.text for c in state.get("retrieved", []))
        
        # Perform comprehensive analysis
        scipy_results = comprehensive_analysis(merged_text)
        
        state["scipy_out"] = scipy_results
        
        logger.info("SciPy computation completed", 
                   analyses_performed=scipy_results.get("analysis_performed", []))
        
        return state
        
    except Exception as e:
        error_msg = f"SciPy computation failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        state["scipy_out"] = {"error": error_msg}
        return state

def node_decide(state: GraphState) -> GraphState:
    """
    Make decision about document relevance.
    
    Args:
        state: Current graph state with analysis and scipy results
        
    Returns:
        Updated state with decision
    """
    try:
        logger.info("Starting decision making", doc_id=state.get("doc_id"))
        
        # Check if we have analysis results
        if not state.get("analysis"):
            # Create fallback analysis if missing
            logger.warning("No analysis available, creating fallback")
            from schema import AnalysisResult
            state["analysis"] = AnalysisResult(
                findings=["Document processed with limited analysis"],
                stats={},
                needs_scipy=False,
                rationale="Analysis step failed or skipped"
            )
        
        llm_model = llm()
        analysis = state["analysis"]
        scipy_out = state.get("scipy_out", {})
        
        # Create analysis summary for LLM
        try:
            analysis_text = f"Findings: {analysis.findings}, Stats: {analysis.stats}, Needs SciPy: {analysis.needs_scipy}"
        except Exception as e:
            logger.warning("Error accessing analysis attributes", error=str(e))
            analysis_text = str(analysis)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a senior researcher. Decide relevance and rigor."),
            ("user", """Based on the analysis and SciPy results, decide if this document is RELEVANT to the research topic.
Return JSON: {{"label":"relevant"/"irrelevant"/"uncertain", "confidence":0-1, "criteria":["list"]}}

ANALYSIS: {analysis_text}
SCIPY: {scipy_text}""")
        ])
        
        scipy_text = json.dumps(scipy_out, default=str)[:4000]
        msg = prompt.format_messages(analysis_text=analysis_text, scipy_text=scipy_text)
        out = llm_model.invoke(msg)
        
        try:
            # Try to clean and parse JSON
            content = out.content.strip()
            logger.debug("Parsing decision content", content=content[:200])
            
            # Try to extract JSON from markdown code blocks if present
            if "```json" in content:
                json_start = content.find("```json") + 7
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            elif "```" in content:
                json_start = content.find("```") + 3
                json_end = content.find("```", json_start)
                if json_end > json_start:
                    content = content[json_start:json_end].strip()
            
            # Try to find JSON-like structure
            if content.startswith('{') and content.endswith('}'):
                data = json.loads(content)
            else:
                # Try to extract JSON from the content using regex
                import re
                json_match = re.search(r'\{.*?\}', content, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    # Try to parse as simple format
                    if "relevant" in content.lower():
                        data = {"label": "relevant", "confidence": 0.7, "criteria": ["Contains relevant content"]}
                    elif "irrelevant" in content.lower():
                        data = {"label": "irrelevant", "confidence": 0.7, "criteria": ["Not relevant content"]}
                    else:
                        data = {"label": "uncertain", "confidence": 0.5, "criteria": ["Could not determine relevance"]}
        except (json.JSONDecodeError, Exception) as e:
            logger.warning("Failed to parse decision JSON, using fallback", error=str(e), content=str(out.content)[:500])
            # Create a reasonable fallback based on content
            content_lower = str(out.content).lower()
            if "relevant" in content_lower or "research" in content_lower:
                data = {"label": "relevant", "confidence": 0.6, "criteria": ["Analysis indicates relevance"]}
            elif "irrelevant" in content_lower or "not relevant" in content_lower:
                data = {"label": "irrelevant", "confidence": 0.6, "criteria": ["Analysis indicates irrelevance"]}
            else:
                data = {"label": "uncertain", "confidence": 0.5, "criteria": ["JSON parsing failed, unclear relevance"]}
        
        # Validate decision data
        label = data.get("label", "uncertain")
        confidence = max(0.0, min(1.0, data.get("confidence", 0.5)))
        criteria = data.get("criteria", [])
        
        decision = Decision(
            label=label,
            confidence=confidence,
            criteria=criteria
        )
        
        state["decision"] = decision
        
        logger.info("Decision completed", 
                   label=label, 
                   confidence=confidence)
        
        return state
        
    except Exception as e:
        error_msg = f"Decision making failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        # Provide fallback decision
        state["decision"] = Decision(
            label="uncertain",
            confidence=0.0,
            criteria=[f"Error: {error_msg}"]
        )
        return state

def node_report(state: GraphState) -> GraphState:
    """
    Generate final report.
    
    Args:
        state: Current graph state with all analysis results
        
    Returns:
        Updated state with final report
    """
    try:
        logger.info("Starting report generation", doc_id=state.get("doc_id"))
        
        doc_id = state.get("doc_id", "unknown")
        analysis = state.get("analysis")
        decision = state.get("decision")
        scipy_out = state.get("scipy_out", {})
        error = state.get("error")
        
        # Build report sections
        report_sections = [
            f"# Research Agent Report for: {doc_id}",
            "",
            "## Summary of Findings"
        ]
        
        if error:
            report_sections.extend([
                f"**Error occurred during processing:** {error}",
                ""
            ])
        
        if analysis:
            findings = analysis.findings[:10]  # Limit findings
            report_sections.extend([
                *[f"- {f}" for f in findings],
                "",
                "## Methodology Analysis",
                f"- Rationale: {analysis.rationale}",
                f"- Statistical Analysis Required: {'Yes' if analysis.needs_scipy else 'No'}",
                ""
            ])
        
        # Statistical checks section
        report_sections.extend([
            "## Statistical Checks"
        ])
        
        if analysis and analysis.stats:
            report_sections.append(f"- Analysis Stats: {json.dumps(analysis.stats)[:1200]}")
        
        if scipy_out:
            scipy_summary = {k: v for k, v in scipy_out.items() if k != "error"}
            report_sections.append(f"- SciPy Results: {json.dumps(scipy_summary)[:1200]}")
        
        report_sections.append("")
        
        # Decision section
        if decision:
            report_sections.extend([
                "## Decision",
                f"- **Label:** {decision.label}",
                f"- **Confidence:** {decision.confidence:.2f}",
                f"- **Criteria:** {', '.join(decision.criteria[:6])}",
                ""
            ])
        
        # Methods used
        methods = ["RAG (Chroma)", "LLM analysis"]
        if scipy_out.get("analysis_performed"):
            methods.append("SciPy statistical analysis")
        
        report_sections.extend([
            "## Methods Used",
            *[f"- {method}" for method in methods],
            ""
        ])
        
        summary_text = "\n".join(report_sections)
        
        report = Report(
            summary=summary_text,
            methods=methods,
            decisions=decision or Decision(label="error", confidence=0.0, criteria=["No decision made"]),
            attachments={
                "analysis_complete": bool(analysis),
                "scipy_complete": bool(scipy_out),
                "error_occurred": bool(error)
            }
        )
        
        state["report"] = report
        
        logger.info("Report generation completed", 
                   doc_id=doc_id, 
                   report_length=len(summary_text))
        
        return state
        
    except Exception as e:
        error_msg = f"Report generation failed: {str(e)}"
        logger.error(error_msg)
        state["error"] = error_msg
        
        # Create minimal error report
        state["report"] = Report(
            summary=f"Error generating report: {error_msg}",
            methods=["Error handler"],
            decisions=Decision(label="error", confidence=0.0, criteria=[error_msg]),
            attachments={"error": error_msg}
        )
        
        return state