"""
Simplified analysis module for Streamlit interface
"""
from typing import Dict, Any, List
import os
import json
import tempfile
from pathlib import Path

from schema import DocChunk, AnalysisResult, Decision, Report
from chunking import smart_chunk
from embeddings import build_or_load_vectorstore
from nodes import llm
from tools_scipy import comprehensive_analysis, decide_need_scipy
from langchain_core.prompts import ChatPromptTemplate
import structlog

logger = structlog.get_logger()

def analyze_text_directly(text: str, doc_id: str = None) -> Dict[str, Any]:
    """
    Analyze text directly without file operations.
    
    Args:
        text: Text content to analyze
        doc_id: Document identifier
        
    Returns:
        Analysis results dictionary
    """
    try:
        if not doc_id:
            doc_id = f"doc_{hash(text[:100]) % 10000}"
        
        logger.info("Starting direct text analysis", doc_id=doc_id, text_length=len(text))
        
        # Step 1: Chunk the text
        chunks = smart_chunk(text, doc_id=doc_id)
        logger.info("Text chunked", chunk_count=len(chunks))
        
        # Step 2: Create embeddings (optional, handle errors gracefully)
        try:
            persist_dir = os.getenv("CHROMA_DIR", "./storage/vectors")
            vs = build_or_load_vectorstore(chunks, persist_dir=persist_dir)
            logger.info("Vector store created")
        except Exception as e:
            logger.warning("Vector store creation failed, continuing", error=str(e))
        
        # Step 3: Select top chunks for analysis
        selected_chunks = chunks[:6] if len(chunks) > 6 else chunks
        combined_text = "\n\n".join([chunk.text for chunk in selected_chunks])
        
        # Step 4: LLM Analysis
        analysis_result = analyze_with_llm(combined_text)
        
        # Step 5: SciPy analysis if needed
        scipy_result = {}
        if analysis_result.needs_scipy:
            logger.info("Performing SciPy analysis")
            scipy_result = comprehensive_analysis(combined_text)
        
        # Step 6: Make decision
        decision = make_decision(analysis_result, scipy_result, combined_text)
        
        # Step 7: Generate report
        report = generate_report(doc_id, analysis_result, scipy_result, decision)
        
        return {
            'doc_id': doc_id,
            'chunks': chunks,
            'analysis': analysis_result,
            'scipy_out': scipy_result,
            'decision': decision,
            'report': report,
            'success': True
        }
        
    except Exception as e:
        logger.error("Direct text analysis failed", error=str(e))
        return {
            'doc_id': doc_id or 'unknown',
            'error': str(e),
            'success': False
        }

def analyze_with_llm(text: str) -> AnalysisResult:
    """Analyze text with LLM."""
    try:
        llm_model = llm()
        
        # Check if it's a mock LLM
        if hasattr(llm_model, 'responses'):
            # Mock LLM - create structured response
            logger.info("Using mock LLM for analysis")
            return AnalysisResult(
                findings=[
                    "تم تحليل المستند باستخدام النظام التجريبي",
                    "يحتوي على محتوى أكاديمي وبحثي",
                    "يتضمن مراجع ومصادر علمية",
                    "يحتوي على بيانات رقمية وإحصائيات"
                ],
                stats={"word_count": len(text.split()), "char_count": len(text)},
                needs_scipy=True,
                rationale="تم تحديد احتياج التحليل الإحصائي بناءً على وجود أرقام ومصطلحات علمية"
            )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "أنت محلل بحثي خبير. حلل النص العلمي بدقة وأرجع JSON صالح."),
            ("user", """حلل النص التالي واستخرج المعلومات الرئيسية.

أرجع النتيجة بصيغة JSON صحيحة بهذا التنسيق بالضبط:
{{
    "findings": ["النتيجة الأولى", "النتيجة الثانية", "النتيجة الثالثة"],
    "stats": {{"word_count": 0, "numbers_found": 0}},
    "needs_scipy": true,
    "rationale": "تبرير القرار"
}}

النص للتحليل:
{text}""")
        ])
        
        # Limit text length to avoid token limits
        limited_text = text[:5000]
        messages = prompt.format_messages(text=limited_text)
        response = llm_model.invoke(messages)
        
        # Handle response properly
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        # Parse response with better error handling
        try:
            content = content.strip()
            logger.debug("LLM response content", content=content[:200])
            
            # Try to extract JSON
            if content.startswith('{') and content.endswith('}'):
                data = json.loads(content)
            else:
                # Look for JSON in the response
                import re
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON structure found")
            
        except Exception as e:
            logger.warning("JSON parsing failed, using content analysis", error=str(e))
            # Analyze content directly
            words = text.split()
            numbers = re.findall(r'\d+\.?\d*', text)
            
            data = {
                "findings": [
                    f"المستند يحتوي على {len(words)} كلمة",
                    f"تم العثور على {len(numbers)} رقم في النص",
                    "يتضمن محتوى أكاديمي" if any(term in text.lower() for term in ['research', 'study', 'analysis', 'بحث', 'دراسة']) else "محتوى عام",
                    content[:200] + "..." if len(content) > 200 else content
                ],
                "stats": {"word_count": len(words), "numbers_found": len(numbers)},
                "needs_scipy": len(numbers) > 5,
                "rationale": "تم التحليل التلقائي بناءً على محتوى النص"
            }
        
        return AnalysisResult(
            findings=data.get("findings", ["لا توجد نتائج"]),
            stats=data.get("stats", {}),
            needs_scipy=data.get("needs_scipy", False),
            rationale=data.get("rationale", "تحليل أساسي")
        )
        
    except Exception as e:
        logger.error("LLM analysis failed completely", error=str(e))
        # Fallback analysis
        words = text.split()
        return AnalysisResult(
            findings=[
                f"تحليل تلقائي: المستند يحتوي على {len(words)} كلمة",
                f"طول النص: {len(text)} حرف",
                "تم التحليل في وضع الأمان بسبب خطأ تقني"
            ],
            stats={"word_count": len(words), "char_count": len(text)},
            needs_scipy=False,
            rationale=f"تحليل آمن بسبب خطأ: {str(e)}"
        )

def make_decision(analysis: AnalysisResult, scipy_result: Dict, text: str) -> Decision:
    """Make relevance decision."""
    try:
        llm_model = llm()
        
        # Check if it's a mock LLM
        if hasattr(llm_model, 'responses'):
            logger.info("Using mock decision making")
            # Smart decision based on content analysis
            research_keywords = ['research', 'study', 'analysis', 'بحث', 'دراسة', 'تحليل', 'نتائج', 'خلاصة']
            relevance_score = sum(1 for keyword in research_keywords if keyword in text.lower()) / len(research_keywords)
            
            if relevance_score > 0.3:
                return Decision(
                    label="relevant",
                    confidence=0.75,
                    criteria=["يحتوي على مصطلحات بحثية", "تركيب أكاديمي واضح", "محتوى علمي متخصص"]
                )
            else:
                return Decision(
                    label="uncertain", 
                    confidence=0.6,
                    criteria=["محتوى عام", "قليل المصطلحات العلمية"]
                )
        
        # Prepare analysis summary safely
        try:
            analysis_summary = f"النتائج: {len(analysis.findings)} نتيجة, يحتاج SciPy: {analysis.needs_scipy}"
        except:
            analysis_summary = "تحليل أساسي متاح"
            
        scipy_summary = "متوفر" if scipy_result and len(scipy_result) > 0 else "غير متوفر"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "أنت خبير في تقييم البحوث العلمية. أرجع JSON صحيح فقط."),
            ("user", f"""قيم مدى صلة هذا المستند بالبحث العلمي.

أرجع JSON بهذا التنسيق بالضبط:
{{
    "label": "relevant",
    "confidence": 0.8,
    "criteria": ["معيار 1", "معيار 2"]
}}

القيم المسموحة لـ label: "relevant" أو "irrelevant" أو "uncertain"
القيم المسموحة لـ confidence: رقم بين 0.0 و 1.0

التحليل: {analysis_summary}
النتائج الإحصائية: {scipy_summary}""")
        ])
        
        messages = prompt.format_messages()
        response = llm_model.invoke(messages)
        
        # Handle response properly
        if hasattr(response, 'content'):
            content = response.content
        elif isinstance(response, str):
            content = response
        else:
            content = str(response)
        
        # Parse response with better error handling
        try:
            content = content.strip()
            logger.debug("Decision response content", content=content[:200])
            
            if content.startswith('{') and content.endswith('}'):
                data = json.loads(content)
            else:
                import re
                json_match = re.search(r'\{[^{}]*\}', content)
                if json_match:
                    data = json.loads(json_match.group())
                else:
                    raise ValueError("No JSON found")
                    
        except Exception as e:
            logger.warning("Decision JSON parsing failed", error=str(e))
            # Fallback decision based on content
            if any(keyword in text.lower() for keyword in ['research', 'study', 'analysis', 'بحث', 'دراسة', 'تحليل']):
                data = {"label": "relevant", "confidence": 0.7, "criteria": ["يحتوي على محتوى بحثي"]}
            else:
                data = {"label": "uncertain", "confidence": 0.5, "criteria": ["محتوى غير واضح الطبيعة العلمية"]}
        
        # Validate data
        label = data.get("label", "uncertain")
        if label not in ["relevant", "irrelevant", "uncertain"]:
            label = "uncertain"
            
        confidence = data.get("confidence", 0.5)
        if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
            confidence = 0.5
            
        criteria = data.get("criteria", ["تحليل أساسي"])
        if not isinstance(criteria, list):
            criteria = [str(criteria)] if criteria else ["لا توجد معايير محددة"]
        
        return Decision(
            label=label,
            confidence=float(confidence),
            criteria=criteria
        )
        
    except Exception as e:
        logger.error("Decision making failed completely", error=str(e))
        return Decision(
            label="uncertain",
            confidence=0.5,
            criteria=[f"خطأ في اتخاذ القرار: {str(e)[:100]}"]
        )

def generate_report(doc_id: str, analysis: AnalysisResult, scipy_result: Dict, decision: Decision) -> Report:
    """Generate final report."""
    try:
        # Build summary
        summary_parts = [
            f"# تقرير تحليل المستند: {doc_id}",
            "",
            "## النتائج الرئيسية"
        ]
        
        if analysis.findings:
            for i, finding in enumerate(analysis.findings[:5], 1):
                summary_parts.append(f"{i}. {finding}")
        
        summary_parts.extend([
            "",
            "## القرار النهائي",
            f"- **التصنيف:** {decision.label}",
            f"- **مستوى الثقة:** {decision.confidence:.0%}",
            f"- **المعايير:** {', '.join(decision.criteria[:3])}"
        ])
        
        if scipy_result.get('analysis_performed'):
            summary_parts.extend([
                "",
                "## التحليل الإحصائي",
                f"- **التحليلات المنجزة:** {', '.join(scipy_result['analysis_performed'])}"
            ])
        
        summary = "\n".join(summary_parts)
        
        return Report(
            summary=summary,
            methods=["تحليل LLM", "تجزئة ذكية"] + (["تحليل SciPy"] if scipy_result else []),
            decisions=decision,
            attachments={
                "analysis_complete": True,
                "scipy_complete": bool(scipy_result),
                "timestamp": str(Path().resolve())
            }
        )
        
    except Exception as e:
        logger.error("Report generation failed", error=str(e))
        return Report(
            summary=f"خطأ في إنشاء التقرير: {str(e)}",
            methods=["معالج الأخطاء"],
            decisions=decision,
            attachments={"error": str(e)}
        )
