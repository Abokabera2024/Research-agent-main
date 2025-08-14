import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

# Ensure environment variables are loaded early
load_dotenv()
import config  # triggers validate_config logic if needed
try:
    config.validate_config()
except Exception:
    pass

# Import our existing modules
from streamlit_analyzer import analyze_text_directly
from embeddings import search_similar, get_collection_info
from loaders import load_pdf_text, validate_pdf_file, assign_doc_id
from nodes import llm
from export_utils import (
    export_to_docx, export_to_pdf_report, export_to_excel, 
    export_chat_to_text, get_available_export_formats
)
from session_manager import (
    save_session, load_session, list_saved_sessions, 
    auto_save_session, load_auto_save, export_session_summary
)
from analytics_utils import (
    generate_analytics_dashboard, create_conversation_timeline, 
    create_question_types_chart, export_analytics_report
)
from langchain_core.prompts import ChatPromptTemplate
import structlog

# Configure logging
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(30),  # WARNING level for cleaner UI
    logger_factory=structlog.WriteLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Page configuration
st.set_page_config(
    page_title="Research Agent - مساعد الباحث",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for RTL and better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1e3a8a, #3b82f6);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .rtl {
        direction: rtl;
        text-align: right;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables."""
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_doc' not in st.session_state:
        st.session_state.current_doc = None
    if 'doc_analysis' not in st.session_state:
        st.session_state.doc_analysis = None
    if 'processed_docs' not in st.session_state:
        st.session_state.processed_docs = []
    if 'auto_save_enabled' not in st.session_state:
        st.session_state.auto_save_enabled = True
    if 'session_start_time' not in st.session_state:
        st.session_state.session_start_time = datetime.now()
    if 'analytics_data' not in st.session_state:
        st.session_state.analytics_data = {
            'questions_asked': 0,
            'exports_created': 0,
            'documents_processed': 0,
            'session_duration': 0
        }

def load_document(uploaded_file):
    """Load and process uploaded PDF."""
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Validate PDF
        if not validate_pdf_file(tmp_path):
            st.error("❌ الملف ليس PDF صالح")
            return None
        
        # Load text
        doc_id = assign_doc_id(tmp_path)
        text = load_pdf_text(tmp_path)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        return {
            'doc_id': doc_id,
            'filename': uploaded_file.name,
            'text': text,
            'upload_time': datetime.now(),
            'size': len(text)
        }
        
    except Exception as e:
        st.error(f"❌ خطأ في تحميل الملف: {str(e)}")
        return None

def analyze_document(doc_info):
    """Analyze document using our simplified pipeline."""
    try:
        with st.spinner("🔬 جاري تحليل المستند..."):
            # Use direct text analysis
            result = analyze_text_directly(doc_info['text'], doc_info['doc_id'])
            return result
            
    except Exception as e:
        st.error(f"❌ خطأ في التحليل: {str(e)}")
        return None

def chat_with_document(user_query, doc_info):
    """Chat with the document using RAG."""
    try:
        # Get LLM instance
        llm_model = llm()
        
        # Search for relevant chunks
        persist_dir = os.getenv("CHROMA_DIR", "./storage/vectors")
        try:
            docs = search_similar(user_query, persist_dir, k=3)
            context = "\n\n".join([doc.page_content for doc in docs])
        except:
            # Fallback to using full text (truncated)
            context = doc_info['text'][:3000]
        
        # Create prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", """أنت مساعد ذكي للباحثين. تجيب على الأسئلة بناءً على محتوى المستند المرفوع.
- أجب باللغة العربية
- كن دقيقاً ومفيداً
- إذا لم تجد المعلومة في المستند، قل ذلك بوضوح
- اقتبس من النص عند الإمكان"""),
            ("user", f"""السؤال: {user_query}

محتوى المستند:
{context}

الرجاء الإجابة على السؤال بناءً على محتوى المستند المذكور أعلاه.""")
        ])
        
        # Get response
        messages = prompt.format_messages()
        response = llm_model.invoke(messages)
        
        # Update analytics
        if 'analytics_data' in st.session_state:
            st.session_state.analytics_data['questions_asked'] += 1
        
        # Handle different response types
        if hasattr(response, 'content'):
            return response.content
        elif isinstance(response, str):
            return response
        else:
            return str(response)
        
    except Exception as e:
        return f"❌ عذراً، حدث خطأ في معالجة سؤالك: {str(e)}"

def display_document_stats(doc_info, analysis_result=None):
    """Display document statistics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 حجم النص", f"{len(doc_info['text']):,} حرف")
    
    with col2:
        word_count = len(doc_info['text'].split())
        st.metric("📝 عدد الكلمات", f"{word_count:,}")
    
    with col3:
        if analysis_result and analysis_result.get('chunks'):
            chunks_count = len(analysis_result['chunks'])
            st.metric("🧩 عدد الأجزاء", chunks_count)
        else:
            st.metric("🧩 عدد الأجزاء", "غير محسوب")
    
    with col4:
        if analysis_result and analysis_result.get('decision'):
            confidence = analysis_result['decision'].confidence
            st.metric("🎯 مستوى الثقة", f"{confidence:.0%}")
        else:
            st.metric("🎯 مستوى الثقة", "غير محسوب")

def display_analysis_results(analysis_result):
    """Display analysis results visually."""
    if not analysis_result:
        return
    
    st.subheader("📊 نتائج التحليل")
    
    # Analysis findings
    if analysis_result.get('analysis') and analysis_result['analysis'].findings:
        st.subheader("🔍 النتائج الرئيسية")
        for i, finding in enumerate(analysis_result['analysis'].findings, 1):
            st.write(f"**{i}.** {finding}")
    
    # Decision
    if analysis_result.get('decision'):
        decision = analysis_result['decision']
        
        col1, col2 = st.columns(2)
        with col1:
            # Decision gauge
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = decision.confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "مستوى الثقة"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 القرار النهائي")
            
            # Color code based on decision
            if decision.label == "relevant":
                st.success(f"✅ **ذو صلة** (ثقة: {decision.confidence:.0%})")
            elif decision.label == "irrelevant":
                st.error(f"❌ **غير ذو صلة** (ثقة: {decision.confidence:.0%})")
            else:
                st.warning(f"⚠️ **غير محدد** (ثقة: {decision.confidence:.0%})")
            
            if decision.criteria:
                st.write("**المعايير المستخدمة:**")
                for criterion in decision.criteria:
                    st.write(f"• {criterion}")
    
    # SciPy results
    if analysis_result.get('scipy_out') and analysis_result['scipy_out'].get('analysis_performed'):
        st.subheader("📈 التحليل الإحصائي")
        
        scipy_results = analysis_result['scipy_out']
        
        # Create tabs for different analyses
        if scipy_results.get('t_test'):
            with st.expander("📊 اختبار T-Test"):
                t_test = scipy_results['t_test']
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("T-Statistic", f"{t_test.get('t_statistic', 0):.3f}")
                    st.metric("P-Value", f"{t_test.get('p_value', 0):.3f}")
                with col2:
                    significant = t_test.get('significant', False)
                    if significant:
                        st.success("نتيجة ذات دلالة إحصائية")
                    else:
                        st.info("نتيجة غير ذات دلالة إحصائية")
        
        if scipy_results.get('correlation'):
            with st.expander("🔗 تحليل الارتباط"):
                corr = scipy_results['correlation']
                if corr.get('pearson'):
                    pearson = corr['pearson']
                    st.write(f"**معامل بيرسون:** {pearson.get('correlation', 0):.3f}")
                    st.write(f"**P-Value:** {pearson.get('p_value', 0):.3f}")

def main():
    """Main Streamlit application."""
    initialize_session_state()
    
    # Check API configuration at startup
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    
    if (openrouter_key and openrouter_key.startswith("sk-or-v1")) or (openai_key and openai_key.startswith("sk-")):
        st.sidebar.success("✅ OpenRouter/OpenAI API متصل")
    else:
        st.sidebar.warning("⚠️ لا يوجد مفتاح API صالح - يتم استخدام استجابات تجريبية")
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🔬 مساعد الباحث - Research Agent</h1>
        <p>واجهة تفاعلية لتحليل ومناقشة المستندات البحثية</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar for file upload and settings
    with st.sidebar:
        st.header("📁 رفع المستندات")
        
        uploaded_file = st.file_uploader(
            "اختر ملف PDF",
            type=['pdf'],
            help="ارفع ملف PDF للتحليل والمناقشة"
        )
        
        if uploaded_file is not None:
            if st.button("🔄 تحليل المستند", type="primary"):
                doc_info = load_document(uploaded_file)
                if doc_info:
                    st.session_state.current_doc = doc_info
                    # Update analytics
                    if 'analytics_data' in st.session_state:
                        st.session_state.analytics_data['documents_processed'] += 1
                    # Start analysis
                    analysis = analyze_document(doc_info)
                    st.session_state.doc_analysis = analysis
                    st.success("✅ تم تحميل المستند بنجاح!")
                    st.rerun()
        
        # Document history
        if st.session_state.processed_docs:
            st.header("📚 المستندات السابقة")
            for doc in st.session_state.processed_docs[-5:]:  # Show last 5
                if st.button(f"📄 {doc['filename'][:20]}...", key=f"doc_{doc['doc_id']}"):
                    st.session_state.current_doc = doc
                    st.rerun()
        
        # Settings
        st.header("⚙️ الإعدادات")
        
        # Language preference
        language = st.selectbox("اللغة", ["العربية", "English"], index=0)
        
        # Analysis depth
        analysis_depth = st.selectbox(
            "عمق التحليل",
            ["سريع", "متوسط", "شامل"],
            index=1
        )
    
    # Main content area
    if st.session_state.current_doc is None:
        # Welcome screen
        st.markdown("### 👋 مرحباً بك في مساعد الباحث!")
        st.markdown("أداة ذكية لتحليل ومناقشة المستندات البحثية بطريقة تفاعلية")
        
        st.markdown("#### 🎯 كيف تستخدم المساعد؟")
        st.markdown("""
        1. 📁 ارفع ملف PDF من الشريط الجانبي
        2. 🔍 انتظر حتى يكتمل التحليل  
        3. 💬 ابدأ في طرح أسئلتك حول المحتوى
        4. 📊 استكشف النتائج والتحليلات
        5. 📄 صدّر النتائج بالصيغة المطلوبة
        """)
        
        st.info("✨ **المميزات المتاحة:** مناقشة تفاعلية، تحليل إحصائي، بحث ذكي، رسوم بيانية، تقارير شاملة")
        
    else:
        # Document is loaded
        doc = st.session_state.current_doc
        analysis = st.session_state.doc_analysis
        
        # Document info header
        st.success(f"📄 **المستند المحمل:** {doc['filename']}")
        
        # Display stats
        display_document_stats(doc, analysis)
        
        # Create tabs for different views
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 المناقشة", "📊 التحليل", "🔍 الاستكشاف", "📄 التقارير", "💾 الجلسات", "📈 التحليلات المتقدمة"])
        
        with tab1:
            st.header("💬 ناقش المستند")
            
            # Add quick question suggestions
            st.subheader("💡 أسئلة مقترحة")
            suggested_questions = [
                "ما هو موضوع هذا البحث؟",
                "ما هي النتائج الرئيسية؟", 
                "ما هي المنهجية المستخدمة؟",
                "هل هناك إحصائيات مهمة؟",
                "ما هي التوصيات؟",
                "من هم المؤلفون؟"
            ]
            
            cols = st.columns(3)
            for i, question in enumerate(suggested_questions):
                with cols[i % 3]:
                    if st.button(question, key=f"suggested_{i}", help="اضغط لطرح هذا السؤال"):
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'role': 'user',
                            'content': question
                        })
                        
                        # Get AI response
                        with st.spinner("🤔 المساعد يفكر..."):
                            response = chat_with_document(question, doc)
                        
                        # Add assistant response
                        st.session_state.chat_history.append({
                            'role': 'assistant',
                            'content': response
                        })
                        
                        st.rerun()
            
            st.divider()
            
            # Chat interface with better styling
            st.subheader("💬 محادثة تفاعلية")
            
            # Display chat history in a container
            chat_container = st.container()
            
            with chat_container:
                if not st.session_state.chat_history:
                    st.info("👋 مرحباً! يمكنك طرح أي سؤال حول المستند أو استخدام الأسئلة المقترحة أعلاه.")
                
                # Display chat messages
                for i, message in enumerate(st.session_state.chat_history):
                    if message['role'] == 'user':
                        with st.chat_message("user", avatar="👤"):
                            st.write(message['content'])
                    else:
                        with st.chat_message("assistant", avatar="🤖"):
                            st.write(message['content'])
            
            # Chat input at the bottom
            user_input = st.chat_input("اكتب سؤالك هنا... (اضغط Enter للإرسال)")
            
            if user_input:
                # Add user message
                st.session_state.chat_history.append({
                    'role': 'user',
                    'content': user_input
                })
                
                # Get AI response
                with st.spinner("🤔 المساعد يفكر..."):
                    response = chat_with_document(user_input, doc)
                
                # Add assistant response
                st.session_state.chat_history.append({
                    'role': 'assistant',
                    'content': response
                })
                
                st.rerun()
            
            # Chat controls
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🗑️ مسح المحادثة", help="مسح جميع الرسائل"):
                    st.session_state.chat_history = []
                    st.rerun()
            
            with col2:
                if st.button("📋 نسخ آخر إجابة", help="نسخ آخر إجابة من المساعد"):
                    if st.session_state.chat_history:
                        last_assistant = None
                        for msg in reversed(st.session_state.chat_history):
                            if msg['role'] == 'assistant':
                                last_assistant = msg['content']
                                break
                        if last_assistant:
                            st.code(last_assistant, language=None)
                        else:
                            st.warning("لا توجد إجابات للنسخ")
                    else:
                        st.warning("لا توجد محادثة")
            
            with col3:
                chat_count = len(st.session_state.chat_history)
                st.metric("عدد الرسائل", chat_count)
        
        with tab2:
            st.header("📊 تحليل المستند")
            display_analysis_results(analysis)
        
        with tab3:
            st.header("🔍 استكشاف المحتوى")
            
            # Search functionality
            search_query = st.text_input("🔍 ابحث في المستند:")
            
            if search_query:
                with st.spinner("🔍 جاري البحث..."):
                    try:
                        persist_dir = os.getenv("CHROMA_DIR", "./storage/vectors")
                        docs = search_similar(search_query, persist_dir, k=5)
                        
                        st.subheader(f"📋 نتائج البحث عن: '{search_query}'")
                        
                        for i, doc_result in enumerate(docs, 1):
                            with st.expander(f"نتيجة {i}"):
                                st.write(doc_result.page_content)
                                if hasattr(doc_result, 'metadata'):
                                    st.caption(f"المصدر: {doc_result.metadata}")
                    except Exception as e:
                        st.warning("⚠️ البحث غير متاح حالياً، يرجى المحاولة لاحقاً")
            
            # Word cloud or frequency analysis could go here
            st.subheader("📈 إحصائيات النص")
            
            if doc['text']:
                words = doc['text'].split()
                word_count = len(words)
                unique_words = len(set(words))
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("إجمالي الكلمات", f"{word_count:,}")
                with col2:
                    st.metric("الكلمات الفريدة", f"{unique_words:,}")
        
        with tab4:
            st.header("📄 التقارير والتصدير المتقدم")
            
            st.subheader("📊 تصدير التقارير")
            
            # Show available formats
            formats = get_available_export_formats()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("### 📄 تقارير شاملة")
                
                # DOCX Export
                if st.button("� تصدير إلى Word (DOCX)", type="primary", key="docx_export"):
                    with st.spinner("📝 جاري إنشاء ملف Word..."):
                        filepath = export_to_docx(doc, analysis, st.session_state.chat_history)
                        if filepath.startswith("خطأ"):
                            st.error(filepath)
                        else:
                            with open(filepath, "rb") as file:
                                st.download_button(
                                    label="📥 تحميل ملف Word",
                                    data=file.read(),
                                    file_name=os.path.basename(filepath),
                                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                                )
                            st.success(f"✅ تم إنشاء ملف Word: {os.path.basename(filepath)}")
                
                # PDF Export
                if st.button("📄 تصدير إلى PDF", type="secondary", key="pdf_export"):
                    with st.spinner("📄 جاري إنشاء ملف PDF..."):
                        filepath = export_to_pdf_report(doc, analysis, st.session_state.chat_history)
                        if filepath.startswith("خطأ"):
                            st.error(filepath)
                        else:
                            with open(filepath, "rb") as file:
                                st.download_button(
                                    label="📥 تحميل ملف PDF",
                                    data=file.read(),
                                    file_name=os.path.basename(filepath),
                                    mime="application/pdf"
                                )
                            st.success(f"✅ تم إنشاء ملف PDF: {os.path.basename(filepath)}")
                
                # Excel Export
                if st.button("📊 تصدير إلى Excel", key="excel_export"):
                    with st.spinner("📊 جاري إنشاء ملف Excel..."):
                        filepath = export_to_excel(doc, analysis, st.session_state.chat_history)
                        if filepath.startswith("خطأ"):
                            st.error(filepath)
                        else:
                            with open(filepath, "rb") as file:
                                st.download_button(
                                    label="📥 تحميل ملف Excel",
                                    data=file.read(),
                                    file_name=os.path.basename(filepath),
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                                )
                            st.success(f"✅ تم إنشاء ملف Excel: {os.path.basename(filepath)}")
            
            with col2:
                st.markdown("### 💬 تصدير المحادثات")
                
                # Chat history export
                if st.button("💬 تصدير سجل المحادثة", key="chat_export"):
                    if st.session_state.chat_history:
                        with st.spinner("💬 جاري تصدير المحادثة..."):
                            filepath = export_chat_to_text(st.session_state.chat_history, doc)
                            if filepath.startswith("خطأ"):
                                st.error(filepath)
                            else:
                                with open(filepath, "r", encoding="utf-8") as file:
                                    st.download_button(
                                        label="📥 تحميل سجل المحادثة",
                                        data=file.read(),
                                        file_name=os.path.basename(filepath),
                                        mime="text/plain"
                                    )
                                st.success(f"✅ تم تصدير المحادثة: {os.path.basename(filepath)}")
                    else:
                        st.warning("⚠️ لا يوجد محادثة للتصدير")
                
                # JSON Export (Enhanced)
                if st.button("� تصدير البيانات الخام (JSON)", key="json_export"):
                    export_data = {
                        'metadata': {
                            'export_time': datetime.now().isoformat(),
                            'version': '2.0',
                            'type': 'research_agent_export'
                        },
                        'document': {
                            'filename': doc['filename'],
                            'doc_id': doc['doc_id'],
                            'text_length': len(doc['text']),
                            'word_count': len(doc['text'].split()),
                            'upload_time': doc['upload_time'].isoformat() if hasattr(doc['upload_time'], 'isoformat') else str(doc['upload_time'])
                        },
                        'analysis': {
                            'success': analysis.get('success', False) if analysis else False,
                            'findings': analysis.get('analysis', {}).findings if analysis and analysis.get('analysis') else [],
                            'decision': {
                                'label': analysis.get('decision', {}).label if analysis and analysis.get('decision') else 'unknown',
                                'confidence': analysis.get('decision', {}).confidence if analysis and analysis.get('decision') else 0,
                                'criteria': analysis.get('decision', {}).criteria if analysis and analysis.get('decision') else []
                            } if analysis and analysis.get('decision') else None,
                            'scipy_results': analysis.get('scipy_out', {}) if analysis else {}
                        },
                        'chat_history': st.session_state.chat_history,
                        'statistics': {
                            'total_messages': len(st.session_state.chat_history),
                            'user_messages': len([m for m in st.session_state.chat_history if m['role'] == 'user']),
                            'assistant_messages': len([m for m in st.session_state.chat_history if m['role'] == 'assistant'])
                        }
                    }
                    
                    st.download_button(
                        label="📥 تحميل البيانات (JSON)",
                        data=json.dumps(export_data, ensure_ascii=False, indent=2, default=str),
                        file_name=f"data_{doc['doc_id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            # Export status and info
            st.subheader("📋 معلومات التصدير")
            
            with st.expander("ℹ️ معلومات حول صيغ التصدير"):
                for format_name, extension, status in formats:
                    if "متاح" in status and "غير" not in status:
                        st.success(f"✅ **{format_name}** (.{extension}) - {status}")
                    else:
                        st.warning(f"⚠️ **{format_name}** (.{extension}) - {status}")
                
                st.info("""
                **ملاحظات هامة:**
                - جميع الملفات المصدرة تُحفظ مع الطابع الزمني
                - يمكن تثبيت المكتبات المفقودة باستخدام pip
                - الملفات المصدرة تحتوي على جميع البيانات والتحليلات
                """)
            
            # Quick stats
            if analysis:
                st.subheader("📈 إحصائيات سريعة")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    findings_count = len(analysis.get('analysis', {}).findings) if analysis.get('analysis') else 0
                    st.metric("عدد النتائج", findings_count)
                
                with col2:
                    chat_count = len(st.session_state.chat_history)
                    st.metric("عدد الرسائل", chat_count)
                
                with col3:
                    confidence = analysis.get('decision', {}).confidence if analysis.get('decision') else 0
                    st.metric("مستوى الثقة", f"{confidence:.0%}")
                
                with col4:
                    scipy_analyses = len(analysis.get('scipy_out', {}).get('analysis_performed', [])) if analysis.get('scipy_out') else 0
                    st.metric("التحليلات الإحصائية", scipy_analyses)

        # Sessions Management Tab
        with tab5:
            st.header("💾 إدارة الجلسات")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🔄 الحفظ والاستعادة")
                
                # Auto-save settings
                auto_save_enabled = st.checkbox(
                    "تمكين الحفظ التلقائي", 
                    value=st.session_state.get('auto_save_enabled', True),
                    help="سيتم حفظ الجلسة تلقائياً كل 5 دقائق"
                )
                st.session_state.auto_save_enabled = auto_save_enabled
                
                # Manual save
                if st.button("💾 حفظ الجلسة الحالية", type="primary"):
                    session_data = {
                        'current_doc': st.session_state.current_doc,
                        'doc_analysis': st.session_state.doc_analysis,
                        'chat_history': st.session_state.chat_history,
                        'processed_docs': st.session_state.processed_docs,
                        'analytics_data': st.session_state.analytics_data
                    }
                    
                    saved_path = save_session(session_data)
                    if saved_path and not saved_path.startswith("خطأ"):
                        st.success(f"✅ تم حفظ الجلسة بنجاح: {Path(saved_path).name}")
                        st.session_state.analytics_data['exports_created'] += 1
                    else:
                        st.error(saved_path)
                
                # Session export
                if st.button("📋 تصدير ملخص الجلسة"):
                    session_data = {
                        'current_doc': st.session_state.current_doc,
                        'chat_history': st.session_state.chat_history,
                        'analytics_data': st.session_state.analytics_data
                    }
                    
                    summary = export_session_summary(session_data)
                    
                    st.download_button(
                        label="📥 تحميل ملخص الجلسة",
                        data=summary,
                        file_name=f"session_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            
            with col2:
                st.subheader("📊 إحصائيات الجلسة")
                
                # Calculate session duration
                session_duration = datetime.now() - st.session_state.session_start_time
                duration_minutes = int(session_duration.total_seconds() / 60)
                
                # Update analytics
                st.session_state.analytics_data['session_duration'] = duration_minutes
                
                # Display metrics
                st.metric("مدة الجلسة", f"{duration_minutes} دقيقة")
                st.metric("الأسئلة المطروحة", st.session_state.analytics_data['questions_asked'])
                st.metric("التصديرات المنشأة", st.session_state.analytics_data['exports_created'])
                st.metric("المستندات المعالجة", st.session_state.analytics_data['documents_processed'])
                
                # Session health
                if duration_minutes > 30:
                    st.info("💡 جلسة طويلة - فكر في حفظ التقدم")
                
                if len(st.session_state.chat_history) > 20:
                    st.info("💬 محادثة مكثفة - قد تحتاج لتصدير النتائج")
            
            st.markdown("---")
            
            # Saved sessions list
            st.subheader("📂 الجلسات المحفوظة")
            
            saved_sessions = list_saved_sessions()
            
            if saved_sessions:
                for session in saved_sessions[:10]:  # Show last 10 sessions
                    with st.expander(f"📄 {session['doc_filename']} - {session['timestamp'][:16]}"):
                        col1, col2, col3 = st.columns([2, 1, 1])
                        
                        with col1:
                            st.write(f"**المستند:** {session['doc_filename']}")
                            st.write(f"**المعرف:** {session['doc_id']}")
                            st.write(f"**عدد الرسائل:** {session['chat_count']}")
                            st.write(f"**حجم الملف:** {session['file_size']:,} بايت")
                        
                        with col2:
                            if st.button(f"📂 استعادة", key=f"load_{session['filename']}"):
                                loaded_session = load_session(session['filepath'])
                                if loaded_session:
                                    st.session_state.current_doc = loaded_session.get('current_doc')
                                    st.session_state.doc_analysis = loaded_session.get('doc_analysis')
                                    st.session_state.chat_history = loaded_session.get('chat_history', [])
                                    st.session_state.processed_docs = loaded_session.get('processed_docs', [])
                                    st.success("✅ تم استعادة الجلسة بنجاح!")
                                    st.rerun()
                                else:
                                    st.error("❌ فشل في استعادة الجلسة")
                        
                        with col3:
                            st.download_button(
                                label="⬇️ تحميل",
                                data=open(session['filepath'], 'r', encoding='utf-8').read(),
                                file_name=session['filename'],
                                mime="application/json",
                                key=f"download_{session['filename']}"
                            )
            else:
                st.info("لا توجد جلسات محفوظة حتى الآن")
            
            # Auto-save status
            if auto_save_enabled and st.session_state.current_doc:
                # Try auto-save
                session_data = {
                    'current_doc': st.session_state.current_doc,
                    'chat_history': st.session_state.chat_history,
                    'doc_analysis': st.session_state.doc_analysis
                }
                
                if auto_save_session(session_data):
                    st.success("💾 تم الحفظ التلقائي", icon="✅")
        
        # Advanced Analytics Tab
        with tab6:
            st.header("📈 التحليلات المتقدمة")
            
            if not st.session_state.current_doc:
                st.warning("⚠️ يرجى تحميل مستند أولاً لعرض التحليلات المتقدمة")
            else:
                # Generate analytics data
                session_data = {
                    'chat_history': st.session_state.chat_history,
                    'current_doc': st.session_state.current_doc,
                    'analytics_data': st.session_state.analytics_data
                }
                
                analytics = generate_analytics_dashboard(session_data, st.session_state.doc_analysis)
                
                # Main analytics overview
                st.subheader("📊 نظرة عامة على التحليلات")
                
                if analytics.get('session_overview'):
                    overview = analytics['session_overview']
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("📝 إجمالي الرسائل", overview.get('total_messages', 0))
                    with col2:
                        st.metric("❓ أسئلة المستخدم", overview.get('user_questions', 0))
                    with col3:
                        st.metric("💭 إجابات النظام", overview.get('assistant_responses', 0))
                    with col4:
                        st.metric("📖 عدد الكلمات", f"{overview.get('word_count', 0):,}")
                
                # Conversation analysis
                if analytics.get('chat_analysis') and st.session_state.chat_history:
                    st.subheader("💬 تحليل المحادثة")
                    
                    chat_analysis = analytics['chat_analysis']
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("متوسط طول الأسئلة", f"{chat_analysis.get('avg_user_message_length', 0):.0f} حرف")
                        st.metric("أطول سؤال", f"{chat_analysis.get('longest_user_message', 0)} حرف")
                    
                    with col2:
                        st.metric("متوسط طول الإجابات", f"{chat_analysis.get('avg_assistant_message_length', 0):.0f} حرف")
                        st.metric("أطول إجابة", f"{chat_analysis.get('longest_assistant_message', 0)} حرف")
                    
                    # Conversation timeline
                    timeline_fig = create_conversation_timeline(st.session_state.chat_history)
                    if timeline_fig:
                        st.plotly_chart(timeline_fig, use_container_width=True)
                
                # Question types analysis
                if analytics.get('question_analysis'):
                    st.subheader("❓ تحليل أنواع الأسئلة")
                    
                    question_chart = create_question_types_chart(analytics['question_analysis'])
                    if question_chart:
                        st.plotly_chart(question_chart, use_container_width=True)
                    else:
                        st.info("لا توجد أسئلة كافية لعرض التحليل")
                
                # Document analysis insights
                if analytics.get('document_analysis'):
                    st.subheader("📄 رؤى تحليل المستند")
                    
                    doc_analysis = analytics['document_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        success = doc_analysis.get('analysis_success', False)
                        st.metric("حالة التحليل", "✅ نجح" if success else "❌ فشل")
                    
                    with col2:
                        findings = doc_analysis.get('findings_count', 0)
                        st.metric("عدد النتائج", findings)
                    
                    with col3:
                        confidence = doc_analysis.get('decision_confidence', 0)
                        st.metric("مستوى الثقة", f"{confidence:.1%}")
                
                # Export analytics
                st.subheader("📤 تصدير التحليلات")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("📊 تصدير JSON", type="secondary"):
                        export_data = export_analytics_report(analytics, 'json')
                        if export_data:
                            st.download_button(
                                label="⬇️ تحميل JSON",
                                data=export_data['content'],
                                file_name=export_data['filename'],
                                mime=export_data['mime_type']
                            )
                
                with col2:
                    if st.button("📝 تصدير نصي", type="secondary"):
                        export_data = export_analytics_report(analytics, 'text')
                        if export_data:
                            st.download_button(
                                label="⬇️ تحميل نص",
                                data=export_data['content'],
                                file_name=export_data['filename'],
                                mime=export_data['mime_type']
                            )
                
                with col3:
                    if st.button("📋 تصدير CSV", type="secondary"):
                        export_data = export_analytics_report(analytics, 'csv')
                        if export_data:
                            st.download_button(
                                label="⬇️ تحميل CSV",
                                data=export_data['content'],
                                file_name=export_data['filename'],
                                mime=export_data['mime_type']
                            )
                
                # Session insights
                if st.session_state.analytics_data:
                    st.subheader("🔍 رؤى الجلسة")
                    
                    session_duration = (datetime.now() - st.session_state.session_start_time).total_seconds() / 60
                    
                    insights = []
                    
                    if session_duration > 30:
                        insights.append("⏰ جلسة طويلة - قد تحتاج لأخذ استراحة")
                    
                    if len(st.session_state.chat_history) > 15:
                        insights.append("💬 محادثة مكثفة - يُنصح بحفظ النتائج")
                    
                    if st.session_state.analytics_data['questions_asked'] > 20:
                        insights.append("❓ عدد كبير من الأسئلة - تفاعل ممتاز مع المحتوى")
                    
                    if analytics.get('chat_analysis', {}).get('avg_user_message_length', 0) > 200:
                        insights.append("📝 أسئلة مفصلة - تظهر فهماً عميقاً للموضوع")
                    
                    if insights:
                        for insight in insights:
                            st.info(insight)
                    else:
                        st.success("✅ جلسة متوازنة وفعالة")

if __name__ == "__main__":
    main()
