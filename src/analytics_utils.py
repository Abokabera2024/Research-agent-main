"""
Advanced analytics and reporting utilities for the Research Agent
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from pathlib import Path
import structlog

logger = structlog.get_logger()

def generate_analytics_dashboard(session_data, analysis_data=None):
    """Generate comprehensive analytics dashboard data."""
    try:
        analytics = {}
        
        # Basic session metrics
        chat_history = session_data.get('chat_history', [])
        current_doc = session_data.get('current_doc', {})
        
        analytics['session_overview'] = {
            'total_messages': len(chat_history),
            'user_questions': len([m for m in chat_history if m['role'] == 'user']),
            'assistant_responses': len([m for m in chat_history if m['role'] == 'assistant']),
            'document_name': current_doc.get('filename', 'Unknown'),
            'document_size': len(current_doc.get('text', '')),
            'word_count': len(current_doc.get('text', '').split()) if current_doc.get('text') else 0
        }
        
        # Chat analysis
        if chat_history:
            # Message lengths
            user_msg_lengths = [len(m['content']) for m in chat_history if m['role'] == 'user']
            assistant_msg_lengths = [len(m['content']) for m in chat_history if m['role'] == 'assistant']
            
            analytics['chat_analysis'] = {
                'avg_user_message_length': sum(user_msg_lengths) / len(user_msg_lengths) if user_msg_lengths else 0,
                'avg_assistant_message_length': sum(assistant_msg_lengths) / len(assistant_msg_lengths) if assistant_msg_lengths else 0,
                'longest_user_message': max(user_msg_lengths) if user_msg_lengths else 0,
                'longest_assistant_message': max(assistant_msg_lengths) if assistant_msg_lengths else 0,
                'total_conversation_length': sum(user_msg_lengths + assistant_msg_lengths)
            }
            
            # Question categories (simple keyword analysis)
            question_keywords = {
                'what': ['ما', 'ماذا', 'what'],
                'how': ['كيف', 'how'],
                'why': ['لماذا', 'why', 'لما'],
                'when': ['متى', 'when'],
                'where': ['أين', 'where'],
                'who': ['من', 'who']
            }
            
            question_types = {k: 0 for k in question_keywords.keys()}
            
            for msg in chat_history:
                if msg['role'] == 'user':
                    content_lower = msg['content'].lower()
                    for q_type, keywords in question_keywords.items():
                        if any(keyword in content_lower for keyword in keywords):
                            question_types[q_type] += 1
                            break
            
            analytics['question_analysis'] = question_types
        
        # Document analysis integration
        if analysis_data:
            analytics['document_analysis'] = {
                'analysis_success': analysis_data.get('success', False),
                'findings_count': len(analysis_data.get('analysis', {}).findings) if analysis_data.get('analysis') else 0,
                'decision_confidence': analysis_data.get('decision', {}).confidence if analysis_data.get('decision') else 0,
                'scipy_analyses': len(analysis_data.get('scipy_out', {}).get('analysis_performed', [])) if analysis_data.get('scipy_out') else 0
            }
        
        return analytics
        
    except Exception as e:
        logger.error("Failed to generate analytics", error=str(e))
        return {}

def create_conversation_timeline(chat_history):
    """Create a timeline visualization of the conversation."""
    try:
        if not chat_history:
            return None
        
        # Prepare data for plotting
        timeline_data = []
        for i, msg in enumerate(chat_history):
            timeline_data.append({
                'message_number': i + 1,
                'role': msg['role'],
                'message_length': len(msg['content']),
                'cumulative_length': sum(len(m['content']) for m in chat_history[:i+1])
            })
        
        df = pd.DataFrame(timeline_data)
        
        # Create timeline plot
        fig = px.line(df, x='message_number', y='cumulative_length', 
                     color='role', title='مسار المحادثة - الطول التراكمي للرسائل')
        
        fig.update_layout(
            xaxis_title='رقم الرسالة',
            yaxis_title='العدد التراكمي للأحرف',
            showlegend=True
        )
        
        return fig
        
    except Exception as e:
        logger.error("Failed to create conversation timeline", error=str(e))
        return None

def create_question_types_chart(question_analysis):
    """Create a chart showing distribution of question types."""
    try:
        if not question_analysis or not any(question_analysis.values()):
            return None
        
        # Filter out zero values
        filtered_data = {k: v for k, v in question_analysis.items() if v > 0}
        
        if not filtered_data:
            return None
        
        # Create pie chart
        fig = px.pie(
            values=list(filtered_data.values()),
            names=list(filtered_data.keys()),
            title='توزيع أنواع الأسئلة'
        )
        
        fig.update_traces(textposition='inside', textinfo='percent+label')
        
        return fig
        
    except Exception as e:
        logger.error("Failed to create question types chart", error=str(e))
        return None

def export_analytics_report(analytics_data, export_format='json'):
    """Export comprehensive analytics report."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if export_format == 'json':
            filename = f"analytics_report_{timestamp}.json"
            content = json.dumps(analytics_data, ensure_ascii=False, indent=2, default=str)
            mime_type = "application/json"
            
        elif export_format == 'text':
            filename = f"analytics_report_{timestamp}.txt"
            content = format_analytics_as_text(analytics_data)
            mime_type = "text/plain"
            
        elif export_format == 'csv':
            filename = f"analytics_report_{timestamp}.csv"
            content = format_analytics_as_csv(analytics_data)
            mime_type = "text/csv"
            
        else:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        return {
            'filename': filename,
            'content': content,
            'mime_type': mime_type
        }
        
    except Exception as e:
        logger.error("Failed to export analytics report", error=str(e))
        return None

def format_analytics_as_text(analytics_data):
    """Format analytics data as readable text."""
    try:
        report = f"""
تقرير التحليلات المتقدم
{'=' * 40}
تاريخ التقرير: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

نظرة عامة على الجلسة:
{'-' * 25}
"""
        
        if 'session_overview' in analytics_data:
            overview = analytics_data['session_overview']
            report += f"""
- إجمالي الرسائل: {overview.get('total_messages', 0)}
- أسئلة المستخدم: {overview.get('user_questions', 0)}
- إجابات النظام: {overview.get('assistant_responses', 0)}
- اسم المستند: {overview.get('document_name', 'غير محدد')}
- حجم المستند: {overview.get('document_size', 0):,} حرف
- عدد الكلمات: {overview.get('word_count', 0):,} كلمة
"""
        
        if 'chat_analysis' in analytics_data:
            chat = analytics_data['chat_analysis']
            report += f"""

تحليل المحادثة:
{'-' * 15}
- متوسط طول رسائل المستخدم: {chat.get('avg_user_message_length', 0):.1f} حرف
- متوسط طول إجابات النظام: {chat.get('avg_assistant_message_length', 0):.1f} حرف
- أطول رسالة مستخدم: {chat.get('longest_user_message', 0)} حرف
- أطول إجابة نظام: {chat.get('longest_assistant_message', 0)} حرف
- إجمالي طول المحادثة: {chat.get('total_conversation_length', 0):,} حرف
"""
        
        if 'question_analysis' in analytics_data:
            questions = analytics_data['question_analysis']
            report += f"""

تحليل أنواع الأسئلة:
{'-' * 20}
"""
            for q_type, count in questions.items():
                if count > 0:
                    report += f"- {q_type}: {count} سؤال\n"
        
        if 'document_analysis' in analytics_data:
            doc_analysis = analytics_data['document_analysis']
            report += f"""

تحليل المستند:
{'-' * 14}
- نجح التحليل: {'نعم' if doc_analysis.get('analysis_success') else 'لا'}
- عدد النتائج: {doc_analysis.get('findings_count', 0)}
- مستوى الثقة: {doc_analysis.get('decision_confidence', 0):.2%}
- التحليلات الإحصائية: {doc_analysis.get('scipy_analyses', 0)}
"""
        
        return report
        
    except Exception as e:
        return f"خطأ في تنسيق التقرير: {str(e)}"

def format_analytics_as_csv(analytics_data):
    """Format analytics data as CSV."""
    try:
        import io
        
        output = io.StringIO()
        
        # Write headers
        output.write("Metric,Value,Category\n")
        
        # Session overview
        if 'session_overview' in analytics_data:
            overview = analytics_data['session_overview']
            for key, value in overview.items():
                output.write(f"{key},{value},session_overview\n")
        
        # Chat analysis
        if 'chat_analysis' in analytics_data:
            chat = analytics_data['chat_analysis']
            for key, value in chat.items():
                output.write(f"{key},{value},chat_analysis\n")
        
        # Question analysis
        if 'question_analysis' in analytics_data:
            questions = analytics_data['question_analysis']
            for key, value in questions.items():
                output.write(f"question_type_{key},{value},question_analysis\n")
        
        # Document analysis
        if 'document_analysis' in analytics_data:
            doc_analysis = analytics_data['document_analysis']
            for key, value in doc_analysis.items():
                output.write(f"{key},{value},document_analysis\n")
        
        return output.getvalue()
        
    except Exception as e:
        return f"Error formatting CSV: {str(e)}"
