"""
Session management and auto-save functionality
"""
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import pickle
import structlog

logger = structlog.get_logger()

def create_sessions_directory():
    """Create sessions directory if it doesn't exist."""
    sessions_dir = Path("./sessions")
    sessions_dir.mkdir(exist_ok=True)
    return sessions_dir

def save_session(session_data: Dict[str, Any]) -> str:
    """Save current session to file."""
    try:
        sessions_dir = create_sessions_directory()
        
        # Create session filename with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        doc_id = session_data.get('current_doc', {}).get('doc_id', 'unknown')
        filename = f"session_{doc_id}_{timestamp}.json"
        filepath = sessions_dir / filename
        
        # Prepare session data for saving
        save_data = {
            'timestamp': datetime.now().isoformat(),
            'version': '2.0',
            'current_doc': session_data.get('current_doc'),
            'doc_analysis': session_data.get('doc_analysis'),
            'chat_history': session_data.get('chat_history', []),
            'processed_docs': session_data.get('processed_docs', [])
        }
        
        # Save as JSON
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2, default=str)
        
        logger.info("Session saved", filepath=str(filepath))
        return str(filepath)
        
    except Exception as e:
        logger.error("Failed to save session", error=str(e))
        return f"خطأ في الحفظ: {str(e)}"

def load_session(filepath: str) -> Dict[str, Any]:
    """Load session from file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            session_data = json.load(f)
        
        logger.info("Session loaded", filepath=filepath)
        return session_data
        
    except Exception as e:
        logger.error("Failed to load session", error=str(e))
        return {}

def list_saved_sessions() -> List[Dict[str, Any]]:
    """List all saved sessions."""
    try:
        sessions_dir = create_sessions_directory()
        sessions = []
        
        for session_file in sessions_dir.glob("session_*.json"):
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    session_data = json.load(f)
                
                # Extract session info
                session_info = {
                    'filepath': str(session_file),
                    'filename': session_file.name,
                    'timestamp': session_data.get('timestamp'),
                    'doc_filename': session_data.get('current_doc', {}).get('filename', 'مجهول'),
                    'doc_id': session_data.get('current_doc', {}).get('doc_id', 'مجهول'),
                    'chat_count': len(session_data.get('chat_history', [])),
                    'file_size': session_file.stat().st_size
                }
                sessions.append(session_info)
                
            except Exception as e:
                logger.warning("Failed to read session file", file=str(session_file), error=str(e))
                continue
        
        # Sort by timestamp (newest first)
        sessions.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return sessions
        
    except Exception as e:
        logger.error("Failed to list sessions", error=str(e))
        return []

def auto_save_session(session_data: Dict[str, Any], interval_minutes: int = 5) -> bool:
    """Auto-save session if enough time has passed."""
    try:
        sessions_dir = create_sessions_directory()
        
        # Check if enough time has passed since last save
        doc_id = session_data.get('current_doc', {}).get('doc_id', 'unknown')
        auto_save_file = sessions_dir / f"autosave_{doc_id}.json"
        
        should_save = False
        
        if not auto_save_file.exists():
            should_save = True
        else:
            # Check last modification time
            last_modified = datetime.fromtimestamp(auto_save_file.stat().st_mtime)
            time_diff = datetime.now() - last_modified
            if time_diff.total_seconds() > (interval_minutes * 60):
                should_save = True
        
        if should_save:
            # Prepare auto-save data
            auto_save_data = {
                'timestamp': datetime.now().isoformat(),
                'type': 'auto_save',
                'current_doc': session_data.get('current_doc'),
                'chat_history': session_data.get('chat_history', []),
                'doc_analysis': session_data.get('doc_analysis')
            }
            
            with open(auto_save_file, 'w', encoding='utf-8') as f:
                json.dump(auto_save_data, f, ensure_ascii=False, indent=2, default=str)
            
            logger.debug("Auto-save completed", file=str(auto_save_file))
            return True
        
        return False
        
    except Exception as e:
        logger.error("Auto-save failed", error=str(e))
        return False

def load_auto_save(doc_id: str) -> Dict[str, Any]:
    """Load auto-saved session for specific document."""
    try:
        sessions_dir = create_sessions_directory()
        auto_save_file = sessions_dir / f"autosave_{doc_id}.json"
        
        if auto_save_file.exists():
            with open(auto_save_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        return {}
        
    except Exception as e:
        logger.error("Failed to load auto-save", error=str(e))
        return {}

def cleanup_old_sessions(days_to_keep: int = 30):
    """Clean up old session files."""
    try:
        sessions_dir = create_sessions_directory()
        cutoff_time = datetime.now().timestamp() - (days_to_keep * 24 * 60 * 60)
        
        cleaned_count = 0
        for session_file in sessions_dir.glob("*.json"):
            if session_file.stat().st_mtime < cutoff_time:
                session_file.unlink()
                cleaned_count += 1
        
        logger.info("Session cleanup completed", cleaned_files=cleaned_count)
        return cleaned_count
        
    except Exception as e:
        logger.error("Session cleanup failed", error=str(e))
        return 0

def export_session_summary(session_data: Dict[str, Any]) -> str:
    """Export session summary as text."""
    try:
        current_doc = session_data.get('current_doc', {})
        chat_history = session_data.get('chat_history', [])
        
        summary = f"""
ملخص الجلسة - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'=' * 50}

معلومات المستند:
- اسم الملف: {current_doc.get('filename', 'غير محدد')}
- معرف المستند: {current_doc.get('doc_id', 'غير محدد')}
- حجم النص: {len(current_doc.get('text', '')):,} حرف

إحصائيات الجلسة:
- عدد الرسائل: {len(chat_history)}
- عدد الأسئلة: {len([m for m in chat_history if m['role'] == 'user'])}
- عدد الإجابات: {len([m for m in chat_history if m['role'] == 'assistant'])}

سجل المحادثة:
{'-' * 30}
"""
        
        for i, message in enumerate(chat_history, 1):
            if message['role'] == 'user':
                summary += f"\nسؤال {i}: {message['content']}\n"
            else:
                summary += f"إجابة {i}: {message['content']}\n{'-' * 30}\n"
        
        return summary
        
    except Exception as e:
        return f"خطأ في إنشاء الملخص: {str(e)}"
