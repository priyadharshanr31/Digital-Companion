# services/database_wrapper.py
"""
Synchronous wrapper for PostgreSQL service to maintain compatibility with existing Streamlit code
"""
import asyncio
from typing import List, Optional, Dict, Any
from services.postgresql_service import PostgreSQLService
from models.user import User, UserRole, UserRelationship
from models.activity import StudentActivity, ActivityType, LearningSession, ProgressMetrics

class DatabaseWrapper:
    """Synchronous wrapper around PostgreSQL service for backward compatibility"""
    
    def __init__(self):
        self.pg_service = PostgreSQLService()
        self.loop = None
        self._ensure_loop()
    
    def _ensure_loop(self):
        """Ensure we have an event loop"""
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
    
    def _run_async(self, coro):
        """Run async coroutine in sync context"""
        if not self.loop.is_running():
            return self.loop.run_until_complete(coro)
        else:
            # If loop is already running (in Streamlit), create a new thread
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, coro)
                return future.result()
    
    # User operations (sync interface)
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username (sync)"""
        return self._run_async(self.pg_service.get_user_by_username(username))
    
    def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID (sync)"""
        return self._run_async(self.pg_service.get_user_by_id(user_id))
    
    def get_students_for_parent(self, parent_id: str) -> List[User]:
        """Get students for parent (sync)"""
        return self._run_async(self.pg_service.get_students_for_parent(parent_id))
    
    def get_all_users(self) -> List[User]:
        """Get all users (sync)"""
        return self._run_async(self.pg_service.get_all_users())
    
    def get_system_analytics(self) -> Dict[str, Any]:
        """Get real system analytics data"""
        return self._run_async(self.pg_service.get_system_analytics())
    
    def get_user_stats(self) -> Dict[str, Any]:
        """Get real user statistics"""
        return self._run_async(self.pg_service.get_user_stats())
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get real knowledge base statistics"""
        return self._run_async(self.pg_service.get_knowledge_base_stats())
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return self._run_async(self.pg_service.get_performance_metrics())
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get security metrics"""
        return self._run_async(self.pg_service.get_security_metrics())
    
    # Activity operations (sync interface)
    def log_activity(self, activity: StudentActivity) -> bool:
        """Log activity (sync)"""
        return self._run_async(self.pg_service.log_activity(activity))
    
    def get_student_activities(self, student_id: str, limit: int = 100) -> List[StudentActivity]:
        """Get student activities (sync)"""
        return self._run_async(self.pg_service.get_student_activities(student_id, limit))
    
    # Caching operations (sync interface)
    def get_cached_response(self, query: str) -> Optional[Dict[str, Any]]:
        """Get cached response (sync)"""
        return self._run_async(self.pg_service.get_cached_response(query))
    
    def cache_response(self, query: str, response_data: Dict[str, Any]) -> bool:
        """Cache response (sync)"""
        return self._run_async(self.pg_service.cache_response(query, response_data))
    
    # Document persistence methods
    def save_document_chunks(self, document_id: str, filename: str, file_type: str, 
                           chunks: List[str], metadata: List[Dict], uploaded_by: str) -> bool:
        """Save document chunks to database for persistence"""
        return self._run_async(self.pg_service.save_document_chunks(
            document_id, filename, file_type, chunks, metadata, uploaded_by))
    
    def load_all_document_chunks(self) -> tuple[List[str], List[Dict]]:
        """Load all document chunks from database"""
        return self._run_async(self.pg_service.load_all_document_chunks())
    
    def delete_document(self, filename: str) -> bool:
        """Delete document and all its chunks"""
        return self._run_async(self.pg_service.delete_document(filename))
    
    def delete_user_completely(self, user_id: str) -> bool:
        """Delete user and all associated data completely"""
        return self._run_async(self.pg_service.delete_user_completely(user_id))
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get all document metadata for management interface"""
        return self._run_async(self.pg_service.get_all_documents())
    
    # Compatibility methods for existing code
    def create_user(self, user: User) -> bool:
        """Create user in PostgreSQL"""
        return self._run_async(self.pg_service.create_user(user))
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, 'loop') and self.loop:
            try:
                self._run_async(self.pg_service.close())
            except:
                pass

# Global database service instance
database_service = DatabaseWrapper()