# services/session_service.py
import os
import json
import hashlib
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from pathlib import Path

class SessionService:
    """Handle persistent user sessions with Redis or file storage"""
    
    def __init__(self, session_dir: str = "sessions"):
        """Initialize session service with Redis or file storage"""
        self.session_timeout = 24 * 60 * 60  # 24 hours in seconds
        self.storage_type = os.getenv('SESSION_STORAGE', 'file')
        
        if self.storage_type == 'redis':
            try:
                import redis
                redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
                self.redis_client = redis.from_url(redis_url)
                # Test connection
                self.redis_client.ping()
                print("✅ Using Redis for session storage")
            except Exception as e:
                print(f"❌ Redis connection failed, falling back to file storage: {e}")
                self.storage_type = 'file'
        
        if self.storage_type == 'file':
            self.session_dir = Path(session_dir)
            self.session_dir.mkdir(exist_ok=True)
            print("✅ Using file storage for sessions")
    
    def _get_session_file(self, session_id: str) -> Path:
        """Get session file path for a session ID"""
        # Hash session ID for security
        hashed_id = hashlib.md5(session_id.encode()).hexdigest()
        return self.session_dir / f"session_{hashed_id}.json"
    
    def create_session(self, user_data: Dict[str, Any]) -> str:
        """Create a new session and return session ID"""
        # Generate unique session ID
        timestamp = str(int(time.time()))
        user_id = user_data.get('id', 'anonymous')
        session_id = hashlib.sha256(f"{user_id}_{timestamp}".encode()).hexdigest()
        
        # Create session data
        session_data = {
            'session_id': session_id,
            'user_data': user_data,
            'created_at': datetime.now().isoformat(),
            'last_accessed': datetime.now().isoformat(),
            'expires_at': (datetime.now() + timedelta(seconds=self.session_timeout)).isoformat()
        }
        
        if self.storage_type == 'redis':
            try:
                self.redis_client.setex(
                    f"session:{session_id}",
                    self.session_timeout,
                    json.dumps(session_data)
                )
                return session_id
            except Exception as e:
                print(f"Error creating Redis session: {e}")
                return None
        else:
            # File storage fallback
            session_file = self._get_session_file(session_id)
            try:
                with open(session_file, 'w') as f:
                    json.dump(session_data, f, indent=2)
                return session_id
            except Exception as e:
                print(f"Error creating file session: {e}")
                return None
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve session data by session ID"""
        if not session_id:
            return None
            
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            return None
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Check if session is expired
            expires_at = datetime.fromisoformat(session_data['expires_at'])
            if datetime.now() > expires_at:
                self.delete_session(session_id)
                return None
            
            # Update last accessed time
            session_data['last_accessed'] = datetime.now().isoformat()
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return session_data['user_data']
        
        except Exception as e:
            print(f"Error reading session: {e}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete a session"""
        if not session_id:
            return False
            
        session_file = self._get_session_file(session_id)
        
        try:
            if session_file.exists():
                session_file.unlink()
            return True
        except Exception as e:
            print(f"Error deleting session: {e}")
            return False
    
    def cleanup_expired_sessions(self):
        """Clean up expired session files"""
        try:
            for session_file in self.session_dir.glob("session_*.json"):
                try:
                    with open(session_file, 'r') as f:
                        session_data = json.load(f)
                    
                    expires_at = datetime.fromisoformat(session_data['expires_at'])
                    if datetime.now() > expires_at:
                        session_file.unlink()
                        
                except Exception:
                    # If we can't read the file, it's corrupted, so delete it
                    session_file.unlink()
        except Exception as e:
            print(f"Error cleaning up sessions: {e}")
    
    def extend_session(self, session_id: str) -> bool:
        """Extend session expiration time"""
        if not session_id:
            return False
            
        session_file = self._get_session_file(session_id)
        
        if not session_file.exists():
            return False
        
        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
            
            # Extend expiration time
            session_data['expires_at'] = (datetime.now() + timedelta(seconds=self.session_timeout)).isoformat()
            session_data['last_accessed'] = datetime.now().isoformat()
            
            with open(session_file, 'w') as f:
                json.dump(session_data, f, indent=2)
            
            return True
        except Exception as e:
            print(f"Error extending session: {e}")
            return False

# Global session service instance
session_service = SessionService()