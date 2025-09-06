# services/auth_service.py
import hashlib
import uuid
import bcrypt
from typing import Optional, Dict, Any, List
from datetime import datetime
from models.user import User, UserRole
# Database service now injected through dependency injection

class AuthService:
    """Enhanced authentication service with role-based access control"""
    
    def __init__(self, db_service):
        self.db = db_service
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user and return user object if valid"""
        user = self.db.get_user_by_username(username)
        if not user or not user.is_active:
            return None
        
        # Verify password - support both bcrypt and legacy SHA256
        try:
            # Check if it's a bcrypt hash (starts with $2b$)
            if user.password_hash.startswith('$2b$'):
                if not bcrypt.checkpw(password.encode('utf-8'), user.password_hash.encode('utf-8')):
                    return None
            else:
                # Legacy SHA256 hash verification
                password_hash = hashlib.sha256(password.encode()).hexdigest()
                if user.password_hash != password_hash:
                    return None
        except Exception as e:
            print(f"Password verification error: {e}")
            return None
        
        # Update last login
        self._update_last_login(user.id)
        return user
    
    def register_user(self, username: str, name: str, email: str, password: str, 
                     role: UserRole, parent_ids: list = None) -> Optional[User]:
        """Register new user with role-based restrictions"""
        
        # Check if username/email already exists
        existing_user = self.db.get_user_by_username(username)
        if existing_user:
            raise ValueError("Username already exists")
        
        # Create new user
        user = User(
            id=str(uuid.uuid4()),
            username=username,
            name=name,
            email=email,
            password_hash=self._hash_password(password),
            role=role,
            parent_ids=parent_ids,
            created_at=datetime.now().isoformat()
        )
        
        success = self.db.create_user(user)
        return user if success else None
    
    def create_user(self, username: str, password: str, name: str, email: str, 
                   role: UserRole) -> bool:
        """Create new user (simplified interface for admin dashboard)"""
        try:
            user = self.register_user(username, name, email, password, role)
            return user is not None
        except Exception as e:
            print(f"Error in create_user: {e}")
            return False
    
    def _hash_password(self, password: str) -> str:
        """Hash password using bcrypt with salt"""
        salt = bcrypt.gensalt(rounds=12)  # Strong hashing with 12 rounds
        hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
        return hashed.decode('utf-8')
    
    def _update_last_login(self, user_id: str):
        """Update user's last login timestamp"""
        # This would be implemented in the database service
        pass
    
    def get_all_users(self) -> List[User]:
        """Get all users (admin only)"""
        try:
            return self.db.get_all_users()
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
    
    def check_upload_permission(self, user: User) -> bool:
        """Check if user can upload documents (admin only)"""
        return user.role == UserRole.ADMIN
    
    def check_analytics_permission(self, user: User) -> bool:
        """Check if user can view analytics"""
        return user.role in [UserRole.ADMIN, UserRole.TEACHER]
    
    def check_student_progress_permission(self, user: User, student_id: str) -> bool:
        """Check if user can view specific student's progress"""
        if user.role == UserRole.ADMIN:
            return True
        elif user.role == UserRole.PARENT:
            # Check if student is linked to this parent
            students = self.db.get_students_for_parent(user.id)
            return any(student.id == student_id for student in students)
        elif user.role == UserRole.TEACHER:
            # For now, teachers can view all students
            # Later: implement class-based restrictions
            return True
        return False
    
    def delete_user(self, username: str) -> bool:
        """Delete user and all associated data (admin only)"""
        try:
            # Get user to delete
            user = self.db.get_user_by_username(username)
            if not user:
                return False
            
            # Delete user's activities, documents, and other data
            self.db.delete_user_completely(user.id)
            return True
        except Exception as e:
            print(f"Error deleting user {username}: {e}")
            return False

# Role-based permission decorators
def admin_required(func):
    """Decorator to require admin role"""
    def wrapper(user: User, *args, **kwargs):
        if user.role != UserRole.ADMIN:
            raise PermissionError("Admin access required")
        return func(user, *args, **kwargs)
    return wrapper

def teacher_or_admin_required(func):
    """Decorator to require teacher or admin role"""
    def wrapper(user: User, *args, **kwargs):
        if user.role not in [UserRole.ADMIN, UserRole.TEACHER]:
            raise PermissionError("Teacher or admin access required")
        return func(user, *args, **kwargs)
    return wrapper