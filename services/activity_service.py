# services/activity_service.py
import uuid
from datetime import datetime
from typing import List, Optional, Dict, Any
from models.activity import StudentActivity, ActivityType, ProgressMetrics
from models.user import User, UserRole
# Database service now injected through dependency injection
import re
import collections

class ActivityService:
    """Service for logging and analyzing student activities"""
    
    def __init__(self, db_service):
        self.db = db_service
        self.topic_keywords = {
            'mathematics': ['math', 'equation', 'algebra', 'geometry', 'calculus', 'number', 'formula'],
            'science': ['science', 'biology', 'chemistry', 'physics', 'experiment', 'molecule', 'cell'],
            'history': ['history', 'war', 'ancient', 'civilization', 'historical', 'past', 'century'],
            'literature': ['literature', 'book', 'novel', 'poetry', 'author', 'writing', 'story'],
            'geography': ['geography', 'country', 'continent', 'climate', 'map', 'location', 'region'],
            'technology': ['technology', 'computer', 'software', 'programming', 'digital', 'internet']
        }
    
    def log_query_activity(self, student_id: str, session_id: str, query: str, 
                          response: str, sources: List[str] = None, 
                          response_time_ms: int = None, 
                          grounding_confidence: float = None) -> bool:
        """Log a student query activity with comprehensive details"""
        
        # Detect topics from query
        detected_topics = self._detect_topics(query)
        
        # Assess difficulty level
        difficulty_level = self._assess_difficulty(query)
        
        activity = StudentActivity(
            id=str(uuid.uuid4()),
            student_id=student_id,
            session_id=session_id,
            activity_type=ActivityType.QUERY,
            timestamp=datetime.now().isoformat(),
            query_text=query,
            response_text=response,
            sources_used=sources,
            response_time_ms=response_time_ms,
            grounding_confidence=grounding_confidence,
            detected_topics=detected_topics,
            difficulty_level=difficulty_level,
            metadata={
                'query_length': len(query),
                'response_length': len(response),
                'sources_count': len(sources) if sources else 0
            }
        )
        
        return self.db.log_activity(activity)
    
    def log_login_activity(self, student_id: str, session_id: str) -> bool:
        """Log student login"""
        activity = StudentActivity(
            id=str(uuid.uuid4()),
            student_id=student_id,
            session_id=session_id,
            activity_type=ActivityType.LOGIN,
            timestamp=datetime.now().isoformat()
        )
        return self.db.log_activity(activity)
    
    def _detect_topics(self, query: str) -> List[str]:
        """Detect topics from query text using keyword matching"""
        query_lower = query.lower()
        detected = []
        
        for topic, keywords in self.topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                detected.append(topic)
        
        return detected if detected else ['general']
    
    def _assess_difficulty(self, query: str) -> str:
        """Assess difficulty level based on query complexity"""
        # Simple heuristic based on question complexity
        complex_words = ['analyze', 'compare', 'evaluate', 'synthesize', 'critique', 'interpret']
        intermediate_words = ['explain', 'describe', 'discuss', 'summarize', 'outline']
        
        query_lower = query.lower()
        
        if any(word in query_lower for word in complex_words):
            return 'advanced'
        elif any(word in query_lower for word in intermediate_words) or len(query) > 100:
            return 'intermediate'
        else:
            return 'basic'
    
    def get_student_progress_summary(self, student_id: str, days_back: int = 7) -> Optional[ProgressMetrics]:
        """Generate progress summary for a student"""
        activities = self.db.get_student_activities(student_id, limit=1000)
        
        if not activities:
            return None
        
        # Filter activities to specified time period
        from datetime import datetime, timedelta
        cutoff_time = datetime.now() - timedelta(days=days_back)
        recent_activities = [
            a for a in activities 
            if datetime.fromisoformat(a.timestamp) >= cutoff_time
        ]
        
        if not recent_activities:
            return None
        
        # Calculate metrics
        total_queries = len([a for a in recent_activities if a.activity_type == ActivityType.QUERY])
        
        # Get unique topics
        all_topics = []
        for activity in recent_activities:
            if activity.detected_topics:
                all_topics.extend(activity.detected_topics)
        unique_topics = len(set(all_topics))
        
        # Calculate average session duration (placeholder)
        avg_session_duration = 300  # Default 5 minutes
        
        # Get most active hours
        hours = [datetime.fromisoformat(a.timestamp).hour for a in recent_activities]
        hour_counts = collections.Counter(hours)
        most_active_hours = [hour for hour, count in hour_counts.most_common(3)]
        
        # Get preferred topics
        topic_counts = collections.Counter(all_topics)
        preferred_topics = [topic for topic, count in topic_counts.most_common(5)]
        
        # Calculate difficulty progression
        difficulties = [a.difficulty_level for a in recent_activities if a.difficulty_level]
        difficulty_progression = list(set(difficulties))
        
        # Sessions per week (estimate)
        sessions_per_week = len(set(a.session_id for a in recent_activities)) * (7 / days_back)
        
        # Average satisfaction (placeholder)
        satisfactions = [a.satisfaction_rating for a in recent_activities if a.satisfaction_rating]
        avg_satisfaction = sum(satisfactions) / len(satisfactions) if satisfactions else 4.0
        
        return ProgressMetrics(
            student_id=student_id,
            period_start=(datetime.now() - timedelta(days=days_back)).isoformat(),
            period_end=datetime.now().isoformat(),
            total_queries=total_queries,
            unique_topics_explored=unique_topics,
            average_session_duration=avg_session_duration,
            most_active_hours=most_active_hours,
            preferred_topics=preferred_topics,
            difficulty_progression=difficulty_progression,
            sessions_per_week=sessions_per_week,
            average_response_satisfaction=avg_satisfaction
        )
    
    def get_students_for_parent_summary(self, parent_id: str) -> List[Dict[str, Any]]:
        """Get progress summary for all students linked to a parent"""
        students = self.db.get_students_for_parent(parent_id)
        summaries = []
        
        for student in students:
            progress = self.get_student_progress_summary(student.id)
            summaries.append({
                'student': student,
                'progress': progress
            })
        
        return summaries
