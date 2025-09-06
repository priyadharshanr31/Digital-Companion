# Core modular imports
from services.database_wrapper import database_service
from services.auth_service import AuthService
from services.activity_service import ActivityService
from services.document_service import DocumentService
from services.rag_service import RAGService
from services.session_service import session_service
from models.user import User, UserRole
from models.activity import StudentActivity, ActivityType
from ui.components import apply_role_theme, render_role_header, render_document_upload_section, render_user_info_sidebar
from ui.parent_dashboard import render_parent_dashboard
from ui.teacher_dashboard import render_teacher_dashboard
import uuid
import time


import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()
import time
import tempfile
import json
import re
import numpy as np
import faiss
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher
import hashlib
import io
import shutil
import yaml
from yaml.loader import SafeLoader

# Core Libraries
try:
    from google import genai
    from google.genai import types
except ImportError:
    st.error("Google GenAI library not found. Install with: pip install google-genai")
    st.stop()

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    st.error("Sentence Transformers not found. Install with: pip install sentence-transformers")
    st.stop()

try:
    import PyPDF2
except ImportError:
    st.error("PyPDF2 not found. Install with: pip install PyPDF2")
    st.stop()

# Video Processing Libraries (FFmpeg-free alternatives)
try:
    from faster_whisper import WhisperModel
except ImportError:
    st.error("faster-whisper not found. Install with: pip install faster-whisper")
    st.stop()

try:
    from moviepy.editor import VideoFileClip
    MOVIEPY_AVAILABLE = True
except ImportError:
    VideoFileClip = None
    MOVIEPY_AVAILABLE = False

try:
    from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled
except ImportError:
    st.error("YouTube Transcript API not found. Install with: pip install youtube-transcript-api")
    st.stop()

# Better YouTube downloader - more reliable than pytube
try:
    import yt_dlp
except ImportError:
    st.error("yt-dlp not found. Install with: pip install yt-dlp")
    st.stop()

try:
    import streamlit_authenticator as stauth
except ImportError:
    st.error("Streamlit Authenticator not found. Install with: pip install streamlit-authenticator")
    st.stop()

# Configure Streamlit page
st.set_page_config(
    page_title="AERO - AI Educational Assistant",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# Create user configuration with proper password hashing
def create_user_config():
    """Create user configuration with hashed passwords"""
    # Create credentials dictionary
    credentials = {
        "usernames": {
            "student1": {
                "email": "alice@student.edu",
                "name": "Alice Johnson",
                "password": "student123",  # Will be hashed automatically
                "role": "student"
            },
            "teacher1": {
                "email": "smith@university.edu",
                "name": "Prof. Smith",
                "password": "teacher123",  # Will be hashed automatically
                "role": "teacher"
            },
            "parent1": {
                "email": "wilson@parent.com",
                "name": "Mrs. Wilson",
                "password": "parent123",  # Will be hashed automatically
                "role": "parent"
            }
        }
    }

    # Hash passwords
    stauth.Hasher.hash_passwords(credentials)

    return {
        "credentials": credentials,
        "cookie": {
            "name": "rag_chatbot_cookie",
            "key": "random_signature_key_2024_advanced",
            "expiry_days": 30
        }
    }


# Initialize session state
session_defaults = {
    'authenticated': False,
    'api_key': None,
    'messages': [],
    'vector_store': None,
    'documents': [],
    'embeddings_model': None,
    'gemini_client': None,
    'grounding_threshold': 0.7,
    'whisper_model': None,
    'user_role': None,
    'username': None,
    'name': None,
    'authentication_status': None,
    'authenticator': None,  # Added for proper logout
    'logout_clicked': False,  # Added to track logout action
    'current_user': None,  # New: current User object
    'session_id': None,  # New: for activity tracking
    'db_service': None,  # New: database service
    'auth_service': None,  # New: auth service
    'activity_service': None,  # New: activity service
    'document_service': None,  # New: document service
    'rag_service': None  # New: RAG service
}

def initialize_session_state():
    """Initialize session state and services"""
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Initialize database and auth services
    if st.session_state.db_service is None:
        st.session_state.db_service = database_service

    if st.session_state.auth_service is None:
        st.session_state.auth_service = AuthService(st.session_state.db_service)

    if st.session_state.activity_service is None:
        st.session_state.activity_service = ActivityService(st.session_state.db_service)

    if st.session_state.document_service is None:
        st.session_state.document_service = DocumentService(st.session_state.db_service)

    if st.session_state.rag_service is None:
        st.session_state.rag_service = RAGService(st.session_state.db_service, st.session_state.activity_service)

    # Generate session ID for activity tracking
    if st.session_state.session_id is None:
        st.session_state.session_id = str(uuid.uuid4())
    
    # Initialize persistent session ID if not exists
    if 'persistent_session_id' not in st.session_state:
        st.session_state.persistent_session_id = None

def check_persistent_session():
    """Check for existing persistent session and restore user state"""
    # Clean up expired sessions first
    session_service.cleanup_expired_sessions()
    
    # Check if we have a persistent session ID in query params
    try:
        persistent_session_id = st.query_params.get('session', None)
    except:
        # Fallback for older Streamlit versions
        query_params = st.experimental_get_query_params() if hasattr(st, 'experimental_get_query_params') else {}
        persistent_session_id = query_params.get('session', [None])[0] if query_params.get('session') else None
    
    # If no session in URL, check session state
    if not persistent_session_id:
        persistent_session_id = st.session_state.persistent_session_id
    
    if persistent_session_id:
        # Try to restore session
        user_data = session_service.get_session(persistent_session_id)
        if user_data:
            # Restore user state
            st.session_state.authenticated = True
            st.session_state.persistent_session_id = persistent_session_id
            
            # Recreate user object
            from models.user import User, UserRole
            st.session_state.current_user = User(
                id=user_data['id'],
                username=user_data['username'],
                name=user_data['name'],
                email=user_data['email'],
                password_hash=user_data['password_hash'],
                role=UserRole(user_data['role'])
            )
            st.session_state.username = user_data['username']
            st.session_state.name = user_data['name']
            st.session_state.user_role = user_data['role']
            
            # Extend session
            session_service.extend_session(persistent_session_id)

def create_persistent_session(user):
    """Create a persistent session for the user"""
    user_data = {
        'id': user.id,
        'username': user.username,
        'name': user.name,
        'email': user.email,
        'password_hash': user.password_hash,
        'role': user.role.value
    }
    
    session_id = session_service.create_session(user_data)
    if session_id:
        st.session_state.persistent_session_id = session_id
        # Update URL to include session (for bookmarking)
        try:
            st.query_params['session'] = session_id
        except:
            # Fallback for older Streamlit versions
            st.experimental_set_query_params(session=session_id) if hasattr(st, 'experimental_set_query_params') else None
    
    return session_id


# Role-based theming moved to ui/components.py


class VideoProcessor:
    """Handles video processing and audio extraction without FFmpeg"""

    def __init__(self):
        self.supported_formats = ['mp4', 'avi', 'mov', 'mkv', 'webm', 'flv']

    def extract_audio_from_video(self, video_file, output_path=None):
        """Extract audio from video using MoviePy (no FFmpeg required)"""
        if not MOVIEPY_AVAILABLE:
            st.error("MoviePy is not available. Cannot process video files.")
            return None
        try:
            if output_path is None:
                output_path = tempfile.mktemp(suffix='.wav')

            # Create temporary video file
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
                temp_video.write(video_file.read())
                temp_video_path = temp_video.name

            # Extract audio using MoviePy
            video_clip = VideoFileClip(temp_video_path)
            audio_clip = video_clip.audio

            # Write audio to WAV file
            audio_clip.write_audiofile(output_path, verbose=False, logger=None)

            # Clean up
            audio_clip.close()
            video_clip.close()
            os.unlink(temp_video_path)

            return output_path

        except Exception as e:
            st.error(f"Error extracting audio from video: {str(e)}")
            return None

    def get_youtube_transcript(self, url):
        """Get YouTube transcript using youtube-transcript-api with better error handling"""
        try:
            # Extract video ID from URL - improved regex
            if 'youtube.com/watch?v=' in url:
                video_id = url.split('watch?v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                video_id = url.split('youtu.be/')[1].split('?')[0]
            elif 'youtube.com/embed/' in url:
                video_id = url.split('embed/')[1].split('?')[0]
            else:
                # Try to extract from the end of the URL
                video_id = url.split('/')[-1].split('?')[0]

            # Try to get transcript with multiple language options
            try:
                transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
                transcript_text = ' '.join([item['text'] for item in transcript_list])
                return transcript_text
            except NoTranscriptFound:
                # Try with auto-generated captions
                try:
                    transcript_list = YouTubeTranscriptApi.get_transcript(video_id, languages=['en-US', 'en-GB'])
                    transcript_text = ' '.join([item['text'] for item in transcript_list])
                    return transcript_text
                except:
                    pass

            return None

        except Exception as e:
            st.warning(f"Could not get YouTube transcript: {str(e)}")
            return None

    def download_youtube_audio(self, url):
        """### FFMPEG_FREE - Download YouTube audio without ffmpeg/ffprobe"""
        try:
            # Use completely ffmpeg-free approach
            temp_path = tempfile.mktemp(suffix='.m4a')

            # Configure yt-dlp to avoid any post-processing that requires ffmpeg
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio/best',  # Prefer m4a, fallback to best
                'outtmpl': temp_path,
                'quiet': True,
                'no_warnings': True,
                'noplaylist': True,
                'postprocessors': [],  # No post-processing to avoid ffmpeg dependency
                'prefer_ffmpeg': False,
                'merge_output_format': None,  # Don't merge
                'writesubtitles': False,
                'writeautomaticsub': False,
                'ignoreerrors': True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

            # Check if file was created
            if os.path.exists(temp_path):
                return temp_path
            else:
                # Try alternative format if m4a fails
                temp_path2 = tempfile.mktemp(suffix='.webm')
                ydl_opts2 = {
                    'format': 'bestaudio[ext=webm]/bestaudio',
                    'outtmpl': temp_path2,
                    'quiet': True,
                    'no_warnings': True,
                    'noplaylist': True,
                    'postprocessors': [],
                    'prefer_ffmpeg': False,
                }

                with yt_dlp.YoutubeDL(ydl_opts2) as ydl:
                    ydl.download([url])

                return temp_path2 if os.path.exists(temp_path2) else None

        except Exception as e:
            st.error(f"Error downloading YouTube audio: {str(e)}")
            return None


class WhisperTranscriber:
    """Handles transcription using faster-whisper (no FFmpeg required)"""

    def __init__(self, model_size="base"):
        self.model_size = model_size
        self.model = None

    def load_model(self):
        """Load WhisperModel - cached to avoid reloading"""
        if self.model is None:
            try:
                self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            except Exception as e:
                st.error(f"Error loading Whisper model: {str(e)}")
                return None
        return self.model

    def transcribe_audio(self, audio_path):
        """Transcribe audio file using faster-whisper"""
        try:
            model = self.load_model()
            if model is None:
                return None

            segments, info = model.transcribe(audio_path, beam_size=5)

            # Combine all segments into full text
            full_text = ""
            for segment in segments:
                full_text += segment.text + " "

            return full_text.strip()

        except Exception as e:
            st.error(f"Error transcribing audio: {str(e)}")
            return None


class GroundingValidator:
    """Validates if responses are properly grounded in provided context"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.min_overlap_threshold = 0.2
        self.semantic_threshold = 0.4

    def calculate_text_overlap(self, response: str, context: str) -> float:
        """Calculate overlap between response and context"""
        response_words = set(response.lower().split())
        context_words = set(context.lower().split())

        if not response_words:
            return 0.0

        overlap = len(response_words.intersection(context_words))
        return overlap / len(response_words)

    def calculate_semantic_similarity(self, response: str, context: str) -> float:
        """Calculate semantic similarity between response and context"""
        try:
            if not response.strip() or not context.strip():
                return 0.0

            response_embedding = self.embedding_model.encode([response])
            context_embedding = self.embedding_model.encode([context])

            # Calculate cosine similarity
            similarity = np.dot(response_embedding[0], context_embedding[0]) / (
                    np.linalg.norm(response_embedding[0]) * np.linalg.norm(context_embedding[0])
            )
            return float(similarity)
        except Exception as e:
            st.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0

    def validate_grounding(self, response: str, context: str) -> Dict[str, Any]:
        """Validate if response is properly grounded in context"""
        if not context.strip():
            return {
                'is_grounded': False,
                'confidence': 0.0,
                'reason': 'No context provided',
                'text_overlap': 0.0,
                'semantic_similarity': 0.0
            }

        # Calculate grounding metrics
        text_overlap = self.calculate_text_overlap(response, context)
        semantic_similarity = self.calculate_semantic_similarity(response, context)

        # Combined confidence score
        confidence = (text_overlap * 0.4) + (semantic_similarity * 0.6)

        # Determine if grounded
        is_grounded = (
                text_overlap >= self.min_overlap_threshold and
                semantic_similarity >= self.semantic_threshold
        )

        return {
            'is_grounded': is_grounded,
            'confidence': confidence,
            'reason': self._get_grounding_reason(text_overlap, semantic_similarity),
            'text_overlap': text_overlap,
            'semantic_similarity': semantic_similarity
        }

    def _get_grounding_reason(self, text_overlap: float, semantic_similarity: float) -> str:
        """Get reason for grounding decision"""
        if text_overlap < self.min_overlap_threshold:
            return f"Low text overlap ({text_overlap:.2f} < {self.min_overlap_threshold})"
        elif semantic_similarity < self.semantic_threshold:
            return f"Low semantic similarity ({semantic_similarity:.2f} < {self.semantic_threshold})"
        else:
            return "Well grounded in provided context"


class DocumentProcessor:
    """Handles document processing and text extraction"""

    def __init__(self):
        self.video_processor = VideoProcessor()
        self.transcriber = WhisperTranscriber()

    def extract_text_from_pdf(self, pdf_file) -> str:
        """Extract text from PDF file"""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return ""

    def extract_text_from_txt(self, txt_file) -> str:
        """Extract text from TXT file"""
        try:
            return txt_file.read().decode('utf-8')
        except Exception as e:
            st.error(f"Error reading text file: {str(e)}")
            return ""

    def extract_text_from_video(self, video_file) -> str:
        """Extract text from video file using faster-whisper"""
        try:
            with st.spinner("Extracting audio from video..."):
                # Extract audio from video
                audio_path = self.video_processor.extract_audio_from_video(video_file)

                if audio_path:
                    with st.spinner("Transcribing audio to text..."):
                        # Transcribe audio to text
                        text = self.transcriber.transcribe_audio(audio_path)

                        # Clean up temporary audio file
                        os.unlink(audio_path)

                        return text if text else ""
                else:
                    return ""

        except Exception as e:
            st.error(f"Error extracting text from video: {str(e)}")
            return ""

    def extract_text_from_youtube(self, youtube_url) -> str:
        """Extract text from YouTube video with improved error handling"""
        try:
            # First try to get transcript directly
            transcript = self.video_processor.get_youtube_transcript(youtube_url)
            if transcript:
                return transcript

            # If no transcript, download audio and transcribe
            with st.spinner("Downloading YouTube audio..."):
                audio_path = self.video_processor.download_youtube_audio(youtube_url)

                if audio_path:
                    with st.spinner("Transcribing YouTube audio..."):
                        text = self.transcriber.transcribe_audio(audio_path)

                        # Clean up temporary audio file
                        try:
                            os.unlink(audio_path)
                        except:
                            pass

                        return text if text else ""
                else:
                    return ""

        except Exception as e:
            st.error(f"Error extracting text from YouTube: {str(e)}")
            return ""

    def chunk_text(self, text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
        """Split text into chunks with overlap - optimized for grounding"""
        chunks = []

        # Split by paragraphs first for better context preservation
        paragraphs = text.split('\n\n')
        current_chunk = ""

        for paragraph in paragraphs:
            if len(current_chunk) + len(paragraph) < chunk_size:
                current_chunk += paragraph + "\n\n"
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = paragraph + "\n\n"

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # If chunks are too large, split them further
        final_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                words = chunk.split()
                current_chunk = ""
                for word in words:
                    if len(current_chunk) + len(word) < chunk_size:
                        current_chunk += word + " "
                    else:
                        if current_chunk.strip():
                            final_chunks.append(current_chunk.strip())
                        current_chunk = word + " "
                if current_chunk.strip():
                    final_chunks.append(current_chunk.strip())
            else:
                final_chunks.append(chunk)

        return final_chunks


class RAGVectorStore:
    """Enhanced vector storage with better relevance scoring"""

    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.index = None
        self.documents = []
        self.embeddings = None
        self.document_metadata = []

    def add_documents(self, documents: List[str], metadata: List[Dict] = None):
        """Add documents to vector store with metadata and context caching"""
        try:
            # Use documents directly (removed context cache complexity)
            cached_documents = documents

            with st.spinner("Creating embeddings..."):
                # Generate embeddings for cached documents
                embeddings = self.embedding_model.encode(cached_documents, show_progress_bar=False)

                if self.index is None:
                    # Create FAISS index
                    dimension = embeddings.shape[1]
                    self.index = faiss.IndexFlatL2(dimension)
                    self.embeddings = embeddings
                    self.documents = cached_documents
                    self.document_metadata = metadata or [{}] * len(cached_documents)
                else:
                    # Add to existing index
                    self.embeddings = np.vstack([self.embeddings, embeddings])
                    self.documents.extend(cached_documents)
                    self.document_metadata.extend(metadata or [{}] * len(cached_documents))

                # Add embeddings to index
                self.index.add(embeddings.astype('float32'))
                return True
        except Exception as e:
            st.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def search(self, query: str, k: int = 5, relevance_threshold: float = 0.3) -> List[Dict[str, Any]]:
        """Enhanced search with relevance filtering"""
        try:
            if self.index is None:
                return []

            # Generate embedding for query
            query_embedding = self.embedding_model.encode([query])

            # Search in FAISS index with more candidates
            search_k = min(k * 2, len(self.documents))
            distances, indices = self.index.search(query_embedding.astype('float32'), search_k)

            results = []
            for distance, idx in zip(distances[0], indices[0]):
                if idx < len(self.documents):
                    # Convert distance to relevance score
                    relevance_score = 1.0 / (1.0 + distance)

                    # Only include results above relevance threshold
                    if relevance_score >= relevance_threshold:
                        results.append({
                            'content': self.documents[idx],
                            'distance': float(distance),
                            'relevance_score': relevance_score,
                            'metadata': self.document_metadata[idx] if idx < len(self.document_metadata) else {}
                        })

            # Sort by relevance score and return top k
            results.sort(key=lambda x: x['relevance_score'], reverse=True)
            return results[:k]

        except Exception as e:
            st.error(f"Error searching vector store: {str(e)}")
            return []


class GroundedGeminiChatbot:
    """AERO - AI Educational Response Oracle with intelligent grounding"""

    def __init__(self, api_key: str, grounding_validator: GroundingValidator):
        try:
            self.client = genai.Client(api_key=api_key)
            self.model_name = "gemini-2.0-flash"
            self.grounding_validator = grounding_validator
            self.max_retries = 2
            st.session_state.gemini_client = self.client
        except Exception as e:
            st.error(f"Error initializing Gemini client: {str(e)}")
            self.client = None

    def _create_grounded_prompt(self, query: str, context: str) -> str:
        """Create a strictly grounded prompt for detailed responses"""
        if not context.strip():
            return self._create_no_context_prompt(query)

        # Enhanced system prompt for strict grounding with detailed responses
        grounded_prompt = f"""You are a helpful assistant that MUST answer questions based ONLY on the provided context. 

CRITICAL INSTRUCTIONS:
1. You can ONLY use information that is explicitly stated in the context below
2. Do NOT use any external knowledge or information not in the context
3. If the context doesn't contain enough information to answer the question, you MUST say "I don't have enough information in the provided context to answer this question"
4. Quote relevant parts of the context when possible
5. Stay strictly within the bounds of the provided information
6. Provide comprehensive, detailed answers when the context allows
7. Use proper formatting with headers, bullet points, and structure when appropriate
8. Include all relevant details from the context
9. Organize your response logically with clear sections

CONTEXT:
{context}

QUESTION: {query}

REQUIREMENTS:
- Base your answer ONLY on the context above
- If information is not in the context, explicitly state that you don't have that information
- Include specific quotes or references from the context to support your answer
- Do not make assumptions or add information not present in the context
- Provide as much detail as possible from the available context
- Structure your answer clearly with proper formatting
- Use markdown formatting for better readability (headers, lists, bold text)
- Be comprehensive and thorough in your response

ANSWER:"""

        return grounded_prompt

    def _create_no_context_prompt(self, query: str) -> str:
        """Create prompt when no context is available"""
        return f"""I don't have any relevant information in my knowledge base to answer your question: "{query}"

Please try:
1. Uploading relevant documents that contain the information you're looking for
2. Rephrasing your question to be more specific
3. Asking a question that relates to the documents you've uploaded

I can only provide answers based on the documents you've provided to me."""

    def _validate_and_improve_response(self, response: str, context: str, query: str) -> Dict[str, Any]:
        """Validate response grounding and improve if needed"""
        # Check grounding
        grounding_result = self.grounding_validator.validate_grounding(response, context)

        # If not well grounded, provide fallback response
        if not grounding_result['is_grounded']:
            fallback_response = self._generate_fallback_response(query, context)
            return {
                'response': fallback_response,
                'grounding_result': grounding_result,
                'is_fallback': True
            }

        return {
            'response': response,
            'grounding_result': grounding_result,
            'is_fallback': False
        }

    def _generate_fallback_response(self, query: str, context: str) -> str:
        """Generate fallback response when grounding fails"""
        if not context.strip():
            return f"I don't have any relevant information in my knowledge base to answer your question about '{query}'. Please upload relevant documents first."

        return f"I cannot provide a complete answer to your question about '{query}' based on the available context. The information in my knowledge base may not be sufficient or directly relevant to your specific question. Please try rephrasing your question or providing more specific documents."

    def generate_response(self, query: str, context: str = "") -> Dict[str, Any]:
        """Generate grounded response with validation - EXTENDED LENGTH (4096 tokens)"""
        try:
            if not self.client:
                return {
                    'response': "Error: Gemini client not initialized.",
                    'grounding_result': None,
                    'is_fallback': True
                }

            # Create grounded prompt
            prompt = self._create_grounded_prompt(query, context)

            # Generate response with enhanced configuration - INCREASED TOKEN LIMIT
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more deterministic responses
                    top_p=0.8,
                    max_output_tokens=4096,  # INCREASED FROM 1024 TO 4096 FOR LONGER DETAILED ANSWERS
                    stop_sequences=["EXTERNAL:", "OUTSIDE:", "GENERAL KNOWLEDGE:"]
                )
            )

            # Validate and improve response
            result = self._validate_and_improve_response(response.text, context, query)

            return result

        except Exception as e:
            return {
                'response': f"Error generating response: {str(e)}",
                'grounding_result': None,
                'is_fallback': True
            }


def authenticate_user():
    """Fixed authentication function for latest streamlit-authenticator"""
    st.markdown('<div class="role-header">🚀 AERO - AI Educational Assistant</div>', unsafe_allow_html=True)

    # Get user configuration
    config = create_user_config()

    # Create authenticator with corrected parameters
    authenticator = stauth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days']
    )

    # ### AUTH_LOGOUT - Store authenticator in session state for proper logout
    st.session_state.authenticator = authenticator

    # **FIXED LOGIN CALL** - Use new syntax without parameters
    try:
        authenticator.login()
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return

    # **FIXED SESSION STATE ACCESS** - Use new session state keys
    if st.session_state.get('authentication_status'):
        st.session_state.authenticated = True
        st.session_state.username = st.session_state.get('username')
        st.session_state.name = st.session_state.get('name')

        # Get user role from credentials
        user_data = config['credentials']['usernames'].get(st.session_state.username, {})
        st.session_state.user_role = user_data.get('role', 'student')

        # Apply role-based theme
        apply_role_theme(st.session_state.user_role)

        st.success(f"✅ Welcome {st.session_state.name}! Role: {st.session_state.user_role.title()}")

        # Show role-specific welcome message
        role_messages = {
            'student': "📚 Access your learning materials and ask questions about uploaded content.",
            'teacher': "👨‍🏫 Manage educational content and track student interactions.",
            'parent': "👨‍👩‍👧‍👦 Review educational materials and monitor learning progress."
        }

        st.info(role_messages.get(st.session_state.user_role, "Welcome to the RAG Chatbot!"))

        # **FIXED LOGOUT CALL** - Use new syntax
        authenticator.logout()

        time.sleep(1)
        st.rerun()

    elif st.session_state.get('authentication_status') is False:
        st.error('❌ Username/password is incorrect')

    elif st.session_state.get('authentication_status') is None:
        st.warning('Please enter your username and password')

    # Demo credentials info
    with st.expander("🔍 Demo Credentials"):
        st.info("""
        **Demo Accounts:**
        - Student: username=`student1`, password=`student123`
        - Teacher: username=`teacher1`, password=`teacher123`  
        - Parent: username=`parent1`, password=`parent123`
        """)


def initialize_models():
    """Initialize embedding model and RAG components"""
    if st.session_state.embeddings_model is None:
        with st.spinner("Loading embedding model..."):
            try:
                st.session_state.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
                st.success("✅ Embedding model loaded successfully!")
            except Exception as e:
                st.error(f"Error loading embedding model: {str(e)}")
                return False

    if st.session_state.vector_store is None:
        st.session_state.vector_store = RAGVectorStore(st.session_state.embeddings_model)

    return True


def get_api_key():
    """Get API key from environment variables, secrets, or user input (in priority order)"""
    st.subheader("🔑 Gemini API Configuration")

    # Priority 1: Environment variable (for Docker/production)
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        st.success(f"✅ API key loaded from environment variables! (length: {len(api_key)})")
        st.session_state.api_key = api_key
        return True
    
    # Priority 2: Streamlit secrets (for local development)
    try:
        api_key = st.secrets.get("GEMINI_API_KEY")
        if api_key:
            st.success("✅ API key loaded from secrets.toml!")
            st.session_state.api_key = api_key
            return True
    except Exception:
        pass  # Secrets not available, continue to manual input
    
    # Priority 3: Manual input (fallback)
    st.info("💡 Enter your API key manually or set GEMINI_API_KEY environment variable")
    api_key = st.text_input(
        "Enter your Gemini API Key:",
        type="password",
        placeholder="Your API key here...",
        help="Get your API key from Google AI Studio"
    )

    if api_key:
        st.session_state.api_key = api_key
        return True

    st.warning("Please enter your Gemini API key to continue.")
    st.info("""
    **To set up automatic API key loading:**
    1. Create folder: `.streamlit` in your project directory
    2. Create file: `secrets.toml` in the `.streamlit` folder
    3. Add your key: `GEMINI_API_KEY = "your_api_key_here"`
    4. Restart the app
    """)
    return False


# Document upload section moved to ui/components.py - use render_document_upload_section()


def process_documents_admin(uploaded_files):
    """Process uploaded documents (Admin only) - shared knowledge base"""
    processor = DocumentProcessor()
    all_chunks = []
    metadata = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(f"Processing {uploaded_file.name}...")

        # Extract text based on file type
        if uploaded_file.type == "application/pdf":
            text = processor.extract_text_from_pdf(uploaded_file)
        elif uploaded_file.type == "text/plain":
            text = processor.extract_text_from_txt(uploaded_file)
        else:
            st.sidebar.error(f"Unsupported file type: {uploaded_file.type}")
            continue

        if text:
            chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
            all_chunks.extend(chunks)

            for chunk in chunks:
                metadata.append({
                    'source_file': uploaded_file.name,
                    'source_type': 'document',
                    'chunk_length': len(chunk),
                    'processing_time': datetime.now().isoformat(),
                    'processed_by': st.session_state.username
                })

        progress_bar.progress((i + 1) / len(uploaded_files))

    if all_chunks:
        if st.session_state.vector_store.add_documents(all_chunks, metadata):
            st.session_state.documents.extend(all_chunks)
            st.sidebar.success(f"✅ Processed {len(all_chunks)} document chunks!")
        else:
            st.sidebar.error("❌ Failed to process documents")

    status_text.empty()
    progress_bar.empty()


def process_videos_admin(uploaded_videos):
    """Process uploaded video files"""
    processor = DocumentProcessor()
    all_chunks = []
    metadata = []

    progress_bar = st.sidebar.progress(0)
    status_text = st.sidebar.empty()

    for i, uploaded_video in enumerate(uploaded_videos):
        status_text.text(f"Processing {uploaded_video.name}...")

        # Extract text from video
        text = processor.extract_text_from_video(uploaded_video)

        if text:
            chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
            all_chunks.extend(chunks)

            for chunk in chunks:
                metadata.append({
                    'source_file': uploaded_video.name,
                    'source_type': 'video',
                    'chunk_length': len(chunk),
                    'processing_time': datetime.now().isoformat(),
                    'processed_by': st.session_state.username
                })

        progress_bar.progress((i + 1) / len(uploaded_videos))

    if all_chunks:
        if st.session_state.vector_store.add_documents(all_chunks, metadata):
            st.session_state.documents.extend(all_chunks)
            st.sidebar.success(f"✅ Processed {len(all_chunks)} video chunks!")
        else:
            st.sidebar.error("❌ Failed to process videos")

    status_text.empty()
    progress_bar.empty()


def process_youtube_admin(youtube_url):
    """FIXED: Process YouTube video with proper spinner usage and better error handling"""
    processor = DocumentProcessor()

    # **FIXED: Use with st.sidebar: and then st.spinner() inside**
    with st.sidebar:
        with st.spinner("Processing YouTube video..."):
            text = processor.extract_text_from_youtube(youtube_url)

            if text:
                chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
                metadata = []

                for chunk in chunks:
                    metadata.append({
                        'source_file': youtube_url,
                        'source_type': 'youtube',
                        'chunk_length': len(chunk),
                        'processing_time': datetime.now().isoformat(),
                        'processed_by': st.session_state.username
                    })

                if st.session_state.vector_store.add_documents(chunks, metadata):
                    st.session_state.documents.extend(chunks)
                    st.sidebar.success(f"✅ Processed {len(chunks)} YouTube chunks!")
                else:
                    st.sidebar.error("❌ Failed to process YouTube video")
            else:
                st.sidebar.error("❌ Could not extract text from YouTube video")


def chat_interface():
    """AERO chat interface - clean and focused"""
    # Simple, clean interface - no unnecessary headers

    # Initialize components
    if 'grounding_validator' not in st.session_state:
        st.session_state.grounding_validator = GroundingValidator(st.session_state.embeddings_model)

    if 'chatbot' not in st.session_state:
        st.session_state.chatbot = GroundedGeminiChatbot(
            st.session_state.api_key,
            st.session_state.grounding_validator
        )

    # Display chat messages with enhanced formatting for longer responses
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Enhanced markdown rendering for longer, detailed responses
            st.markdown(message["content"], unsafe_allow_html=True)

            if message["role"] == "assistant":
                # Show grounding information
                if "grounding_result" in message and message["grounding_result"]:
                    grounding = message["grounding_result"]

                    # Color code based on grounding quality
                    if grounding['confidence'] >= 0.8:
                        confidence_color = "🟢"
                    elif grounding['confidence'] >= 0.6:
                        confidence_color = "🟡"
                    else:
                        confidence_color = "🔴"

                    st.markdown(f"{confidence_color} **Grounding Confidence:** {grounding['confidence']:.2f}")

                    with st.expander("🔍 Grounding Details"):
                        st.write(f"**Well Grounded:** {'Yes' if grounding['is_grounded'] else 'No'}")
                        st.write(f"**Text Overlap:** {grounding['text_overlap']:.2f}")
                        st.write(f"**Semantic Similarity:** {grounding['semantic_similarity']:.2f}")
                        st.write(f"**Reason:** {grounding['reason']}")

                # Show sources if available
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"**Source {i + 1}** (Relevance: {source['relevance_score']:.2f})")
                            st.markdown(f"*Type:* {source['metadata'].get('source_type', 'Unknown')}")
                            st.markdown(f"*File:* {source['metadata'].get('source_file', 'Unknown')}")
                            st.code(
                                source['content'][:300] + "..." if len(source['content']) > 300 else source['content'])

    # Role-based chat input prompts
    prompts = {
        'student': "Ask detailed questions about your study materials...",
        'teacher': "Query educational content for comprehensive analysis...",
        'parent': "Ask about learning materials and educational content..."
    }

    # Chat input
    # Make sure prompts exists
    prompts = prompts if 'prompts' in locals() else {}

    # Chat input (placeholder pulled from prompts["user"] if available)
    if prompt := st.chat_input(prompts.get("user", "Ask me anything...")):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response using RAG service
        with st.chat_message("assistant"):
            with st.spinner("Generating comprehensive response..."):
                current_user = st.session_state.current_user

                response_data = st.session_state.rag_service.generate_response_with_logging(
                    prompt,
                    current_user,
                    st.session_state.session_id,
                    st.session_state.chatbot,
                    st.session_state.vector_store
                )

                response = response_data['response']
                grounding_result = response_data.get('grounding_result')
                is_fallback = response_data.get('is_fallback', False)
                sources = response_data.get('sources', [])
                is_cached = response_data.get('is_cached', False)

                # Show cache indicator
                if is_cached:
                    st.info("⚡ Cached response for faster performance")

                # Display response with enhanced formatting
                st.markdown(response, unsafe_allow_html=True)

                # Show grounding information using modular component
                from ui.components import render_grounding_info, render_sources_info
                render_grounding_info(grounding_result)

                # Show fallback warning if needed
                if is_fallback:
                    st.warning("⚠️ Fallback response used due to poor grounding")

                # Display sources using modular component
                if sources:
                    formatted_sources = []
                    for source in sources:
                        if isinstance(source, dict):
                            formatted_sources.append(source)
                        else:
                            formatted_sources.append({
                                'content': str(source),
                                'metadata': {'source_file': 'Unknown', 'source_type': 'document'},
                                'relevance_score': 0.5
                            })
                    render_sources_info(formatted_sources)

        # Add assistant message to chat history
        assistant_message = {
            "role": "assistant",
            "content": response,
            "grounding_result": grounding_result,
            "is_fallback": is_fallback
        }
        if sources:
            assistant_message["sources"] = sources
        st.session_state.messages.append(assistant_message)



def sidebar_controls():
    """Clean AERO sidebar controls"""
    # User info (only show if authenticated)
    if st.session_state.get("authenticated") and st.session_state.get("current_user"):
        user = st.session_state.current_user
        st.sidebar.markdown(f"""
        <div style="padding: 15px; background: #f8f9fa; border-radius: 8px; margin: 10px 0;">
            <strong>👤 {user.name}</strong><br>
            <small style="color: #666;">{user.role.value.title()}</small>
        </div>
        """, unsafe_allow_html=True)

    # Clean logout button
    st.sidebar.markdown("---")
    if st.sidebar.button("🚪 Logout", type="secondary"):
        # Clear persistent session
        if st.session_state.get('persistent_session_id'):
            session_service.delete_session(st.session_state.persistent_session_id)
            try:
                st.query_params.clear()
            except:
                # Fallback for older Streamlit versions
                st.experimental_set_query_params() if hasattr(st, 'experimental_set_query_params') else None
        
        # Clear session state directly
        for key in ['authenticated', 'username', 'name', 'user_role', 'authentication_status', 'current_user', 'persistent_session_id']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()


    # Clean model info (collapsed by default)
    with st.sidebar.expander("ℹ️ About AERO"):
        st.info("**AERO** - AI Educational Response Oracle\nIntelligent learning assistant powered by advanced AI.")

    # Simple session stats (optional)
    if st.session_state.get("authenticated") and st.session_state.messages:
        total_messages = len([m for m in st.session_state.messages if m["role"] == "user"])
        if total_messages > 0:
            with st.sidebar.expander(f"💬 {total_messages} questions asked"):
                st.write("✅ Session active")

    # Controls (only show if authenticated)
    if st.session_state.get("authenticated"):
        st.sidebar.header("🎮 Controls")

        col1, col2 = st.sidebar.columns(2)

        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()

        with col2:
            if st.button("💾 Export Chat"):
                export_chat()


def export_chat():
    """Export chat history with enhanced metadata"""
    if st.session_state.messages:
        chat_data = {
            "export_timestamp": datetime.now().isoformat(),
            "user_info": {
                "username": st.session_state.username,
                "name": st.session_state.name,
                "role": st.session_state.user_role
            },
            "messages": st.session_state.messages,
            "settings": {
                "grounding_threshold": st.session_state.grounding_threshold,
                "total_documents": len(st.session_state.documents),
                "max_tokens": 4096
            },
            "session_stats": {
                "total_messages": len(st.session_state.messages),
                "document_count": len(st.session_state.documents)
            }
        }

        json_str = json.dumps(chat_data, indent=2)

        st.sidebar.download_button(
            label="📥 Download Chat",
            data=json_str,
            file_name=f"chat_export_{st.session_state.username}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )


def main():
    """Main application logic with modular authentication"""
    
    # Initialize session state first
    initialize_session_state()
    
    # Check for persistent session before showing login
    check_persistent_session()
    
    # Create demo users on first run
    from ui.auth_page import create_demo_users
    create_demo_users(st.session_state.auth_service)

    # Apply role-based theme if authenticated
    if st.session_state.authenticated and st.session_state.current_user:
        apply_role_theme(st.session_state.current_user.role.value)

    # Check authentication - if not authenticated, show login and stop
    if not st.session_state.authenticated:
        from ui.auth_page import render_auth_page
        render_auth_page(st.session_state.auth_service)
        return

    # Initialize models
    if not initialize_models():
        st.error("Failed to initialize models. Please refresh the page.")
        return

    # Get API key
    if not get_api_key():
        return

    # Render role-specific header
    current_user = st.session_state.current_user
    render_role_header(current_user.role.value, current_user.name)
    
    # Role-specific main interface
    if current_user.role == UserRole.PARENT:
        # Parent dashboard with sidebar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            render_parent_dashboard(current_user, st.session_state.activity_service)
        
        with col2:
            render_user_info_sidebar(current_user)
            sidebar_controls()
    
    elif current_user.role == UserRole.TEACHER:
        # Teacher analytics dashboard
        col1, col2 = st.columns([3, 1])
        
        with col1:
            render_teacher_dashboard(current_user, st.session_state.activity_service)
        
        with col2:
            render_user_info_sidebar(current_user)
            sidebar_controls()
    
    elif current_user.role == UserRole.ADMIN:
        # Admin dashboard with full system management and sidebar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            from ui.admin_dashboard import render_admin_dashboard
            render_admin_dashboard(
                current_user,
                st.session_state.auth_service,
                st.session_state.activity_service
            )
        
        with col2:
            render_document_upload_section(current_user, st.session_state.document_service)
            render_user_info_sidebar(current_user)
            sidebar_controls()
    
    else:
        # Main chat interface for student
        col1, col2 = st.columns([3, 1])

        with col1:
            chat_interface()

        with col2:
            render_document_upload_section(current_user, st.session_state.document_service)
            render_user_info_sidebar(current_user)
            sidebar_controls()

    # Role-based footer
    role_footers = {
        'student': '📚 Enhanced Learning Experience',
        'teacher': '🎓 Educational Content Management',
        'parent': '👨‍👩‍👧‍👦 Family Learning Support'
    }

    # Clean footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #999; padding: 10px;'>
        <p>🚀 <strong>AERO</strong> - AI Educational Assistant</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
