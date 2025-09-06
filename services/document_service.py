# services/document_service.py
import os
import tempfile
import streamlit as st
from typing import List, Dict, Any
from datetime import datetime
from models.user import User, UserRole

# Import existing classes from main app
try:
    from DIGITAL_COMPANION_APP import DocumentProcessor, VideoProcessor, WhisperTranscriber
except ImportError:
    # Will import these when we refactor the main app
    pass

class DocumentService:
    """Service for handling document operations with admin-only uploads"""
    
    def __init__(self, db_service):
        self.db = db_service
        self.processor = None  # Will initialize when needed
    
    def _get_processor(self):
        """Lazy initialization of document processor"""
        if self.processor is None:
            # We'll use a simpler approach for now
            # Import here to avoid circular imports during development
            try:
                import sys
                import importlib
                main_app = importlib.import_module('DIGITAL_COMPANION_APP')
                self.processor = main_app.DocumentProcessor()
            except Exception as e:
                st.error(f"Error initializing document processor: {e}")
                return None
        return self.processor
    
    def process_documents_admin(self, uploaded_files: List, current_user: User, vector_store) -> bool:
        """Process uploaded documents (Admin only) - shared knowledge base"""
        if current_user.role != UserRole.ADMIN:
            st.error("⚠️ Only administrators can upload documents")
            return False
            
        processor = self._get_processor()
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

                # Store document info in database
                doc_id = self._store_document_info(
                    uploaded_file.name, 
                    uploaded_file.type,
                    current_user.id,
                    chunks
                )

                for chunk in chunks:
                    metadata.append({
                        'source_file': uploaded_file.name,
                        'source_type': 'document',
                        'chunk_length': len(chunk),
                        'processing_time': datetime.now().isoformat(),
                        'processed_by': current_user.username,
                        'document_id': doc_id
                    })

            progress_bar.progress((i + 1) / len(uploaded_files))

        success = False
        if all_chunks:
            if vector_store.add_documents(all_chunks, metadata):
                st.session_state.documents.extend(all_chunks)
                st.sidebar.success(f"✅ Processed {len(all_chunks)} document chunks!")
                success = True
            else:
                st.sidebar.error("❌ Failed to process documents")

        status_text.empty()
        progress_bar.empty()
        return success
    
    def process_videos_admin(self, uploaded_videos: List, current_user: User, vector_store) -> bool:
        """Process uploaded video files (Admin only)"""
        if current_user.role != UserRole.ADMIN:
            st.error("⚠️ Only administrators can upload videos")
            return False
            
        processor = self._get_processor()
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

                # Store document info in database
                doc_id = self._store_document_info(
                    uploaded_video.name, 
                    "video",
                    current_user.id,
                    chunks
                )

                for chunk in chunks:
                    metadata.append({
                        'source_file': uploaded_video.name,
                        'source_type': 'video',
                        'chunk_length': len(chunk),
                        'processing_time': datetime.now().isoformat(),
                        'processed_by': current_user.username,
                        'document_id': doc_id
                    })

            progress_bar.progress((i + 1) / len(uploaded_videos))

        success = False
        if all_chunks:
            if vector_store.add_documents(all_chunks, metadata):
                st.session_state.documents.extend(all_chunks)
                st.sidebar.success(f"✅ Processed {len(all_chunks)} video chunks!")
                success = True
            else:
                st.sidebar.error("❌ Failed to process videos")

        status_text.empty()
        progress_bar.empty()
        return success
    
    def process_youtube_admin(self, youtube_url: str, current_user: User, vector_store) -> bool:
        """Process YouTube video (Admin only)"""
        if current_user.role != UserRole.ADMIN:
            st.error("⚠️ Only administrators can process YouTube videos")
            return False
            
        processor = self._get_processor()

        with st.sidebar:
            with st.spinner("Processing YouTube video..."):
                text = processor.extract_text_from_youtube(youtube_url)

                if text:
                    chunks = processor.chunk_text(text, chunk_size=400, overlap=50)
                    
                    # Store document info in database
                    doc_id = self._store_document_info(
                        youtube_url, 
                        "youtube",
                        current_user.id,
                        chunks
                    )
                    
                    metadata = []
                    for chunk in chunks:
                        metadata.append({
                            'source_file': youtube_url,
                            'source_type': 'youtube',
                            'chunk_length': len(chunk),
                            'processing_time': datetime.now().isoformat(),
                            'processed_by': current_user.username,
                            'document_id': doc_id
                        })

                    if vector_store.add_documents(chunks, metadata):
                        st.session_state.documents.extend(chunks)
                        st.sidebar.success(f"✅ Processed {len(chunks)} YouTube chunks!")
                        return True
                    else:
                        st.sidebar.error("❌ Failed to process YouTube video")
                else:
                    st.sidebar.error("❌ Could not extract text from YouTube video")
        
        return False
    
    def _store_document_info(self, filename: str, doc_type: str, uploaded_by: str, chunks: List[str]) -> str:
        """Store document metadata in database"""
        import hashlib
        import uuid
        
        # Create content hash from all chunks
        content_hash = hashlib.sha256(''.join(chunks).encode()).hexdigest()
        doc_id = str(uuid.uuid4())
        
        # This would store in the database
        # For now, we'll implement this in the database service
        return doc_id
    
    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the shared knowledge base"""
        return {
            'total_chunks': len(st.session_state.documents),
            'total_documents': 0,  # TODO: Get from database
            'document_types': [],  # TODO: Get from database
            'last_updated': datetime.now().isoformat()
        }