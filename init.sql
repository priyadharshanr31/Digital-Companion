-- init.sql - PostgreSQL database initialization
-- This runs automatically when the PostgreSQL container starts

-- Create extensions for better performance
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create optimized users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(128) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'student', 'teacher', 'parent')),
    parent_ids UUID[],
    student_ids UUID[],
    class_ids UUID[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_login TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance (remove CONCURRENTLY for init script)
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
CREATE INDEX IF NOT EXISTS idx_users_created_at ON users(created_at);

-- User relationships table
CREATE TABLE IF NOT EXISTS user_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    parent_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    child_user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    relationship_type VARCHAR(20) NOT NULL DEFAULT 'parent',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE,
    UNIQUE(parent_user_id, child_user_id, relationship_type)
);

CREATE INDEX IF NOT EXISTS idx_relationships_parent ON user_relationships(parent_user_id);
CREATE INDEX IF NOT EXISTS idx_relationships_child ON user_relationships(child_user_id);

-- Documents table with full-text search
CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    filename VARCHAR(255) NOT NULL,
    content_hash VARCHAR(64) NOT NULL,
    content_text TEXT, -- Full document text for search
    -- content_vector VECTOR(384), -- Will be added with pgvector extension later
    document_type VARCHAR(50) NOT NULL,
    file_size INTEGER,
    upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    uploaded_by UUID NOT NULL REFERENCES users(id),
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE,
    search_vector tsvector -- Full-text search
);

-- Full-text search indexes
CREATE INDEX IF NOT EXISTS idx_documents_search ON documents USING GIN(search_vector);
CREATE INDEX IF NOT EXISTS idx_documents_uploaded_by ON documents(uploaded_by);
CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_upload_time ON documents(upload_timestamp);

-- Trigger to update search vector
CREATE OR REPLACE FUNCTION update_document_search_vector()
RETURNS TRIGGER AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', COALESCE(NEW.filename, '') || ' ' || COALESCE(NEW.content_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER documents_search_vector_update
    BEFORE INSERT OR UPDATE ON documents
    FOR EACH ROW EXECUTE FUNCTION update_document_search_vector();

-- Partitioned student activities table for high performance
CREATE TABLE IF NOT EXISTS student_activities (
    id UUID DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES users(id),
    session_id UUID NOT NULL,
    activity_type VARCHAR(20) NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    query_text TEXT,
    response_text TEXT,
    sources_used TEXT[],
    response_time_ms INTEGER,
    grounding_confidence REAL,
    detected_topics TEXT[],
    difficulty_level VARCHAR(20),
    session_duration_sec INTEGER,
    follow_up_questions INTEGER,
    satisfaction_rating INTEGER CHECK (satisfaction_rating BETWEEN 1 AND 5),
    ip_address INET,
    user_agent TEXT,
    metadata JSONB DEFAULT '{}'
) PARTITION BY RANGE (timestamp);

-- Create monthly partitions for student activities (better performance)
CREATE TABLE student_activities_2025_01 PARTITION OF student_activities
    FOR VALUES FROM ('2025-01-01') TO ('2025-02-01');
CREATE TABLE student_activities_2025_02 PARTITION OF student_activities
    FOR VALUES FROM ('2025-02-01') TO ('2025-03-01');
CREATE TABLE student_activities_2025_03 PARTITION OF student_activities
    FOR VALUES FROM ('2025-03-01') TO ('2025-04-01');

-- Indexes on partitioned table
CREATE INDEX IF NOT EXISTS idx_activities_student ON student_activities(student_id);
CREATE INDEX IF NOT EXISTS idx_activities_session ON student_activities(session_id);
CREATE INDEX IF NOT EXISTS idx_activities_timestamp ON student_activities(timestamp);
CREATE INDEX IF NOT EXISTS idx_activities_type ON student_activities(activity_type);

-- Learning sessions table
CREATE TABLE IF NOT EXISTS learning_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    student_id UUID NOT NULL REFERENCES users(id),
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    total_queries INTEGER DEFAULT 0,
    unique_topics TEXT[],
    average_confidence REAL,
    session_quality_score REAL,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_sessions_student ON learning_sessions(student_id);
CREATE INDEX IF NOT EXISTS idx_sessions_start_time ON learning_sessions(start_time);

-- High-performance response cache with TTL
CREATE TABLE IF NOT EXISTS response_cache (
    query_hash VARCHAR(64) PRIMARY KEY,
    query_text TEXT NOT NULL,
    response_data JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP DEFAULT (CURRENT_TIMESTAMP + INTERVAL '1 hour'),
    hit_count INTEGER DEFAULT 1,
    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_cache_expires ON response_cache(expires_at);
CREATE INDEX IF NOT EXISTS idx_cache_created ON response_cache(created_at);

-- Automatic cache cleanup function
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM response_cache WHERE expires_at < CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- Insert default admin user
INSERT INTO users (username, name, email, password_hash, role) 
VALUES (
    'admin',
    'System Administrator',
    'admin@digitalcompanion.com',
    'ef92b778bafe771e89245b89ecbc08a44a4e166c06659911881f383d4473e94f', -- SHA256 of 'admin123'
    'admin'
) ON CONFLICT (username) DO NOTHING;

-- Insert demo users
INSERT INTO users (username, name, email, password_hash, role) VALUES
    ('student1', 'Alice Johnson', 'alice@student.edu', 'ba3253876aed6bc22d4a6ff53d8406c6ad864195ed144ab5c87621b6c233b548baeae6956df346ec8c17f5ea10f35ee3cbc514797ed7ddd3145464e2a0bab413', 'student'),
    ('teacher1', 'Prof. Smith', 'smith@university.edu', 'cb8379ac2098aa165029e3938a51da0bcecfc008fd6795f401178647f96c5b34', 'teacher'),
    ('parent1', 'Mrs. Wilson', 'wilson@parent.com', 'b03ddf3ca2e714a6548e7495e2a03f5e824eaac9837cd7c5262cc730f2969847', 'parent')
ON CONFLICT (username) DO NOTHING;

-- Performance monitoring views
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT 
    u.username,
    u.role,
    COUNT(sa.id) as total_queries,
    AVG(sa.response_time_ms) as avg_response_time,
    MAX(sa.timestamp) as last_activity
FROM users u
LEFT JOIN student_activities sa ON u.id = sa.student_id
WHERE u.role = 'student'
GROUP BY u.id, u.username, u.role;

CREATE OR REPLACE VIEW system_performance AS
SELECT 
    COUNT(*) as total_users,
    COUNT(*) FILTER (WHERE role = 'student') as student_count,
    COUNT(*) FILTER (WHERE last_login > CURRENT_DATE - INTERVAL '7 days') as active_users_week,
    (SELECT COUNT(*) FROM student_activities WHERE timestamp > CURRENT_DATE) as queries_today,
    (SELECT AVG(response_time_ms) FROM student_activities WHERE timestamp > CURRENT_DATE - INTERVAL '1 hour') as avg_response_time_hour
FROM users;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dc_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dc_user;

-- Set up automatic vacuum and analyze for performance
ALTER TABLE users SET (autovacuum_vacuum_scale_factor = 0.1);
ALTER TABLE student_activities SET (autovacuum_vacuum_scale_factor = 0.05);
ALTER TABLE documents SET (autovacuum_vacuum_scale_factor = 0.1);