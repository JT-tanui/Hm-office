import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class ConversationDB:
    def __init__(self, db_path='conversations.db'):
        self.db_path = db_path
        self.init_db()

    def _connect(self) -> sqlite3.Connection:
        """Create a DB connection with foreign keys enforced."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        return conn
    
    def init_db(self):
        """Initialize database with tables"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                use_as_context INTEGER DEFAULT 1,
                summary TEXT DEFAULT NULL,
                folder_id TEXT DEFAULT NULL,
                profile_id TEXT DEFAULT 'default',
                user_id TEXT DEFAULT 'anonymous'
            )
        ''')
        
        # Migrate existing tables if needed
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'use_as_context' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN use_as_context INTEGER DEFAULT 1")
        
        if 'summary' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN summary TEXT DEFAULT NULL")
        
        if 'pinned' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN pinned INTEGER DEFAULT 0")
        
        if 'archived' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN archived INTEGER DEFAULT 0")
        if 'folder_id' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN folder_id TEXT DEFAULT NULL")
        if 'user_id' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT 'anonymous'")
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                created_at INTEGER NOT NULL,
                pinned INTEGER DEFAULT 0,
                user_id TEXT DEFAULT 'anonymous',
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        ''')
        cursor.execute("PRAGMA table_info(messages)")
        message_columns = [column[1] for column in cursor.fetchall()]
        if 'pinned' not in message_columns:
            cursor.execute("ALTER TABLE messages ADD COLUMN pinned INTEGER DEFAULT 0")
        if 'user_id' not in message_columns:
            cursor.execute("ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT 'anonymous'")
        cursor.execute("PRAGMA table_info(rag_chunks)")
        rag_cols = [column[1] for column in cursor.fetchall()]
        if 'embedding' not in rag_cols:
            try:
                cursor.execute("ALTER TABLE rag_chunks ADD COLUMN embedding TEXT")
            except Exception:
                pass
        
        # User Settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS user_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                voice_style TEXT DEFAULT '',
                voice_speed REAL DEFAULT 1.2,
                auto_speak INTEGER DEFAULT 1,
                wake_word_enabled INTEGER DEFAULT 0,
                updated_at INTEGER
            )
        ''')
        
        # Initialize default settings if not exists
        cursor.execute("INSERT OR IGNORE INTO user_settings (id, voice_style, voice_speed, auto_speak, wake_word_enabled, updated_at) VALUES (1, '', 1.2, 1, 0, ?)", (int(datetime.now().timestamp()),))
        
        # Tags table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS tags (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                color TEXT DEFAULT '#3B82F6',
                icon TEXT DEFAULT '🏷️',
                created_at INTEGER NOT NULL
            )
        ''')
        
        # Conversation-Tag mapping table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversation_tags (
                conversation_id TEXT NOT NULL,
                tag_id TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                PRIMARY KEY (conversation_id, tag_id),
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
                FOREIGN KEY (tag_id) REFERENCES tags(id) ON DELETE CASCADE
            )
        ''')
        
        # Voice Memos table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_memos (
                id TEXT PRIMARY KEY,
                title TEXT,
                audio_path TEXT,
                transcription TEXT,
                duration REAL DEFAULT 0,
                created_at INTEGER NOT NULL,
                tags TEXT DEFAULT '[]',
                profile_id TEXT DEFAULT 'default'
            )
        ''')
        
        # Profiles table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                avatar_path TEXT,
                created_at INTEGER NOT NULL,
                is_default INTEGER DEFAULT 0,
                user_id TEXT DEFAULT 'anonymous'
            )
        ''')

        # Voice Profiles (metadata only)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS voice_profiles (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                sample_path TEXT,
                provider TEXT DEFAULT 'supertonic',
                cloned INTEGER DEFAULT 0,
                created_at INTEGER NOT NULL
            )
        ''')

        # Profile Settings table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profile_settings (
                profile_id TEXT PRIMARY KEY,
                voice_style TEXT DEFAULT '',
                voice_speed REAL DEFAULT 1.2,
                auto_speak INTEGER DEFAULT 1,
                wake_word_enabled INTEGER DEFAULT 0,
                wake_word_sensitivity REAL DEFAULT 0.7,
                activation_sound_path TEXT DEFAULT NULL,
                updated_at INTEGER,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')

        # Wake Words table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS wake_words (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                word TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')

        # Integrations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrations (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                service TEXT NOT NULL,
                access_token TEXT,
                refresh_token TEXT,
                expires_at INTEGER,
                config TEXT DEFAULT '{}',
                created_at INTEGER NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')

        # Folders table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS folders (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                name TEXT NOT NULL,
                color TEXT DEFAULT '#64748b',
                icon TEXT DEFAULT '📁',
                created_at INTEGER NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')

        # Users table (email/passwordless)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                email TEXT UNIQUE NOT NULL,
                login_code TEXT,
                code_expires INTEGER,
                session_token TEXT,
                created_at INTEGER NOT NULL
            )
        ''')

        # RAG sources
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_sources (
                id TEXT PRIMARY KEY,
                profile_id TEXT NOT NULL,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                status TEXT DEFAULT 'ready',
                meta TEXT DEFAULT '{}',
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS rag_chunks (
                id TEXT PRIMARY KEY,
                source_id TEXT NOT NULL,
                profile_id TEXT NOT NULL,
                content TEXT NOT NULL,
                meta TEXT DEFAULT '{}',
                created_at INTEGER NOT NULL,
                FOREIGN KEY (source_id) REFERENCES rag_sources(id) ON DELETE CASCADE,
                FOREIGN KEY (profile_id) REFERENCES profiles(id) ON DELETE CASCADE
            )
        ''')
        
        # Ensure default profile exists
        cursor.execute("SELECT count(*) FROM profiles WHERE id = 'default'")
        if cursor.fetchone()[0] == 0:
            cursor.execute("INSERT INTO profiles (id, name, created_at, is_default) VALUES ('default', 'Default User', ?, 1)", (int(datetime.now().timestamp()),))
            # Copy existing settings to default profile if they exist
            cursor.execute("INSERT OR IGNORE INTO profile_settings (profile_id, voice_style, voice_speed, auto_speak, wake_word_enabled, updated_at) SELECT 'default', voice_style, voice_speed, auto_speak, wake_word_enabled, updated_at FROM user_settings WHERE id = 1")

        # Migrate existing tables
        cursor.execute("PRAGMA table_info(conversations)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'profile_id' not in columns:
            cursor.execute("ALTER TABLE conversations ADD COLUMN profile_id TEXT DEFAULT 'default'")
            
        cursor.execute("PRAGMA table_info(voice_memos)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'profile_id' not in columns:
            cursor.execute("ALTER TABLE voice_memos ADD COLUMN profile_id TEXT DEFAULT 'default'")

        # Add user_id to profiles if missing
        cursor.execute("PRAGMA table_info(profiles)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'user_id' not in columns:
            cursor.execute("ALTER TABLE profiles ADD COLUMN user_id TEXT DEFAULT 'anonymous'")

        cursor.execute("PRAGMA table_info(profile_settings)")
        columns = [column[1] for column in cursor.fetchall()]
        if 'wake_word_sensitivity' not in columns:
            cursor.execute("ALTER TABLE profile_settings ADD COLUMN wake_word_sensitivity REAL DEFAULT 0.7")
        if 'activation_sound_path' not in columns:
            cursor.execute("ALTER TABLE profile_settings ADD COLUMN activation_sound_path TEXT DEFAULT NULL")

        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memos_created ON voice_memos(created_at DESC)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_profile ON conversations(profile_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_user ON messages(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memos_profile ON voice_memos(profile_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_folder ON conversations(folder_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_profiles_user ON profiles(user_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_chunks_source ON rag_chunks(source_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_rag_chunks_profile ON rag_chunks(profile_id)')
        
        # Create FTS5 virtual table for full-text search
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
                content,
                conversation_id UNINDEXED,
                role UNINDEXED,
                content='messages',
                content_rowid='rowid'
            )
        ''')
        
        # Create triggers to keep FTS table in sync
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
                INSERT INTO messages_fts(rowid, content, conversation_id, role)
                VALUES (new.rowid, new.content, new.conversation_id, new.role);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.rowid;
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
                DELETE FROM messages_fts WHERE rowid = old.rowid;
                INSERT INTO messages_fts(rowid, content, conversation_id, role)
                VALUES (new.rowid, new.content, new.conversation_id, new.role);
            END
        ''')
        
        # Create FTS5 for voice memos
        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS memos_fts USING fts5(
                title,
                transcription,
                content='voice_memos',
                content_rowid='rowid'
            )
        ''')

        cursor.execute('''
            CREATE VIRTUAL TABLE IF NOT EXISTS rag_chunks_fts USING fts5(
                content,
                source_id UNINDEXED,
                profile_id UNINDEXED,
                content='rag_chunks',
                content_rowid='rowid'
            )
        ''')
        
        # Triggers for memos FTS
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS memos_ai AFTER INSERT ON voice_memos BEGIN
                INSERT INTO memos_fts(rowid, title, transcription)
                VALUES (new.rowid, new.title, new.transcription);
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS memos_ad AFTER DELETE ON voice_memos BEGIN
                DELETE FROM memos_fts WHERE rowid = old.rowid;
            END
        ''')
        
        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS memos_au AFTER UPDATE ON voice_memos BEGIN
                DELETE FROM memos_fts WHERE rowid = old.rowid;
                INSERT INTO memos_fts(rowid, title, transcription)
                VALUES (new.rowid, new.title, new.transcription);
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS rag_chunks_ai AFTER INSERT ON rag_chunks BEGIN
                INSERT INTO rag_chunks_fts(rowid, content, source_id, profile_id)
                VALUES (new.rowid, new.content, new.source_id, new.profile_id);
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS rag_chunks_ad AFTER DELETE ON rag_chunks BEGIN
                DELETE FROM rag_chunks_fts WHERE rowid = old.rowid;
            END
        ''')

        cursor.execute('''
            CREATE TRIGGER IF NOT EXISTS rag_chunks_au AFTER UPDATE ON rag_chunks BEGIN
                DELETE FROM rag_chunks_fts WHERE rowid = old.rowid;
                INSERT INTO rag_chunks_fts(rowid, content, source_id, profile_id)
                VALUES (new.rowid, new.content, new.source_id, new.profile_id);
            END
        ''')
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, conversation_id: str, title: str = "New Chat", profile_id: str = "default", folder_id: Optional[str] = None, user_id: str = "anonymous") -> Dict:
        """Create a new conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        cursor.execute(
            'INSERT INTO conversations (id, title, created_at, updated_at, profile_id, folder_id, user_id) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (conversation_id, title, now, now, profile_id, folder_id, user_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            'id': conversation_id,
            'title': title,
            'created_at': now,
            'updated_at': now,
            'profile_id': profile_id,
            'folder_id': folder_id,
            'user_id': user_id
        }
    
    def get_conversations(self, profile_id: str = "default", folder_id: Optional[str] = None, user_id: Optional[str] = None) -> List[Dict]:
        """Get all conversations for a profile (and user) with message counts"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = '''
            SELECT c.*, COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            WHERE c.profile_id = ?
        '''
        params: List = [profile_id]
        if user_id:
            query += ' AND c.user_id = ?'
            params.append(user_id)
        if folder_id:
            query += ' AND c.folder_id = ?'
            params.append(folder_id)

        query += ' GROUP BY c.id ORDER BY c.pinned DESC, c.updated_at DESC'
        cursor.execute(query, tuple(params))
        
        results = [dict(row) for row in cursor.fetchall()]
        
        # Parse tags as JSON
        for conv in results:
            if conv.get('tags'):
                conv['tags'] = json.loads(conv['tags'])
            else:
                conv['tags'] = []
        
        conn.close()
        return results
    
    def create_conversation_with_title(self, title: str = "New Conversation", profile_id: str = "default") -> str:
        """Legacy helper to create a conversation with generated ID"""
        conversation_id = str(int(datetime.now().timestamp() * 1000))
        self.create_conversation(conversation_id, title, profile_id)
        return conversation_id
    
    def get_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """Get a single conversation with its messages"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get conversation
        if user_id:
            cursor.execute('SELECT * FROM conversations WHERE id = ? AND user_id = ?', (conversation_id, user_id))
        else:
            cursor.execute('SELECT * FROM conversations WHERE id = ?', (conversation_id,))
        conv_row = cursor.fetchone()
        
        if not conv_row:
            conn.close()
            return None
        
        conversation = dict(conv_row)
        
        # Get messages
        cursor.execute(
            'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        messages = [dict(row) for row in cursor.fetchall()]
        
        conversation['messages'] = messages
        
        conn.close()
        return conversation
    
    def update_conversation(self, conversation_id: str, title: str = None, user_id: Optional[str] = None) -> bool:
        """Update conversation title"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        
        if title:
            if user_id:
                cursor.execute(
                    'UPDATE conversations SET title = ?, updated_at = ? WHERE id = ? AND user_id = ?',
                    (title, now, conversation_id, user_id)
                )
            else:
                cursor.execute(
                    'UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?',
                    (title, now, conversation_id)
                )
        else:
            if user_id:
                cursor.execute(
                    'UPDATE conversations SET updated_at = ? WHERE id = ? AND user_id = ?',
                    (now, conversation_id, user_id)
                )
            else:
                cursor.execute(
                    'UPDATE conversations SET updated_at = ? WHERE id = ?',
                    (now, conversation_id)
                )
        
        success = cursor.rowcount > 0
        conn.commit()
        conn.close()
        
        return success
    
    def delete_conversation(self, conversation_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a conversation and all its messages"""
        conn = self._connect()
        cursor = conn.cursor()
        
        if user_id:
            cursor.execute('DELETE FROM conversations WHERE id = ? AND user_id = ?', (conversation_id, user_id))
        else:
            cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success
    
    def add_message(self, message_id: str, conversation_id: str, role: str, content: str, model: str = None, pinned: bool = False, user_id: Optional[str] = None) -> Dict:
        """Add a message to a conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)

        if user_id:
            cursor.execute('SELECT 1 FROM conversations WHERE id = ? AND user_id = ?', (conversation_id, user_id))
            if not cursor.fetchone():
                conn.close()
                raise ValueError("Conversation not found for user")
        
        cursor.execute(
            'INSERT INTO messages (id, conversation_id, role, content, model, created_at, pinned, user_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (message_id, conversation_id, role, content, model, now, 1 if pinned else 0, user_id or 'anonymous')
        )
        
        # Update conversation updated_at
        cursor.execute(
            'UPDATE conversations SET updated_at = ? WHERE id = ?',
            (now, conversation_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            'id': message_id,
            'conversation_id': conversation_id,
            'role': role,
            'content': content,
            'model': model,
            'created_at': now
        }
    
    def get_messages(self, conversation_id: str) -> List[Dict]:
        """Get all messages for a conversation"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        
        messages = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return messages

    def pin_message(self, message_id: str, pinned: bool = True) -> bool:
        """Toggle pin on a message"""
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute(
            'UPDATE messages SET pinned = ? WHERE id = ?',
            (1 if pinned else 0, message_id)
        )
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated
    
    def toggle_context_usage(self, conversation_id: str, enabled: bool) -> bool:
        """Toggle whether a conversation is used as context"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations
            SET use_as_context = ?
            WHERE id = ?
        ''', (1 if enabled else 0, conversation_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success

    # ================= FOLDERS =================

    def create_folder(self, profile_id: str, name: str, color: str = "#64748b", icon: str = "📁") -> Dict:
        conn = self._connect()
        cursor = conn.cursor()
        folder_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        cursor.execute(
            'INSERT INTO folders (id, profile_id, name, color, icon, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            (folder_id, profile_id, name, color, icon, created_at)
        )
        conn.commit()
        conn.close()
        return {
            "id": folder_id,
            "profile_id": profile_id,
            "name": name,
            "color": color,
            "icon": icon,
            "created_at": created_at
        }

    def get_folders(self, profile_id: str) -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM folders WHERE profile_id = ? ORDER BY name ASC', (profile_id,))
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def delete_folder(self, folder_id: str, profile_id: str) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('UPDATE conversations SET folder_id = NULL WHERE folder_id = ?', (folder_id,))
        cursor.execute('DELETE FROM folders WHERE id = ? AND profile_id = ?', (folder_id, profile_id))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def set_conversation_folder(self, conversation_id: str, folder_id: Optional[str]) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('UPDATE conversations SET folder_id = ? WHERE id = ?', (folder_id, conversation_id))
        updated = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return updated

    # ================= AUTH =================
    def request_login_code(self, email: str, code: str, expires_at: int):
        conn = self._connect()
        cursor = conn.cursor()
        user_id = str(int(datetime.now().timestamp() * 1000))
        cursor.execute('INSERT OR IGNORE INTO users (id, email, created_at) VALUES (?, ?, ?)', (user_id, email, int(datetime.now().timestamp())))
        cursor.execute('UPDATE users SET login_code = ?, code_expires = ? WHERE email = ?', (code, expires_at, email))
        conn.commit()
        conn.close()

    def verify_login_code(self, email: str, code: str) -> Optional[str]:
        conn = self._connect()
        cursor = conn.cursor()
        now = int(datetime.now().timestamp())
        cursor.execute('SELECT id, login_code, code_expires FROM users WHERE email = ?', (email,))
        row = cursor.fetchone()
        if not row:
            conn.close()
            return None
        user_id, stored_code, expires = row
        if stored_code != code or (expires and expires < now):
            conn.close()
            return None
        session = str(int(datetime.now().timestamp() * 1000))
        cursor.execute('UPDATE users SET session_token = ?, login_code = NULL, code_expires = NULL WHERE id = ?', (session, user_id))
        conn.commit()
        conn.close()
        return session

    def get_user_by_session(self, session_token: str) -> Optional[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE session_token = ?', (session_token,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    # ================= RAG SOURCES =================

    def list_rag_sources(self, profile_id: str) -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM rag_sources WHERE profile_id = ? ORDER BY created_at DESC', (profile_id,))
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def upsert_rag_source(self, source_id: str, profile_id: str, name: str, type_: str, status: str = "ready", meta: Optional[Dict] = None) -> Dict:
        conn = self._connect()
        cursor = conn.cursor()
        now = int(datetime.now().timestamp())
        cursor.execute('SELECT id FROM rag_sources WHERE id = ?', (source_id,))
        meta_json = json.dumps(meta or {})
        if cursor.fetchone():
            cursor.execute('UPDATE rag_sources SET name = ?, type = ?, status = ?, meta = ?, updated_at = ? WHERE id = ?', (name, type_, status, meta_json, now, source_id))
        else:
            cursor.execute('INSERT INTO rag_sources (id, profile_id, name, type, status, meta, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)', (source_id, profile_id, name, type_, status, meta_json, now, now))
        conn.commit()
        conn.close()
        return {"id": source_id, "profile_id": profile_id, "name": name, "type": type_, "status": status, "meta": meta or {}, "created_at": now, "updated_at": now}

    def delete_rag_source(self, source_id: str, profile_id: Optional[str] = None) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        if profile_id:
            cursor.execute('DELETE FROM rag_sources WHERE id = ? AND profile_id = ?', (source_id, profile_id))
        else:
            cursor.execute('DELETE FROM rag_sources WHERE id = ?', (source_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

    def add_rag_chunk(self, source_id: str, profile_id: str, content: str, meta: Optional[Dict] = None, embedding: Optional[List[float]] = None):
        conn = self._connect()
        cursor = conn.cursor()
        chunk_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        cursor.execute('INSERT INTO rag_chunks (id, source_id, profile_id, content, meta, created_at, embedding) VALUES (?, ?, ?, ?, ?, ?, ?)',
                       (chunk_id, source_id, profile_id, content, json.dumps(meta or {}), created_at, json.dumps(embedding) if embedding else None))
        conn.commit()
        conn.close()

    def clear_rag_chunks(self, source_id: str):
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM rag_chunks WHERE source_id = ?', (source_id,))
        conn.commit()
        conn.close()

    def search_rag(self, query: str, profile_id: str, limit: int = 5) -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.id, c.source_id, c.content, snippet(rag_chunks_fts, 1, '<b>', '</b>', '...', 32) AS snippet
            FROM rag_chunks_fts
            JOIN rag_chunks c ON c.rowid = rag_chunks_fts.rowid
            WHERE rag_chunks_fts MATCH ? AND c.profile_id = ?
            LIMIT ?
        ''', (query, profile_id, limit))
        results = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return results

    def get_recent_rag_chunks(self, profile_id: str, limit: int = 400) -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM rag_chunks WHERE profile_id = ? ORDER BY created_at DESC LIMIT ?', (profile_id, limit))
        results = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return results
    
    def update_summary(self, conversation_id: str, summary: str) -> bool:
        """Update conversation summary"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE conversations
            SET summary = ?, updated_at = ?
            WHERE id = ?
        ''', (summary, int(datetime.now().timestamp() * 1000), conversation_id))
        
        conn.commit()
        success = cursor.rowcount > 0
        conn.close()
        return success
    
    def get_context_conversations(self, exclude_id: Optional[str] = None) -> List[Dict]:
        """Get all conversations that should be used as context"""
        conn = self._connect()
        cursor = conn.cursor()
        
        if exclude_id:
            cursor.execute('''
                SELECT id, title, summary
                FROM conversations
                WHERE use_as_context = 1 AND id != ?
                ORDER BY updated_at DESC
            ''', (exclude_id,))
        else:
            cursor.execute('''
                SELECT id, title, summary
                FROM conversations
                WHERE use_as_context = 1
                ORDER BY updated_at DESC
            ''')
        
        conversations = []
        for row in cursor.fetchall():
            conversations.append({
                'id': row[0],
                'title': row[1],
                'summary': row[2]
            })
        
        conn.close()
        return conversations
    
    def get_all_context_messages(self, exclude_id: Optional[str] = None, limit: int = 50, profile_id: str = "default") -> List[Dict]:
        """Get messages from context-enabled conversations for cross-conversation context"""
        conn = self._connect()
        cursor = conn.cursor()
        
        if exclude_id:
            cursor.execute('''
                SELECT m.role, m.content, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.use_as_context = 1 AND c.id != ? AND c.profile_id = ?
                ORDER BY m.created_at DESC
                LIMIT ?
            ''', (exclude_id, profile_id, limit))
        else:
            cursor.execute('''
                SELECT m.role, m.content, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.use_as_context = 1 AND c.profile_id = ?
                ORDER BY m.created_at DESC
                LIMIT ?
            ''', (profile_id, limit))
        
        messages = []
        for row in cursor.fetchall():
            messages.append({
                'role': row[0],
                'content': row[1],
                'timestamp': row[2],
                'conversation_title': row[3]
            })
        
        conn.close()
        # Return in chronological order
        return list(reversed(messages))

    # ================= SETTINGS =================
    
    def get_settings(self) -> Dict:
        """Get user settings"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM user_settings WHERE id = 1")
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return {
            "voice_style": "",
            "voice_speed": 1.2,
            "auto_speak": 1,
            "wake_word_enabled": 0
        }

    def update_settings(self, settings: Dict) -> bool:
        """Update user settings"""
        conn = self._connect()
        cursor = conn.cursor()
        
        fields = []
        values = []
        
        if 'voice_style' in settings:
            fields.append("voice_style = ?")
            values.append(settings['voice_style'])
            
        if 'voice_speed' in settings:
            fields.append("voice_speed = ?")
            values.append(float(settings['voice_speed']))
            
        if 'auto_speak' in settings:
            fields.append("auto_speak = ?")
            values.append(1 if settings['auto_speak'] else 0)
            
        if 'wake_word_enabled' in settings:
            fields.append("wake_word_enabled = ?")
            values.append(1 if settings['wake_word_enabled'] else 0)
            
        if not fields:
            conn.close()
            return False
            
        fields.append("updated_at = ?")
        values.append(int(datetime.now().timestamp()))
        values.append(1) # WHERE id = 1
        
        query = f"UPDATE user_settings SET {', '.join(fields)} WHERE id = ?"
        
        try:
            cursor.execute(query, tuple(values))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating settings: {e}")
            conn.close()
            return False
    
    def search_messages(self, query: str, limit: int = 50, profile_id: str = "default") -> List[Dict]:
        """Search messages using FTS5 full-text search"""
        if not query or not query.strip():
            return []
        
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Use FTS5 MATCH for full-text search
        cursor.execute('''
            SELECT 
                m.id,
                m.conversation_id,
                m.role,
                m.content,
                m.model,
                m.created_at,
                c.title as conversation_title,
                c.updated_at as conversation_updated_at,
                snippet(messages_fts, 0, '<mark>', '</mark>', '...', 40) as highlighted_content
            FROM messages_fts
            JOIN messages m ON messages_fts.rowid = m.rowid
            JOIN conversations c ON m.conversation_id = c.id
            WHERE messages_fts MATCH ? AND c.profile_id = ?
            ORDER BY m.created_at DESC
            LIMIT ?
        ''', (query, profile_id, limit))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'conversation_id': row['conversation_id'],
                'conversation_title': row['conversation_title'],
                'role': row['role'],
                'content': row['content'],
                'highlighted_content': row['highlighted_content'],
                'model': row['model'],
                'created_at': row['created_at'],
                'conversation_updated_at': row['conversation_updated_at']
            })
        
        conn.close()
        return results
    
    def filter_conversations(self, start_date: Optional[int] = None, end_date: Optional[int] = None, model: Optional[str] = None, profile_id: str = "default", folder_id: Optional[str] = None) -> List[Dict]:
        """Filter conversations by date range and/or model"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.use_as_context
            FROM conversations c
        '''
        
        conditions = ['c.profile_id = ?']
        params = [profile_id]
        
        if model:
            query += ' JOIN messages m ON c.id = m.conversation_id'
            conditions.append('m.model = ?')
            params.append(model)
        
        if start_date:
            conditions.append('c.created_at >= ?')
            params.append(start_date)
        
        if end_date:
            conditions.append('c.created_at <= ?')
            params.append(end_date)

        if folder_id:
            conditions.append('c.folder_id = ?')
            params.append(folder_id)
        
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
        
        query += ' ORDER BY c.updated_at DESC'
        
        cursor.execute(query, tuple(params))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'use_as_context': bool(row['use_as_context'])
            })
        
        conn.close()
        return results

    def get_context_audit(self, profile_id: str = "default") -> List[Dict]:
        """Return list of conversations used as context for a profile with message counts"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
            SELECT c.id, c.title, c.updated_at, c.created_at, c.summary,
                   COUNT(m.id) AS message_count
            FROM conversations c
            LEFT JOIN messages m ON m.conversation_id = c.id
            WHERE c.profile_id = ? AND c.use_as_context = 1
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''', (profile_id,))
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return data
    
    # Tag Management Methods
    def create_tag(self, tag_id: str, name: str, color: str = '#3B82F6', icon: str = '🏷️') -> Dict:
        """Create a new tag"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        try:
            cursor.execute(
                'INSERT INTO tags (id, name, color, icon, created_at) VALUES (?, ?, ?, ?, ?)',
                (tag_id, name, color, icon, now)
            )
            conn.commit()
            conn.close()
            return {'id': tag_id, 'name': name, 'color': color, 'icon': icon, 'created_at': now}
        except sqlite3.IntegrityError:
            conn.close()
            return None
    
    def get_tags(self) -> List[Dict]:
        """Get all tags with usage counts"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT t.id, t.name, t.color, t.icon, t.created_at,
                   COUNT(ct.conversation_id) as usage_count
            FROM tags t
            LEFT JOIN conversation_tags ct ON t.id = ct.tag_id
            GROUP BY t.id
            ORDER BY t.name ASC
        ''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'name': row['name'],
                'color': row['color'],
                'icon': row['icon'],
                'created_at': row['created_at'],
                'usage_count': row['usage_count']
            })
        
        conn.close()
        return results
    
    def delete_tag(self, tag_id: str) -> bool:
        """Delete a tag"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tags WHERE id = ?', (tag_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def add_tag_to_conversation(self, conversation_id: str, tag_id: str) -> bool:
        """Add a tag to a conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        try:
            cursor.execute(
                'INSERT INTO conversation_tags (conversation_id, tag_id, created_at) VALUES (?, ?, ?)',
                (conversation_id, tag_id, now)
            )
            conn.commit()
            conn.close()
            return True
        except sqlite3.IntegrityError:
            conn.close()
            return False
    
    def remove_tag_from_conversation(self, conversation_id: str, tag_id: str) -> bool:
        """Remove a tag from a conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            'DELETE FROM conversation_tags WHERE conversation_id = ? AND tag_id = ?',
            (conversation_id, tag_id)
        )
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def get_conversation_tags(self, conversation_id: str) -> List[Dict]:
        """Get all tags for a conversation"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT t.id, t.name, t.color, t.icon
            FROM tags t
            JOIN conversation_tags ct ON t.id = ct.tag_id
            WHERE ct.conversation_id = ?
            ORDER BY t.name ASC
        ''', (conversation_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'name': row['name'],
                'color': row['color'],
                'icon': row['icon']
            })
        
        conn.close()
        return results
    
    def get_conversations_by_tag(self, tag_id: str) -> List[Dict]:
        """Get all conversations with a specific tag"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.title, c.created_at, c.updated_at, c.pinned, c.archived
            FROM conversations c
            JOIN conversation_tags ct ON c.id = ct.conversation_id
            WHERE ct.tag_id = ?
            ORDER BY c.updated_at DESC
        ''', (tag_id,))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                'id': row['id'],
                'title': row['title'],
                'created_at': row['created_at'],
                'updated_at': row['updated_at'],
                'pinned': bool(row['pinned']),
                'archived': bool(row['archived'])
            })
        
        conn.close()
        return results
    
    def pin_conversation(self, conversation_id: str, pinned: bool = True) -> bool:
        """Pin or unpin a conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE conversations SET pinned = ? WHERE id = ?',
            (1 if pinned else 0, conversation_id)
        )
        updated = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return updated
    
    def archive_conversation(self, conversation_id: str, archived: bool = True) -> bool:
        """Archive or unarchive a conversation"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute(
            'UPDATE conversations SET archived = ? WHERE id = ?',
            (1 if archived else 0, conversation_id)
        )
        updated = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return updated

    def create_voice_memo(self, title: str, audio_path: str, transcription: str = "", duration: float = 0, tags: List[str] = None, profile_id: str = "default") -> str:
        """Create a new voice memo"""
        conn = self._connect()
        cursor = conn.cursor()
        
        memo_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        tags_json = json.dumps(tags or [])
        
        cursor.execute(
            'INSERT INTO voice_memos (id, title, audio_path, transcription, duration, created_at, tags, profile_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (memo_id, title, audio_path, transcription, duration, created_at, tags_json, profile_id)
        )
        
        conn.commit()
        conn.close()
        return memo_id

    def get_voice_memos(self, profile_id: str = "default", limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get recent voice memos for a profile"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM voice_memos WHERE profile_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?',
            (profile_id, limit, offset)
        )
        
        memos = []
        for row in cursor.fetchall():
            memo = dict(row)
            memo['tags'] = json.loads(memo['tags']) if memo['tags'] else []
            memos.append(memo)
            
        conn.close()
        return memos

    def get_voice_memo(self, memo_id: str) -> Optional[Dict]:
        """Get a specific voice memo"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM voice_memos WHERE id = ?', (memo_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            memo = dict(row)
            memo['tags'] = json.loads(memo['tags']) if memo['tags'] else []
            return memo
        return None

    def delete_voice_memo(self, memo_id: str) -> bool:
        """Delete a voice memo"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM voice_memos WHERE id = ?', (memo_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted

    def search_voice_memos(self, query: str, limit: int = 20, profile_id: str = "default") -> List[Dict]:
        """Search voice memos using FTS5"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search in FTS table and join with main table
        cursor.execute('''
            SELECT m.*, snippet(memos_fts, 1, '<b>', '</b>', '...', 64) as snippet
            FROM voice_memos m
            JOIN memos_fts fts ON m.rowid = fts.rowid
            WHERE fts MATCH ? AND m.profile_id = ?
            ORDER BY m.created_at DESC
            LIMIT ?
        ''', (query, profile_id, limit))
        
        results = []
        for row in cursor.fetchall():
            memo = dict(row)
            memo['tags'] = json.loads(memo['tags']) if memo['tags'] else []
            results.append(memo)
            
        conn.close()
        return results

    # ================= PROFILE SETTINGS =================

    def get_profile_settings(self, profile_id: str) -> Dict:
        """Get settings for a profile"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM profile_settings WHERE profile_id = ?", (profile_id,))
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
            
        # Return defaults if not found
        return {
            "voice_style": "",
            "voice_speed": 1.2,
            "auto_speak": 1,
            "wake_word_enabled": 0,
            "wake_word_sensitivity": 0.7,
            "activation_sound_path": None
        }

    def update_profile_settings(self, profile_id: str, settings: Dict) -> bool:
        """Update profile settings"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Check if settings exist
        cursor.execute("SELECT 1 FROM profile_settings WHERE profile_id = ?", (profile_id,))
        exists = cursor.fetchone() is not None
        
        now = int(datetime.now().timestamp())
        
        if not exists:
            # Create default settings first
            cursor.execute(
                "INSERT INTO profile_settings (profile_id, updated_at) VALUES (?, ?)",
                (profile_id, now)
            )
        
        fields = []
        values = []
        
        if 'voice_style' in settings:
            fields.append("voice_style = ?")
            values.append(settings['voice_style'])
            
        if 'voice_speed' in settings:
            fields.append("voice_speed = ?")
            values.append(float(settings['voice_speed']))
            
        if 'auto_speak' in settings:
            fields.append("auto_speak = ?")
            values.append(1 if settings['auto_speak'] else 0)
            
        if 'wake_word_enabled' in settings:
            fields.append("wake_word_enabled = ?")
            values.append(1 if settings['wake_word_enabled'] else 0)

        if 'wake_word_sensitivity' in settings:
            fields.append("wake_word_sensitivity = ?")
            values.append(float(settings['wake_word_sensitivity']))

        if 'activation_sound_path' in settings:
            fields.append("activation_sound_path = ?")
            values.append(settings['activation_sound_path'])
            
        if not fields:
            conn.close()
            return False
            
        fields.append("updated_at = ?")
        values.append(now)
        values.append(profile_id)
        
        query = f"UPDATE profile_settings SET {', '.join(fields)} WHERE profile_id = ?"
        
        try:
            cursor.execute(query, tuple(values))
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            print(f"Error updating profile settings: {e}")
            conn.close()
            return False

    # ================= WAKE WORDS =================

    def add_wake_word(self, profile_id: str, word: str) -> Dict:
        """Add a new wake word for a profile"""
        conn = self._connect()
        cursor = conn.cursor()
        
        ww_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        
        cursor.execute(
            'INSERT INTO wake_words (id, profile_id, word, created_at) VALUES (?, ?, ?, ?)',
            (ww_id, profile_id, word, created_at)
        )
        
        conn.commit()
        conn.close()
        
        return {
            'id': ww_id,
            'profile_id': profile_id,
            'word': word,
            'created_at': created_at
        }

    def get_wake_words(self, profile_id: str) -> List[Dict]:
        """Get all wake words for a profile"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM wake_words WHERE profile_id = ? ORDER BY created_at DESC',
            (profile_id,)
        )
        
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results

    def delete_wake_word(self, wake_word_id: str) -> bool:
        """Delete a wake word"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM wake_words WHERE id = ?', (wake_word_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted

    # ================= PROFILES =================

    def get_profiles(self, user_id: Optional[str] = None) -> List[Dict]:
        """Get all profiles for a user"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if user_id:
            cursor.execute('SELECT * FROM profiles WHERE user_id = ? ORDER BY created_at ASC', (user_id,))
        else:
            cursor.execute('SELECT * FROM profiles ORDER BY created_at ASC')
        results = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return results

    def get_profile(self, profile_id: str, user_id: Optional[str] = None) -> Optional[Dict]:
        """Get a specific profile, optionally scoped to a user"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        if user_id:
            cursor.execute('SELECT * FROM profiles WHERE id = ? AND user_id = ?', (profile_id, user_id))
        else:
            cursor.execute('SELECT * FROM profiles WHERE id = ?', (profile_id,))
        row = cursor.fetchone()
        
        conn.close()
        return dict(row) if row else None

    def create_profile(self, name: str, avatar_path: str = "", user_id: str = "anonymous") -> str:
        """Create a new profile for a user"""
        conn = self._connect()
        cursor = conn.cursor()
        
        profile_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        
        cursor.execute(
            'INSERT INTO profiles (id, name, avatar_path, created_at, is_default, user_id) VALUES (?, ?, ?, ?, ?, ?)',
            (profile_id, name, avatar_path, created_at, 0, user_id)
        )
        
        # Create default settings for this profile
        cursor.execute(
            'INSERT INTO profile_settings (profile_id) VALUES (?)',
            (profile_id,)
        )
        
        conn.commit()
        conn.close()
        return profile_id

    def delete_profile(self, profile_id: str, user_id: Optional[str] = None) -> bool:
        """Delete a profile"""
        conn = self._connect()
        cursor = conn.cursor()

        if user_id:
            cursor.execute('DELETE FROM profiles WHERE id = ? AND user_id = ?', (profile_id, user_id))
        else:
            cursor.execute('DELETE FROM profiles WHERE id = ?', (profile_id,))
        deleted = cursor.rowcount > 0

        conn.commit()
        conn.close()
        return deleted

    def ensure_user_default_profile(self, user_id: str, email: str) -> str:
        """Ensure a default profile exists for a given user; return its id"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id FROM profiles WHERE user_id = ? ORDER BY created_at ASC LIMIT 1', (user_id,))
        row = cursor.fetchone()
        if row:
            conn.close()
            return row['id']

        profile_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        cursor.execute(
            'INSERT INTO profiles (id, name, avatar_path, created_at, is_default, user_id) VALUES (?, ?, ?, ?, ?, ?)',
            (profile_id, email.split("@")[0] if "@" in email else email, "", created_at, 1, user_id)
        )
        cursor.execute(
            'INSERT INTO profile_settings (profile_id) VALUES (?)',
            (profile_id,)
        )
        conn.commit()
        conn.close()
        return profile_id

    def profile_belongs_to_user(self, profile_id: str, user_id: str) -> bool:
        conn = self._connect()
        cursor = conn.cursor()
        cursor.execute('SELECT 1 FROM profiles WHERE id = ? AND user_id = ?', (profile_id, user_id))
        ok = cursor.fetchone() is not None
        conn.close()
        return ok

    # ================= INTEGRATIONS =================

    def create_integration(self, profile_id: str, service: str, access_token: str = None, refresh_token: str = None, expires_at: int = None, config: Dict = None) -> str:
        """Create or update an integration"""
        conn = self._connect()
        cursor = conn.cursor()
        
        # Check if exists for this service and profile
        cursor.execute('SELECT id FROM integrations WHERE profile_id = ? AND service = ?', (profile_id, service))
        existing = cursor.fetchone()
        
        now = int(datetime.now().timestamp())
        config_json = json.dumps(config or {})
        
        if existing:
            # Update
            integration_id = existing[0]
            cursor.execute('''
                UPDATE integrations 
                SET access_token = ?, refresh_token = ?, expires_at = ?, config = ?
                WHERE id = ?
            ''', (access_token, refresh_token, expires_at, config_json, integration_id))
        else:
            # Create
            integration_id = str(int(datetime.now().timestamp() * 1000))
            cursor.execute('''
                INSERT INTO integrations (id, profile_id, service, access_token, refresh_token, expires_at, config, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (integration_id, profile_id, service, access_token, refresh_token, expires_at, config_json, now))
            
        conn.commit()
        conn.close()
        return integration_id

    def get_integrations(self, profile_id: str) -> List[Dict]:
        """Get all integrations for a profile"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM integrations WHERE profile_id = ?', (profile_id,))
        
        results = []
        for row in cursor.fetchall():
            integration = dict(row)
            integration['config'] = json.loads(integration['config']) if integration['config'] else {}
            # Don't return sensitive tokens in list view if possible, or handle carefully
            results.append(integration)
            
        conn.close()
        return results

    def get_integration(self, integration_id: str) -> Optional[Dict]:
        """Get a specific integration"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM integrations WHERE id = ?', (integration_id,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row:
            integration = dict(row)
            integration['config'] = json.loads(integration['config']) if integration['config'] else {}
            return integration
        return None

    def delete_integration(self, integration_id: str) -> bool:
        """Delete an integration"""
        conn = self._connect()
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM integrations WHERE id = ?', (integration_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted

    # ================= DATA PORTABILITY =================

    def export_profile_data(self, profile_id: str) -> Dict:
        """Export conversations, messages, tags and settings for a profile"""
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM conversations WHERE profile_id = ?', (profile_id,))
        conversations = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM messages WHERE conversation_id IN (SELECT id FROM conversations WHERE profile_id = ?)', (profile_id,))
        messages = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM tags')
        tags = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM conversation_tags WHERE conversation_id IN (SELECT id FROM conversations WHERE profile_id = ?)', (profile_id,))
        conversation_tags = [dict(row) for row in cursor.fetchall()]

        cursor.execute('SELECT * FROM profile_settings WHERE profile_id = ?', (profile_id,))
        settings = cursor.fetchone()
        settings_dict = dict(settings) if settings else {}

        conn.close()

        return {
            "conversations": conversations,
            "messages": messages,
            "tags": tags,
            "conversation_tags": conversation_tags,
            "settings": settings_dict
        }

    def import_profile_data(self, profile_id: str, payload: Dict) -> bool:
        """Import profile data (idempotent inserts where possible)"""
        conversations = payload.get("conversations", [])
        messages = payload.get("messages", [])
        tags = payload.get("tags", [])
        conversation_tags = payload.get("conversation_tags", [])
        settings = payload.get("settings", {})

        conn = self._connect()
        cursor = conn.cursor()
        try:
            for tag in tags:
                cursor.execute(
                    'INSERT OR IGNORE INTO tags (id, name, color, icon, created_at) VALUES (?, ?, ?, ?, ?)',
                    (tag["id"], tag["name"], tag.get("color", "#3B82F6"), tag.get("icon", "🏷️"), tag.get("created_at", int(datetime.now().timestamp()*1000)))
                )
            for conv in conversations:
                cursor.execute(
                    'INSERT OR IGNORE INTO conversations (id, title, created_at, updated_at, use_as_context, summary, pinned, archived, profile_id) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        conv["id"], conv.get("title", "Conversation"), conv.get("created_at", int(datetime.now().timestamp())),
                        conv.get("updated_at", conv.get("created_at", int(datetime.now().timestamp()))), conv.get("use_as_context", 1),
                        conv.get("summary"), conv.get("pinned", 0), conv.get("archived", 0), conv.get("profile_id", profile_id)
                    )
                )
            for msg in messages:
                cursor.execute(
                    'INSERT OR IGNORE INTO messages (id, conversation_id, role, content, model, created_at, pinned) VALUES (?, ?, ?, ?, ?, ?, ?)',
                    (
                        msg["id"], msg["conversation_id"], msg["role"], msg["content"], msg.get("model"),
                        msg.get("created_at", int(datetime.now().timestamp()*1000)), msg.get("pinned", 0)
                    )
                )
            for ct in conversation_tags:
                cursor.execute(
                    'INSERT OR IGNORE INTO conversation_tags (conversation_id, tag_id, created_at) VALUES (?, ?, ?)',
                    (ct["conversation_id"], ct["tag_id"], ct.get("created_at", int(datetime.now().timestamp()*1000)))
                )
            if settings:
                cursor.execute(
                    'INSERT OR REPLACE INTO profile_settings (profile_id, voice_style, voice_speed, auto_speak, wake_word_enabled, wake_word_sensitivity, activation_sound_path, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        profile_id,
                        settings.get("voice_style", ""),
                        settings.get("voice_speed", 1.2),
                        settings.get("auto_speak", 1),
                        settings.get("wake_word_enabled", 0),
                        settings.get("wake_word_sensitivity", 0.7),
                        settings.get("activation_sound_path"),
                        settings.get("updated_at", int(datetime.now().timestamp()))
                    )
                )
            conn.commit()
            return True
        except Exception as e:
            print(f"Import error: {e}")
            conn.rollback()
            return False
        finally:
            conn.close()

    # ================= VOICE PROFILES =================

    def create_voice_profile(self, name: str, description: str, sample_path: str = "", provider: str = "supertonic", cloned: bool = False) -> str:
        conn = self._connect()
        cursor = conn.cursor()
        voice_id = str(int(datetime.now().timestamp() * 1000))
        created_at = int(datetime.now().timestamp())
        cursor.execute(
            'INSERT INTO voice_profiles (id, name, description, sample_path, provider, cloned, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (voice_id, name, description, sample_path, provider, 1 if cloned else 0, created_at)
        )
        conn.commit()
        conn.close()
        return voice_id

    def list_voice_profiles(self) -> List[Dict]:
        conn = self._connect()
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM voice_profiles ORDER BY created_at DESC')
        data = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return data
