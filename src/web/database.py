import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Optional

class ConversationDB:
    def __init__(self, db_path='conversations.db'):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database with tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Conversations table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at INTEGER NOT NULL,
                updated_at INTEGER NOT NULL,
                use_as_context INTEGER DEFAULT 1,
                summary TEXT DEFAULT NULL
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
        
        # Messages table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                model TEXT,
                created_at INTEGER NOT NULL,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
            )
        ''')
        
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
                is_default INTEGER DEFAULT 0
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
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_memos_profile ON voice_memos(profile_id)')
        
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
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, conversation_id: str, title: str = "New Chat", profile_id: str = "default") -> Dict:
        """Create a new conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        cursor.execute(
            'INSERT INTO conversations (id, title, created_at, updated_at, profile_id) VALUES (?, ?, ?, ?, ?)',
            (conversation_id, title, now, now, profile_id)
        )
        
        conn.commit()
        conn.close()
        
        return {
            'id': conversation_id,
            'title': title,
            'created_at': now,
            'updated_at': now,
            'profile_id': profile_id
        }
    
    def get_conversations(self, profile_id: str = "default") -> List[Dict]:
        """Get all conversations for a profile with message counts"""
    
    def get_conversation(self, conversation_id: str) -> Optional[Dict]:
        """Get a single conversation with its messages"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get conversation
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
    
    def update_conversation(self, conversation_id: str, title: str = None) -> bool:
        """Update conversation title"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        
        if title:
            cursor.execute(
                'UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?',
                (title, now, conversation_id)
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
    
    def delete_conversation(self, conversation_id: str) -> bool:
        """Delete a conversation and all its messages"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM conversations WHERE id = ?', (conversation_id,))
        success = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        
        return success
    
    def add_message(self, message_id: str, conversation_id: str, role: str, content: str, model: str = None) -> Dict:
        """Add a message to a conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        
        cursor.execute(
            'INSERT INTO messages (id, conversation_id, role, content, model, created_at) VALUES (?, ?, ?, ?, ?, ?)',
            (message_id, conversation_id, role, content, model, now)
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
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC',
            (conversation_id,)
        )
        
        messages = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return messages
    
    def toggle_context_usage(self, conversation_id: str, enabled: bool) -> bool:
        """Toggle whether a conversation is used as context"""
        conn = sqlite3.connect(self.db_path)
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
    
    def update_summary(self, conversation_id: str, summary: str) -> bool:
        """Update conversation summary"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
    
    def get_all_context_messages(self, exclude_id: Optional[str] = None, limit: int = 50) -> List[Dict]:
        """Get messages from context-enabled conversations for cross-conversation context"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if exclude_id:
            cursor.execute('''
                SELECT m.role, m.content, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.use_as_context = 1 AND c.id != ?
                ORDER BY m.created_at DESC
                LIMIT ?
            ''', (exclude_id, limit))
        else:
            cursor.execute('''
                SELECT m.role, m.content, m.created_at, c.title
                FROM messages m
                JOIN conversations c ON m.conversation_id = c.id
                WHERE c.use_as_context = 1
                ORDER BY m.created_at DESC
                LIMIT ?
            ''', (limit,))
        
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
    
    def search_messages(self, query: str, limit: int = 50) -> List[Dict]:
        """Search messages using FTS5 full-text search"""
        if not query or not query.strip():
            return []
        
        conn = sqlite3.connect(self.db_path)
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
            WHERE messages_fts MATCH ?
            ORDER BY m.created_at DESC
            LIMIT ?
        ''', (query, limit))
        
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
    
    def filter_conversations(self, start_date: Optional[int] = None, end_date: Optional[int] = None, model: Optional[str] = None) -> List[Dict]:
        """Filter conversations by date range and/or model"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
            SELECT DISTINCT c.id, c.title, c.created_at, c.updated_at, c.use_as_context
            FROM conversations c
        '''
        
        conditions = []
        params = []
        
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
    
    # Tag Management Methods
    def create_tag(self, tag_id: str, name: str, color: str = '#3B82F6', icon: str = '🏷️') -> Dict:
        """Create a new tag"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM tags WHERE id = ?', (tag_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted
    
    def add_tag_to_conversation(self, conversation_id: str, tag_id: str) -> bool:
        """Add a tag to a conversation"""
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM voice_memos WHERE id = ?', (memo_id,))
        deleted = cursor.rowcount > 0
        
        conn.commit()
        conn.close()
        return deleted

    def search_voice_memos(self, query: str, limit: int = 20) -> List[Dict]:
        """Search voice memos using FTS5"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Search in FTS table and join with main table
        cursor.execute('''
            SELECT m.*, snippet(memos_fts, 1, '<b>', '</b>', '...', 64) as snippet
            FROM voice_memos m
            JOIN memos_fts fts ON m.rowid = fts.rowid
            WHERE fts MATCH ?
            ORDER BY rank
            LIMIT ?
        ''', (query, limit))
        
        results = []
        for row in cursor.fetchall():
            memo = dict(row)
            memo['tags'] = json.loads(memo['tags']) if memo['tags'] else []
            results.append(memo)
            
        conn.close()
        return results
