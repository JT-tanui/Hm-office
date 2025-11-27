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
        
        # Create indices
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC)')
        
        conn.commit()
        conn.close()
    
    def create_conversation(self, conversation_id: str, title: str = "New Chat") -> Dict:
        """Create a new conversation"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = int(datetime.now().timestamp() * 1000)
        cursor.execute(
            'INSERT INTO conversations (id, title, created_at, updated_at) VALUES (?, ?, ?, ?)',
            (conversation_id, title, now, now)
        )
        
        conn.commit()
        conn.close()
        
        return {
            'id': conversation_id,
            'title': title,
            'created_at': now,
            'updated_at': now
        }
    
    def get_conversations(self) -> List[Dict]:
        """Get all conversations with message counts"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT c.id, c.title, c.created_at, c.updated_at, c.use_as_context, c.summary,
                   COUNT(m.id) as message_count
            FROM conversations c
            LEFT JOIN messages m ON c.id = m.conversation_id
            GROUP BY c.id
            ORDER BY c.updated_at DESC
        ''')
        
        conversations = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return conversations
    
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
