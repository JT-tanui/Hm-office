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
            SELECT c.id, c.title, c.created_at, c.updated_at,
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
