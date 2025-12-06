"""
One-off migration helper to backfill user ownership fields in the SQLite DB.

Usage:
    python3 scripts/migrate_user_ownership.py [path_to_db]

Defaults to conversations.db in repo root.
"""

import sqlite3
import sys
from pathlib import Path


def run(db_path: Path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # Ensure columns exist (defensive)
    cur.execute("PRAGMA table_info(conversations)")
    cols = {row[1] for row in cur.fetchall()}
    if "user_id" not in cols:
        cur.execute("ALTER TABLE conversations ADD COLUMN user_id TEXT DEFAULT 'anonymous'")

    cur.execute("PRAGMA table_info(messages)")
    cols = {row[1] for row in cur.fetchall()}
    if "user_id" not in cols:
        cur.execute("ALTER TABLE messages ADD COLUMN user_id TEXT DEFAULT 'anonymous'")

    cur.execute("PRAGMA table_info(profiles)")
    cols = {row[1] for row in cur.fetchall()}
    if "user_id" not in cols:
        cur.execute("ALTER TABLE profiles ADD COLUMN user_id TEXT DEFAULT 'anonymous'")

    # Backfill profiles.user_id to 'anonymous' where NULL
    cur.execute("UPDATE profiles SET user_id = COALESCE(user_id, 'anonymous')")

    # Backfill conversations.user_id from profile, else anonymous
    cur.execute(
        """
        UPDATE conversations
        SET user_id = COALESCE((
            SELECT p.user_id FROM profiles p WHERE p.id = conversations.profile_id
        ), 'anonymous')
        WHERE user_id IS NULL OR user_id = ''
        """
    )

    # Backfill messages.user_id from owning conversation
    cur.execute(
        """
        UPDATE messages
        SET user_id = COALESCE((
            SELECT c.user_id FROM conversations c WHERE c.id = messages.conversation_id
        ), 'anonymous')
        WHERE user_id IS NULL OR user_id = ''
        """
    )

    conn.commit()
    conn.close()
    print(f"Migration complete for {db_path}")


if __name__ == "__main__":
    db_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("conversations.db")
    run(db_path)
