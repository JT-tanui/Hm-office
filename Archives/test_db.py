import sys
import os
sys.path.append(os.path.abspath("src/web"))
from database import ConversationDB

try:
    print("Initializing DB...")
    db = ConversationDB()
    print("DB Initialized.")
    
    print("Fetching conversations...")
    conversations = db.get_conversations()
    print(f"Success! Found {len(conversations)} conversations.")
    for c in conversations[:3]:
        print(f" - {c['title']} (Context: {c.get('use_as_context')})")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
