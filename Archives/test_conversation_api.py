"""Test script to verify conversation API flow"""
import requests
import json

BASE_URL = "http://127.0.0.1:5000"
PROFILE_ID = "default"

print("="*60)
print("Testing Conversation API Flow")
print("="*60)

# Test 1: Get conversations
print("\n1. Testing GET /api/conversations")
headers = {"X-Profile-ID": PROFILE_ID}
response = requests.get(f"{BASE_URL}/api/conversations", headers=headers)
print(f"   Status: {response.status_code}")
if response.ok:
    data = response.json()
    print(f"   Found {len(data.get('conversations', []))} conversations")
    for conv in data.get('conversations', [])[:3]:
        print(f"     - ID: {conv['id']}, Title: {conv['title']}, Messages: {conv.get('message_count', 0)}")
else:
    print(f"   Error: {response.text}")

#Test 2: Create new conversation
print("\n2. Testing POST /api/conversations")
new_conv_id = str(int(__import__('time').time() * 1000))
response = requests.post(
    f"{BASE_URL}/api/conversations",
    headers={"Content-Type": "application/json", "X-Profile-ID": PROFILE_ID},
    json={"id": new_conv_id, "title": "Test Conversation"}
)
print(f"   Status: {response.status_code}")
if response.ok:
    print(f"   Created conversation: {new_conv_id}")
else:
    print(f"   Error: {response.text}")

# Test 3: Get specific conversation
print(f"\n3. Testing GET /api/conversations/{new_conv_id}")
response = requests.get(f"{BASE_URL}/api/conversations/{new_conv_id}")
print(f"   Status: {response.status_code}")
if response.ok:
    data = response.json()
    print(f"   Retrieved: {data.get('conversation', {}).get('title')}")
    print(f"   Messages: {len(data.get('conversation', {}).get('messages', []))}")
else:
    print(f"   Error: {response.text}")

# Test 4: Add message
print(f"\n4. Testing POST /api/conversations/{new_conv_id}/messages")
response = requests.post(
    f"{BASE_URL}/api/conversations/{new_conv_id}/messages",
    headers={"Content-Type": "application/json", "X-Profile-ID": PROFILE_ID},
    json={
        "id": str(int(__import__('time').time() * 1000)),
        "role": "user",
        "content": "Hello, this is a test message!",
        "model": "test"
    }
)
print(f"   Status: {response.status_code}")
if response.ok:
    print(f"   Message added successfully")
else:
    print(f"   Error: {response.text}")

# Test 5: Verify message was added
print(f"\n5. Verifying message was saved")
response = requests.get(f"{BASE_URL}/api/conversations/{new_conv_id}")
if response.ok:
    data = response.json()
    messages = data.get('conversation', {}).get('messages', [])
    print(f"   Status: {response.status_code}")
    print(f"   Messages in conversation: {len(messages)}")
    if messages:
        print(f"   Last message: {messages[-1].get('content', '')[:50]}")
else:
    print(f"   Error: {response.text}")

print("\n" + "="*60)
print("Test Complete!")
print("="*60)
