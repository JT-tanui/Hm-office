import requests
import json
import time

BASE_URL = "http://localhost:5000/api"
PROFILE_ID = "default"

def test_integrations():
    print("Testing Integration Hub API...")
    
    # 1. Get Integrations (should be empty or have dummy)
    print("\n1. Getting integrations...")
    try:
        res = requests.get(f"{BASE_URL}/integrations", headers={"X-Profile-ID": PROFILE_ID})
        if res.status_code == 200:
            data = res.json()
            print("Success:", json.dumps(data, indent=2))
            integrations = data.get("integrations", [])
            dummy = next((i for i in integrations if i["id"] == "dummy"), None)
            if not dummy:
                print("ERROR: Dummy integration not found in available list")
                return
        else:
            print(f"ERROR: Failed to get integrations. Status: {res.status_code}")
            print(res.text)
            return
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        return

    # 2. Connect Dummy Integration
    print("\n2. Connecting dummy integration...")
    try:
        res = requests.post(
            f"{BASE_URL}/integrations/connect",
            headers={"X-Profile-ID": PROFILE_ID, "Content-Type": "application/json"},
            json={"service": "dummy", "config": {"test": True}}
        )
        if res.status_code == 200:
            data = res.json()
            print("Success:", json.dumps(data, indent=2))
            integration_id = data.get("id")
        else:
            print(f"ERROR: Failed to connect. Status: {res.status_code}")
            print(res.text)
            return
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")
        return

    # 3. Verify Connection
    print("\n3. Verifying connection status...")
    try:
        res = requests.get(f"{BASE_URL}/integrations", headers={"X-Profile-ID": PROFILE_ID})
        if res.status_code == 200:
            data = res.json()
            integrations = data.get("integrations", [])
            dummy = next((i for i in integrations if i["id"] == "dummy"), None)
            if dummy and dummy.get("connected"):
                print("Success: Dummy integration is connected")
            else:
                print("ERROR: Dummy integration not marked as connected")
        else:
            print(f"ERROR: Failed to get integrations. Status: {res.status_code}")
    except Exception as e:
        print(f"ERROR: Connection failed: {e}")

    # 4. Disconnect Integration
    if integration_id:
        print(f"\n4. Disconnecting integration {integration_id}...")
        try:
            res = requests.delete(f"{BASE_URL}/integrations/{integration_id}")
            if res.status_code == 200:
                print("Success: Disconnected")
            else:
                print(f"ERROR: Failed to disconnect. Status: {res.status_code}")
                print(res.text)
        except Exception as e:
            print(f"ERROR: Connection failed: {e}")

if __name__ == "__main__":
    test_integrations()
