import os
from typing import Dict, Any
import requests
from .base import Integration

class NotionIntegration(Integration):
    """Notion integration (OAuth + pages scaffold)"""

    def get_id(self) -> str:
        return "notion"

    def get_name(self) -> str:
        return "Notion"

    def get_auth_url(self, redirect_uri: str) -> str:
        client_id = os.getenv("NOTION_CLIENT_ID", "")
        scope = "read%20write"
        return f"https://api.notion.com/v1/oauth/authorize?owner=user&client_id={client_id}&redirect_uri={redirect_uri}&response_type=code&scope={scope}"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        client_id = os.getenv("NOTION_CLIENT_ID")
        client_secret = os.getenv("NOTION_CLIENT_SECRET")
        if not client_id or not client_secret:
            return {
                "access_token": None,
                "refresh_token": None,
                "expires_at": 0,
                "error": "NOTION_CLIENT_ID/SECRET not set"
            }
        resp = requests.post(
            "https://api.notion.com/v1/oauth/token",
            auth=(client_id, client_secret),
            json={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri
            },
            timeout=15
        )
        data = resp.json()
        return {
            "access_token": data.get("access_token"),
            "refresh_token": data.get("refresh_token"),
            "expires_at": data.get("expires_in", 0),
            "bot_id": data.get("bot_id"),
            "workspace_name": data.get("workspace_name")
        }

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": self.is_connected
        }
