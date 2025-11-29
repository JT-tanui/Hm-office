from typing import Dict, Any
import os
import requests
from .base import Integration

class GoogleDriveIntegration(Integration):
    """Google Drive integration (OAuth + upload scaffold)"""

    def get_id(self) -> str:
        return "google_drive"

    def get_name(self) -> str:
        return "Google Drive"

    def get_auth_url(self, redirect_uri: str) -> str:
        client_id = os.getenv("GOOGLE_CLIENT_ID", "")
        scope = "https%3A//www.googleapis.com/auth/drive.file"
        return f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&scope={scope}&redirect_uri={redirect_uri}&client_id={client_id}&access_type=offline&prompt=consent"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        client_id = os.getenv("GOOGLE_CLIENT_ID")
        client_secret = os.getenv("GOOGLE_CLIENT_SECRET")
        if not client_id or not client_secret:
            return {"error": "GOOGLE_CLIENT_ID/SECRET not set"}
        resp = requests.post(
            "https://oauth2.googleapis.com/token",
            data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code"
            },
            timeout=15
        )
        data = resp.json()
        return {
            "access_token": data.get("access_token"),
            "refresh_token": data.get("refresh_token"),
            "expires_at": data.get("expires_in"),
            "token_type": data.get("token_type")
        }

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.is_connected}
