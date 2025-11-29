from typing import Dict, Any
from .base import Integration

class GoogleKeepIntegration(Integration):
    """Google Keep integration scaffold"""

    def get_id(self) -> str:
        return "google_keep"

    def get_name(self) -> str:
        return "Google Keep"

    def get_auth_url(self, redirect_uri: str) -> str:
        scope = "https%3A//www.googleapis.com/auth/keep"
        return f"https://accounts.google.com/o/oauth2/v2/auth?response_type=code&scope={scope}&redirect_uri={redirect_uri}&client_id={{GOOGLE_CLIENT_ID}}"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        return {"access_token": "keep_token_placeholder", "refresh_token": "keep_refresh_placeholder", "expires_at": 0}

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.is_connected}
