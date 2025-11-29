from typing import Dict, Any
from .base import Integration

class TrelloIntegration(Integration):
    """Trello integration scaffold"""

    def get_id(self) -> str:
        return "trello"

    def get_name(self) -> str:
        return "Trello"

    def get_auth_url(self, redirect_uri: str) -> str:
        return f"https://trello.com/1/authorize?response_type=token&key={{TRELLO_KEY}}&return_url={redirect_uri}&scope=read,write"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        return {"access_token": "trello_token_placeholder", "refresh_token": None, "expires_at": 0}

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.is_connected}
