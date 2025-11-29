from typing import Dict, Any
from .base import Integration

class AsanaIntegration(Integration):
    """Asana integration scaffold"""

    def get_id(self) -> str:
        return "asana"

    def get_name(self) -> str:
        return "Asana"

    def get_auth_url(self, redirect_uri: str) -> str:
        scope = "default"
        return f"https://app.asana.com/-/oauth_authorize?client_id={{ASANA_CLIENT_ID}}&redirect_uri={redirect_uri}&response_type=code&scope={scope}"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        return {"access_token": "asana_token_placeholder", "refresh_token": "asana_refresh_placeholder", "expires_at": 0}

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.is_connected}
