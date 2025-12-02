from typing import Dict, Any
from .base import Integration

class MicrosoftGraphIntegration(Integration):
    """Microsoft 365 Graph integration scaffold"""

    def get_id(self) -> str:
        return "microsoft_graph"

    def get_name(self) -> str:
        return "Microsoft 365"

    def get_auth_url(self, redirect_uri: str) -> str:
        scope = "offline_access%20User.Read%20Tasks.ReadWrite%20Files.ReadWrite"
        return f"https://login.microsoftonline.com/common/oauth2/v2.0/authorize?response_type=code&scope={scope}&client_id={{MS_CLIENT_ID}}&redirect_uri={redirect_uri}"

    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        return {"access_token": code, "refresh_token": None, "expires_at": 0}

    def disconnect(self) -> bool:
        return True

    def get_status(self) -> Dict[str, Any]:
        return {"connected": self.is_connected}
