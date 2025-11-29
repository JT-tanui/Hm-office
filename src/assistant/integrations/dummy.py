from typing import Dict, Any
from .base import Integration

class DummyIntegration(Integration):
    """Dummy integration for testing purposes"""
    
    def get_id(self) -> str:
        return "dummy"
        
    def get_name(self) -> str:
        return "Dummy Service"
        
    def get_auth_url(self, redirect_uri: str) -> str:
        return f"http://localhost:5000/dummy-auth?redirect_uri={redirect_uri}"
        
    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        return {
            "access_token": "dummy_access_token",
            "refresh_token": "dummy_refresh_token",
            "expires_at": 3600
        }
        
    def disconnect(self) -> bool:
        return True
        
    def get_status(self) -> Dict[str, Any]:
        return {
            "connected": True,
            "username": "test_user"
        }
