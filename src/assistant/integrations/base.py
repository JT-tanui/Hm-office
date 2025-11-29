from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

class Integration(ABC):
    """Abstract base class for all integrations"""
    
    def __init__(self, profile_id: str, config: Dict[str, Any] = None):
        self.profile_id = profile_id
        self.config = config or {}
        self.is_connected = False
        
    @abstractmethod
    def get_id(self) -> str:
        """Return unique identifier for this integration type (e.g., 'google', 'notion')"""
        pass
        
    @abstractmethod
    def get_name(self) -> str:
        """Return human-readable name"""
        pass
        
    @abstractmethod
    def get_auth_url(self, redirect_uri: str) -> str:
        """Return OAuth authorization URL"""
        pass
        
    @abstractmethod
    def handle_callback(self, code: str, redirect_uri: str) -> Dict[str, Any]:
        """Handle OAuth callback and return tokens"""
        pass
        
    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect integration and cleanup"""
        pass
        
    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return current status and metadata"""
        pass
