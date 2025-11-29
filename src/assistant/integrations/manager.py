from typing import Dict, List, Optional, Type
from .base import Integration

class IntegrationManager:
    """Manages available and active integrations"""
    
    def __init__(self, db):
        self.db = db
        self._registry: Dict[str, Type[Integration]] = {}
        self._instances: Dict[str, Integration] = {}
        
    def register(self, integration_cls: Type[Integration]):
        """Register a new integration class"""
        instance = integration_cls("dummy") # Temporary instance to get ID
        self._registry[instance.get_id()] = integration_cls
        
    def get_available_integrations(self) -> List[Dict]:
        """Get list of supported integrations"""
        available = []
        for integration_cls in self._registry.values():
            instance = integration_cls("dummy")
            available.append({
                "id": instance.get_id(),
                "name": instance.get_name(),
                "description": (instance.__doc__ or "").strip()
            })
        return available
        
    def get_integration_instance(self, service_id: str, profile_id: str) -> Optional[Integration]:
        """Get or create an instance of an integration for a profile"""
        key = f"{service_id}:{profile_id}"
        
        if key in self._instances:
            return self._instances[key]
            
        if service_id not in self._registry:
            return None
            
        # Load config from DB
        # This part assumes we can fetch integration details from DB
        # For now, we'll just instantiate with empty config
        # In a real implementation, we'd fetch tokens here
        
        cls = self._registry[service_id]
        instance = cls(profile_id)
        self._instances[key] = instance
        return instance
