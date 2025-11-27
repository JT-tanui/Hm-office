import shutil
import subprocess
import logging
from typing import List

logger = logging.getLogger(__name__)

def check_ollama_running() -> bool:
    """Checks if Ollama is accessible via CLI."""
    if not shutil.which("ollama"):
        logger.error("Ollama CLI not found in PATH.")
        return False
    
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        return True
    except subprocess.CalledProcessError:
        logger.error("Ollama is installed but failed to list models. Is the server running?")
        return False
    except Exception as e:
        logger.error(f"Error checking Ollama: {e}")
        return False

def get_installed_models() -> List[str]:
    """Returns a list of installed Ollama models."""
    if not shutil.which("ollama"):
        return []
    
    try:
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.strip().split('\n')
        # Skip header
        if len(lines) > 0 and "NAME" in lines[0]:
            lines = lines[1:]
        
        models = [line.split()[0] for line in lines if line.strip()]
        return models
    except Exception:
        return []
