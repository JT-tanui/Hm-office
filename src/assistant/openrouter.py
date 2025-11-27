import logging
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

class OpenRouterClient:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://openrouter.ai/api/v1"
    
    def generate(self, prompt: str, model: str = "minimax/minimax-01", system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generates a response from OpenRouter API.
        
        Args:
            prompt: User prompt
            model: Model ID (e.g., "minimax/minimax-01")
            system: Optional system message
            options: Optional parameters (temperature, max_tokens, etc.)
        
        Returns:
            Response text or None on error
        """
        if not self.api_key:
            logger.error("OpenRouter API key not provided")
            return None
        
        url = f"{self.base_url}/chat/completions"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        # Add optional parameters
        if options:
            payload.update(options)
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:5000",  # Optional: for analytics
            "X-Title": "Local Talk Assistant"  # Optional: for analytics
        }
        
        req = urllib.request.Request(url, data=data, headers=headers)
        
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    # OpenRouter returns OpenAI-compatible format
                    if "choices" in result and len(result["choices"]) > 0:
                        return result["choices"][0]["message"]["content"]
                    else:
                        logger.error(f"Unexpected response format: {result}")
                        return None
                else:
                    logger.error(f"OpenRouter API returned status {response.status}")
                    return None
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            logger.error(f"OpenRouter HTTP Error {e.code}: {error_body}")
            return None
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to OpenRouter: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating response from OpenRouter: {e}")
            return None
    
    def stream_generate(self, prompt: str, model: str = "minimax/minimax-01", system: Optional[str] = None):
        """
        Stream responses from OpenRouter.
        Yields text chunks as they arrive.
        """
        if not self.api_key:
            logger.error("OpenRouter API key not provided")
            return
        
        url = f"{self.base_url}/chat/completions"
        
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": True
        }
        
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:5000",
            "X-Title": "Local Talk Assistant"
        }
        
        req = urllib.request.Request(url, data=data, headers=headers)
        
        buffer = ""  # Initialize buffer before try block
        MAX_BUFFER_SIZE = 300
        
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line or not line.startswith("data: "):
                        continue
                    
                    if line == "data: [DONE]":
                        if buffer.strip():
                            yield buffer.strip()
                        break
                    
                    try:
                        data_str = line[6:]  # Remove "data: " prefix
                        chunk = json.loads(data_str)
                        
                        if "choices" in chunk and len(chunk["choices"]) > 0:
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                text = delta["content"]
                                buffer += text
                                
                                # Force yield if buffer gets too large (prevents unbounded growth)
                                if len(buffer) > MAX_BUFFER_SIZE:
                                    yield buffer.strip()
                                    buffer = ""
                                    continue
                                
                                # Yield on newlines (paragraph breaks)
                                if '\n' in buffer:
                                    parts = buffer.split('\n')
                                    for part in parts[:-1]:
                                        if part.strip():
                                            yield part.strip()
                                    buffer = parts[-1]
                                    continue
                                
                                # Yield complete sentences
                                import re
                                sentences = re.split(r'(?<=[.!?;:])\s+', buffer)
                                
                                # Yield all complete sentences
                                for sentence in sentences[:-1]:
                                    if sentence.strip():
                                        yield sentence.strip()
                                
                                # Keep incomplete sentence in buffer
                                buffer = sentences[-1] if sentences else ""
                                
                    except json.JSONDecodeError:
                        continue
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else str(e)
            logger.error(f"OpenRouter HTTP Error {e.code}: {error_body}")
            # Yield error message as a sentence
            yield f"Error: OpenRouter returned error {e.code}. Please check your API key and rate limits."
        except Exception as e:
            logger.error(f"Error streaming from OpenRouter: {e}")
            if buffer.strip():
                yield buffer.strip()
            else:
                yield f"Error: {str(e)}"

