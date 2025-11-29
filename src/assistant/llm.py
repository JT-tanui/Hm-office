import logging
import json
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from .config import (
    OLLAMA_HOST, DEFAULT_MODEL, CODING_MODEL_SMALL, 
    CODING_MODEL_MEDIUM, CODING_MODEL_LARGE, VISION_MODEL, CLOUD_MODEL
)
from .config import OPENAI_API_KEY

logger = logging.getLogger(__name__)

class OpenAIClient:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.base_url = base_url

    def generate(self, prompt: str, model: str = "gpt-4o-mini", system: Optional[str] = None, temperature: float = 0.7) -> Optional[str]:
        """Non-streaming chat completion."""
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            return None

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(url, data=data, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    if "choices" in result and result["choices"]:
                        return result["choices"][0]["message"]["content"]
                logger.error(f"OpenAI API returned status {response.status}")
        except Exception as e:
            logger.error(f"OpenAI generate error: {e}")
        return None

    def stream_generate(self, prompt: str, model: str = "gpt-4o-mini", system: Optional[str] = None, temperature: float = 0.7):
        """Stream chat completion chunks."""
        if not self.api_key:
            logger.error("OpenAI API key not provided")
            return

        url = f"{self.base_url}/chat/completions"
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": True
        }

        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

        req = urllib.request.Request(url, data=data, headers=headers)
        buffer = ""
        MAX_BUFFER_SIZE = 300

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                for raw_line in response:
                    line = raw_line.decode("utf-8").strip()
                    if not line or not line.startswith("data:"):
                        continue
                    if line.strip() == "data: [DONE]":
                        if buffer.strip():
                            yield buffer.strip()
                        break
                    try:
                        payload_str = line[len("data:"):].strip()
                        chunk = json.loads(payload_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        if "content" in delta:
                            buffer += delta["content"]
                            # flush buffer on size/newlines/sentences
                            if len(buffer) > MAX_BUFFER_SIZE:
                                yield buffer.strip()
                                buffer = ""
                                continue
                            if "\n" in buffer:
                                parts = buffer.split("\n")
                                for part in parts[:-1]:
                                    if part.strip():
                                        yield part.strip()
                                buffer = parts[-1]
                                continue
                            import re
                            sentences = re.split(r'(?<=[.!?;:])\s+', buffer)
                            for sentence in sentences[:-1]:
                                if sentence.strip():
                                    yield sentence.strip()
                            buffer = sentences[-1] if sentences else ""
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error(f"OpenAI stream error: {e}")
            if buffer.strip():
                yield buffer.strip()
            else:
                yield f"Error: {e}"

class OllamaClient:
    def __init__(self, host: str = OLLAMA_HOST):
        self.host = host
        self.base_url = f"{host}/api"

    def generate(self, prompt: str, model: str = DEFAULT_MODEL, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """Generates a response from the LLM."""
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                if response.status == 200:
                    result = json.loads(response.read().decode("utf-8"))
                    return result.get("response")
                else:
                    logger.error(f"Ollama API returned status {response.status}")
                    return None
        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return None
    
    def stream_generate(self, prompt: str, model: str = DEFAULT_MODEL, system: Optional[str] = None, options: Optional[Dict[str, Any]] = None):
        """
        Stream responses from Ollama.
        Yields text chunks as they arrive.
        """
        url = f"{self.base_url}/generate"  # base_url already includes /api
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        if system:
            payload["system"] = system
        if options:
            payload["options"] = options
        
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        req = urllib.request.Request(url, data=data, headers=headers)
        
        logger.info(f"Streaming from Ollama: URL={url}, Model={model}")
        
        buffer = ""
        try:
            # Increase timeout to 300s for slow local models
            with urllib.request.urlopen(req, timeout=300) as response:
                MAX_BUFFER_SIZE = 300  # Match Supertonic's chunk limit
                
                for line in response:
                    line = line.decode("utf-8").strip()
                    if not line:
                        continue
                    
                    try:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            text = chunk["response"]
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
                        
                        if chunk.get("done", False):
                            # Yield remaining buffer
                            if buffer.strip():
                                yield buffer.strip()
                            break
                    except json.JSONDecodeError:
                        continue

        except urllib.error.URLError as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            yield f"Error: Could not connect to Ollama. Is it running? ({e})"
                        
        except Exception as e:
            logger.error(f"Error streaming from Ollama: {e}")
            if buffer and buffer.strip():
                yield buffer.strip()
            yield f"Error: {str(e)}"


class LLMService:
    def __init__(self):
        self.ollama_client = OllamaClient()
    
    def get_response(self, prompt: str, model_preference: str = "chat", provider: str = "ollama", api_key: str = None, system_prompt: str = None, temperature: float = 0.7) -> str:
        """
        High-level method to get a response with provider support.
        
        Args:
            prompt: User prompt
            model_preference: Model ID or preference key
            provider: "ollama" or "openrouter"
            api_key: OpenRouter API key (required if provider is openrouter)
            system_prompt: Optional system instructions
            temperature: Decoding temperature
        
        Returns:
            Response text
        """
        temperature = max(0.0, min(temperature, 2.0))
        if provider == "openrouter":
            from .openrouter import OpenRouterClient
            
            if not api_key:
                return "Error: OpenRouter API key is required."
            
            # Use model_preference directly as model ID
            model_id = model_preference
            client = OpenRouterClient(api_key)
            
            logger.info(f"Using OpenRouter model: {model_id}")
            response = client.generate(prompt, model=model_id, system=system_prompt, temperature=temperature)
            
            if response:
                return response
            
            # Fallback to free models if primary fails
            logger.warning(f"Model {model_id} failed. Trying free fallback models...")
            free_models = [
                "meta-llama/llama-3.2-3b-instruct:free",
                "google/gemini-flash-1.5:free",
                "qwen/qwen-2-7b-instruct:free",
                "microsoft/phi-3-mini-128k-instruct:free"
            ]
            
            for fallback_model in free_models:
                logger.info(f"Trying free fallback: {fallback_model}")
                response = client.generate(prompt, model=fallback_model, system=system_prompt, temperature=temperature)
                if response:
                    logger.info(f"Success with fallback model: {fallback_model}")
                    return response
            
            return "I'm sorry, I couldn't generate a response from OpenRouter. Please check your API key and model selection, or try a different model."
        elif provider == "openai":
            client = OpenAIClient(api_key or OPENAI_API_KEY)
            logger.info(f"Using OpenAI model: {model_preference}")
            response = client.generate(prompt, model=model_preference, system=system_prompt, temperature=temperature)
            if response:
                return response
            return "I'm sorry, I couldn't generate a response from OpenAI. Please check your API key and model selection."
        
        else:  # Ollama (default)
            # For Ollama, use model_preference directly as model name
            # This allows cloud models like "minimax-m2:cloud" to work
            logger.info(f"Using Ollama model: {model_preference}")
            response = self.ollama_client.generate(
                prompt,
                model=model_preference,
                system=system_prompt,
                options={"temperature": temperature}
            )
            
            if response:
                return response
            
            # Try fallback strategy for known preference keys
            strategy = self._get_strategy(model_preference)
            if strategy and strategy[0] != model_preference:
                for model in strategy:
                    logger.info(f"Trying fallback Ollama model: {model}")
                    response = self.ollama_client.generate(prompt, model=model, system=system_prompt)
                    if response:
                        return response
                    logger.warning(f"Model {model} failed or timed out. Falling back...")
            
            return "I'm sorry, I couldn't generate a response with any of the available Ollama models."
    
    def _get_strategy(self, preference: str) -> list[str]:
        if preference == "coding_small":
            return [CODING_MODEL_SMALL, CODING_MODEL_MEDIUM, DEFAULT_MODEL]
        elif preference == "coding_medium":
            return [CODING_MODEL_MEDIUM, CODING_MODEL_SMALL, DEFAULT_MODEL]
        elif preference == "coding_large":
            return [CODING_MODEL_LARGE, CODING_MODEL_MEDIUM, CODING_MODEL_SMALL]
        elif preference == "vision":
            return [VISION_MODEL]
        elif preference == "cloud":
             return [CLOUD_MODEL]
        else: # chat
            return [DEFAULT_MODEL, CODING_MODEL_SMALL]
