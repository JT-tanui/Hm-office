import os
from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
SUPERTONIC_ROOT = PROJECT_ROOT / "supertonic"
ASSETS_DIR = SUPERTONIC_ROOT / "assets"

# LLM Configuration
# LLM Configuration
DEFAULT_MODEL = "qwen2.5-coder:3b"
CODING_MODEL_SMALL = "qwen2.5-coder:3b"
CODING_MODEL_MEDIUM = "qwen2.5-coder:3b" # Fallback to small if medium not available
CODING_MODEL_LARGE = "qwen2.5-coder:3b" # Fallback to small if large not available
VISION_MODEL = "llava:13b"
CLOUD_MODEL = "minimax-m2:cloud"

MODEL_PREFERENCES = {
    "chat": DEFAULT_MODEL,
    "coding_small": CODING_MODEL_SMALL,
    "coding_medium": CODING_MODEL_MEDIUM,
    "coding_large": CODING_MODEL_LARGE,
    "vision": VISION_MODEL
}

# OpenRouter Models
OPENROUTER_MODELS = {
    "minimax-m2": "minimax/minimax-01",
    "claude-sonnet": "anthropic/claude-3.5-sonnet",
    "gpt-4": "openai/gpt-4-turbo",
    "gemini-pro": "google/gemini-pro",
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct"
}

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# TTS Configuration
TTS_MODEL_PATH = ASSETS_DIR / "model.onnx" # Verify exact name in assets
TTS_CONFIG_PATH = ASSETS_DIR / "config.json" # Verify exact name
SAMPLE_RATE = 24000 # Typical for Supertonic/Piper-like models, verify

# Audio Configuration
AUDIO_DEVICE_ID = None # Default device
