import unittest
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))

from assistant.llm import LLMService
from assistant.tts_supertonic import SupertonicTTS
from utils.audio_io import list_audio_devices

class TestComponents(unittest.TestCase):
    def test_llm_connection(self):
        print("\nTesting LLM Connection...")
        llm = LLMService()
        # Use a very simple prompt to be fast
        response = llm.get_response("Hello", model_preference="chat")
        print(f"LLM Response: {response}")
        self.assertIsNotNone(response)
        self.assertNotEqual(response, "")

    def test_tts_initialization(self):
        print("\nTesting TTS Initialization...")
        try:
            tts = SupertonicTTS()
            self.assertIsNotNone(tts)
        except Exception as e:
            self.fail(f"TTS Initialization failed: {e}")

    def test_audio_devices(self):
        print("\nTesting Audio Devices...")
        try:
            list_audio_devices()
        except Exception as e:
            self.fail(f"Audio device listing failed: {e}")

if __name__ == "__main__":
    unittest.main()
