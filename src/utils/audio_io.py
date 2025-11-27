import sounddevice as sd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def play_audio(audio_data: np.ndarray, sample_rate: int):
    """Plays audio data using sounddevice."""
    try:
        sd.play(audio_data, sample_rate)
        sd.wait()
    except Exception as e:
        logger.error(f"Error playing audio: {e}")

def list_audio_devices():
    """Lists available audio devices."""
    print(sd.query_devices())
