import logging
import os
from .llm import LLMService
from .tts_supertonic import SupertonicTTS
from utils.audio_io import play_audio

logger = logging.getLogger(__name__)

class AssistantPipeline:
    def __init__(self):
        logger.info("Initializing Assistant Pipeline...")
        self.llm = LLMService()
        try:
            self.tts = SupertonicTTS()
            logger.info("TTS initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize TTS: {e}")
            self.tts = None

    def generate_response(self, user_input: str, model_preference: str = "chat", voice_style: str = None, voice_speed: float = 1.0, provider: str = "ollama", api_key: str = None, system_prompt: str = None, temperature: float = 0.7) -> dict:
        """
        Generates response and audio (if available).
        Returns:
            dict: {
                "text": str,
                "audio": np.ndarray or None,
                "sample_rate": int or None,
                "model_used": str
            }
        """
        # 1. LLM
        response_text = self.llm.get_response(
            user_input, 
            model_preference, 
            provider=provider, 
            api_key=api_key,
            system_prompt=system_prompt,
            temperature=temperature
        )
        
        result = {
            "text": response_text,
            "audio": None,
            "sample_rate": None
        }

        # 2. TTS
        if self.tts and response_text:
            try:
                # Clean text for TTS (remove markdown, special chars)
                clean_text = self._clean_text_for_tts(response_text)
                
                # Resolve voice style path if provided
                style_path = None
                if voice_style:
                    style_path = os.path.join(self.tts.assets_dir, "voice_styles", f"{voice_style}.json")
                
                # Setup voice speed (allow override via parameter)
                speed = voice_speed
                if voice_speed == 1.0 and voice_style:  # Only auto-adjust if not explicitly set
                    # Male voices slightly slower for smoothness
                    if voice_style.startswith('M'):
                        speed = 0.95
                
                # Chunk long text - VERY SMALL CHUNKS for reliability
                # Supertonic is sensitive to length
                max_chunk_length = 100  # Increased to 100 for better performance
                
                if len(clean_text) > max_chunk_length:
                    logger.info(f"Text is {len(clean_text)} chars, chunking into {max_chunk_length} char pieces")
                    chunks = self._chunk_text(clean_text, max_chunk_length)
                    logger.info(f"Created {len(chunks)} chunks")
                    audio_chunks = []
                    
                    for i, chunk in enumerate(chunks):
                        logger.info(f"Chunk {i+1}/{len(chunks)}: '{chunk}'")
                        try:
                            chunk_audio, sample_rate = self.tts.synthesize(chunk, style_path=style_path, speed=speed)
                            audio_chunks.append(chunk_audio)
                            result["sample_rate"] = sample_rate
                            logger.info(f"Successfully generated audio for chunk {i+1}")
                        except Exception as chunk_error:
                            logger.error(f"Failed to synthesize chunk {i+1} with style {style_path}: {chunk_error}")
                            # Fallback: Try without style if it failed (might be a style issue)
                            if style_path:
                                try:
                                    logger.info(f"Retrying chunk {i+1} with default style...")
                                    chunk_audio, sample_rate = self.tts.synthesize(chunk, style_path=None, speed=1.0)
                                    audio_chunks.append(chunk_audio)
                                    result["sample_rate"] = sample_rate
                                    logger.info(f"Fallback success for chunk {i+1}")
                                    continue
                                except Exception as fallback_error:
                                    logger.error(f"Fallback failed for chunk {i+1}: {fallback_error}")
                            continue
                    
                    if not audio_chunks:
                        logger.error("No audio chunks were successfully generated")
                        return result
                    
                    # Concatenate chunks with small silence
                    import numpy as np
                    # Create 1D silence array to match TTS output shape
                    silence = np.zeros(int(0.3 * sample_rate), dtype=np.float32)
                    audio_parts = []
                    for i, chunk in enumerate(audio_chunks):
                        audio_parts.append(chunk)
                        if i < len(audio_chunks) - 1:
                            audio_parts.append(silence)
                    
                    logger.info(f"Concatenating {len(audio_parts)} audio parts")
                    # Use axis=0 for 1D arrays
                    result["audio"] = np.concatenate(audio_parts, axis=0)
                    logger.info("Audio concatenation successful")
                else:
                    audio, sample_rate = self.tts.synthesize(clean_text, style_path=style_path, speed=speed)
                    result["audio"] = audio
                    result["sample_rate"] = sample_rate
            except Exception as e:
                logger.error(f"TTS error: {e}")
        
        return result
    
    def _clean_text_for_tts(self, text: str) -> str:
        """Remove markdown formatting and special characters that confuse TTS"""
        import re
        
        # Remove markdown bold/italic
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **bold**
        text = re.sub(r'\*(.+?)\*', r'\1', text)      # *italic*
        text = re.sub(r'__(.+?)__', r'\1', text)      # __bold__
        text = re.sub(r'_(.+?)_', r'\1', text)        # _italic_
        
        # Normalize smart quotes
        text = text.replace('“', '"').replace('”', '"')
        text = text.replace('‘', "'").replace('’', "'")
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Convert bullet points to natural speech
        text = re.sub(r'^[\-\*]\s+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        # Expand common abbreviations
        text = re.sub(r'\bDr\.', 'Doctor', text)
        text = re.sub(r'\bMr\.', 'Mister', text)
        text = re.sub(r'\bMrs\.', 'Missus', text)
        text = re.sub(r'\bMs\.', 'Miss', text)
        text = re.sub(r'\bProf\.', 'Professor', text)
        text = re.sub(r'\betc\.', 'et cetera', text)
        text = re.sub(r'\be\.g\.', 'for example', text)
        text = re.sub(r'\bi\.e\.', 'that is', text)
        text = re.sub(r'\bvs\.', 'versus', text)
        
        # Fix acronym pronunciation
        text = re.sub(r'\bAI\b', 'A.I.', text)
        text = re.sub(r'\bLLM\b', 'L.L.M.', text)
        text = re.sub(r'\bAPI\b', 'A.P.I.', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://\S+', 'link', text)
        
        # Remove emojis and special symbols (keep only word chars and basic punctuation)
        # This is the most reliable way to ensure TTS doesn't choke
        text = re.sub(r'[^\w\s.,!?;:\'"()-]', '', text)
        
        # Remove excessive punctuation
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'!{2,}', '!', text)
        text = re.sub(r'\?{2,}', '?', text)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _chunk_text(self, text: str, max_length: int) -> list:
        """Chunk text at sentence boundaries"""
        import re
        
        # Split into sentences (multiple delimiters)
        sentences = re.split(r'(?<=[.!?;:])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # If a single sentence is too long, split it further
            if len(sentence) > max_length:
                # Split at commas or other natural breaks
                subsentences = re.split(r'(?<=,)\s+', sentence)
                for sub in subsentences:
                    sub = sub.strip()
                    if len(current_chunk) + len(sub) + 1 <= max_length:
                        current_chunk += (" " if current_chunk else "") + sub
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        # If even a sub-sentence is too long, force split
                        if len(sub) > max_length:
                            words = sub.split()
                            word_chunk = ""
                            for word in words:
                                if len(word_chunk) + len(word) + 1 <= max_length:
                                    word_chunk += (" " if word_chunk else "") + word
                                else:
                                    if word_chunk:
                                        chunks.append(word_chunk)
                                    word_chunk = word
                            current_chunk = word_chunk
                        else:
                            current_chunk = sub
            else:
                # Normal sentence processing
                if len(current_chunk) + len(sentence) + 1 <= max_length:
                    current_chunk += (" " if current_chunk else "") + sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text[:max_length]]

    def process_input(self, user_input: str, model_preference: str = "chat"):
        """
        CLI wrapper for processing input.
        """
        print(f"Thinking ({model_preference})...")
        
        result = self.generate_response(user_input, model_preference)
        
        print(f"Assistant: {result['text']}")

        if result["audio"] is not None:
            try:
                play_audio(result["audio"], result["sample_rate"])
            except Exception as e:
                logger.error(f"Playback error: {e}")
        elif self.tts is None:
             logger.warning("TTS not available, skipping audio.")
