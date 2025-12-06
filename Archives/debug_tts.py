import sys
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(os.path.join(os.getcwd(), "src"))

from assistant.pipeline import AssistantPipeline

def test_tts():
    print("Initializing pipeline...")
    pipeline = AssistantPipeline()
    
    if not pipeline.tts:
        print("ERROR: TTS not initialized!")
        return

    text = """Job descriptions are kind of all over the place these days, aren't they? I notice they often get really bloated with requirements and buzzwords. Sometimes it feels like companies are listing every possible skill they might want someday, rather than what they actually need.

The job descriptions that work best are the ones that are clear about what success looks like in the role. Like instead of just listing "5 years of X experience," they'll say what you're actually supposed to accomplish day to day.

I think remote work has also changed how companies write job descriptions. Some are still figuring out how to communicate their culture and work environment remotely.

What got you thinking about job descriptions? Are you job hunting or just curious?"""

    print(f"\nProcessing text ({len(text)} chars)...")
    
    # 1. Test cleaning
    clean_text = pipeline._clean_text_for_tts(text)
    print(f"\nCleaned text ({len(clean_text)} chars):")
    print(clean_text)
    
    # 2. Test chunking
    max_len = 80
    chunks = pipeline._chunk_text(clean_text, max_len)
    print(f"\nGenerated {len(chunks)} chunks:")
    for i, c in enumerate(chunks):
        print(f"  {i+1}: [{len(c)}] {c}")
        
    # 3. Test synthesis
    print("\nSynthesizing...")
    try:
        # We'll just call generate_response which does the whole flow
        # But to be specific, let's call the internal logic if possible, 
        # or just use generate_response with a mock LLM response?
        # Actually, let's just use the logic from pipeline.py manually to isolate TTS
        
        audio_chunks = []
        for i, chunk in enumerate(chunks):
            print(f"  Synthesizing chunk {i+1}...")
            try:
                audio, rate = pipeline.tts.synthesize(chunk, speed=1.0)
                print(f"    Success! Shape: {audio.shape}")
                audio_chunks.append(audio)
            except Exception as e:
                print(f"    FAILED: {e}")
                
        if audio_chunks:
            print(f"\nSuccessfully generated {len(audio_chunks)} audio chunks.")
        else:
            print("\nFAILED to generate any audio chunks.")
            
    except Exception as e:
        print(f"Global error: {e}")

if __name__ == "__main__":
    test_tts()
