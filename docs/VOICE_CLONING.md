# Voice Cloning Guide for Supertonic TTS

## Overview
This guide explains how to create custom voice styles for the Local Talk Assistant using Supertonic TTS. Voice styles enable natural, human-like speech with different personalities and characteristics.

## Current Voice Styles
The system includes 4 pre-trained voices:
- **M1** - Male Voice 1 (Neutral)
- **M2** - Male Voice 2 (Deeper)
- **F1** - Female Voice 1 (Warm)
- **F2** - Female Voice 2 (Energetic)

## Voice Style Architecture

### Voice File Structure
Each voice is a JSON file (~434KB) containing neural embeddings:
```
supertonic/assets/voice_styles/
├── M1.json
├── M2.json
├── F1.json
└── F2.json
```

### Embedding Format
```json
{
  "style_ttl": {
    "data": [
      [ /* 256-dimensional embedding vectors */ ]
    ]
  }
}
```

## Creating Custom Voices

### Option 1: Using Existing Voices with Personas

The quickest way to create custom voices is to use existing voices with modified TTS parameters:

**In `src/assistant/tts_supertonic.py`:**
```python
def synthesize(self, text: str, style_path: str = None, speed: float = 1.0, pitch: float = 1.0):
    # Add pitch control (requires Supertonic mod)
    # speed: 0.5-2.0 (slower to faster)
    # pitch: 0.5-2.0 (lower to higher)
```

**Example Presets:**
```python
# Professional Male (jm)
voice='M1', speed=1.0, pitch=0.95  # Slightly deeper, measured

# GenZ Female (jf)  
voice='F2', speed=1.15, pitch=1.05  # Slightly faster, higher energy
```

### Option 2: Voice Cloning (Advanced)

To create entirely new voices from audio samples:

#### Requirements
1. **Audio Samples**: 10-30 minutes of clean speech
   - Format: WAV, 22050Hz, mono
   - Quality: Studio/podcast quality
   - Content: Diverse sentences, emotions
   - No background noise/music

2. **Supertonic Training Environment**
   - Python 3.10+
   - PyTorch with CUDA (GPU recommended)
   - 8GB+ RAM
   - Supertonic training scripts

#### Step 1: Prepare Audio Data

```bash
cd supertonic/training

# Organize your audio files
mkdir -p data/raw/your_voice_name
# Copy WAV files to data/raw/your_voice_name/

# Preprocess audio
python preprocess_audio.py \
  --input data/raw/your_voice_name \
  --output data/processed/your_voice_name \
  --sample_rate 22050
```

#### Step 2: Extract Voice Embeddings

```bash
# Run voice embedding extraction
python extract_embeddings.py \
  --audio_dir data/processed/your_voice_name \
  --output embeddings/your_voice_name.json \
  --model_path models/supertonic_base.pt
```

This generates the 256-dimensional embeddings that define the voice.

#### Step 3: Train Voice Model (Optional)

For best quality, fine-tune on your specific voice:

```bash
python train_voice.py \
  --data data/processed/your_voice_name \
  --embedding embeddings/your_voice_name.json \
  --epochs 100 \
  --output models/your_voice_name.pt
```

#### Step 4: Export Voice Style

```bash
# Convert trained model to voice style JSON
python export_voice_style.py \
  --model models/your_voice_name.pt \
  --embedding embeddings/your_voice_name.json \
  --output ../assets/voice_styles/your_voice_name.json
```

#### Step 5: Register in Application

**Update `src/web/static/index.html`:**
```html
<select id="voice-select">
    <option value="">Default</option>
    <option value="M1">Male Voice 1</option>
    <option value="M2">Male Voice 2</option>
    <option value="F1">Female Voice 1</option>
    <option value="F2">Female Voice 2</option>
    <option value="your_voice_name">Your Custom Voice</option>
</select>
```

### Option 3: Using Pre-trained Voice Models

Download community voices from Supertonic Voice Hub:

```bash
# Browse available voices
python download_voice.py --list

# Download specific voice
python download_voice.py \
  --voice "professional_british_male" \
  --output assets/voice_styles/pbm.json
```

## Voice Quality Optimization

### For Maximum Naturalness

1. **Audio Quality Matters**
   - Use high-quality audio (studio mic preferred)
   - Remove noise with Audacity/RX
   - Normalize levels

2. **Diverse Training Data**
   - Include questions, statements, exclamations
   - Vary emotions and tones
   - Cover different speech patterns

3. **Optimal Settings**
   - Speed: 0.9-1.1 (too fast/slow sounds robotic)
   - Chunk size: 100-300 chars (current: good)
   - Sample rate: 22050Hz (default: optimal)

### Testing Your Voice

```python
# Test script
from assistant.tts_supertonic import SupertonicTTS

tts = SupertonicTTS()
audio, sr = tts.synthesize(
    "Hello, this is a test of my custom voice.",
    style_path="assets/voice_styles/your_voice.json",
    speed=1.0
)

# Save test audio
import scipy.io.wavfile as wav
wav.write("test_output.wav", sr, audio)
```

## Troubleshooting

### Voice Sounds Robotic
- Reduce speed (try 0.95)
- Use more training data
- Check audio preprocessing

### Artifacts/Glitches
- Clean source audio
- Increase training epochs
- Verify embeddings integrity

### Voice Doesn't Match Sample
- Need more training data (20+ minutes)
- Ensure consistent audio quality
- Re-extract embeddings

## Commercial Use

**Important**: Voice cloning requires explicit consent:
- ✅ Your own voice
- ✅ Licensed voice actors
- ✅ Open-source voice datasets
- ❌ Celebrity/public figures without permission
- ❌ Copyrighted audio

## Advanced: Creating Persona Voices

For corporate vs. GenZ voices without full training:

```python
# src/assistant/tts_supertonic.py

VOICE_PERSONAS = {
    'jm': {  # Journey Male - Corporate
        'base': 'M1',
        'speed': 1.0,
        'pitch': 0.95,
        'emphasis': 1.2  # Stronger emphasis on keywords
    },
    'jf': {  # Journey Female - GenZ
        'base': 'F2',
        'speed': 1.15,
        'pitch': 1.05,
        'variation': 1.3  # More pitch variation
    }
}
```

## Resources

- **Supertonic Docs**: `supertonic/README.md`
- **Training Scripts**: `supertonic/training/`
- **Example Voices**: `supertonic/examples/`
- **Voice Hub**: https://supertonic-voice-hub.example.com (if available)

## Support

For issues or questions:
1. Check logs in `assistant.log`
2. Verify voice JSON format
3. Test with default voices first
4. Consult Supertonic documentation

---

**Note**: This guide assumes Supertonic training tools are available. If you only have the inference model, stick to Option 1 (parameter tuning) or Option 3 (download pre-trained voices).
