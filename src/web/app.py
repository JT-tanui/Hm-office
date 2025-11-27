import os
import sys
import logging
import base64
import numpy as np
import urllib.request
from flask import Flask, render_template, request, jsonify, send_from_directory

# Add src to path so we can import assistant modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistant.pipeline import AssistantPipeline
from assistant.config import MODEL_PREFERENCES
from database import ConversationDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder=".")

# Initialize pipeline
pipeline = AssistantPipeline()
logger.info("Pipeline initialized successfully")

# Initialize database
db = ConversationDB()
logger.info("Database initialized successfully")

@app.route("/")
def index():
    return app.send_static_file("index.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("text")
    model_preference = data.get("model", "chat")
    voice_style = data.get("voice")
    voice_speed = data.get("speed", 1.0)  # Voice speed control
    enable_tts = data.get("tts", True)
    provider = data.get("provider", "ollama")
    api_key = data.get("api_key")
    system_prompt = data.get("system_prompt")  # System instructions
    conversation_history = data.get("conversation_history", [])  # Get conversation history

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    # Build conversation context if history exists
    if conversation_history:
        # Format history as conversation context
        context_messages = []
        for msg in conversation_history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                context_messages.append(f"User: {content}")
            else:
                context_messages.append(f"Assistant: {content}")
        
        # Combine history with current input
        full_context = "\n".join(context_messages) + f"\nUser: {user_input}"
        logger.info(f"Using conversation history with {len(conversation_history)} messages")
    else:
        full_context = user_input

    result = pipeline.generate_response(
        full_context, 
        model_preference, 
        voice_style=voice_style,
        voice_speed=voice_speed,
        provider=provider,
        api_key=api_key,
        system_prompt=system_prompt
    )
    
    response_data = {
        "text": result["text"],
        "model": model_preference,
    }

    if enable_tts and result["audio"] is not None:
        logger.info("TTS enabled and audio generated, saving to file...")
        # Save audio to file for better performance/caching
        import io
        import uuid
        import scipy.io.wavfile as wav
        
        # Ensure static/audio exists
        audio_dir = os.path.join(app.static_folder, "audio")
        os.makedirs(audio_dir, exist_ok=True)
        logger.info(f"Audio directory: {audio_dir}")
        
        # Cleanup old files (older than 1 hour)
        try:
            import time
            current_time = time.time()
            for f in os.listdir(audio_dir):
                f_path = os.path.join(audio_dir, f)
                if os.path.isfile(f_path):
                    # If older than 1 hour (3600 seconds)
                    if current_time - os.path.getmtime(f_path) > 3600:
                        os.remove(f_path)
                        logger.info(f"Cleaned up old file: {f}")
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        # Generate unique filename
        filename = f"{uuid.uuid4()}.wav"
        filepath = os.path.join(audio_dir, filename)
        
        # Normalize to 16-bit PCM
        audio_data = result["audio"]
        if audio_data.dtype == np.float32:
            audio_data = (audio_data * 32767).astype(np.int16)
            
        # Write to file
        wav.write(filepath, result["sample_rate"], audio_data)
        logger.info(f"Audio file saved: {filepath}")
        
        # Return URL
        response_data["audio"] = f"/static/audio/{filename}"
        logger.info(f"Audio URL: {response_data['audio']}")
    elif enable_tts:
        logger.warning("TTS enabled but no audio was generated!")
    else:
        logger.info("TTS is disabled")

    return jsonify(response_data)

@app.route("/api/chat/stream", methods=["POST"])
def chat_stream():
    """True streaming - streams LLM response and generates audio per sentence"""
    from flask import Response, stream_with_context
    import json as json_module
    
    data = request.json
    user_input = data.get("text")
    model_preference = data.get("model", "chat")
    voice_style = data.get("voice")
    enable_tts = data.get("tts", True)
    provider = data.get("provider", "ollama")
    api_key = data.get("api_key")
    system_prompt = data.get("system_prompt")
    conversation_history = data.get("conversation_history", [])  # Get conversation history
    conversation_id = data.get("conversation_id")  # Get current conversation ID
    use_cross_context = data.get("use_cross_context", True)  # Enable cross-context by default

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        try:
            # Log all incoming parameters for debugging
            logger.info(f"=== STREAM REQUEST START ===")
            logger.info(f"User input: {user_input[:100]}...")
            logger.info(f"Model: {model_preference}, Provider: {provider}")
            logger.info(f"Voice style: {voice_style}, TTS enabled: {enable_tts}")
            logger.info(f"Conversation ID: {conversation_id}, Cross-context: {use_cross_context}")
            logger.info(f"History length: {len(conversation_history) if conversation_history else 0}")
            
            full_text = ""
            chunk_index = 0
            
            # Determine context limit based on model
            # Approx chars per token = 4. We use a safe buffer.
            if provider == "openrouter":
                if any(x in model_preference.lower() for x in ['claude', 'gemini', 'gpt-4', '128k', '200k']):
                    max_context_chars = 100000  # Large context models
                else:
                    max_context_chars = 16000   # Standard OpenRouter models
            else:  # Ollama
                # Most local models are 4k-8k context
                if any(x in model_preference.lower() for x in ['llama3', 'mistral', 'mixtral', 'qwen']):
                    max_context_chars = 8000    # Reduced from 12000
                else:
                    max_context_chars = 4000    # Reduced from 6000 (Conservative)
            
            logger.info(f"Dynamic context limit for {model_preference}: {max_context_chars} chars")

            # Build conversation context with budget
            context_parts = []
            current_chars = len(user_input) + (len(system_prompt) if system_prompt else 0) + 500 # Reserve 500 for system overhead
            
            # 1. Add current user input (already counted)
            # We'll append it at the end, but track size now
            
            # 2. Add current conversation history (reverse to get most recent first)
            history_context = []
            if conversation_history:
                for msg in reversed(conversation_history):
                    role = "User" if msg['role'] == 'user' else "Assistant"
                    msg_text = f"{role}: {msg['content']}"
                    msg_len = len(msg_text) + 2 # newline
                    
                    if current_chars + msg_len < max_context_chars:
                        history_context.insert(0, msg_text) # Prepend to keep order
                        current_chars += msg_len
                    else:
                        logger.info(f"Context limit reached at {current_chars} chars, stopping history")
                        break
                
                if history_context:
                    context_parts.append("\n# Current conversation:")
                    context_parts.extend(history_context)
                    logger.info(f"Added {len(history_context)} messages from current history")

            # 3. Add cross-conversation context if enabled and we still have space
            # Only if we haven't used more than 70% of budget on current history
            if use_cross_context and conversation_id and current_chars < (max_context_chars * 0.7):
                try:
                    # Fetch more candidates, but only use what fits
                    cross_messages = db.get_all_context_messages(exclude_id=conversation_id, limit=20)
                    if cross_messages:
                        cross_context = []
                        for msg in cross_messages:
                            role = "User" if msg['role'] == 'user' else "Assistant"
                            msg_text = f"{role}: {msg['content']}"
                            msg_len = len(msg_text) + 2
                            
                            if current_chars + msg_len < max_context_chars:
                                cross_context.append(msg_text)
                                current_chars += msg_len
                            else:
                                break
                        
                        if cross_context:
                            # Insert at the very beginning
                            context_parts.insert(0, "# Previous conversations context:")
                            # Insert items after the header
                            for i, msg in enumerate(cross_context):
                                context_parts.insert(i+1, msg)
                            logger.info(f"Added {len(cross_context)} messages from other conversations")
                except Exception as e:
                    logger.error(f"Error loading cross-conversation context: {e}")
            
            # Add current user input at the end
            context_parts.append(f"\nUser: {user_input}")
            
            # Build final context
            if context_parts:
                full_context = "\n".join(context_parts)
            else:
                full_context = user_input
            
            # Log context size for debugging
            context_length = len(full_context)
            logger.info(f"Full context length: {context_length} characters")
            if context_length > 8000:
                logger.warning(f"Context is very large ({context_length} chars), this may cause issues")
            
            # Setup voice
            style_path = None
            if voice_style and pipeline.tts:
                style_path = os.path.join(pipeline.tts.assets_dir, "voice_styles", f"{voice_style}.json")
            speed = 1.15 if voice_style and voice_style.startswith('M') else 1.2
            
            # Stream from LLM
            if provider == "openrouter":
                from assistant.openrouter import OpenRouterClient
                if not api_key:
                    yield f"data: {json_module.dumps({'type': 'error', 'message': 'API key required'})}\n\n"
                    return
                
                client = OpenRouterClient(api_key)
                stream = client.stream_generate(full_context, model=model_preference, system=system_prompt)
            else:  # Ollama
                # Map preference key to actual model name if it exists in config
                from assistant.config import MODEL_PREFERENCES
                actual_model = MODEL_PREFERENCES.get(model_preference, model_preference)
                
                # Log the mapping for debugging
                if actual_model != model_preference:
                    logger.info(f"Mapped model preference '{model_preference}' to '{actual_model}'")
                
                stream = pipeline.llm.ollama_client.stream_generate(full_context, model=actual_model, system=system_prompt)
            
            # Process each sentence from LLM
            for sentence in stream:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                full_text += (" " if full_text else "") + sentence
                
                # Send text chunk immediately
                yield f"data: {json_module.dumps({'type': 'text_chunk', 'content': sentence})}\n\n"
                
                # Generate and send audio for this sentence if TTS enabled
                if enable_tts and pipeline.tts:
                    try:
                        # Clean text for TTS
                        clean_sentence = pipeline._clean_text_for_tts(sentence)
                        
                        if len(clean_sentence) > 5:  # Skip very short fragments
                            logger.info(f"Generating audio for: {clean_sentence[:50]}...")
                            
                            # SAFEGUARD: Chunk long sentences before TTS (Supertonic limit: 300 chars)
                            if len(clean_sentence) > 300:
                                logger.warning(f"Sentence too long ({len(clean_sentence)} chars), chunking...")
                                mini_chunks = pipeline._chunk_text(clean_sentence, 300)
                                for mchunk in mini_chunks:
                                    audio, sample_rate = pipeline.tts.synthesize(mchunk, style_path=style_path, speed=speed)
                                    
                                    # Save audio to file
                                    import uuid
                                    import scipy.io.wavfile as wav_module
                                    
                                    # Ensure static/audio exists
                                    audio_dir = os.path.join(app.static_folder, "audio")
                                    os.makedirs(audio_dir, exist_ok=True)
                                    
                                    # Generate unique filename
                                    filename = f"{uuid.uuid4()}.wav"
                                    filepath = os.path.join(audio_dir, filename)
                                    
                                    # Normalize to 16-bit PCM
                                    audio_data = audio
                                    if audio_data.dtype == np.float32:
                                        audio_data = (audio_data * 32767).astype(np.int16)
                                    
                                    # Write to file
                                    wav_module.write(filepath, sample_rate, audio_data)
                                    
                                    # Send audio chunk with file URL
                                    chunk_data = {
                                        'type': 'audio',
                                        'index': chunk_index,
                                        'audio': f"/static/audio/{filename}"
                                    }
                                    yield f"data: {json_module.dumps(chunk_data)}\n\n"
                                    chunk_index += 1
                            else:
                                audio, sample_rate = pipeline.tts.synthesize(clean_sentence, style_path=style_path, speed=speed)
                                
                                # Save audio to file
                                import uuid
                                import scipy.io.wavfile as wav_module
                                
                                # Ensure static/audio exists
                                audio_dir = os.path.join(app.static_folder, "audio")
                                os.makedirs(audio_dir, exist_ok=True)
                                
                                # Generate unique filename
                                filename = f"{uuid.uuid4()}.wav"
                                filepath = os.path.join(audio_dir, filename)
                                
                                # Normalize to 16-bit PCM
                                audio_data = audio
                                if audio_data.dtype == np.float32:
                                    audio_data = (audio_data * 32767).astype(np.int16)
                                
                                # Write to file
                                wav_module.write(filepath, sample_rate, audio_data)
                                
                                # Send audio chunk with file URL
                                chunk_data = {
                                    'type': 'audio',
                                    'index': chunk_index,
                                    'audio': f"/static/audio/{filename}"
                                }
                                yield f"data: {json_module.dumps(chunk_data)}\n\n"
                                chunk_index += 1
                    except Exception as e:
                        logger.error(f"Error generating audio for sentence: {e}")
                        continue
            
            # Send completion signal with full text
            yield f"data: {json_module.dumps({'type': 'done', 'full_text': full_text})}\n\n"
            logger.info(f"=== STREAM REQUEST COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json_module.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return Response(stream_with_context(generate()), mimetype='text/event-stream')


@app.route("/api/config", methods=["GET"])
def get_config():
    return jsonify({
        "models": list(MODEL_PREFERENCES.keys())
    })

@app.route("/api/settings", methods=["GET"])
def get_settings_api():
    """Get user settings"""
    try:
        settings = db.get_settings()
        return jsonify(settings)
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/settings", methods=["POST"])
def update_settings_api():
    """Update user settings"""
    try:
        data = request.json
        success = db.update_settings(data)
        if success:
            return jsonify({"success": True, "settings": db.get_settings()})
        else:
            return jsonify({"error": "Failed to update settings"}), 500
    except Exception as e:
        logger.error(f"Failed to update settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/models", methods=["GET"])
def get_models():
    """Get available models for a specific provider"""
    import subprocess
    import json as json_module
    
    provider = request.args.get("provider", "ollama")
    
    if provider == "openrouter":
        # Fetch from OpenRouter API
        try:
            req = urllib.request.Request("https://openrouter.ai/api/v1/models")
            with urllib.request.urlopen(req, timeout=10) as response:
                data = json_module.loads(response.read().decode("utf-8"))
                models = []
                for model in data.get("data", []):
                    model_id = model.get("id", "")
                    name = model.get("name", model_id)
                    # Include context length for user reference
                    context = model.get("context_length", 0)
                    models.append({
                        "id": model_id,
                        "name": f"{name} ({context}k)" if context else name
                    })
                return jsonify({"models": models})
        except Exception as e:
            logger.error(f"Failed to fetch OpenRouter models: {e}")
            # Fallback to some popular models
            return jsonify({
                "models": [
                    {"id": "minimax/minimax-01", "name": "MiniMax M2"},
                    {"id": "anthropic/claude-3.5-sonnet", "name": "Claude 3.5 Sonnet"},
                    {"id": "openai/gpt-4-turbo", "name": "GPT-4 Turbo"},
                    {"id": "google/gemini-pro", "name": "Gemini Pro"}
                ]
            })
    else:  # ollama
        # Fetch from ollama list command
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                models = []
                
                # Skip header line
                for line in lines[1:]:
                    if line.strip():
                        # Parse: NAME ID SIZE MODIFIED
                        parts = line.split()
                        if parts:
                            model_name = parts[0]
                            models.append({
                                "id": model_name,
                                "name": model_name
                            })
                
                logger.info(f"Found Ollama models: {models}")
                return jsonify({"models": models})
            else:
                logger.error(f"Ollama list failed: {result.stderr}")
                # Fallback to config
                from assistant.config import MODEL_PREFERENCES
                models = [{"id": k, "name": k} for k in MODEL_PREFERENCES.keys()]
                logger.info(f"Using fallback models: {models}")
                return jsonify({"models": models})
        except Exception as e:
            logger.error(f"Failed to fetch Ollama models: {e}")
            # Fallback to config
            from assistant.config import MODEL_PREFERENCES
            models = [{"id": k, "name": k} for k in MODEL_PREFERENCES.keys()]
            return jsonify({"models": models})

@app.route("/api/conversation-styles", methods=["GET"])
def get_conversation_styles():
    """Return predefined conversation style templates"""
    styles = {
        "casual": {
            "name": "Casual",
            "icon": "🎭",
            "prompt": "You're a friendly, laid-back assistant. Use casual language, contractions, and a relaxed tone. Keep responses conversational and approachable."
        },
        "professional": {
            "name": "Professional",
            "icon": "💼",
            "prompt": "You're a professional assistant. Be formal, concise, and precise. Use proper grammar and maintain a business-appropriate tone."
        },
        "friendly": {
            "name": "Friendly",
            "icon": "🤗",
            "prompt": "You're warm and empathetic. Show enthusiasm and support. Use encouraging language and make the user feel comfortable."
        },
        "analytical": {
            "name": "Analytical",
            "icon": "🧠",
            "prompt": "You're thoughtful and detail-oriented. Break down complex topics systematically. Provide thorough explanations with reasoning."
        },
        "teacher": {
            "name": "Teacher",
            "icon": "🎓",
            "prompt": "You're a patient educator. Explain concepts clearly with examples. Check for understanding and encourage questions."
        }
    }
    return jsonify({"styles": styles})

@app.route("/api/voices", methods=["GET"])
def get_voices():
    # List json files in assets/voice_styles
    # We can access assets dir via pipeline.tts.assets_dir if initialized
    if pipeline.tts:
        voice_dir = os.path.join(pipeline.tts.assets_dir, "voice_styles")
        if os.path.exists(voice_dir):
            voices = [f.replace(".json", "") for f in os.listdir(voice_dir) if f.endswith(".json")]
            return jsonify({"voices": sorted(voices)})
    return jsonify({"voices": []})

# ============= Conversation API Endpoints =============

@app.route("/api/conversations", methods=["GET"])
def get_conversations_api():
    """Get all conversations"""
    try:
        conversations = db.get_conversations()
        return jsonify({"conversations": conversations})
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations", methods=["POST"])
def create_conversation_api():
    """Create a new conversation"""
    try:
        data = request.json
        conversation_id = data.get("id")
        title = data.get("title", "New Chat")
        
        if not conversation_id:
            return jsonify({"error": "id is required"}), 400
        
        conversation = db.create_conversation(conversation_id, title)
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>", methods=["GET"])
def get_conversation_api(conversation_id):
    """Get a single conversation with messages"""
    try:
        conversation = db.get_conversation(conversation_id)
        if not conversation:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify(conversation)
    except Exception as e:
        logger.error(f"Failed to get conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>", methods=["PATCH"])
def update_conversation_api(conversation_id):
    """Update conversation title"""
    try:
        data = request.json
        title = data.get("title")
        
        success = db.update_conversation(conversation_id, title)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to update conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/toggle-context", methods=["PATCH"])
def toggle_context_api(conversation_id):
    """Toggle whether conversation is used as context"""
    try:
        data = request.json
        enabled = data.get("enabled", True)
        
        success = db.toggle_context_usage(conversation_id, enabled)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"success": True, "enabled": enabled})
    except Exception as e:
        logger.error(f"Failed to toggle context: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>", methods=["DELETE"])
def delete_conversation_api(conversation_id):
    """Delete a conversation"""
    try:
        success = db.delete_conversation(conversation_id)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/messages", methods=["POST"])
def add_message_api(conversation_id):
    """Add a message to a conversation"""
    try:
        data = request.json
        message_id = data.get("id")
        role = data.get("role")
        content = data.get("content")
        model = data.get("model")
        
        if not all([message_id, role, content]):
            return jsonify({"error": "id, role, and content are required"}), 400
        
        message = db.add_message(message_id, conversation_id, role, content, model)
        return jsonify(message)
    except Exception as e:
        logger.error(f"Failed to add message: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("Starting Web Interface on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
