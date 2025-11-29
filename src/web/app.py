import os
import sys
import logging
import base64
import json as json_module
import sqlite3
import numpy as np
import urllib.request
import urllib.parse
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
import tempfile
import zipfile
import requests
import random

# Add src to path so we can import assistant modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from assistant.pipeline import AssistantPipeline
from assistant.config import MODEL_PREFERENCES
from assistant.integrations.manager import IntegrationManager
from assistant.integrations.dummy import DummyIntegration
from assistant.integrations.notion import NotionIntegration
from assistant.integrations.google_drive import GoogleDriveIntegration
from assistant.integrations.google_keep import GoogleKeepIntegration
from assistant.integrations.microsoft_graph import MicrosoftGraphIntegration
from assistant.integrations.trello import TrelloIntegration
from assistant.integrations.asana import AsanaIntegration
from database import ConversationDB

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder=".")

# Configure uploads
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'memos')
VOICE_UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'voices')
WAKE_SOUND_FOLDER = os.path.join(os.path.dirname(__file__), 'static', 'uploads', 'wake_sounds')
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'webm', 'ogg', 'm4a', 'txt', 'md', 'markdown', 'zip', 'pdf', 'docx'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VOICE_UPLOAD_FOLDER, exist_ok=True)
os.makedirs(WAKE_SOUND_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['VOICE_UPLOAD_FOLDER'] = VOICE_UPLOAD_FOLDER
app.config['WAKE_SOUND_FOLDER'] = WAKE_SOUND_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_session_profile():
    token = request.cookies.get("session_token")
    if not token:
        return None
    user = db.get_user_by_session(token)
    return user

def _extract_text_from_file(path: str) -> list:
    """Return list of (name, text) from a file path (txt/md/pdf/docx)."""
    texts = []
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.txt', '.md', '.markdown']:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            texts.append((os.path.basename(path), f.read()))
    elif ext == '.pdf':
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(path)
            content = "\n".join([page.extract_text() or "" for page in reader.pages])
            texts.append((os.path.basename(path), content))
        except Exception as e:
            logger.warning(f"PDF extraction failed for {path}: {e}")
    elif ext == '.docx':
        try:
            import docx
            document = docx.Document(path)
            content = "\n".join([p.text for p in document.paragraphs])
            texts.append((os.path.basename(path), content))
        except Exception as e:
            logger.warning(f"DOCX extraction failed for {path}: {e}")
    return texts

def _chunk_text_for_rag(text: str, max_len: int = 800) -> list:
    chunks = []
    buffer = []
    length = 0
    for line in text.splitlines():
        if length + len(line) > max_len and buffer:
            chunks.append("\n".join(buffer))
            buffer = []
            length = 0
        buffer.append(line)
        length += len(line)
    if buffer:
        chunks.append("\n".join(buffer))
    return chunks

def _hash_embed(text: str, dim: int = 256) -> list:
    import hashlib
    vec = [0.0] * dim
    for token in text.lower().split():
        h = int(hashlib.sha256(token.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0
    # L2 normalize
    import math
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]

def _get_embedding(text: str) -> list:
    from openai import OpenAI
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    client = OpenAI(api_key=api_key)
    resp = client.embeddings.create(model="text-embedding-3-small", input=text[:6000])
    return resp.data[0].embedding

def semantic_rank(query: str, chunks: list, top_k: int = 5) -> list:
    q_vec = None
    try:
        q_vec = _get_embedding(query)
    except Exception as e:
        logger.warning(f"Embedding query failed, falling back to hash: {e}")
        q_vec = _hash_embed(query)
    scored = []
    for row in chunks:
        emb = row.get("embedding")
        if emb:
            try:
                import json as _json
                c_vec = _json.loads(emb)
            except Exception:
                c_vec = None
        else:
            c_vec = None
        if not c_vec:
            try:
                c_vec = _get_embedding(row.get("content", ""))
            except Exception:
                c_vec = _hash_embed(row.get("content", ""))
        score = sum(q * c for q, c in zip(q_vec, c_vec))
        scored.append((score, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for score, row in scored[:top_k] if score > 0]

def get_web_search_context(query: str, limit: int = 3) -> str:
    """Lightweight DuckDuckGo search helper."""
    try:
        params = urllib.parse.urlencode({
            "q": query,
            "format": "json",
            "no_redirect": "1",
            "no_html": "1"
        })
        url = f"https://api.duckduckgo.com/?{params}"
        with urllib.request.urlopen(url, timeout=5) as resp:
            data = json_module.loads(resp.read().decode("utf-8"))
            topics = data.get("RelatedTopics", [])[:limit]
            lines = []
            for idx, item in enumerate(topics, start=1):
                text = item.get("Text") or ""
                link = item.get("FirstURL") or ""
                if text or link:
                    lines.append(f"{idx}. {text} ({link})")
            if lines:
                return "Web search results:\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"Web search failed: {e}")
    return ""

def get_rag_context(query: str, profile_id: str, limit: int = 5, source_id: str = None) -> str:
    """Retrieve top chunks for lightweight RAG (FS-backed or conversation search)."""
    try:
        # Semantic-ish search over recent chunks using hashed embeddings
        recent = db.get_recent_rag_chunks(profile_id, limit=400)
        if source_id:
            recent = [r for r in recent if r.get("source_id") == source_id]
        ranked = semantic_rank(query, recent, top_k=limit)
        if not ranked:
            # fallback to messages
            results = db.search_messages(query, limit, profile_id=profile_id)
            if not results:
                return ""
            lines = []
            for idx, row in enumerate(results, start=1):
                snippet = row.get("highlighted_content") or row.get("content") or ""
                title = row.get("conversation_title") or "Conversation"
                lines.append(f"{idx}. [{title}] {snippet}")
            return "Knowledge base snippets:\n" + "\n".join(lines)

        lines = []
        for idx, row in enumerate(ranked, start=1):
            snippet = row.get("snippet") or row.get("content") or ""
            lines.append(f"{idx}. {snippet}")
        return "Knowledge base snippets:\n" + "\n".join(lines)
    except Exception as e:
        logger.warning(f"RAG context failed: {e}")
        return ""

# Initialize pipeline
pipeline = AssistantPipeline()
logger.info("Pipeline initialized successfully")

# Initialize database
db = ConversationDB()
logger.info("Database initialized successfully")

# Integrations
integration_manager = IntegrationManager(db)
integration_manager.register(DummyIntegration)
integration_manager.register(NotionIntegration)
integration_manager.register(GoogleDriveIntegration)
integration_manager.register(GoogleKeepIntegration)
integration_manager.register(MicrosoftGraphIntegration)
integration_manager.register(TrelloIntegration)
integration_manager.register(AsanaIntegration)

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
    temperature = float(data.get("temperature", 0.7))
    use_web_search = data.get("use_web_search", False)
    use_rag = data.get("use_rag", False)
    rag_source_id = data.get("rag_source_id")
    rag_source_id = data.get("rag_source_id")
    tone = data.get("tone")
    profile_id = request.headers.get("X-Profile-ID", "default")

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    if tone:
        system_prompt = (system_prompt or "") + f"\nTone preference: {tone}"

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

    # Tool contexts (web search + RAG)
    tool_contexts = []
    if use_web_search:
        search_ctx = get_web_search_context(user_input)
        if search_ctx:
            tool_contexts.append(search_ctx)
    if use_rag:
        rag_ctx = get_rag_context(user_input, profile_id, source_id=rag_source_id)
        if rag_ctx:
            tool_contexts.append(rag_ctx)
    if tool_contexts:
        full_context = "\n\n".join(tool_contexts) + "\n\nUser query:\n" + full_context

    result = pipeline.generate_response(
        full_context, 
        model_preference, 
        voice_style=voice_style,
        voice_speed=voice_speed,
        provider=provider,
        api_key=api_key,
        system_prompt=system_prompt,
        temperature=temperature
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
    
    data = request.json
    session_user = get_session_profile()
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
    use_web_search = data.get("use_web_search", False)
    use_rag = data.get("use_rag", False)
    tone = data.get("tone")
    temperature = float(data.get("temperature", 0.7))
    profile_id = request.headers.get("X-Profile-ID", "default")

    if not user_input:
        return jsonify({"error": "No text provided"}), 400

    def generate():
        try:
            local_system = system_prompt
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

            if tone:
                local_system = (local_system or "") + f"\nTone preference: {tone}"

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
                    cross_messages = db.get_all_context_messages(
                        exclude_id=conversation_id,
                        limit=20,
                        profile_id=profile_id
                    )
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

            # Tooling contexts
            if use_web_search:
                search_ctx = get_web_search_context(user_input)
                if search_ctx:
                    context_parts.insert(0, "# Web search results")
                    context_parts.insert(1, search_ctx)
            if use_rag:
                rag_ctx = get_rag_context(user_input, profile_id, source_id=rag_source_id)
                if rag_ctx:
                    context_parts.insert(0, "# Retrieved knowledge")
                    context_parts.insert(1, rag_ctx)
            
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
                stream = client.stream_generate(full_context, model=model_preference, system=local_system, temperature=temperature)
            elif provider == "openai":
                from assistant.llm import OpenAIClient
                if not api_key:
                    yield f"data: {json_module.dumps({'type': 'error', 'message': 'OpenAI API key required'})}\n\n"
                    return
                client = OpenAIClient(api_key)
                stream = client.stream_generate(full_context, model=model_preference, system=local_system, temperature=temperature)
            else:  # Ollama
                # Map preference key to actual model name if it exists in config
                from assistant.config import MODEL_PREFERENCES
                actual_model = MODEL_PREFERENCES.get(model_preference, model_preference)
                
                # Log the mapping for debugging
                if actual_model != model_preference:
                    logger.info(f"Mapped model preference '{model_preference}' to '{actual_model}'")
                
                stream = pipeline.llm.ollama_client.stream_generate(
                    full_context,
                    model=actual_model,
                    system=local_system,
                    options={"temperature": temperature}
                )
            
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
        session_user = get_session_profile()
        profile_id = request.headers.get("X-Profile-ID", "default")
        settings = db.get_settings()
        profile_settings = db.get_profile_settings(profile_id)
        merged = {
            **settings,
            "wake_word_sensitivity": profile_settings.get("wake_word_sensitivity", 0.7),
            "activation_sound_path": profile_settings.get("activation_sound_path")
        }
        return jsonify(merged)
    except Exception as e:
        logger.error(f"Failed to get settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/settings", methods=["POST"])
def update_settings_api():
    """Update user settings"""
    try:
        session_user = get_session_profile()
        data = request.json or {}
        profile_id = request.headers.get("X-Profile-ID", "default")
        success_global = db.update_settings(data)
        profile_updates = {}
        for key in ["voice_style", "voice_speed", "auto_speak", "wake_word_enabled", "wake_word_sensitivity", "activation_sound_path"]:
            if key in data and data.get(key) is not None:
                profile_updates[key] = data.get(key)
        profile_success = db.update_profile_settings(profile_id, profile_updates) if profile_updates else False
        if success_global or profile_success:
            merged = db.get_settings()
            profile_settings = db.get_profile_settings(profile_id)
            merged["wake_word_sensitivity"] = profile_settings.get("wake_word_sensitivity", 0.7)
            merged["activation_sound_path"] = profile_settings.get("activation_sound_path")
            return jsonify({"success": True, "settings": merged})
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
    elif provider == "openai":
        # Return common OpenAI chat models
        return jsonify({
            "models": [
                {"id": "gpt-4o-mini", "name": "GPT-4o Mini"},
                {"id": "gpt-4o", "name": "GPT-4o"},
                {"id": "gpt-3.5-turbo", "name": "GPT-3.5 Turbo"}
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

# Wake Word Management
@app.route("/api/wake-words", methods=["GET", "POST"])
def wake_words_api():
    profile_id = request.headers.get("X-Profile-ID", "default")
    try:
        if request.method == "GET":
            wake_words = db.get_wake_words(profile_id)
            return jsonify({"wake_words": wake_words})

        data = request.json or {}
        word = data.get("word")
        if not word:
            return jsonify({"error": "word is required"}), 400
        new_word = db.add_wake_word(profile_id, word)
        return jsonify(new_word), 201
    except Exception as e:
        logger.error(f"Wake word error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/wake-words/<wake_word_id>", methods=["DELETE"])
def delete_wake_word_api(wake_word_id):
    try:
        success = db.delete_wake_word(wake_word_id)
        if not success:
            return jsonify({"error": "Wake word not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete wake word: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/wake-words/activation-sound", methods=["POST"])
def upload_activation_sound():
    """Upload a custom wake-word activation sound and store path in profile settings"""
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique = f"{int(datetime.now().timestamp())}_{filename}"
            filepath = os.path.join(app.config['WAKE_SOUND_FOLDER'], unique)
            file.save(filepath)

            relative_path = f"/static/uploads/wake_sounds/{unique}"
            # Persist to profile settings
            db.update_profile_settings(profile_id, {"activation_sound_path": relative_path})

            return jsonify({"success": True, "path": relative_path}), 201

        return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        logger.error(f"Failed to upload activation sound: {e}")
        return jsonify({"error": str(e)}), 500

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

@app.route("/api/voice-profiles", methods=["GET", "POST"])
def voice_profiles():
    """List or create voice profiles (metadata only)"""
    if request.method == "GET":
        try:
            profiles = db.list_voice_profiles()
            return jsonify({"profiles": profiles})
        except Exception as e:
            logger.error(f"Failed to list voice profiles: {e}")
            return jsonify({"error": str(e)}), 500
    else:
        try:
            payload_json = request.json if request.is_json else None
            name = (payload_json or {}).get("name") if payload_json else request.form.get("name")
            description = (payload_json or {}).get("description", "") if payload_json else request.form.get("description", "")
            provider = (payload_json or {}).get("provider", "supertonic") if payload_json else request.form.get("provider", "supertonic")
            cloned = (payload_json or {}).get("cloned", False) if payload_json else request.form.get("cloned", "0")
            sample_path = ""

            if not payload_json and 'sample' in request.files:
                file = request.files['sample']
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    unique = f"{int(datetime.now().timestamp())}_{filename}"
                    filepath = os.path.join(app.config['VOICE_UPLOAD_FOLDER'], unique)
                    file.save(filepath)
                    sample_path = f"/static/uploads/voices/{unique}"

            if not name:
                return jsonify({"error": "Name required"}), 400

            voice_id = db.create_voice_profile(name, description, sample_path, provider, cloned in [True, "1", "true"])
            return jsonify({
                "success": True,
                "voice": {
                    "id": voice_id,
                    "name": name,
                    "description": description,
                    "sample_path": sample_path,
                    "provider": provider,
                    "cloned": cloned in [True, "1", "true"]
                }
            }), 201
        except Exception as e:
            logger.error(f"Failed to create voice profile: {e}")
            return jsonify({"error": str(e)}), 500

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

@app.route("/api/conversations/<conversation_id>/folder", methods=["PATCH"])
def set_conversation_folder(conversation_id):
    """Assign or clear a folder on a conversation"""
    try:
        data = request.json or {}
        folder_id = data.get("folder_id")
        success = db.set_conversation_folder(conversation_id, folder_id)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        return jsonify({"success": True, "folder_id": folder_id})
    except Exception as e:
        logger.error(f"Failed to set folder: {e}")
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

@app.route("/api/search", methods=["GET"])
def search_messages_api():
    """Full-text search across all messages"""
    try:
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 50))
        profile_id = request.headers.get("X-Profile-ID", "default")
        
        if not query:
            return jsonify({"results": []}), 200
        
        results = db.search_messages(query, limit, profile_id=profile_id)
        return jsonify({"results": results, "query": query, "count": len(results)})
    except Exception as e:
        logger.error(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500

# Integrations API
@app.route("/api/integrations", methods=["GET"])
def list_integrations():
    """List available integrations and connection state for the current profile"""
    try:
        session_user = get_session_profile()
        if not session_user:
            return jsonify({"error": "auth required"}), 401
        profile_id = request.headers.get("X-Profile-ID", "default")
        available = integration_manager.get_available_integrations()
        connected = db.get_integrations(profile_id)
        connected_by_service = {item["service"]: item for item in connected}

        merged = []
        for item in available:
            connected_item = connected_by_service.get(item["id"])
            connect_url = None
            if item["id"] == "notion":
                host = request.host_url.rstrip('/')
                callback = f"{host}/api/integrations/callback/notion"
                connect_url = NotionIntegration("dummy").get_auth_url(callback) + f"&state={profile_id}"
            if item["id"] == "google_drive":
                host = request.host_url.rstrip('/')
                callback = f"{host}/api/integrations/callback/google_drive"
                connect_url = GoogleDriveIntegration("dummy").get_auth_url(callback) + f"&state={profile_id}"
            merged.append({
                **item,
                "connected": connected_item is not None,
                "connected_at": connected_item.get("created_at") if connected_item else None,
                "integration_id": connected_item.get("id") if connected_item else None,
                "config": connected_item.get("config") if connected_item else {},
                "connect_url": connect_url
            })
        return jsonify({"integrations": merged})
    except Exception as e:
        logger.error(f"Failed to list integrations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrations/connect", methods=["POST"])
def connect_integration():
    """Create or update an integration connection"""
    try:
        session_user = get_session_profile()
        if not session_user:
            return jsonify({"error": "auth required"}), 401
        profile_id = request.headers.get("X-Profile-ID", "default")
        data = request.json or {}
        service = data.get("service")
        config = data.get("config", {})
        code = data.get("code")
        redirect_uri = data.get("redirect_uri", "")
        if not service:
            return jsonify({"error": "service is required"}), 400

        instance = integration_manager.get_integration_instance(service, profile_id)
        if not instance:
            return jsonify({"error": "Unknown integration"}), 404

        tokens = {}
        if hasattr(instance, "handle_callback"):
            tokens = instance.handle_callback(code or "dummy_code", redirect_uri)

        integration_id = db.create_integration(
            profile_id,
            service,
            access_token=tokens.get("access_token"),
            refresh_token=tokens.get("refresh_token"),
            expires_at=tokens.get("expires_at"),
            config=config
        )
        return jsonify({"success": True, "integration_id": integration_id}), 201
    except Exception as e:
        logger.error(f"Failed to connect integration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrations/<integration_id>", methods=["DELETE"])
def delete_integration_api(integration_id):
    """Disconnect an integration"""
    try:
        session_user = get_session_profile()
        if not session_user:
            return jsonify({"error": "auth required"}), 401
        success = db.delete_integration(integration_id)
        if not success:
            return jsonify({"error": "Integration not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete integration: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrations/callback/<service>")
def integration_callback(service):
    """OAuth callback for integrations"""
    profile_id = request.args.get("state") or request.args.get("profile_id", "default")
    code = request.args.get("code")
    redirect_uri = request.base_url
    instance = integration_manager.get_integration_instance(service, profile_id)
    if not instance or not code:
        return jsonify({"error": "Invalid callback"}), 400
    tokens = instance.handle_callback(code, redirect_uri)
    db.create_integration(
        profile_id,
        service,
        access_token=tokens.get("access_token"),
        refresh_token=tokens.get("refresh_token"),
        expires_at=tokens.get("expires_at"),
        config={}
    )
    return redirect("/?integration=success")

# Notion actions
def _get_integration_token(profile_id: str, service: str) -> str:
    integrations = db.get_integrations(profile_id)
    for item in integrations:
        if item.get("service") == service:
            return item.get("access_token")
    return None

@app.route("/api/integrations/google_drive/folders", methods=["GET"])
def google_drive_folders():
    profile_id = request.headers.get("X-Profile-ID", "default")
    session_user = get_session_profile()
    if not session_user:
        return jsonify({"folders": []}), 401
    token = _get_integration_token(profile_id, "google_drive")
    if not token:
        return jsonify({"folders": []})
    try:
        headers = {"Authorization": f"Bearer {token}"}
        params = {"q": "mimeType='application/vnd.google-apps.folder' and trashed=false", "pageSize": 50, "fields": "files(id,name)"}
        resp = requests.get("https://www.googleapis.com/drive/v3/files", headers=headers, params=params, timeout=15)
        data = resp.json()
        return jsonify({"folders": data.get("files", [])})
    except Exception as e:
        logger.error(f"Drive folders failed: {e}")
        return jsonify({"folders": []})


@app.route("/api/integrations/notion/pages", methods=["GET", "POST"])
def notion_pages():
    profile_id = request.headers.get("X-Profile-ID", "default")
    session_user = get_session_profile()
    if not session_user:
        return jsonify({"error": "auth required"}), 401
    token = _get_integration_token(profile_id, "notion")
    if not token:
        return jsonify({"error": "Notion not connected"}), 400
    headers = {
        "Authorization": f"Bearer {token}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json"
    }
    try:
        if request.method == "GET":
            import requests
            resp = requests.post("https://api.notion.com/v1/search", headers=headers, json={"page_size": 10}, timeout=15)
            return jsonify(resp.json())
        else:
            data = request.json or {}
            parent = data.get("parent_id")
            title = data.get("title", "From Tanui")
            if not parent:
                return jsonify({"error": "parent_id required"}), 400
            import requests
            payload = {
                "parent": {"page_id": parent},
                "properties": {"title": {"title": [{"text": {"content": title}}]}},
                "children": data.get("children", [])
            }
            resp = requests.post("https://api.notion.com/v1/pages", headers=headers, json=payload, timeout=15)
            return jsonify(resp.json())
    except Exception as e:
        logger.error(f"Notion action failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/integrations/google_drive/upload", methods=["POST"])
def google_drive_upload():
    """Upload a text snippet to Google Drive"""
    profile_id = request.headers.get("X-Profile-ID", "default")
    session_user = get_session_profile()
    if not session_user:
        return jsonify({"error": "auth required"}), 401
    token = _get_integration_token(profile_id, "google_drive")
    if not token:
        return jsonify({"error": "Google Drive not connected"}), 400
    data = request.json or {}
    content = data.get("content", "")
    filename = data.get("filename", "tanui_note.txt")
    parent_id = data.get("parent_id")
    try:
        metadata = {"name": filename}
        if parent_id:
            metadata["parents"] = [parent_id]
        files = {
            'data': ('metadata', json_module.dumps(metadata), 'application/json'),
            'file': ('content', content, 'text/plain')
        }
        headers = {"Authorization": f"Bearer {token}"}
        resp = requests.post(
            "https://www.googleapis.com/upload/drive/v3/files?uploadType=multipart",
            headers=headers,
            files=files,
            timeout=15
        )
        if resp.status_code >= 300:
            return jsonify({"error": "Upload failed", "details": resp.text}), 500
        return jsonify({"success": True, "file": resp.json()})
    except Exception as e:
        logger.error(f"Drive upload failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/filter", methods=["GET"])
def filter_conversations_api():
    """Filter conversations by date and/or model"""
    try:
        start_date = request.args.get("start_date", type=int)
        end_date = request.args.get("end_date", type=int)
        model = request.args.get("model")
        folder_id = request.args.get("folder_id")
        profile_id = request.headers.get("X-Profile-ID", "default")
        
        results = db.filter_conversations(start_date, end_date, model, profile_id=profile_id, folder_id=folder_id)
        return jsonify({"conversations": results, "count": len(results)})
    except Exception as e:
        logger.error(f"Filter failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/context/audit", methods=["GET"])
def context_audit():
    """List conversations used as context for a profile"""
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        data = db.get_context_audit(profile_id)
        return jsonify({"conversations": data})
    except Exception as e:
        logger.error(f"Context audit failed: {e}")
        return jsonify({"error": str(e)}), 500

# Tag Management Endpoints
@app.route("/api/tags", methods=["GET"])
def get_tags_api():
    """Get all tags"""
    try:
        tags = db.get_tags()
        return jsonify({"tags": tags})
    except Exception as e:
        logger.error(f"Failed to get tags: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tags", methods=["POST"])
def create_tag_api():
    """Create a new tag"""
    try:
        data = request.json
        tag_id = data.get("id")
        name = data.get("name")
        color = data.get("color", "#3B82F6")
        icon = data.get("icon", "🏷️")
        
        if not all([tag_id, name]):
            return jsonify({"error": "id and name are required"}), 400
        
        tag = db.create_tag(tag_id, name, color, icon)
        if not tag:
            return jsonify({"error": "Tag with this name already exists"}), 409
        
        return jsonify(tag), 201
    except Exception as e:
        logger.error(f"Failed to create tag: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/tags/<tag_id>", methods=["DELETE"])
def delete_tag_api(tag_id):
    """Delete a tag"""
    try:
        success = db.delete_tag(tag_id)
        if not success:
            return jsonify({"error": "Tag not found"}), 404
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete tag: {e}")
        return jsonify({"error": str(e)}), 500

# Folder Management
@app.route("/api/folders", methods=["GET", "POST"])
def folders_api():
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        if request.method == "GET":
            folders = db.get_folders(profile_id)
            return jsonify({"folders": folders})
        data = request.json or {}
        name = data.get("name")
        color = data.get("color", "#64748b")
        icon = data.get("icon", "📁")
        if not name:
            return jsonify({"error": "name is required"}), 400
        folder = db.create_folder(profile_id, name, color, icon)
        return jsonify(folder), 201
    except Exception as e:
        logger.error(f"Folder error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/folders/<folder_id>", methods=["DELETE"])
def delete_folder(folder_id):
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        success = db.delete_folder(folder_id, profile_id)
        if not success:
            return jsonify({"error": "Folder not found"}), 404
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete folder: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/tags", methods=["POST"])
def add_tag_to_conversation_api(conversation_id):
    """Add a tag to a conversation"""
    try:
        data = request.json
        tag_id = data.get("tag_id")
        
        if not tag_id:
            return jsonify({"error": "tag_id is required"}), 400
        
        success = db.add_tag_to_conversation(conversation_id, tag_id)
        if not success:
            return jsonify({"error": "Tag already added or not found"}), 409
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to add tag: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/tags/<tag_id>", methods=["DELETE"])
def remove_tag_from_conversation_api(conversation_id, tag_id):
    """Remove a tag from a conversation"""
    try:
        success = db.remove_tag_from_conversation(conversation_id, tag_id)
        if not success:
            return jsonify({"error": "Tag not found on conversation"}), 404
        
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to remove tag: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/tags", methods=["GET"])
def get_conversation_tags_api(conversation_id):
    """Get all tags for a conversation"""
    try:
        tags = db.get_conversation_tags(conversation_id)
        return jsonify({"tags": tags})
    except Exception as e:
        logger.error(f"Failed to get tags: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/messages/<message_id>/pin", methods=["PATCH"])
def pin_message_api(message_id):
    """Pin or unpin a single message"""
    try:
        data = request.json or {}
        pinned = data.get("pinned", True)
        success = db.pin_message(message_id, pinned)
        if not success:
            return jsonify({"error": "Message not found"}), 404
        return jsonify({"success": True, "pinned": pinned})
    except Exception as e:
        logger.error(f"Failed to pin message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/pin", methods=["PATCH"])
def pin_conversation_api(conversation_id):
    """Pin or unpin a conversation"""
    try:
        data = request.json
        pinned = data.get("pinned", True)
        
        success = db.pin_conversation(conversation_id, pinned)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"success": True, "pinned": pinned})
    except Exception as e:
        logger.error(f"Failed to pin conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/archive", methods=["PATCH"])
def archive_conversation_api(conversation_id):
    """Archive or unarchive a conversation"""
    try:
        data = request.json
        archived = data.get("archived", True)
        
        success = db.archive_conversation(conversation_id, archived)
        if not success:
            return jsonify({"error": "Conversation not found"}), 404
        
        return jsonify({"success": True, "archived": archived})
    except Exception as e:
        logger.error(f"Failed to archive conversation: {e}")
        return jsonify({"error": str(e)}), 500

# Voice Memos API

@app.route("/api/voice-memos", methods=["GET"])
def get_voice_memos():
    """Get recent voice memos"""
    try:
        limit = int(request.args.get("limit", 50))
        offset = int(request.args.get("offset", 0))
        profile_id = request.headers.get('X-Profile-ID', 'default')
        memos = db.get_voice_memos(profile_id=profile_id, limit=limit, offset=offset)
        return jsonify({"memos": memos})
    except Exception as e:
        logger.error(f"Failed to get voice memos: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice-memos", methods=["POST"])
def create_voice_memo():
    """Create a new voice memo (upload audio)"""
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided"}), 400
            
        file = request.files['audio']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
            
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            # Add timestamp to filename to prevent collisions
            timestamp = int(datetime.now().timestamp())
            unique_filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(filepath)
            
            # Relative path for frontend
            relative_path = f"/static/uploads/memos/{unique_filename}"
            
            title = request.form.get('title', f"Voice Memo {datetime.now().strftime('%Y-%m-%d %H:%M')}")
            duration = float(request.form.get('duration', 0))
            transcription = request.form.get('transcription', "")
            profile_id = request.headers.get('X-Profile-ID', 'default')
            
            # Parse tags if provided
            tags_json = request.form.get('tags', '[]')
            try:
                tags = json_module.loads(tags_json)
            except:
                tags = []
            
            memo_id = db.create_voice_memo(title, relative_path, transcription, duration, tags, profile_id=profile_id)
            
            return jsonify({
                "success": True, 
                "memo": {
                    "id": memo_id,
                    "title": title,
                    "audio_path": relative_path,
                    "transcription": transcription,
                    "duration": duration,
                    "tags": tags,
                    "created_at": timestamp,
                    "profile_id": profile_id
                }
            }), 201
            
        return jsonify({"error": "File type not allowed"}), 400
    except Exception as e:
        logger.error(f"Failed to create voice memo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice-memos/<memo_id>", methods=["GET"])
def get_voice_memo(memo_id):
    """Get a specific voice memo"""
    try:
        memo = db.get_voice_memo(memo_id)
        if not memo:
            return jsonify({"error": "Memo not found"}), 404
        return jsonify({"memo": memo})
    except Exception as e:
        logger.error(f"Failed to get voice memo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice-memos/<memo_id>", methods=["DELETE"])
def delete_voice_memo(memo_id):
    """Delete a voice memo"""
    try:
        # Get memo to find file path
        memo = db.get_voice_memo(memo_id)
        if not memo:
            return jsonify({"error": "Memo not found"}), 404
            
        # Delete from DB
        success = db.delete_voice_memo(memo_id)
        
        # Delete file if exists
        if success and memo.get('audio_path'):
            # Convert relative path back to absolute
            filename = os.path.basename(memo['audio_path'])
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.exists(filepath):
                os.remove(filepath)
                
        return jsonify({"success": success})
    except Exception as e:
        logger.error(f"Failed to delete voice memo: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/voice-memos/search", methods=["GET"])
def search_voice_memos():
    """Search voice memos"""
    try:
        query = request.args.get("q", "")
        limit = int(request.args.get("limit", 20))
        
        if not query:
            return jsonify({"results": []})
            
        results = db.search_voice_memos(query, limit)
        return jsonify({"results": results})
    except Exception as e:
        logger.error(f"Failed to search voice_memos: {e}")
        return jsonify({"error": str(e)}), 500

# RAG sources
@app.route("/api/rag/sources", methods=["GET", "POST"])
def rag_sources():
    profile_id = request.headers.get("X-Profile-ID", "default")
    session_user = get_session_profile()
    if not session_user:
        return jsonify({"error": "auth required"}), 401
    if request.method == "GET":
        try:
            sources = db.list_rag_sources(profile_id)
            return jsonify({"sources": sources})
        except Exception as e:
            logger.error(f"Failed to list RAG sources: {e}")
            return jsonify({"error": str(e)}), 500

    # POST - upload a file or zip
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        name = request.form.get("name") or file.filename
        source_id = str(int(datetime.now().timestamp() * 1000))
        db.upsert_rag_source(source_id, profile_id, name, "upload", status="processing")

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, secure_filename(file.filename))
            file.save(filepath)

            texts = []
            if zipfile.is_zipfile(filepath):
                with zipfile.ZipFile(filepath, 'r') as zf:
                    for member in zf.namelist():
                        if member.endswith('/'):
                            continue
                        if not any(member.lower().endswith(ext) for ext in ['.txt', '.md', '.markdown', '.pdf', '.docx']):
                            continue
                        with zf.open(member) as f:
                            try:
                                extracted_path = os.path.join(tmpdir, secure_filename(member))
                                os.makedirs(os.path.dirname(extracted_path), exist_ok=True)
                                with open(extracted_path, 'wb') as out:
                                    out.write(f.read())
                                texts.extend(_extract_text_from_file(extracted_path))
                            except Exception as e:
                                logger.warning(f"Skip {member}: {e}")
            else:
                if not any(filepath.lower().endswith(ext) for ext in ['.txt', '.md', '.markdown', '.pdf', '.docx']):
                    return jsonify({"error": "Only txt/md/pdf/docx/zip supported for now"}), 400
                texts.extend(_extract_text_from_file(filepath))

        # Clear existing chunks (if any)
        db.clear_rag_chunks(source_id)
        for fname, text in texts:
            for chunk in _chunk_text_for_rag(text):
                embedding = None
                try:
                    embedding = _get_embedding(chunk)
                except Exception as e:
                    logger.warning(f"Embedding failed, using hash: {e}")
                    embedding = _hash_embed(chunk)
                db.add_rag_chunk(source_id, profile_id, chunk, meta={"file": fname}, embedding=embedding)

        db.upsert_rag_source(source_id, profile_id, name, "upload", status="ready", meta={"files": len(texts)})
        return jsonify({"success": True, "id": source_id})
    except Exception as e:
        logger.error(f"Failed to ingest RAG source: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/rag/sources/<source_id>", methods=["DELETE"])
def rag_source_delete(source_id):
    try:
        db.delete_rag_source(source_id)
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete source: {e}")
        return jsonify({"error": str(e)}), 500

# Profile API

@app.route("/api/profiles", methods=["GET"])
def get_profiles():
    """Get all profiles"""
    try:
        profiles = db.get_profiles()
        return jsonify({"profiles": profiles})
    except Exception as e:
        logger.error(f"Failed to get profiles: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles", methods=["POST"])
def create_profile():
    """Create a new profile"""
    try:
        data = request.json
        name = data.get("name")
        if not name:
            return jsonify({"error": "Name is required"}), 400
            
        avatar_path = data.get("avatar_path", "")
        profile_id = db.create_profile(name, avatar_path)
        
        return jsonify({"success": True, "id": profile_id, "name": name}), 201
    except Exception as e:
        logger.error(f"Failed to create profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>", methods=["GET"])
def get_profile(profile_id):
    """Get a specific profile"""
    try:
        profile = db.get_profile(profile_id)
        if not profile:
            return jsonify({"error": "Profile not found"}), 404
        return jsonify({"profile": profile})
    except Exception as e:
        logger.error(f"Failed to get profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id):
    """Delete a profile"""
    try:
        if profile_id == 'default':
            return jsonify({"error": "Cannot delete default profile"}), 400
            
        success = db.delete_profile(profile_id)
        if not success:
            return jsonify({"error": "Profile not found"}), 404
            
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to delete profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>/export", methods=["GET"])
def export_profile(profile_id):
    """Export all data for a profile"""
    try:
        payload = db.export_profile_data(profile_id)
        return jsonify(payload)
    except Exception as e:
        logger.error(f"Failed to export profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>/import", methods=["POST"])
def import_profile(profile_id):
    """Import profile data payload"""
    try:
        data = request.json or {}
        success = db.import_profile_data(profile_id, data)
        if not success:
            return jsonify({"error": "Import failed"}), 400
        return jsonify({"success": True})
    except Exception as e:
        logger.error(f"Failed to import profile: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>/settings", methods=["GET"])
def get_profile_settings(profile_id):
    """Get settings for a profile"""
    try:
        settings = db.get_profile_settings(profile_id)
        return jsonify({"settings": settings})
    except Exception as e:
        logger.error(f"Failed to get profile settings: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/profiles/<profile_id>/settings", methods=["PUT"])
def update_profile_settings(profile_id):
    """Update settings for a profile"""
    try:
        settings = request.json
        success = db.update_profile_settings(profile_id, settings)
        return jsonify({"success": success})
    except Exception as e:
        logger.error(f"Failed to update profile settings: {e}")
        return jsonify({"error": str(e)}), 500

# Conversation API

@app.route("/api/conversations", methods=["GET"])
def get_conversations():
    """Get all conversations for a profile"""
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        folder_id = request.args.get("folder_id")
        conversations = db.get_conversations(profile_id, folder_id=folder_id)
        return jsonify({"conversations": conversations})
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations", methods=["POST"])
def create_conversation():
    """Create a new conversation"""
    try:
        from datetime import datetime
        profile_id = request.headers.get("X-Profile-ID", "default")
        data = request.json or {}
        
        # Use provided ID from frontend
        conversation_id = data.get("id", str(int(datetime.now().timestamp() * 1000)))
        title = data.get("title", "New Chat")
        folder_id = data.get("folder_id")
        
        # Create conversation in database with the provided ID
        db.create_conversation(conversation_id, title, profile_id, folder_id)
        logger.info(f"Created conversation {conversation_id} for profile {profile_id}")
        return jsonify({"id": conversation_id, "title": title, "folder_id": folder_id}), 201
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>", methods=["GET", "DELETE"])
def conversation_by_id(conversation_id):
    """Get or delete a specific conversation"""
    if request.method == "GET":
        try:
            conversation = db.get_conversation(conversation_id)
            if not conversation:
                return jsonify({"error": "Conversation not found"}), 404
            return jsonify({"conversation": conversation})
        except Exception as e:
            logger.error(f"Failed to get conversation: {e}")
            return jsonify({"error": str(e)}), 500
    else:  # DELETE
        try:
            success = db.delete_conversation(conversation_id)
            if not success:
                return jsonify({"error": "Conversation not found"}), 404
            return jsonify({"success": True})
        except Exception as e:
            logger.error(f"Failed to delete conversation: {e}")
            return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/<conversation_id>/messages", methods=["POST"])
def add_message(conversation_id):
    """Add a message to a conversation"""
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        data = request.json
        
        message_id = data.get("id", str(int(datetime.now().timestamp() * 1000)))
        role = data.get("role")
        content = data.get("content")
        model = data.get("model")
        
        if not all([role, content]):
            return jsonify({"error": "Missing required fields"}), 400
        
        message = db.add_message(message_id, conversation_id, role, content, model)
        return jsonify({"success": True, "message": message}), 201
    except Exception as e:
        logger.error(f"Failed to add message: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/conversations/import", methods=["POST"])
def import_conversation():
    """Import a conversation from JSON payload"""
    try:
        profile_id = request.headers.get("X-Profile-ID", "default")
        payload = request.json or {}
        convo = payload.get("conversation") or payload
        if not isinstance(convo, dict):
            return jsonify({"error": "conversation payload required"}), 400

        conversation_id = convo.get("id") or str(int(datetime.now().timestamp() * 1000))
        title = convo.get("title") or "Imported Chat"
        folder_id = convo.get("folder_id")
        db.create_conversation(conversation_id, title, profile_id, folder_id)

        for msg in convo.get("messages", []):
            db.add_message(
                msg.get("id", str(int(datetime.now().timestamp() * 1000))),
                conversation_id,
                msg.get("role", "assistant"),
                msg.get("content", ""),
                msg.get("model")
            )

        return jsonify({"success": True, "id": conversation_id})
    except Exception as e:
        logger.error(f"Failed to import conversation: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/request-code", methods=["POST"])
def auth_request_code():
    try:
        data = request.json or {}
        email = data.get("email")
        if not email:
            return jsonify({"error": "email required"}), 400
        code = str(random.randint(100000, 999999))
        expires = int(datetime.now().timestamp()) + 600
        db.request_login_code(email, code, expires)
        logger.info(f"Login code for {email}: {code}")
        return jsonify({"success": True, "code": code})
    except Exception as e:
        logger.error(f"Auth code error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/verify", methods=["POST"])
def auth_verify():
    try:
        data = request.json or {}
        email = data.get("email")
        code = data.get("code")
        if not email or not code:
            return jsonify({"error": "email and code required"}), 400
        session_token = db.verify_login_code(email, code)
        if not session_token:
            return jsonify({"error": "Invalid or expired code"}), 400
        resp = jsonify({"success": True, "session": session_token})
        resp.set_cookie("session_token", session_token, httponly=True, samesite='Lax')
        return resp
    except Exception as e:
        logger.error(f"Auth verify error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    token = request.cookies.get("session_token")
    if not token:
        return jsonify({"user": None})
    user = db.get_user_by_session(token)
    if not user:
        return jsonify({"user": None})
    return jsonify({"user": {"id": user["id"], "email": user["email"]}})


if __name__ == "__main__":
    print("Starting Web Interface on http://localhost:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)
