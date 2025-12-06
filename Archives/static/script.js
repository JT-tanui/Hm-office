document.addEventListener('DOMContentLoaded', () => {
    // UI Elements
    const chatHistory = document.getElementById('chat-history');
    const userInput = document.getElementById('user-input');
    const sendBtn = document.getElementById('send-btn');
    const voiceBtn = document.getElementById('voice-input-btn');
    const providerSelect = document.getElementById('provider-select');
    const apiKeyGroup = document.getElementById('api-key-group');
    const apiKeyInput = document.getElementById('api-key-input');
    const toggleKeyBtn = document.getElementById('toggle-key-visibility');
    const modelSelect = document.getElementById('model-select');
    const voiceSelect = document.getElementById('voice-select');
    const ttsToggle = document.getElementById('tts-toggle');
    const captionsToggle = document.getElementById('captions-toggle');
    const systemPromptInput = document.getElementById('system-prompt-input');

    // Call Mode Elements
    const callModeToggle = document.getElementById('call-mode-toggle');
    const chatView = document.getElementById('chat-view');
    const callView = document.getElementById('call-view');
    const avatarPulse = document.getElementById('avatar-pulse');
    const callStatusText = document.getElementById('call-status-text');
    const callTranscript = document.getElementById('call-transcript');
    const callMicBtn = document.getElementById('call-mic-btn');
    const callEndBtn = document.getElementById('call-end-btn');

    let recognition = null;
    let isRecording = false;
    let searchInput = null;
    let isCallMode = false;

    // Conversation history for context
    let conversationHistory = [];

    const BASE_SYSTEM_PROMPT = `You are a helpful AI assistant. When responding:
- Keep your language clear and conversational
- Avoid excessive punctuation or special characters
- Use natural speech patterns
- Break complex ideas into digestible sentences
- Avoid markdown formatting, code blocks, or technical symbols in your explanations
- If you need to reference code or technical terms, spell them out naturally`;

    // Load saved settings
    const savedSystemPrompt = localStorage.getItem('system_prompt');
    if (savedSystemPrompt) systemPromptInput.value = savedSystemPrompt;

    const savedApiKey = localStorage.getItem('openrouter_api_key');
    if (savedApiKey) apiKeyInput.value = savedApiKey;

    // Event Listeners
    systemPromptInput.addEventListener('change', () => localStorage.setItem('system_prompt', systemPromptInput.value));
    apiKeyInput.addEventListener('change', () => localStorage.setItem('openrouter_api_key', apiKeyInput.value));

    toggleKeyBtn.addEventListener('click', () => {
        apiKeyInput.type = apiKeyInput.type === 'password' ? 'text' : 'password';
        toggleKeyBtn.textContent = apiKeyInput.type === 'password' ? '👁️' : '🙈';
    });

    // Initialize Speech Recognition
    if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        recognition = new SpeechRecognition();
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'en-US';

        recognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            if (isCallMode) {
                callTranscript.textContent = `You: ${transcript}`;
                sendMessage(transcript);
            } else {
                userInput.value = transcript;
                userInput.focus();
            }
        };

        recognition.onend = () => {
            isRecording = false;
            voiceBtn.classList.remove('recording');
            if (isCallMode) {
                callMicBtn.classList.remove('recording');
                // In call mode, we might want to auto-restart listening after response
                // But for now, let's keep it manual or triggered by silence
            }
        };

        recognition.onerror = (event) => {
            console.error('Speech recognition error:', event.error);
            isRecording = false;
            voiceBtn.classList.remove('recording');
            if (isCallMode) callStatusText.textContent = "Error: " + event.error;
        };
    } else {
        voiceBtn.disabled = true;
        voiceBtn.title = 'Voice input not supported in this browser';
        if (callMicBtn) callMicBtn.disabled = true;
    }

    // Voice Input Handlers
    voiceBtn.addEventListener('click', toggleVoiceInput);
    callMicBtn.addEventListener('click', toggleVoiceInput);

    function toggleVoiceInput() {
        if (!recognition) return;
        if (isRecording) {
            recognition.stop();
        } else {
            recognition.start();
            isRecording = true;
            voiceBtn.classList.add('recording');
            if (isCallMode) {
                callMicBtn.classList.add('recording');
                callStatusText.textContent = "Listening...";
            }
        }
    }

    // Call Mode Toggle
    callModeToggle.addEventListener('click', () => {
        isCallMode = !isCallMode;
        if (isCallMode) {
            chatView.style.display = 'none';
            callView.style.display = 'flex';
            callModeToggle.textContent = '💬 Switch to Chat Mode';
            // Auto-start voice in call mode
            toggleVoiceInput();
        } else {
            chatView.style.display = 'flex';
            callView.style.display = 'none';
            callModeToggle.textContent = '📞 Switch to Call Mode';
            if (isRecording) recognition.stop();
        }
    });

    callEndBtn.addEventListener('click', () => {
        // End call and switch back
        if (isRecording) recognition.stop();
        isCallMode = false;
        chatView.style.display = 'flex';
        callView.style.display = 'none';
        callModeToggle.textContent = '📞 Switch to Call Mode';
    });

    // Clear History button
    const clearHistoryBtn = document.getElementById('clear-history-btn');
    if (clearHistoryBtn) {
        clearHistoryBtn.addEventListener('click', () => {
            conversationHistory = [];
            console.log("Conversation history cleared");
            alert("Conversation history cleared! Starting fresh.");
        });
    }

    // Provider Selection
    providerSelect.addEventListener('change', async () => {
        const provider = providerSelect.value;
        apiKeyGroup.style.display = provider === 'openrouter' ? 'flex' : 'none';
        await loadModels(provider);
    });

    // Model Loading
    async function loadModels(provider) {
        try {
            const response = await fetch(`/api/models?provider=${provider}`);
            const data = await response.json();

            modelSelect.innerHTML = '';
            data.models.forEach(model => {
                const option = document.createElement('option');
                option.value = typeof model === 'string' ? model : model.id;
                option.text = typeof model === 'string' ? model : model.name;
                modelSelect.add(option);
            });

            if (provider === 'openrouter') enableModelSearch();
            else disableModelSearch();
        } catch (err) {
            console.error('Failed to load models:', err);
        }
    }

    // Model Search
    function enableModelSearch() {
        if (searchInput) return;
        searchInput = document.createElement('input');
        searchInput.type = 'text';
        searchInput.placeholder = 'Search models...';
        searchInput.style.cssText = `width: 100%; padding: 8px; margin-bottom: 5px; border-radius: 5px; border: 1px solid #45475a; background-color: var(--input-bg); color: var(--text-color); outline: none;`;
        modelSelect.parentNode.insertBefore(searchInput, modelSelect);
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            Array.from(modelSelect.options).forEach(opt => {
                opt.style.display = opt.text.toLowerCase().includes(term) ? '' : 'none';
            });
        });
    }

    function disableModelSearch() {
        if (searchInput) {
            searchInput.remove();
            searchInput = null;
        }
        Array.from(modelSelect.options).forEach(opt => opt.style.display = '');
    }

    // Voice Loading
    fetch('/api/voices').then(res => res.json()).then(data => {
        voiceSelect.innerHTML = '';
        if (data.voices.length === 0) {
            voiceSelect.add(new Option("No voices found", ""));
            return;
        }
        data.voices.forEach(voice => voiceSelect.add(new Option(voice, voice)));
    }).catch(err => console.error("Failed to load voices:", err));

    // Chat Logic
    userInput.addEventListener('input', function () {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
    });

    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });

    sendBtn.addEventListener('click', () => sendMessage());

    async function sendMessage(textOverride = null) {
        const text = textOverride || userInput.value.trim();
        if (!text) return;

        const provider = providerSelect.value;
        const apiKey = provider === 'openrouter' ? apiKeyInput.value : null;

        if (provider === 'openrouter' && !apiKey) {
            alert('Please enter your OpenRouter API key.');
            return;
        }

        if (!isCallMode) {
            appendMessage(text, 'user');
            userInput.value = '';
            userInput.style.height = 'auto';
            userInput.disabled = true;
            sendBtn.disabled = true;
        } else {
            callStatusText.textContent = "Thinking...";
        }

        const loadingId = !isCallMode ? appendLoading() : null;

        let fullSystemPrompt = BASE_SYSTEM_PROMPT;
        const customPrompt = systemPromptInput.value.trim();
        if (customPrompt) fullSystemPrompt += `\n\nAdditional Instructions:\n${customPrompt}`;

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: text,
                    model: modelSelect.value,
                    voice: voiceSelect.value,
                    provider: provider,
                    api_key: apiKey,
                    system_prompt: fullSystemPrompt,
                    tts: ttsToggle.checked,
                    conversation_history: conversationHistory  // Add conversation context
                })
            });

            const data = await response.json();

            if (!isCallMode) removeMessage(loadingId);

            if (data.error) {
                if (!isCallMode) appendMessage(`Error: ${data.error}`, 'assistant');
                else callStatusText.textContent = "Error: " + data.error;
            } else {
                if (!isCallMode) {
                    const msgId = appendMessage(data.text, 'assistant', data.model);
                    if (data.audio && ttsToggle.checked) playAudio(data.audio, msgId);

                    // Store in conversation history
                    conversationHistory.push({ role: 'user', content: text });
                    conversationHistory.push({ role: 'assistant', content: data.text });
                } else {
                    callStatusText.textContent = "Speaking...";
                    avatarPulse.classList.add('speaking');
                    callTranscript.textContent = data.text; // Show AI response in transcript
                    if (data.audio && ttsToggle.checked) playAudio(data.audio, 'call-mode');

                    // Store in conversation history (call mode)
                    conversationHistory.push({ role: 'user', content: text });
                    conversationHistory.push({ role: 'assistant', content: data.text });
                }
            }

        } catch (err) {
            if (!isCallMode) {
                removeMessage(loadingId);
                appendMessage(`Network Error: ${err.message}`, 'assistant');
            } else {
                callStatusText.textContent = "Network Error";
            }
        } finally {
            if (!isCallMode) {
                userInput.disabled = false;
                sendBtn.disabled = false;
                userInput.focus();
            }
        }
    }

    function appendMessage(text, sender, model = null) {
        const id = Date.now().toString();
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${sender}`;
        msgDiv.id = `msg-${id}`;

        // Use marked.parse for markdown rendering
        const formattedText = marked.parse(text);

        let content = `<div class="bubble">${formattedText}</div>`;

        if (sender === 'assistant') {
            content += `
                <div class="meta">
                    <span>${model || 'Unknown Model'}</span>
                    <button class="read-aloud-btn" onclick="readAloud('${id}', \`${text.replace(/`/g, '\\`').replace(/\$/g, '\\$')}\`)" title="Read aloud">
                        🔊 Read Aloud
                    </button>
                    <span class="speaking-indicator" id="speaking-${id}">Speaking...</span>
                </div>
            `;
        }

        msgDiv.innerHTML = content;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;

        if (sender === 'assistant') updateCaptionsVisibility();
        return id;
    }

    function appendLoading() {
        const id = Date.now().toString();
        const msgDiv = document.createElement('div');
        msgDiv.className = 'message assistant';
        msgDiv.id = `msg-${id}`;
        msgDiv.innerHTML = `<div class="bubble">Thinking...</div>`;
        chatHistory.appendChild(msgDiv);
        chatHistory.scrollTop = chatHistory.scrollHeight;
        return id;
    }

    function removeMessage(id) {
        const el = document.getElementById(`msg-${id}`);
        if (el) el.remove();
    }

    // Audio playback with stop functionality
    let currentAudio = null;

    function stopSpeaking() {
        if (currentAudio) {
            currentAudio.pause();
            currentAudio.currentTime = 0;
            currentAudio = null;
        }

        // Hide all speaking indicators
        const indicators = document.querySelectorAll('.speaking-indicator');
        indicators.forEach(ind => ind.style.display = 'none');
    }

    function playAudio(audioSource, msgId) {
        console.log("Playing audio for", msgId);

        // Stop any existing audio
        stopSpeaking();

        try {
            const audio = new Audio(audioSource);
            currentAudio = audio;

            let indicator = null;
            if (msgId && msgId !== 'call-mode') {
                indicator = document.getElementById(`speaking-${msgId}`);
                if (indicator) indicator.style.display = 'block';
            }

            audio.onended = () => {
                console.log("Playback ended");
                stopSpeaking();
            };

            audio.onerror = (e) => {
                console.error("Audio error:", e);
                stopSpeaking();
            };

            audio.play().catch(e => console.error("Playback failed:", e));
        } catch (e) {
            console.error("Error creating audio:", e);
        }
    }

    function updateCaptionsVisibility() {
        const bubbles = document.querySelectorAll('.message.assistant .bubble');
        bubbles.forEach(b => {
            b.style.display = captionsToggle.checked ? 'block' : 'none';
        });
    }

    captionsToggle.addEventListener('change', updateCaptionsVisibility);

    // Read Aloud - Generate TTS on demand
    window.readAloud = async function (msgId, text) {
        console.log("Read Aloud requested for:", msgId);
        const indicator = document.getElementById(`speaking-${msgId}`);
        if (indicator) indicator.style.display = 'block';

        try {
            const provider = providerSelect.value;
            const apiKey = provider === 'openrouter' ? apiKeyInput.value : null;

            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    text: "Generate TTS for: " + text,
                    model: modelSelect.value,
                    voice: voiceSelect.value,
                    provider: provider,
                    api_key: apiKey,
                    tts: true
                })
            });

            const data = await response.json();

            if (data.audio) {
                playAudio(data.audio, msgId);
            } else {
                console.error("No audio returned");
                if (indicator) indicator.style.display = 'none';
            }
        } catch (err) {
            console.error("Read Aloud error:", err);
            if (indicator) indicator.style.display = 'none';
        }
    };

    // Initial load
    loadModels(providerSelect.value);
});
