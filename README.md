# Tanui Assistant 🎙️

A modern AI voice assistant with Text-to-Speech (TTS), wake word detection, and multi-provider LLM support.

## Features

✨ **Voice Interaction**
- Voice input with Web Speech API
- Natural-sounding TTS (Supertonic voices)
- Wake word detection ("Hey Tanui")
- Call Mode for hands-free conversation

🤖 **Multi-Provider LLM**
- Local models via Ollama
- Cloud models via OpenRouter
- Dynamic context management
- Cross-conversation memory

💬 **Conversation Management**
- Persistent chat history (SQLite)
- Multi-conversation support
- Export functionality
- Context toggle per conversation

🎨 **Modern UI**
- Responsive design (desktop & mobile)
- Mobile-friendly drawer sidebar
- Real-time streaming responses
- Dark mode interface

## Quick Start

### Prerequisites

- Python 3.11+
- Node.js 20+
- Ollama (for local models) or OpenRouter API key (for cloud models)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd Assistant
   ```

2. **Backend Setup**
   ```bash
   pip install -r requirements.txt
   ```

3. **Frontend Setup**
   ```bash
   cd src/web/client
   npm install
   cd ../../..
   ```

4. **Environment Configuration**
   ```bash
   cp .env.example .env
   # Edit .env and add your OPENROUTER_API_KEY
   ```

### Running with Docker

```bash
# Build and start all services
docker-compose up --build

# Access the application
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

### Running Locally

1. **Start Ollama** (if using local models)
   ```bash
   ollama serve
   ```

2. **Start Backend**
   ```bash
   python src/web/app.py
   ```

3. **Start Frontend** (in a new terminal)
   ```bash
   cd src/web/client
   npm run dev
   ```

4. **Access**: Open http://localhost:3000

## Configuration

### Ollama Models

Download models for local inference:
```bash
# Fast, lightweight
ollama pull phi3:mini

# Coding-focused
ollama pull qwen2.5-coder:7b

# General purpose
ollama pull llama3:8b
```

### OpenRouter

1. Get API key from https://openrouter.ai
2. Add to `.env`: `OPENROUTER_API_KEY=sk-or-v1-...`
3. Select "OpenRouter" as provider in settings

## Architecture

```
Assistant/
├── src/
│   ├── assistant/          # Core AI logic
│   │   ├── llm.py         # Ollama client
│   │   ├── openrouter.py  # OpenRouter client
│   │   ├── pipeline.py    # Main processing pipeline
│   │   └── tts.py         # Text-to-Speech
│   └── web/
│       ├── app.py         # Flask backend
│       ├── database.py    # SQLite operations
│       └── client/        # Next.js frontend
├── docker-compose.yml
└── requirements.txt
```

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | OpenRouter API key | - |
| `OLLAMA_HOST` | Ollama server URL | `http://localhost:11434` |
| `FLASK_ENV` | Flask environment | `development` |
| `DATABASE_PATH` | SQLite database path | `conversations.db` |

## Development

### Frontend Development
```bash
cd src/web/client
npm run dev
```

### Backend Development
```bash
# Enable debug mode
export FLASK_DEBUG=1
python src/web/app.py
```

## Troubleshooting

**Ollama timeouts?**
- Use smaller models (`phi3:mini`)
- Reduce context in settings
- Check `ollama serve` is running

**OpenRouter rate limits?**
- Wait 60 seconds between requests (free tier)
- Use different free models
- Add credits to your account

**No audio?**
- Enable TTS in settings
- Check browser console for errors
- Ensure HTTPS or localhost

## License

MIT

## Contributing

Pull requests welcome! Please open an issue first to discuss major changes.
