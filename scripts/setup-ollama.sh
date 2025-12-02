#!/bin/bash
# Setup Ollama with cloud model on Leonel Group server

set -e

echo "Setting up Ollama..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
fi

# Start Ollama service
echo "Starting Ollama service..."
systemctl start ollama || service ollama start || true
systemctl enable ollama || true

# Wait for Ollama to be ready
echo "Waiting for Ollama to be ready..."
sleep 3

# Pull the minimax-m2:cloud model
echo "Pulling minimax-m2:cloud model (this may take a few minutes)..."
ollama pull minimax-m2:cloud || ollama run minimax-m2:cloud

echo ""
echo "✅ Ollama setup complete!"
echo ""
echo "Test the model:"
echo "  ollama run minimax-m2:cloud 'Hello, how are you?'"
echo ""
echo "Check available models:"
echo "  ollama list"
