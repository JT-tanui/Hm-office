#!/bin/bash
# Initial setup script for DigitalOcean Droplet
# Run this once on the droplet to prepare it for deployments

set -e

echo "Setting up Assistant deployment environment..."

# Create app directory
mkdir -p /root/assistant
cd /root/assistant

# Install Docker Compose plugin if not already installed
if ! docker compose version &> /dev/null; then
    echo "Installing Docker Compose plugin..."
    apt-get update
    apt-get install -y docker-compose-plugin
fi

# Ensure Ollama is running
if ! systemctl is-active --quiet ollama; then
    echo "Starting Ollama service..."
    systemctl start ollama
    systemctl enable ollama
fi

# Create volumes directories
mkdir -p /root/assistant/data
mkdir -p /root/assistant/audio

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Add the following secrets to your GitHub repository:"
echo "   - DROPLET_HOST: 138.68.59.201"
echo "   - DROPLET_USER: root"
echo "   - DROPLET_SSH_KEY: (your SSH private key)"
echo ""
echo "2. Generate SSH key pair if you haven't:"
echo "   ssh-keygen -t ed25519 -C 'github-actions'"
echo ""
echo "3. Add the public key to /root/.ssh/authorized_keys"
echo ""
echo "4. Push to your repository to trigger deployment"
echo ""
