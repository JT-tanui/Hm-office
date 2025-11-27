import sys
import logging
import argparse
from .pipeline import AssistantPipeline
from utils.env_utils import check_ollama_running

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("assistant.log"),
        logging.StreamHandler(sys.stdout) # Optional: print logs to stdout too, or keep it clean for UI
    ]
)
# Reduce noise from libraries
logging.getLogger("onnxruntime").setLevel(logging.WARNING)

def main():
    print("=== Local Talk Assistant ===")
    
    if not check_ollama_running():
        print("Error: Ollama is not running. Please start Ollama first.")
        return

    pipeline = AssistantPipeline()

    print("\nType 'exit' to quit.")
    print("Commands: /chat, /code-small, /code-medium, /code-large, /vision")
    
    mode = "chat"

    while True:
        try:
            user_input = input(f"\n[{mode}] You: ").strip()
        except KeyboardInterrupt:
            print("\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            break

        # Mode switching
        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd == "/chat":
                mode = "chat"
                print("Switched to General Chat mode.")
            elif cmd == "/code-small":
                mode = "coding_small"
                print("Switched to Coding (Small) mode.")
            elif cmd == "/code-medium":
                mode = "coding_medium"
                print("Switched to Coding (Medium) mode.")
            elif cmd == "/code-large":
                mode = "coding_large"
                print("Switched to Coding (Large) mode.")
            elif cmd == "/vision":
                mode = "vision"
                print("Switched to Vision mode.")
            else:
                print(f"Unknown command: {cmd}")
            continue

        pipeline.process_input(user_input, model_preference=mode)

if __name__ == "__main__":
    main()
