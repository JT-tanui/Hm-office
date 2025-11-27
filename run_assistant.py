import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from assistant.main import main

if __name__ == "__main__":
    main()
