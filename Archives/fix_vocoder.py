import os
import requests
import sys

# URL for the raw LFS file
URL = "https://huggingface.co/Supertone/supertonic/resolve/main/onnx/vocoder.onnx?download=true"
DEST = r"j:\Assistant\supertonic\assets\onnx\vocoder.onnx"
EXPECTED_SIZE_MB = 95 # It should be around 100MB

def download_vocoder():
    print(f"Downloading vocoder from {URL}...")
    if os.path.exists(DEST):
        os.remove(DEST)
        print("Removed existing corrupt file.")

    try:
        with requests.get(URL, stream=True) as r:
            r.raise_for_status()
            total_length = r.headers.get('content-length')
            
            with open(DEST, 'wb') as f:
                if total_length is None: # no content length header
                    f.write(r.content)
                else:
                    dl = 0
                    total_length = int(total_length)
                    for data in r.iter_content(chunk_size=4096):
                        dl += len(data)
                        f.write(data)
                        done = int(50 * dl / total_length)
                        sys.stdout.write(f"\r[{'=' * done}{' ' * (50-done)}] {dl/1024/1024:.2f} MB")
                        sys.stdout.flush()
            
        print("\nDownload complete.")
        
        # Check size
        size_mb = os.path.getsize(DEST) / (1024 * 1024)
        print(f"File size: {size_mb:.2f} MB")
        
        if size_mb < EXPECTED_SIZE_MB:
            print("WARNING: File seems too small. Download might be incomplete or an LFS pointer.")
        else:
            print("Size looks correct.")

    except Exception as e:
        print(f"\nError downloading: {e}")

if __name__ == "__main__":
    # Ensure requests is installed (it usually is, but just in case)
    try:
        import requests
    except ImportError:
        print("Installing requests...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "requests"])
        import requests

    download_vocoder()
