import os
import urllib.request

BASE_URL = "https://huggingface.co/Supertone/supertonic/resolve/main/onnx"
DEST_DIR = r"j:\Assistant\supertonic\assets\onnx"

FILES = [
    "vector_estimator.onnx",
    "vocoder.onnx"
]

def download_file(filename):
    url = f"{BASE_URL}/{filename}?download=true"
    dest_path = os.path.join(DEST_DIR, filename)
    print(f"Downloading {filename} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"Successfully downloaded {filename}")
    except Exception as e:
        print(f"Failed to download {filename}: {e}")

if __name__ == "__main__":
    if not os.path.exists(DEST_DIR):
        print(f"Directory {DEST_DIR} does not exist!")
    else:
        for f in FILES:
            download_file(f)
