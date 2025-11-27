import os
import onnxruntime as ort

ASSETS_DIR = r"j:\Assistant\supertonic\assets\onnx"

MODELS = [
    "duration_predictor.onnx",
    "text_encoder.onnx",
    "vector_estimator.onnx",
    "vocoder.onnx"
]

def check_models():
    print("Checking ONNX models...")
    all_good = True
    for model_name in MODELS:
        path = os.path.join(ASSETS_DIR, model_name)
        if not os.path.exists(path):
            print(f"[MISSING] {model_name} not found at {path}")
            all_good = False
            continue
            
        try:
            sess = ort.InferenceSession(path)
            print(f"[OK] {model_name} loaded successfully.")
        except Exception as e:
            print(f"[FAIL] {model_name} failed to load: {e}")
            all_good = False
    
    if all_good:
        print("\nAll models verified!")
    else:
        print("\nSome models failed verification.")

if __name__ == "__main__":
    check_models()
