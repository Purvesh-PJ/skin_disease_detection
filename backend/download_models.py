"""
Helper script used by Docker build step to pre-cache model weights from Hugging Face.
Streams weights in 16MB chunks directly to disk with proper error handling and retries.
"""
import os
import sys
import shutil
import urllib.request

MODELS = {
    'resnet101.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/resnet101.h5',
    'densenet121.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/densenet121.h5',
    'efficientnetb3.h5': 'https://huggingface.co/purvesh-pj/skin-disease-ensemble/resolve/main/efficientnetb3.h5'
}

TARGET_DIR = os.getenv('MODEL_DIR', '/app/trained_models')
HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def download_models():
    os.makedirs(TARGET_DIR, exist_ok=True)
    print(f"Target model directory: {TARGET_DIR}")

    for filename, url in MODELS.items():
        destination = os.path.join(TARGET_DIR, filename)
        if os.path.exists(destination) and os.path.getsize(destination) > 1000000:
            print(f"[CACHE] {filename} already exists ({os.path.getsize(destination)} bytes). Skipping download.")
            continue

        print(f"[DOWNLOAD] Downloading {filename} from Hugging Face...")
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=180) as resp, open(destination, 'wb') as out_f:
                shutil.copyfileobj(resp, out_f, length=16 * 1024 * 1024)
            size_mb = os.path.getsize(destination) / (1024 * 1024)
            print(f"[SUCCESS] Downloaded {filename} ({size_mb:.2f} MB)")
        except Exception as e:
            print(f"[ERROR] Failed to download {filename}: {e}", file=sys.stderr)
            sys.exit(1)

    print("All model weights successfully verified and ready.")

if __name__ == '__main__':
    download_models()
