import os
import sys


SUPPORTED_EXTENSIONS = {".wav", ".mp4", ".m4a"}


def validate_wav(file_path: str) -> str:
    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        sys.exit(1)

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        print(f"[ERROR] Unsupported file type '{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(1)

    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    if size_mb > 25:
        print(f"[ERROR] File exceeds 25 MB limit (Whisper API max): {size_mb:.1f} MB")
        sys.exit(1)

    print(f"[OK] Input file  : {file_path}")
    print(f"[OK] File size   : {size_mb:.2f} MB")
    return file_path


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        print("[ERROR] OPENAI_API_KEY environment variable is not set.")
        print("        Set it with: $env:OPENAI_API_KEY = 'sk-...'")
        sys.exit(1)
    print("[OK] API key     : found")
    return key
