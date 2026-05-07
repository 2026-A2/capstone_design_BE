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
    print(f"[OK] Input file  : {file_path}")
    print(f"[OK] File size   : {size_mb:.2f} MB")
    return file_path