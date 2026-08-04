"""Shared video upload/persistence for Streamlit UI."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = PROJECT_ROOT / ".streamlit_uploads"


def persist_uploaded_video(uploaded_file) -> Path:
    """Save uploaded video bytes to a stable local path for OpenCV."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    dest = UPLOAD_DIR / Path(uploaded_file.name).name
    dest.write_bytes(uploaded_file.getbuffer())
    return dest
