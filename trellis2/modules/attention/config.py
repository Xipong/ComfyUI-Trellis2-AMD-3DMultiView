"""Dense attention configuration, including the ROCm/Aule backend."""
import os

SUPPORTED_BACKENDS = ("xformers", "flash_attn", "flash_attn_3", "sdpa", "naive", "aule")
# Importing the extension must not require an NVIDIA attention package. The
# ComfyUI loader resolves auto to the appropriate backend before inference.
BACKEND = os.environ.get("ATTN_BACKEND", "sdpa")
if BACKEND not in SUPPORTED_BACKENDS:
    BACKEND = "sdpa"
DEBUG = os.environ.get("ATTN_DEBUG") == "1"


def set_backend(backend):
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unknown dense attention backend: {backend!r}")
    global BACKEND
    BACKEND = backend


def set_debug(debug):
    global DEBUG
    DEBUG = bool(debug)
