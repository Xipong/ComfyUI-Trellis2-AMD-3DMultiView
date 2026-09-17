"""Backend selection for the AMD integration; no ComfyUI imports here."""
import importlib.util
import logging

log = logging.getLogger(__name__)
BACKENDS = ("auto", "aule", "sdpa", "flash_attn", "xformers", "flash_attn_3")
MODULES = {"aule": "aule", "flash_attn": "flash_attn", "xformers": "xformers",
           "flash_attn_3": "flash_attn_interface"}


def has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def select_backend(requested, *, rocm, available=has_module):
    """Resolve auto and old NVIDIA workflow settings before model loading.

    Aule is preferred on ROCm because unfused SDPA can exhaust RDNA2 VRAM.
    Explicit Aule requests never silently turn into a quadratic-memory fallback.
    """
    if requested not in BACKENDS:
        raise ValueError(f"Unknown attention backend: {requested!r}")
    preferred = "aule" if rocm else "flash_attn"
    automatic = preferred if available(MODULES[preferred]) else "sdpa"
    if requested == "auto":
        return automatic
    if requested == "aule" and not available("aule"):
        raise RuntimeError("Aule is not installed. Install requirements-amd.txt in the "
                           "ComfyUI Python environment, or explicitly select sdpa.")
    # Existing upstream workflows often serialize flash_attn. Do not wait for a
    # large sparse attention call to discover that its NVIDIA extension is absent.
    if rocm and requested in ("flash_attn", "xformers", "flash_attn_3"):
        if requested == "flash_attn_3" or not available(MODULES[requested]):
            log.warning("TRELLIS.2: %s is unavailable on this ROCm setup; using %s.",
                        requested, automatic)
            return automatic
    return requested
