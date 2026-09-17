"""Sparse convolution and attention settings for AMD and upstream backends."""
import os

CONV_BACKENDS = ("none", "spconv", "torchsparse", "flex_gemm")
ATTN_BACKENDS = ("xformers", "flash_attn", "flash_attn_3", "sdpa", "aule")
CONV = os.environ.get("SPARSE_CONV_BACKEND", "flex_gemm")
if CONV not in CONV_BACKENDS:
    CONV = "flex_gemm"
ATTN = os.environ.get("SPARSE_ATTN_BACKEND", os.environ.get("ATTN_BACKEND", "sdpa"))
if ATTN not in ATTN_BACKENDS:
    ATTN = "sdpa"
DEBUG = os.environ.get("SPARSE_DEBUG") == "1"


def set_conv_backend(backend):
    if backend not in CONV_BACKENDS:
        raise ValueError(f"Unknown sparse convolution backend: {backend!r}")
    global CONV
    CONV = backend


def set_attn_backend(backend):
    if backend not in ATTN_BACKENDS:
        raise ValueError(f"Unknown sparse attention backend: {backend!r}")
    global ATTN
    ATTN = backend


def set_debug(debug):
    global DEBUG
    DEBUG = bool(debug)
