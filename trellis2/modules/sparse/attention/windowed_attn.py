"""Window attention with explicit Aule/SDPA support and coordinate-safe pairing.

The AMD upstream added Aule to the selector but left both window dispatchers
without an Aule branch. Partitions here are backend-independent, so switching
attention backends cannot reuse a cached xformers-only attention mask.
"""
import logging
import torch
from .. import SparseTensor, config

__all__ = ["sparse_windowed_scaled_dot_product_self_attention",
           "sparse_windowed_scaled_dot_product_cross_attention"]
log = logging.getLogger(__name__)


def _window_coords(tensor, window_size, shift_window):
    dims = tensor.coords.shape[1] - 1
    size = (window_size,) * dims if isinstance(window_size, int) else tuple(window_size)
    shift = (shift_window,) * dims if isinstance(shift_window, int) else tuple(shift_window)
    if len(size) != dims or len(shift) != dims or any(s <= 0 for s in size):
        raise ValueError("Window sizes must be positive and match the coordinate dimensions")
    coords = tensor.coords.to(torch.int64).clone()
    coords[:, 1:] = torch.div(
        coords[:, 1:] + coords.new_tensor(shift), coords.new_tensor(size),
        rounding_mode="floor")
    return coords


def _partition(inverse, count):
    fwd = torch.argsort(inverse, stable=True)
    bwd = torch.empty_like(fwd)
    bwd[fwd] = torch.arange(len(fwd), device=fwd.device)
    lengths = torch.bincount(inverse, minlength=count)
    return fwd, bwd, lengths


def calc_window_partition(tensor: SparseTensor, window_size, shift_window=0):
    coords = _window_coords(tensor, window_size, shift_window)
    unique, inverse = torch.unique(coords, dim=0, sorted=True, return_inverse=True)
    fwd, bwd, lengths = _partition(inverse, len(unique))
    cumulative = torch.cat((lengths.new_zeros(1), lengths.cumsum(0))).to(torch.int32)
    return fwd, bwd, lengths, {
        "cu_seqlens": cumulative,
        "max_seqlen": int(lengths.max().item()) if lengths.numel() else 0,
    }


def _attend(q, k, v):
    """[L,H,C] in and out. Never mix tokens from unrelated sparse windows."""
    if q.shape[0] == 0 or k.shape[0] == 0:
        return q.new_zeros((q.shape[0], q.shape[1], v.shape[-1]))
    backend = config.ATTN
    if backend == "aule":
        from aule import flash_attention_triton
        args = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
        return flash_attention_triton(*args, causal=False)[0].transpose(0, 1)
    if backend in ("flash_attn", "flash_attn_3", "xformers"):
        try:
            if backend == "xformers":
                from xformers.ops import memory_efficient_attention
                result = memory_efficient_attention(q[None], k[None], v[None])
            elif backend == "flash_attn":
                from flash_attn import flash_attn_func
                result = flash_attn_func(q[None], k[None], v[None])
            else:
                from flash_attn_interface import flash_attn_func
                result = flash_attn_func(q[None], k[None], v[None])
                if isinstance(result, tuple):
                    result = result[0]
            return result[0]
        except (ImportError, NotImplementedError) as exc:
            # Do not swallow OOM, invalid inputs, or native kernel faults.
            log.warning("TRELLIS.2 window attention: %s unavailable (%s); using SDPA", backend, exc)
    elif backend != "sdpa":
        raise ValueError(f"Unknown sparse window attention backend: {backend!r}")
    args = [x.transpose(0, 1).unsqueeze(0) for x in (q, k, v)]
    return torch.nn.functional.scaled_dot_product_attention(
        *args, dropout_p=0.0, is_causal=False)[0].transpose(0, 1)


def sparse_windowed_scaled_dot_product_self_attention(
        qkv: SparseTensor, window_size: int, shift_window=(0, 0, 0)):
    if qkv.feats.ndim != 4 or qkv.feats.shape[1] != 3:
        raise ValueError("qkv features must have shape [T,3,H,C]")
    if qkv.feats.shape[0] == 0:
        return qkv.replace(qkv.feats.new_empty((0, *qkv.feats.shape[2:])))
    name = f"amd_window_v1_{window_size}_{shift_window}"
    partition = qkv.get_spatial_cache(name)
    if partition is None:
        partition = calc_window_partition(qkv, window_size, shift_window)
        qkv.register_spatial_cache(name, partition)
    fwd, bwd, lengths, _ = partition
    outputs = [_attend(*chunk.unbind(1))
               for chunk in qkv.feats[fwd].split(lengths.tolist())]
    return qkv.replace(torch.cat(outputs, dim=0)[bwd])


def sparse_windowed_scaled_dot_product_cross_attention(
        q: SparseTensor, kv: SparseTensor, q_window_size: int, kv_window_size: int,
        q_shift_window=(0, 0, 0), kv_shift_window=(0, 0, 0)):
    if q.feats.ndim != 3 or kv.feats.ndim != 4 or kv.feats.shape[1] != 2:
        raise ValueError("Expected q features [T,H,C] and kv features [T,2,H,C]")
    if q.shape[0] != kv.shape[0] or q.device != kv.device:
        raise ValueError("q and kv must share batch size and device")
    if q.feats.shape[0] == 0:
        return q.replace(q.feats.new_empty((0, q.feats.shape[1], kv.feats.shape[-1])))
    # Pair by (batch, x-window, y-window, z-window), NOT by the ordinal
    # positions of nonempty windows. Q and KV may have different occupancy.
    qc = _window_coords(q, q_window_size, q_shift_window)
    kc = _window_coords(kv, kv_window_size, kv_shift_window)
    unique, inverse = torch.unique(torch.cat((qc, kc)), dim=0, sorted=True,
                                   return_inverse=True)
    qf, qb, qlengths = _partition(inverse[:len(qc)], len(unique))
    kf, _, klengths = _partition(inverse[len(qc):], len(unique))
    outputs = []
    for qs, ks in zip(q.feats[qf].split(qlengths.tolist()),
                      kv.feats[kf].split(klengths.tolist())):
        if qs.shape[0]:
            outputs.append(_attend(qs, *ks.unbind(1)))
    return q.replace(torch.cat(outputs, dim=0)[qb])
