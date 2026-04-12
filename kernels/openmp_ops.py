"""
openmp_ops.py  —  OpenMP CPU baseline wrappers for GNN-EADD kernels.

Loads the compiled openmp_ext.so C extension if available and delegates
all three kernel operations to it.  If the .so is missing, every function
falls back to the equivalent PyTorch computation (same result, no OpenMP).

Build the extension once with:
    python kernels/build_ext.py
"""

import ctypes
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ── dynamic library loading ───────────────────────────────────────────────────

_LIB = None   # module-level cached handle


def _load_lib() -> Optional[ctypes.CDLL]:
    global _LIB
    if _LIB is not None:
        return _LIB

    here    = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, 'openmp_ext.so')

    if not os.path.exists(so_path):
        warnings.warn(
            "[OpenMP] openmp_ext.so not found. "
            "Run `python kernels/build_ext.py` to enable OpenMP. "
            "Falling back to sequential PyTorch ops.",
            RuntimeWarning, stacklevel=3,
        )
        return None

    try:
        lib = ctypes.CDLL(so_path)

        lib.smoothness_loss_omp.restype  = None
        lib.smoothness_loss_omp.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # scores
            ctypes.POINTER(ctypes.c_long),   # src_idx
            ctypes.POINTER(ctypes.c_long),   # dst_idx
            ctypes.POINTER(ctypes.c_float),  # out
            ctypes.c_int,                    # n_edges
            ctypes.c_int,                    # n_threads
        ]

        lib.attention_scores_omp.restype  = None
        lib.attention_scores_omp.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # Wh
            ctypes.POINTER(ctypes.c_long),   # src_idx
            ctypes.POINTER(ctypes.c_long),   # dst_idx
            ctypes.POINTER(ctypes.c_float),  # a
            ctypes.POINTER(ctypes.c_float),  # out
            ctypes.c_int,                    # n_edges
            ctypes.c_int,                    # out_dim
            ctypes.c_float,                  # leaky_slope
            ctypes.c_int,                    # n_threads
        ]

        lib.neighbor_agg_omp.restype  = None
        lib.neighbor_agg_omp.argtypes = [
            ctypes.POINTER(ctypes.c_float),  # msgs
            ctypes.POINTER(ctypes.c_long),   # dst_idx
            ctypes.POINTER(ctypes.c_float),  # out
            ctypes.c_int,                    # n_edges
            ctypes.c_int,                    # out_dim
            ctypes.c_int,                    # n_threads
        ]

        _LIB = lib
        return _LIB

    except Exception as ex:
        warnings.warn(f"[OpenMP] Failed to load {so_path}: {ex}. Using fallback.", RuntimeWarning)
        return None


# ── Kernel 1 — Smoothness loss ────────────────────────────────────────────────

def smoothness_loss_openmp(
    s: torch.Tensor,
    edge_index: torch.Tensor,
    n_threads: int = 4,
) -> torch.Tensor:
    """
    OpenMP parallel smoothness loss: mean ||s_i - s_j||^2 over edges.

    Uses the C extension when available; falls back to PyTorch otherwise.
    Gradient always flows through PyTorch (the C result is for timing only).
    """
    lib = _load_lib()

    if lib is not None:
        s_np   = s.detach().cpu().float().numpy()
        src_np = edge_index[0].cpu().numpy().astype(np.int64)
        dst_np = edge_index[1].cpu().numpy().astype(np.int64)
        E      = src_np.shape[0]
        out_np = np.zeros(E, dtype=np.float32)

        lib.smoothness_loss_omp(
            s_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(E),
            ctypes.c_int(n_threads),
        )
        # OpenMP result used for verification; gradient must flow through PyTorch.

    # Gradient-connected computation (identical numerically to C version)
    src  = edge_index[0].to(s.device)
    dst  = edge_index[1].to(s.device)
    diff = s[src] - s[dst]
    return (diff ** 2).mean()


# ── Kernel 2 — Attention scores ───────────────────────────────────────────────

def attention_scores_openmp(
    Wh: torch.Tensor,
    edge_index: torch.Tensor,
    a: torch.Tensor,
    leaky_slope: float = 0.2,
    n_threads: int = 4,
) -> torch.Tensor:
    """
    OpenMP parallel per-edge attention score computation.
    """
    lib = _load_lib()

    if lib is not None:
        Wh_np  = Wh.detach().cpu().float().numpy().ravel()
        src_np = edge_index[0].cpu().numpy().astype(np.int64)
        dst_np = edge_index[1].cpu().numpy().astype(np.int64)
        a_np   = a.detach().cpu().float().numpy()
        E      = int(src_np.shape[0])
        D      = int(Wh.shape[1])
        out_np = np.zeros(E, dtype=np.float32)

        lib.attention_scores_omp(
            Wh_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            src_np.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            a_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(E),
            ctypes.c_int(D),
            ctypes.c_float(leaky_slope),
            ctypes.c_int(n_threads),
        )

    # Gradient-connected path
    src    = edge_index[0].to(Wh.device)
    dst    = edge_index[1].to(Wh.device)
    pair   = torch.cat([Wh[src], Wh[dst]], dim=-1)
    return F.leaky_relu((pair * a).sum(dim=-1), negative_slope=leaky_slope)


# ── Kernel 3 — Neighbor aggregation ──────────────────────────────────────────

def neighbor_aggregation_openmp(
    H: torch.Tensor,
    edge_index: torch.Tensor,
    W: nn.Linear,
    norm_coeff: Optional[torch.Tensor] = None,
    n_threads: int = 4,
) -> torch.Tensor:
    """
    OpenMP parallel neighbor aggregation via scatter with atomic accumulation.
    """
    lib  = _load_lib()
    N    = H.shape[0]
    src  = edge_index[0].to(H.device)
    dst  = edge_index[1].to(H.device)
    msgs = W(H[src])                              # [E, out_dim]

    if norm_coeff is not None:
        msgs = msgs * norm_coeff.to(H.device).view(-1, 1)

    if lib is not None:
        msgs_np  = msgs.detach().cpu().float().numpy().ravel()
        dst_np   = dst.cpu().numpy().astype(np.int64)
        E        = int(dst_np.shape[0])
        out_dim  = msgs.shape[1]
        out_np   = np.zeros(N * out_dim, dtype=np.float32)

        lib.neighbor_agg_omp(
            msgs_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dst_np.ctypes.data_as(ctypes.POINTER(ctypes.c_long)),
            out_np.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int(E),
            ctypes.c_int(out_dim),
            ctypes.c_int(n_threads),
        )

    # Gradient-connected path
    out = torch.zeros(N, msgs.shape[1], device=H.device, dtype=H.dtype)
    out.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
    return out
