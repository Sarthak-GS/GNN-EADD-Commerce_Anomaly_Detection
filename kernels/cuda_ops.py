"""
cuda_ops.py  —  Thread-per-edge parallel kernels for GNN-EADD.

Implements the three target operations from the paper using sparse,
edge-indexed computation instead of dense N x N matrix ops.

When CUDA is available the tensors live on GPU and these run on GPU.
When CUDA is unavailable they fall back to CPU with identical code.

Kernel 1 — Smoothness loss  L_unsup = sum_{(i,j) in E} ||s_i - s_j||^2
Kernel 2 — Per-edge attention scores  e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
Kernel 3 — Neighbor aggregation   out_v = sum_{u in N(v)} coeff * W * h_u

The thread-per-edge model: each 'thread' handles exactly one edge entry,
all edges computed concurrently — no sequential loop over N x N pairs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# Attempt to load the raw CUDA extension
try:
    import gnn_cuda_ext
    _HAS_CUDA_EXT = True
except ImportError:
    _HAS_CUDA_EXT = False


def smoothness_loss_parallel(
    s: torch.Tensor,           # [N]   anomaly scores (on device)
    edge_index: torch.Tensor,  # [2,E] edge list (src, dst)
) -> torch.Tensor:
    """
    Kernel 1: Thread-per-edge squared score difference, then mean reduction.

    Replaces the dense  (A * (s_i - s_j)^2).mean()  which allocates an
    N x N matrix.  Here we only iterate over existing edges — O(E) not O(N^2).

    CUDA mapping:
        blockDim.x = 256
        gridDim.x  = ceil(E / 256)
        thread tid = blockIdx.x * blockDim.x + threadIdx.x
        if tid < E:
            diff = s[src[tid]] - s[dst[tid]]
            out[tid] = diff * diff
        then reduce out[] to scalar
    """
    if _HAS_CUDA_EXT and s.is_cuda:
        return gnn_cuda_ext.smoothness_loss(s, edge_index)

    src, dst = edge_index[0], edge_index[1]   # [E] each
    diff = s[src] - s[dst]                    # [E]  — one element per edge
    return (diff ** 2).mean()


def attention_scores_parallel(
    Wh: torch.Tensor,           # [N, out_dim]  linearly-transformed node features
    edge_index: torch.Tensor,   # [2, E]
    a: torch.Tensor,            # [2 * out_dim]  attention parameter vector
    leaky_slope: float = 0.2,
) -> torch.Tensor:
    """
    Kernel 2: Thread-per-edge raw attention score.

    Replaces the dense [N, N, 2*out_dim] broadcast that allocates
    O(N^2 * d) memory with a sparse [E, 2*out_dim] computation.

    CUDA mapping:
        tid = blockIdx.x * blockDim.x + threadIdx.x
        if tid < E:
            pair = concat( Wh[src[tid]], Wh[dst[tid]] )  // 2*out_dim values
            e[tid] = LeakyReLU( dot(a, pair) )
    """
    if _HAS_CUDA_EXT and Wh.is_cuda:
        return gnn_cuda_ext.attention_scores(Wh, edge_index, a, leaky_slope)

    src, dst = edge_index[0], edge_index[1]
    Wh_src = Wh[src]                              # [E, out_dim]
    Wh_dst = Wh[dst]                              # [E, out_dim]
    pair   = torch.cat([Wh_src, Wh_dst], dim=-1)  # [E, 2*out_dim]
    e      = F.leaky_relu(
        (pair * a).sum(dim=-1),                   # [E]
        negative_slope=leaky_slope,
    )
    return e                                      # [E] raw attention logits


def neighbor_aggregation_parallel(
    H: torch.Tensor,            # [N, in_dim]   node feature matrix
    edge_index: torch.Tensor,   # [2, E]        src -> dst
    W: nn.Linear,               # weight matrix W ∈ R^{in_dim x out_dim}
    norm_coeff: torch.Tensor = None,  # [E] per-edge normalization weight
) -> torch.Tensor:
    """
    Kernel 3: Thread-per-edge scatter-based neighbor aggregation.

    Replaces  A_norm @ H  (dense matmul) with an explicit edge loop
    using scatter_add — identical to how CUDA atomics accumulate one
    edge contribution per thread into the destination node's output slot.

    CUDA mapping:
        tid = blockIdx.x * blockDim.x + threadIdx.x
        if tid < E:
            msg = W * H[src[tid]] * coeff[tid]   // transform + scale
            atomicAdd( &out[dst[tid]], msg )      // safe concurrent write
    """
    N = H.shape[0]
    if _HAS_CUDA_EXT and H.is_cuda:
        # Pre-transform messages
        msgs = W(H[edge_index[0]])
        if norm_coeff is not None:
            msgs = msgs * norm_coeff.view(-1, 1)
        return gnn_cuda_ext.neighbor_agg(msgs, edge_index[1], N)

    src, dst = edge_index[0], edge_index[1]  # [E] each

    # Apply weight transform on source features — [E, out_dim]
    msgs = W(H[src])

    # Apply per-edge normalization coefficient
    if norm_coeff is not None:
        msgs = msgs * norm_coeff.view(-1, 1)   # [E, out_dim]

    out_dim = msgs.shape[1]
    out = torch.zeros(N, out_dim, device=H.device, dtype=H.dtype)

    # scatter_add: out[dst[e]] += msgs[e]  for all e in 0..E-1
    # In CUDA: each thread does one atomic accumulation
    out.scatter_add_(0, dst.unsqueeze(1).expand_as(msgs), msgs)
    return out                                 # [N, out_dim]
