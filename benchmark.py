"""
benchmark.py  —  Timing comparison for all three parallel kernels.

Measures wall-clock time for each of the three target operations
across three execution modes: sequential, OpenMP (CPU), and CUDA-mode (GPU or CPU).

Output is a formatted table:
    Kernel             | Sequential (ms) | OpenMP (ms) | CUDA-mode (ms) | Speedup (OMP) | Speedup (CUDA)

Usage:
    python benchmark.py
    python benchmark.py --n 500 --runs 20 --n_threads 8
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
import numpy as np
import argparse
import time
import warnings
from pathlib import Path

from kernels.cuda_ops  import (
    smoothness_loss_parallel,
    attention_scores_parallel,
    neighbor_aggregation_parallel,
)
from kernels.openmp_ops import (
    smoothness_loss_openmp,
    attention_scores_openmp,
    neighbor_aggregation_openmp,
)

# Attempt to load the raw CUDA extension for benchmarking
try:
    import gnn_cuda_ext
    HAS_RAW_CUDA = True
except ImportError:
    HAS_RAW_CUDA = False


# ── helpers ────────────────────────────────────────────────────────────────────

def _time_fn(fn, n_runs: int) -> float:
    """Returns mean wall-clock time in milliseconds over n_runs calls."""
    # warm-up
    for _ in range(2):
        fn()
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(times))


def _make_random_graph(N: int, avg_degree: int, device: torch.device):
    """Generates a random edge_index [2, E] with ~N*avg_degree edges."""
    E   = N * avg_degree
    src = torch.randint(0, N, (E,), device=device)
    dst = torch.randint(0, N, (E,), device=device)
    return torch.stack([src, dst], dim=0)   # [2, E]


# ── Kernel 1: Smoothness loss ──────────────────────────────────────────────────

def benchmark_k1(N: int, E_idx: torch.Tensor, n_runs: int, n_threads: int, device: torch.device):
    s    = torch.rand(N, requires_grad=False, device=device)

    # Sequential (dense N x N)
    def seq():
        diff = s.unsqueeze(1) - s.unsqueeze(0)
        return (diff ** 2).mean()

    # CUDA-mode (sparse, thread-per-edge)
    def cuda_mode():
        return smoothness_loss_parallel(s, E_idx)

    # OpenMP (sparse, thread-per-edge via C extension or fallback)
    def omp():
        return smoothness_loss_openmp(s, E_idx, n_threads=n_threads)

    # Raw CUDA (Custom Kernel)
    def raw_cuda():
        if HAS_RAW_CUDA and device.type == 'cuda':
            return gnn_cuda_ext.smoothness_loss(s, E_idx)
        return torch.tensor(0.0)

    t_seq  = _time_fn(seq,       n_runs)
    t_cuda = _time_fn(cuda_mode, n_runs)
    t_omp  = _time_fn(omp,       n_runs)
    t_raw  = _time_fn(raw_cuda,  n_runs) if HAS_RAW_CUDA and device.type == 'cuda' else 0.0
    return t_seq, t_omp, t_cuda, t_raw


# ── Kernel 2: Per-edge attention scores ───────────────────────────────────────

def benchmark_k2(N: int, D: int, E_idx: torch.Tensor, n_runs: int, n_threads: int, device: torch.device):
    Wh = torch.rand(N, D, device=device)
    a  = torch.rand(2 * D, device=device)

    # Sequential (dense N x N x 2D broadcast)
    def seq():
        N_ = Wh.shape[0]
        Wh_i = Wh.unsqueeze(1).expand(N_, N_, D)
        Wh_j = Wh.unsqueeze(0).expand(N_, N_, D)
        pair = torch.cat([Wh_i, Wh_j], dim=-1)
        return torch.nn.functional.leaky_relu((pair * a).sum(dim=-1), 0.2)

    # CUDA-mode (sparse)
    def cuda_mode():
        return attention_scores_parallel(Wh, E_idx, a)

    # OpenMP (sparse)
    def omp():
        return attention_scores_openmp(Wh, E_idx, a, n_threads=n_threads)

    # Raw CUDA (Custom Kernel)
    def raw_cuda():
        if HAS_RAW_CUDA and device.type == 'cuda':
            return gnn_cuda_ext.attention_scores(Wh, E_idx, a, 0.2)
        return torch.tensor(0.0)

    t_seq  = _time_fn(seq,       n_runs)
    t_cuda = _time_fn(cuda_mode, n_runs)
    t_omp  = _time_fn(omp,       n_runs)
    t_raw  = _time_fn(raw_cuda,  n_runs) if HAS_RAW_CUDA and device.type == 'cuda' else 0.0
    return t_seq, t_omp, t_cuda, t_raw


# ── Kernel 3: Neighbor aggregation ────────────────────────────────────────────

def benchmark_k3(N: int, in_dim: int, out_dim: int, E_idx: torch.Tensor,
                 n_runs: int, n_threads: int, device: torch.device):
    H = torch.rand(N, in_dim, device=device)
    W = nn.Linear(in_dim, out_dim, bias=False).to(device)

    # Sequential (dense matmul: A_norm @ H, then W)
    A_dense = torch.zeros(N, N, device=device)
    A_dense[E_idx[0], E_idx[1]] = 1.0 / (E_idx.shape[1] ** 0.5)

    def seq():
        msg = A_dense @ H
        return W(msg)

    # CUDA-mode (sparse scatter)
    def cuda_mode():
        return neighbor_aggregation_parallel(H, E_idx, W)

    # OpenMP (sparse scatter via C extension or fallback)
    def omp():
        return neighbor_aggregation_openmp(H, E_idx, W, n_threads=n_threads)

    # Raw CUDA (Custom Kernel)
    def raw_cuda():
        if HAS_RAW_CUDA and device.type == 'cuda':
            msgs = W(H[E_idx[0]])
            return gnn_cuda_ext.neighbor_agg(msgs, E_idx[1], N)
        return torch.tensor(0.0)

    t_seq  = _time_fn(seq,       n_runs)
    t_cuda = _time_fn(cuda_mode, n_runs)
    t_omp  = _time_fn(omp,       n_runs)
    t_raw  = _time_fn(raw_cuda,  n_runs) if HAS_RAW_CUDA and device.type == 'cuda' else 0.0
    return t_seq, t_omp, t_cuda, t_raw


# ── Main ───────────────────────────────────────────────────────────────────────

def run_benchmark(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[Benchmark] Device: {device}")
    print(f"[Benchmark] N={args.n}, avg_degree={args.avg_degree}, "
          f"runs={args.runs}, threads={args.n_threads}\n")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        E_idx = _make_random_graph(args.n, args.avg_degree, device)
        E     = E_idx.shape[1]
        D     = 64    # out_dim for attention / aggregation

    print(f"  Graph stats:  N={args.n} nodes,  E={E} edges,  "
          f"density={E / (args.n ** 2):.5f}\n")

    rows = []

    # Kernel 1
    t_seq, t_omp, t_cuda, t_raw = benchmark_k1(args.n, E_idx, args.runs, args.n_threads, device)
    rows.append(("K1  Smoothness Loss",  t_seq, t_omp, t_cuda, t_raw))

    # Kernel 2
    t_seq, t_omp, t_cuda, t_raw = benchmark_k2(args.n, D, E_idx, args.runs, args.n_threads, device)
    rows.append(("K2  Attention Scores", t_seq, t_omp, t_cuda, t_raw))

    # Kernel 3
    t_seq, t_omp, t_cuda, t_raw = benchmark_k3(args.n, D, D, E_idx, args.runs, args.n_threads, device)
    rows.append(("K3  Neighbor Agg.",    t_seq, t_omp, t_cuda, t_raw))

    # ── Print table ────────────────────────────────────────────────────────────
    W_name = 25
    col_w  = 16
    hdr    = (f"  {'Kernel':<{W_name}}  "
              f"{'Seq (ms)':>{col_w-4}}  "
              f"{'OMP (ms)':>{col_w-4}}  "
              f"{'PyTorch (ms)':>{col_w-2}}  "
              f"{'Raw CUDA (ms)':>{col_w-2}}  "
              f"{'Speedup':>{col_w-4}}")
    sep    = "  " + "-" * (W_name + 5 * (col_w + 0))
    print(hdr)
    print(sep)
    for name, t_seq, t_omp, t_pyg, t_raw in rows:
        # Show best speedup vs sequential
        best_t = min(t_omp, t_pyg)
        if HAS_RAW_CUDA and t_raw > 0:
            best_t = min(best_t, t_raw)
        
        speedup = t_seq / max(best_t, 1e-9)
        
        print(f"  {name:<{W_name}}  "
              f"{t_seq:>{col_w-4}.3f}  "
              f"{t_omp:>{col_w-4}.3f}  "
              f"{t_pyg:>{col_w-2}.3f}  "
              f"{t_raw:>{col_w-2}.3f}  "
              f"{speedup:>{col_w-4}.2f}x")
    print(sep)
    print(f"\n  Note: Speedup is relative to sequential baseline. "
          f"Raw CUDA is available: {HAS_RAW_CUDA}\n")


def parse_args():
    p = argparse.ArgumentParser(description="GNN-EADD Kernel Benchmark")
    p.add_argument('--n',          type=int, default=300,
                   help='Number of graph nodes')
    p.add_argument('--avg_degree', type=int, default=10,
                   help='Average node degree for random test graph')
    p.add_argument('--runs',       type=int, default=10,
                   help='Number of timing repetitions')
    p.add_argument('--n_threads',  type=int, default=4,
                   help='OpenMP thread count')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(args)
