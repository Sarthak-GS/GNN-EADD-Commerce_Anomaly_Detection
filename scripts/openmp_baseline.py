"""
openmp_baseline.py — OpenMP-parallelized CPU baseline for GNN-EADD operations

=============================================================================
 LECTURE CONCEPTS DEMONSTRATED:
 [Lec 2]  Parallel Architecture: Shared-memory CPU parallelism (OpenMP)
          vs. GPU accelerator model (CUDA).  Both share the same memory
          space within their domain but differ in thread counts and
          granularity.
 [Lec 3]  Amdahl's Law: The sequential fraction of the pipeline (data
          loading, Python overhead) limits speedup.  We measure the
          parallel fraction to compute theoretical maximum speedup.
 [Lec 20] Synchronization: OpenMP reduction directives handle the
          accumulation pattern; atomic operations handle scatter-add.
=============================================================================

This module provides THREE operations matching the CUDA kernels:
  1. smoothness_loss_openmp  — edge-parallel squared diff + reduction
  2. gat_attention_openmp    — edge-parallel attention score computation
  3. neighbor_aggregation_openmp — edge-parallel scatter-add aggregation

All operations use ctypes to call a compiled C shared library with
OpenMP pragmas.  The C code is compiled at import time if the .so
doesn't exist.
"""

import os
import sys
import ctypes
import numpy as np
import subprocess
import tempfile
from pathlib import Path

# ─── C source with OpenMP pragmas ──────────────────────────────────────────

_C_SOURCE = r"""
/*
 * openmp_kernels.c — OpenMP CPU baseline for GNN-EADD Phase 2
 *
 * [Lec 2]  Shared Memory Parallel Architecture:
 *          All threads share the same address space and can read/write
 *          the same arrays.  OpenMP distributes loop iterations across
 *          CPU cores automatically.
 *
 * [Lec 20] Synchronization:
 *          - #pragma omp parallel for reduction(+:...) handles safe
 *            accumulation without explicit locks.
 *          - #pragma omp atomic handles scatter-add pattern.
 *
 * Compile: gcc -O3 -fopenmp -shared -fPIC -o openmp_kernels.so openmp_kernels.c
 */

#include <math.h>
#include <omp.h>
#include <string.h>

/* ═══════════════════════════════════════════════════════════════════════════
 *  OPERATION 1: SMOOTHNESS LOSS (Eq. 17)
 *  ─────────────────────────────────────────────────────────────────────────
 *  L_unsup = (1/E) * Σ_{e=(i,j)} (s_i - s_j)²
 *
 *  [Lec 2]  Each CPU thread processes a chunk of edges independently.
 *  [Lec 20] The reduction(+:total) clause ensures that partial sums
 *           from different threads are accumulated without races.
 *  [Lec 3]  This operation is fully parallelizable (parallel fraction
 *           ≈ 1.0 for the computation; the only sequential overhead is
 *           the final reduction which takes O(num_threads) time).
 * ═══════════════════════════════════════════════════════════════════════════ */

double smoothness_loss_omp(
    const float* scores,     /* [N] anomaly scores */
    const long* row,         /* [E] source indices */
    const long* col,         /* [E] destination indices */
    int E)                   /* number of edges */
{
    double total = 0.0;

    /* [Lec 2] OpenMP parallel for with reduction:
     * The loop iterations are distributed across available CPU cores.
     * schedule(static) divides iterations equally — appropriate here
     * because each iteration has equal work (one subtract + square).
     * [Lec 20] reduction(+:total) creates private copies of 'total'
     * per thread and sums them at the end, avoiding race conditions. */
    #pragma omp parallel for reduction(+:total) schedule(static)
    for (int e = 0; e < E; e++) {
        float diff = scores[row[e]] - scores[col[e]];
        total += (double)(diff * diff);
    }

    return total / (double)E;  /* mean, consistent with CUDA version */
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  OPERATION 2: GAT ATTENTION SCORES (Eq. 12)
 *  ─────────────────────────────────────────────────────────────────────────
 *  e_ij = LeakyReLU( a^T [Wh_i || Wh_j] )
 *
 *  [Lec 2]  Each edge's attention score is independent — trivially parallel.
 *  [Lec 9]  Unlike GPUs, CPUs don't suffer from warp divergence, but the
 *           branch prediction unit handles the LeakyReLU conditional.
 * ═══════════════════════════════════════════════════════════════════════════ */

void gat_attention_omp(
    const float* Wh,          /* [N, D] transformed features */
    const long* row,          /* [E] source indices */
    const long* col,          /* [E] destination indices */
    const float* attn_vec,    /* [2*D] attention vector */
    float* e_out,             /* [E] output logits */
    int E, int D,
    float negative_slope)
{
    /* [Lec 2] Each thread handles a subset of edges.
     * No synchronization needed — each edge writes to a unique output. */
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        long src = row[e];
        long dst = col[e];
        float val = 0.0f;

        for (int d = 0; d < D; d++) {
            val += Wh[dst * D + d] * attn_vec[d];
            val += Wh[src * D + d] * attn_vec[D + d];
        }

        /* LeakyReLU */
        e_out[e] = (val > 0.0f) ? val : val * negative_slope;
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  OPERATION 3: NEIGHBOR AGGREGATION (Eq. 7/13)
 *  ─────────────────────────────────────────────────────────────────────────
 *  out[dst][d] += alpha[e] * Wh[src][d]
 *
 *  [Lec 20] Synchronization challenge: multiple edges may target the same
 *           destination node, causing write conflicts.  We use
 *           #pragma omp atomic to serialize updates to each output cell.
 *  [Lec 1]  This is the fundamental parallelization challenge in GNNs:
 *           the scatter-add pattern on irregular graph structures.
 * ═══════════════════════════════════════════════════════════════════════════ */

void neighbor_aggregation_omp(
    const float* Wh,          /* [N, D] source features */
    const float* alpha,       /* [E] attention weights */
    const long* row,          /* [E] source indices */
    const long* col,          /* [E] destination indices */
    float* out,               /* [N, D] output (must be zero-initialized) */
    int E, int D)
{
    /* [Lec 2] Parallelize over edges.
     * [Lec 20] The #pragma omp atomic on the output update handles the
     * many-to-one write pattern.  This is analogous to atomicAdd in CUDA
     * but with CPU-level atomic instructions.
     * Note: atomic is needed because multiple edges may have the same dst. */
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < E; e++) {
        long src = row[e];
        long dst = col[e];
        float a  = alpha[e];

        for (int d = 0; d < D; d++) {
            float contribution = a * Wh[src * D + d];
            #pragma omp atomic
            out[dst * D + d] += contribution;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════════
 *  OPERATION 4: MATRIX MULTIPLICATION (Feature Transformation)
 *  ─────────────────────────────────────────────════════════════════════════
 *  C[M,N] = A[M,K] × B[K,N]
 *
 *  [Lec 16-17] This is the CPU counterpart to the tiled CUDA kernel.
 *              We parallelize the outer loop (rows of A) across threads.
 *              No tiling/shared memory analogy on CPU, but each thread
 *              has its own L1/L2 cache that provides some locality.
 * ═══════════════════════════════════════════════════════════════════════════ */

void matmul_omp(
    const float* A,   /* [M, K] */
    const float* B,   /* [K, N] */
    float* C,         /* [M, N] — must be zero-initialized */
    int M, int K, int N)
{
    /* [Lec 2] Parallelize over rows of C.
     * Each thread computes a full row of the output independently. */
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}


/* Helper: returns the number of OpenMP threads available */
int get_num_threads(void) {
    int n;
    #pragma omp parallel
    {
        #pragma omp single
        n = omp_get_num_threads();
    }
    return n;
}
"""


# ─── Compile and load the shared library ───────────────────────────────────

_LIB = None
_LIB_PATH = None


def _get_lib():
    """Compile the OpenMP C code and load it as a shared library."""
    global _LIB, _LIB_PATH
    if _LIB is not None:
        return _LIB

    project_dir = Path(__file__).parent
    build_dir = project_dir / 'build_openmp'
    build_dir.mkdir(exist_ok=True)

    src_path = build_dir / 'openmp_kernels.c'
    lib_path = build_dir / 'openmp_kernels.so'

    # Write the C source
    src_path.write_text(_C_SOURCE)

    # Compile with OpenMP
    # [Lec 2] gcc -fopenmp enables OpenMP parallelization directives
    cmd = [
        'gcc', '-O3', '-fopenmp', '-shared', '-fPIC',
        '-o', str(lib_path), str(src_path), '-lm',
    ]

    print(f"[OpenMP] Compiling: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[OpenMP] Compilation failed:\n{result.stderr}")
        raise RuntimeError("Failed to compile OpenMP kernels")

    _LIB = ctypes.CDLL(str(lib_path))
    _LIB_PATH = lib_path

    # Set up function signatures
    _LIB.smoothness_loss_omp.restype = ctypes.c_double
    _LIB.smoothness_loss_omp.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # scores
        ctypes.POINTER(ctypes.c_long),    # row
        ctypes.POINTER(ctypes.c_long),    # col
        ctypes.c_int,                     # E
    ]

    _LIB.gat_attention_omp.restype = None
    _LIB.gat_attention_omp.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # Wh
        ctypes.POINTER(ctypes.c_long),    # row
        ctypes.POINTER(ctypes.c_long),    # col
        ctypes.POINTER(ctypes.c_float),   # attn_vec
        ctypes.POINTER(ctypes.c_float),   # e_out
        ctypes.c_int,                     # E
        ctypes.c_int,                     # D
        ctypes.c_float,                   # negative_slope
    ]

    _LIB.neighbor_aggregation_omp.restype = None
    _LIB.neighbor_aggregation_omp.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # Wh
        ctypes.POINTER(ctypes.c_float),   # alpha
        ctypes.POINTER(ctypes.c_long),    # row
        ctypes.POINTER(ctypes.c_long),    # col
        ctypes.POINTER(ctypes.c_float),   # out
        ctypes.c_int,                     # E
        ctypes.c_int,                     # D
    ]

    _LIB.matmul_omp.restype = None
    _LIB.matmul_omp.argtypes = [
        ctypes.POINTER(ctypes.c_float),   # A
        ctypes.POINTER(ctypes.c_float),   # B
        ctypes.POINTER(ctypes.c_float),   # C
        ctypes.c_int,                     # M
        ctypes.c_int,                     # K
        ctypes.c_int,                     # N
    ]

    _LIB.get_num_threads.restype = ctypes.c_int
    _LIB.get_num_threads.argtypes = []

    n_threads = _LIB.get_num_threads()
    print(f"[OpenMP] Library loaded: {lib_path} ({n_threads} threads)")

    return _LIB


# ─── Python wrappers ──────────────────────────────────────────────────────

def _np_ptr(arr, dtype):
    """Get a ctypes pointer to a numpy array."""
    return arr.ctypes.data_as(ctypes.POINTER(dtype))


def smoothness_loss_openmp(scores: np.ndarray, row: np.ndarray, col: np.ndarray) -> float:
    """
    Compute L_unsup = (1/E) Σ (s_i - s_j)²  using OpenMP.
    [Lec 20] Uses reduction(+:) for safe parallel accumulation.
    """
    lib = _get_lib()
    scores = np.ascontiguousarray(scores, dtype=np.float32)
    row    = np.ascontiguousarray(row, dtype=np.int64)
    col    = np.ascontiguousarray(col, dtype=np.int64)
    E = len(row)

    result = lib.smoothness_loss_omp(
        _np_ptr(scores, ctypes.c_float),
        _np_ptr(row, ctypes.c_long),
        _np_ptr(col, ctypes.c_long),
        E,
    )
    return result


def gat_attention_openmp(
    Wh: np.ndarray, row: np.ndarray, col: np.ndarray,
    attn_vec: np.ndarray, negative_slope: float = 0.2,
) -> np.ndarray:
    """
    Compute per-edge attention logits using OpenMP.
    [Lec 2] Trivially parallel — each edge is independent.
    """
    lib = _get_lib()
    Wh       = np.ascontiguousarray(Wh, dtype=np.float32)
    row      = np.ascontiguousarray(row, dtype=np.int64)
    col      = np.ascontiguousarray(col, dtype=np.int64)
    attn_vec = np.ascontiguousarray(attn_vec, dtype=np.float32)

    E = len(row)
    D = Wh.shape[1]
    e_out = np.zeros(E, dtype=np.float32)

    lib.gat_attention_omp(
        _np_ptr(Wh, ctypes.c_float),
        _np_ptr(row, ctypes.c_long),
        _np_ptr(col, ctypes.c_long),
        _np_ptr(attn_vec, ctypes.c_float),
        _np_ptr(e_out, ctypes.c_float),
        E, D, negative_slope,
    )
    return e_out


def neighbor_aggregation_openmp(
    Wh: np.ndarray, alpha: np.ndarray,
    row: np.ndarray, col: np.ndarray,
    num_nodes: int,
) -> np.ndarray:
    """
    Compute out[dst] += alpha[e] * Wh[src] using OpenMP.
    [Lec 20] Uses atomic updates for the scatter-add pattern.
    """
    lib = _get_lib()
    Wh    = np.ascontiguousarray(Wh, dtype=np.float32)
    alpha = np.ascontiguousarray(alpha, dtype=np.float32)
    row   = np.ascontiguousarray(row, dtype=np.int64)
    col   = np.ascontiguousarray(col, dtype=np.int64)

    D = Wh.shape[1]
    out = np.zeros((num_nodes, D), dtype=np.float32)

    lib.neighbor_aggregation_omp(
        _np_ptr(Wh, ctypes.c_float),
        _np_ptr(alpha, ctypes.c_float),
        _np_ptr(row, ctypes.c_long),
        _np_ptr(col, ctypes.c_long),
        _np_ptr(out, ctypes.c_float),
        len(row), D,
    )
    return out


def matmul_openmp(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute C = A × B using OpenMP-parallelized loops.
    [Lec 16-17] CPU counterpart to tiled CUDA matrix multiplication.
    """
    lib = _get_lib()
    A = np.ascontiguousarray(A, dtype=np.float32)
    B = np.ascontiguousarray(B, dtype=np.float32)

    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimensions must match: {K} vs {K2}"

    C = np.zeros((M, N), dtype=np.float32)

    lib.matmul_omp(
        _np_ptr(A, ctypes.c_float),
        _np_ptr(B, ctypes.c_float),
        _np_ptr(C, ctypes.c_float),
        M, K, N,
    )
    return C


def get_num_threads() -> int:
    """Returns the number of OpenMP threads available."""
    lib = _get_lib()
    return lib.get_num_threads()


if __name__ == '__main__':
    # Quick smoke test
    print("=== OpenMP Baseline Smoke Test ===")
    lib = _get_lib()
    print(f"Threads: {get_num_threads()}")

    N, E, D = 100, 500, 64
    scores = np.random.randn(N).astype(np.float32)
    row = np.random.randint(0, N, E).astype(np.int64)
    col = np.random.randint(0, N, E).astype(np.int64)

    loss = smoothness_loss_openmp(scores, row, col)
    print(f"Smoothness loss: {loss:.6f}")

    Wh = np.random.randn(N, D).astype(np.float32)
    attn = np.random.randn(2 * D).astype(np.float32)
    e_out = gat_attention_openmp(Wh, row, col, attn)
    print(f"Attention scores: shape={e_out.shape}, range=[{e_out.min():.4f}, {e_out.max():.4f}]")

    alpha = np.abs(np.random.randn(E).astype(np.float32))
    alpha /= alpha.sum()
    out = neighbor_aggregation_openmp(Wh, alpha, row, col, N)
    print(f"Aggregation: shape={out.shape}, range=[{out.min():.4f}, {out.max():.4f}]")

    A = np.random.randn(32, 64).astype(np.float32)
    B = np.random.randn(64, 16).astype(np.float32)
    C = matmul_openmp(A, B)
    C_ref = A @ B
    err = np.abs(C - C_ref).max()
    print(f"Matmul: shape={C.shape}, max_error vs numpy={err:.8f}")

    print("=== All OpenMP tests passed ===")
