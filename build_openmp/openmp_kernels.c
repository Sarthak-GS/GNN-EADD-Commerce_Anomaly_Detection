
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
