/*
 * openmp_kernels.c  —  OpenMP CPU parallel kernels for GNN-EADD.
 *
 * Three kernels matching the three CUDA ops:
 *   Kernel 1: smoothness_loss_omp   — per-edge ||s_i - s_j||^2
 *   Kernel 2: attention_scores_omp  — per-edge e_ij = LeakyReLU(a^T [Wh_i | Wh_j])
 *   Kernel 3: neighbor_agg_omp      — scatter accumulation into destination nodes
 *
 * Compile:
 *   gcc -O3 -fopenmp -shared -fPIC -o openmp_ext.so openmp_kernels.c -lm
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>


/* ─────────────────────────────────────────────────────────────────────────── */
/* Kernel 1: Smoothness loss — parallel over edges                            */
/* ─────────────────────────────────────────────────────────────────────────── */

void smoothness_loss_omp(
    const float* scores,    /* [N]  node anomaly scores            */
    const long*  src_idx,   /* [E]  source node indices            */
    const long*  dst_idx,   /* [E]  destination node indices       */
    float*       out,       /* [E]  per-edge squared diff (output) */
    int          n_edges,
    int          n_threads
) {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_edges; e++) {
        float diff = scores[src_idx[e]] - scores[dst_idx[e]];
        out[e] = diff * diff;
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Kernel 2: Attention scores — parallel over edges                           */
/* ─────────────────────────────────────────────────────────────────────────── */

void attention_scores_omp(
    const float* Wh,         /* [N * out_dim]  flattened transformed features */
    const long*  src_idx,    /* [E]  source indices                           */
    const long*  dst_idx,    /* [E]  destination indices                      */
    const float* a,          /* [2 * out_dim]  attention parameter vector     */
    float*       out,        /* [E]  raw attention logits (output)            */
    int          n_edges,
    int          out_dim,
    float        leaky_slope,
    int          n_threads
) {
    omp_set_num_threads(n_threads);
    #pragma omp parallel for schedule(static)
    for (int e = 0; e < n_edges; e++) {
        long src = src_idx[e];
        long dst = dst_idx[e];
        float sum = 0.0f;
        /* dot product: a^T [Wh_src || Wh_dst] */
        for (int d = 0; d < out_dim; d++) {
            sum += Wh[src * out_dim + d] * a[d];
            sum += Wh[dst * out_dim + d] * a[out_dim + d];
        }
        /* LeakyReLU */
        out[e] = (sum > 0.0f) ? sum : sum * leaky_slope;
    }
}


/* ─────────────────────────────────────────────────────────────────────────── */
/* Kernel 3: Neighbor aggregation — parallel scatter with atomics             */
/* ─────────────────────────────────────────────────────────────────────────── */

void neighbor_agg_omp(
    const float* msgs,       /* [E * out_dim]  pre-transformed per-edge msgs */
    const long*  dst_idx,    /* [E]  destination node indices                 */
    float*       out,        /* [N * out_dim]  output (caller must zero-init) */
    int          n_edges,
    int          out_dim,
    int          n_threads
) {
    omp_set_num_threads(n_threads);
    /* Parallel over edges; many threads may write to same dst node
       — use OpenMP atomic to prevent data races (mirrors CUDA atomicAdd). */
    #pragma omp parallel for schedule(dynamic, 64)
    for (int e = 0; e < n_edges; e++) {
        long dst = dst_idx[e];
        for (int d = 0; d < out_dim; d++) {
            #pragma omp atomic
            out[dst * out_dim + d] += msgs[e * out_dim + d];
        }
    }
}
